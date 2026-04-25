use std::ptr::NonNull;

use serde::{Deserialize, Serialize};

use crate::error::{Result, TransferError};
use crate::rc_backend::{GetOrPrepareResult, RcBackend};

/// RDMA operation type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransferOp {
    Read,
    Write,
}

/// A memory region to register for RDMA access.
#[derive(Clone, Copy, Debug)]
pub struct MemoryRegion {
    pub ptr: NonNull<u8>,
    pub len: usize,
}

/// A single RDMA transfer descriptor.
#[derive(Clone, Copy, Debug)]
pub struct TransferDesc {
    pub local_ptr: NonNull<u8>,
    pub remote_ptr: NonNull<u8>,
    pub len: usize,
}

/// Receiver-side RDMA WRITE-with-immediate completion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImmCompletion {
    pub nic_idx: usize,
    pub local_qpn: u32,
    pub imm_data: u32,
}

pub type ImmCompletionReceiver = mea::mpsc::UnboundedReceiver<ImmCompletion>;

/// RC queue pair endpoint info exchanged during handshake.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct RcEndpoint {
    pub(crate) gid: [u8; 16],
    pub(crate) lid: u16,
    pub(crate) qp_num: u32,
    pub(crate) psn: u32,
}

/// A registered memory region descriptor (base pointer, length, remote key).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct RegisteredMemoryRegion {
    pub(crate) base_ptr: u64,
    pub(crate) len: u64,
    pub(crate) rkey: u32,
}

/// Per-NIC handshake data: endpoint + memory region snapshot.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct NicHandshake {
    pub(crate) endpoint: RcEndpoint,
    pub(crate) memory_regions: Vec<RegisteredMemoryRegion>,
    pub(crate) signal_region: RegisteredMemoryRegion,
}

/// Opaque handshake metadata exchanged between peers via gRPC.
///
/// Contains one [`NicHandshake`] per NIC. NICs are 1:1 mapped by index
/// between two machines (mlx5_0↔mlx5_0, etc.).
#[derive(Clone, Debug)]
pub struct HandshakeMetadata {
    pub(crate) nics: Vec<NicHandshake>,
}

#[derive(Serialize, Deserialize)]
struct WireHandshakeMetadata {
    nics: Vec<NicHandshake>,
}

impl HandshakeMetadata {
    pub fn to_bytes(&self) -> Vec<u8> {
        let wire = WireHandshakeMetadata {
            nics: self.nics.clone(),
        };
        bincode::serde::encode_to_vec(&wire, bincode::config::standard())
            .expect("handshake metadata serialization should not fail")
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let (wire, consumed) = bincode::serde::decode_from_slice::<WireHandshakeMetadata, _>(
            bytes,
            bincode::config::standard(),
        )
        .map_err(|_| TransferError::InvalidArgument("invalid handshake metadata"))?;
        if consumed != bytes.len() {
            return Err(TransferError::InvalidArgument(
                "trailing bytes in handshake metadata",
            ));
        }
        Ok(Self { nics: wire.nics })
    }
}

/// Connection status for a remote peer.
pub enum ConnectionStatus {
    /// Already connected; call batch_transfer_async directly.
    Existing,
    /// A handshake to this peer is already in progress.
    Connecting,
    /// Not connected. Exchange this local metadata with remote peer via gRPC,
    /// then call complete_handshake. On failure, call abort_handshake.
    Prepared(HandshakeMetadata),
}

pub struct TransferEngine {
    backend: RcBackend,
}

impl TransferEngine {
    pub fn new(nic_names: &[String]) -> Result<Self> {
        Ok(Self {
            backend: RcBackend::new(nic_names)?,
        })
    }

    pub fn register_memory(&self, regions: &[MemoryRegion]) -> Result<()> {
        for region in regions {
            self.backend.register_memory(region.ptr, region.len)?;
        }
        Ok(())
    }

    pub fn unregister_memory(&self, ptrs: &[NonNull<u8>]) -> Result<()> {
        for &ptr in ptrs {
            self.backend.unregister_memory(ptr)?;
        }
        Ok(())
    }

    /// Check if already connected to the remote peer; if not, prepare local
    /// QPs and return the handshake metadata that must be exchanged via gRPC.
    pub fn get_or_prepare(&self, remote_addr: &str) -> Result<ConnectionStatus> {
        match self.backend.get_or_prepare(remote_addr)? {
            GetOrPrepareResult::Existing => Ok(ConnectionStatus::Existing),
            GetOrPrepareResult::AlreadyConnecting => Ok(ConnectionStatus::Connecting),
            GetOrPrepareResult::NeedHandshake(nics) => {
                Ok(ConnectionStatus::Prepared(HandshakeMetadata { nics }))
            }
        }
    }

    /// Complete a connection after exchanging handshake metadata with the peer.
    pub fn complete_handshake(
        &self,
        remote_addr: &str,
        local_meta: &HandshakeMetadata,
        remote_meta: &HandshakeMetadata,
    ) -> Result<()> {
        self.backend
            .complete_handshake_for(remote_addr, local_meta.nics.clone(), &remote_meta.nics)
    }

    /// Drop pending sessions created by get_or_prepare when handshake failed.
    pub fn abort_handshake(&self, remote_addr: &str, local_meta: &HandshakeMetadata) {
        self.backend.abort_handshake(remote_addr, &local_meta.nics);
    }

    /// Return cached local handshake metadata for an established connection.
    pub fn local_meta_for(&self, remote_addr: &str) -> Option<HandshakeMetadata> {
        self.backend
            .local_meta_for_addr(remote_addr)
            .map(|nics| HandshakeMetadata { nics })
    }

    /// Remove cached connection state on transfer failure.
    pub fn invalidate_connection(&self, remote_addr: &str) {
        self.backend.invalidate_connection(remote_addr);
    }

    /// Return the runtime NIC index closest to a CUDA GPU, if topology is known.
    pub fn preferred_nic_index_for_gpu(&self, device_id: u32) -> Option<usize> {
        self.backend.preferred_nic_index_for_gpu(device_id)
    }

    /// Submit a batch of RDMA READ or WRITE operations against a connected peer.
    ///
    /// Ops are NUMA-aware distributed across NICs. Returns one receiver per
    /// active NIC; each yields the bytes transferred on that NIC.
    ///
    /// The connection must be established via `get_or_prepare` +
    /// `complete_handshake` before calling this method.
    pub fn batch_transfer_async(
        &self,
        op: TransferOp,
        remote_addr: &str,
        descs: &[TransferDesc],
    ) -> Result<Vec<mea::oneshot::Receiver<Result<usize>>>> {
        self.backend.batch_transfer_async(op, remote_addr, descs)
    }

    /// Submit a batch and keep the NIC index attached to each completion.
    pub fn batch_transfer_async_with_nic(
        &self,
        op: TransferOp,
        remote_addr: &str,
        descs: &[TransferDesc],
    ) -> Result<Vec<(usize, mea::oneshot::Receiver<Result<usize>>)>> {
        self.backend
            .batch_transfer_async_with_nic(op, remote_addr, descs)
    }

    /// Submit a batch on one explicit runtime NIC index.
    pub fn batch_transfer_async_on_nic(
        &self,
        op: TransferOp,
        remote_addr: &str,
        descs: &[TransferDesc],
        nic_idx: usize,
    ) -> Result<mea::oneshot::Receiver<Result<usize>>> {
        self.backend
            .batch_transfer_async_on_nic(op, remote_addr, descs, nic_idx)
    }

    /// Submit a final RDMA WRITE-with-immediate signal against every QP for a
    /// connected peer. Returns send-side completions.
    pub fn write_imm_async(
        &self,
        remote_addr: &str,
        imm_data: u32,
    ) -> Result<Vec<mea::oneshot::Receiver<Result<usize>>>> {
        self.backend.write_imm_async(remote_addr, imm_data)
    }

    /// Take the single receiver for RDMA WRITE-with-immediate completions.
    ///
    /// The transfer layer does not interpret immediate data. Higher layers own
    /// demuxing `imm_data` into leases, fan-in counters, and request lifecycle.
    pub fn take_imm_receiver(&self) -> Option<ImmCompletionReceiver> {
        self.backend.take_imm_receiver()
    }

    /// Number of active RC queue pairs across all NICs.
    pub fn num_qps(&self) -> usize {
        self.backend.num_qps()
    }

    /// Number of local RDMA NICs configured for this endpoint.
    pub fn nic_count(&self) -> usize {
        self.backend.nic_count()
    }
}

#[cfg(test)]
mod tests {
    use super::{HandshakeMetadata, NicHandshake, RcEndpoint, RegisteredMemoryRegion};

    #[test]
    fn handshake_metadata_roundtrip() {
        let meta = HandshakeMetadata {
            nics: vec![NicHandshake {
                endpoint: RcEndpoint {
                    gid: [7u8; 16],
                    lid: 0,
                    qp_num: 200,
                    psn: 0x1111,
                },
                memory_regions: vec![RegisteredMemoryRegion {
                    base_ptr: 0x1000,
                    len: 0x2000,
                    rkey: 42,
                }],
                signal_region: RegisteredMemoryRegion {
                    base_ptr: 0x3000,
                    len: 8,
                    rkey: 43,
                },
            }],
        };
        let bytes = meta.to_bytes();
        let decoded = HandshakeMetadata::from_bytes(&bytes).expect("decode");
        assert_eq!(decoded.nics.len(), 1);
        assert_eq!(decoded.nics[0].endpoint, meta.nics[0].endpoint);
        assert_eq!(decoded.nics[0].memory_regions, meta.nics[0].memory_regions);
    }

    #[test]
    fn handshake_metadata_multi_nic_roundtrip() {
        let meta = HandshakeMetadata {
            nics: vec![
                NicHandshake {
                    endpoint: RcEndpoint {
                        gid: [1u8; 16],
                        lid: 0,
                        qp_num: 100,
                        psn: 0x1000,
                    },
                    memory_regions: vec![RegisteredMemoryRegion {
                        base_ptr: 0x1000,
                        len: 0x2000,
                        rkey: 10,
                    }],
                    signal_region: RegisteredMemoryRegion {
                        base_ptr: 0x3000,
                        len: 8,
                        rkey: 11,
                    },
                },
                NicHandshake {
                    endpoint: RcEndpoint {
                        gid: [2u8; 16],
                        lid: 0,
                        qp_num: 200,
                        psn: 0x2000,
                    },
                    memory_regions: vec![RegisteredMemoryRegion {
                        base_ptr: 0x1000,
                        len: 0x2000,
                        rkey: 20,
                    }],
                    signal_region: RegisteredMemoryRegion {
                        base_ptr: 0x4000,
                        len: 8,
                        rkey: 21,
                    },
                },
            ],
        };
        let bytes = meta.to_bytes();
        let decoded = HandshakeMetadata::from_bytes(&bytes).expect("decode");
        assert_eq!(decoded.nics.len(), 2);
        assert_eq!(decoded.nics[0].endpoint.qp_num, 100);
        assert_eq!(decoded.nics[0].memory_regions[0].rkey, 10);
        assert_eq!(decoded.nics[1].endpoint.qp_num, 200);
        assert_eq!(decoded.nics[1].memory_regions[0].rkey, 20);
    }

    #[test]
    fn handshake_metadata_rejects_garbage() {
        assert!(HandshakeMetadata::from_bytes(&[1, 2, 3]).is_err());
    }
}
