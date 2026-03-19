use serde::{Deserialize, Serialize};

use crate::error::{Result, TransferError};
use crate::rc_backend::RcBackend;

/// RDMA operation type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransferOp {
    Read,
    Write,
}

/// RC queue pair endpoint info exchanged during handshake.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RcEndpoint {
    pub gid: [u8; 16],
    pub lid: u16,
    pub qp_num: u32,
    pub psn: u32,
}

/// A registered memory region descriptor (base pointer, length, remote key).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegisteredMemoryRegion {
    pub base_ptr: u64,
    pub len: u64,
    pub rkey: u32,
}

/// Per-NIC handshake data: endpoint + memory region snapshot.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NicHandshake {
    pub endpoint: RcEndpoint,
    pub memory_regions: Vec<RegisteredMemoryRegion>,
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

pub struct TransferEngine {
    backend: RcBackend,
}

impl TransferEngine {
    pub fn new(nic_names: &[String]) -> Result<Self> {
        Ok(Self {
            backend: RcBackend::new(nic_names)?,
        })
    }

    pub fn register_memory(&self, ptrs: &[u64], lens: &[usize]) -> Result<()> {
        if ptrs.len() != lens.len() {
            return Err(TransferError::BatchLengthMismatch {
                ptrs: ptrs.len(),
                lens: lens.len(),
            });
        }
        for (&ptr, &len) in ptrs.iter().zip(lens.iter()) {
            self.backend.register_memory(ptr, len)?;
        }
        Ok(())
    }

    pub fn unregister_memory(&self, ptrs: &[u64]) -> Result<()> {
        for &ptr in ptrs {
            self.backend.unregister_memory(ptr)?;
        }
        Ok(())
    }

    /// Create one RC QP per NIC (INIT state) and return handshake metadata
    /// containing per-NIC endpoints and snapshots of all registered memory.
    ///
    /// Call after `register_memory`, serialize with [`HandshakeMetadata::to_bytes`],
    /// and send to the remote peer.
    pub fn prepare_handshake(&self) -> Result<HandshakeMetadata> {
        let nics = self.backend.prepare_handshake()?;
        Ok(HandshakeMetadata { nics })
    }

    /// Connect all prepared QPs to the remote peer (RTR→RTS).
    ///
    /// Called by the **responder** after receiving the initiator's metadata.
    /// The responder should call [`prepare_handshake`](Self::prepare_handshake) first
    /// to create its own QPs, then `accept_handshake` to immediately connect them.
    pub fn accept_handshake(&self, remote: &HandshakeMetadata) -> Result<()> {
        self.backend.accept_handshake(&remote.nics)
    }

    /// Submit a batch of RDMA READ or WRITE operations against a peer.
    ///
    /// Ops are round-robin distributed across NICs. Returns a `Receiver`
    /// that yields the total bytes transferred once all RDMA operations complete.
    /// The caller decides when to block on `.recv()`.
    ///
    /// On the **initiator** side, the first call lazily connects QPs to the
    /// peer. Subsequent calls reuse the existing long-lived connections.
    pub fn batch_transfer_async(
        &self,
        op: TransferOp,
        peer: &HandshakeMetadata,
        local_ptrs: &[u64],
        remote_ptrs: &[u64],
        lens: &[usize],
    ) -> Result<std::sync::mpsc::Receiver<Result<usize>>> {
        if local_ptrs.len() != remote_ptrs.len() {
            return Err(TransferError::BatchLengthMismatch {
                ptrs: local_ptrs.len(),
                lens: remote_ptrs.len(),
            });
        }
        if local_ptrs.len() != lens.len() {
            return Err(TransferError::BatchLengthMismatch {
                ptrs: local_ptrs.len(),
                lens: lens.len(),
            });
        }

        self.backend
            .batch_transfer_async(op, &peer.nics, local_ptrs, remote_ptrs, lens)
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
