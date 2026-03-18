use serde::{Deserialize, Serialize};
use std::time::Instant;

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

/// Opaque handshake metadata exchanged between peers via gRPC.
///
/// Contains the RC endpoint and registered memory regions needed to
/// establish an RDMA connection and perform transfers.
#[derive(Clone, Debug)]
pub struct HandshakeMetadata {
    pub(crate) endpoint: RcEndpoint,
    pub(crate) memory_regions: Vec<RegisteredMemoryRegion>,
}

#[derive(Serialize, Deserialize)]
struct WireHandshakeMetadata {
    endpoint: RcEndpoint,
    memory_regions: Vec<RegisteredMemoryRegion>,
}

impl HandshakeMetadata {
    pub fn to_bytes(&self) -> Vec<u8> {
        let wire = WireHandshakeMetadata {
            endpoint: self.endpoint,
            memory_regions: self.memory_regions.clone(),
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
        Ok(Self {
            endpoint: wire.endpoint,
            memory_regions: wire.memory_regions,
        })
    }
}

pub struct TransferEngine {
    backend: RcBackend,
}

impl TransferEngine {
    pub fn new() -> Self {
        Self {
            backend: RcBackend::new(),
        }
    }

    pub fn initialize(&mut self, nic_name: impl Into<String>) -> Result<()> {
        self.backend.initialize(&nic_name.into())
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

    /// Create an RC QP (INIT state) and return handshake metadata containing
    /// the local endpoint and a snapshot of all registered memory regions.
    ///
    /// Call after `register_memory`, serialize with [`HandshakeMetadata::to_bytes`],
    /// and send to the remote peer.
    pub fn prepare_handshake(&self) -> Result<HandshakeMetadata> {
        let endpoint = self.backend.prepare_handshake()?;
        let memory_regions = self.backend.snapshot_registered_memory();
        Ok(HandshakeMetadata {
            endpoint,
            memory_regions,
        })
    }

    /// Connect the most recently prepared QP to the remote peer (RTR→RTS).
    ///
    /// Called by the **responder** after receiving the initiator's metadata.
    /// The responder should call [`prepare_handshake`](Self::prepare_handshake) first
    /// to create its own QP, then `accept_handshake` to immediately connect it.
    pub fn accept_handshake(&self, remote: &HandshakeMetadata) -> Result<()> {
        self.backend
            .accept_handshake(&remote.endpoint, &remote.memory_regions)
    }

    /// Perform a batch of RDMA READ or WRITE operations against a peer.
    ///
    /// On the **initiator** side, the first call lazily connects the QP to the
    /// peer. Subsequent calls reuse the existing long-lived connection.
    pub fn batch_transfer(
        &self,
        op: TransferOp,
        peer: &HandshakeMetadata,
        local_ptrs: &[u64],
        remote_ptrs: &[u64],
        lens: &[usize],
    ) -> Result<usize> {
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

        let started_at = Instant::now();
        let transferred = self.backend.batch_transfer(
            op,
            &peer.endpoint,
            &peer.memory_regions,
            local_ptrs,
            remote_ptrs,
            lens,
        )?;
        let elapsed = started_at.elapsed();
        let elapsed_secs = elapsed.as_secs_f64();
        if elapsed_secs > 0.0 {
            let gbps = (transferred as f64 * 8.0) / elapsed_secs / 1e9;
            let gib_per_sec = (transferred as f64) / elapsed_secs / (1024.0 * 1024.0 * 1024.0);
            log::debug!(
                "batch_transfer {:?} e2e: bytes={}, chunks={}, elapsed_ms={:.3}, bw_gbps={:.3}, bw_gibps={:.3}",
                op,
                transferred,
                lens.len(),
                elapsed_secs * 1000.0,
                gbps,
                gib_per_sec
            );
        }
        Ok(transferred)
    }
}

impl Default for TransferEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        HandshakeMetadata, RcEndpoint, RegisteredMemoryRegion, TransferEngine, TransferOp,
    };
    use crate::error::TransferError;

    #[test]
    fn register_batch_len_mismatch_fails() {
        let engine = TransferEngine::new();
        let err = engine
            .register_memory(&[0x1000, 0x2000], &[4096])
            .expect_err("must fail for mismatch");
        assert_eq!(err, TransferError::BatchLengthMismatch { ptrs: 2, lens: 1 });
    }

    #[test]
    fn transfer_batch_len_mismatch_fails() {
        let engine = TransferEngine::new();
        let meta = HandshakeMetadata {
            endpoint: RcEndpoint {
                gid: [1u8; 16],
                lid: 0,
                qp_num: 42,
                psn: 0,
            },
            memory_regions: vec![],
        };
        let err = engine
            .batch_transfer(
                TransferOp::Write,
                &meta,
                &[0x1000, 0x2000],
                &[0x3000],
                &[128],
            )
            .expect_err("must fail for mismatch");
        assert_eq!(err, TransferError::BatchLengthMismatch { ptrs: 2, lens: 1 });
    }

    #[test]
    fn handshake_metadata_roundtrip() {
        let meta = HandshakeMetadata {
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
        };
        let bytes = meta.to_bytes();
        let decoded = HandshakeMetadata::from_bytes(&bytes).expect("decode");
        assert_eq!(decoded.endpoint, meta.endpoint);
        assert_eq!(decoded.memory_regions, meta.memory_regions);
    }

    #[test]
    fn handshake_metadata_rejects_garbage() {
        assert!(HandshakeMetadata::from_bytes(&[1, 2, 3]).is_err());
    }
}
