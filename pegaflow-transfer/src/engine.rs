use crate::{
    api::WorkerConfig,
    control_protocol::RegisteredMemoryRegion,
    domain_address::DomainAddress,
    error::{Result, TransferError},
    sideway_backend::SidewayBackend,
};
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// RDMA operation type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransferOp {
    Read,
    Write,
}

/// Opaque handshake metadata exchanged between peers via gRPC.
///
/// Contains everything needed to establish an RDMA connection and perform
/// transfers: the peer's UD address and its registered memory regions.
///
/// Call [`TransferEngine::handshake_metadata`] after registering memory,
/// serialize with [`to_bytes`](Self::to_bytes), and send to the remote peer.
#[derive(Clone, Debug)]
pub struct HandshakeMetadata {
    pub(crate) ud_address: DomainAddress,
    pub(crate) memory_regions: Vec<RegisteredMemoryRegion>,
}

#[derive(Serialize, Deserialize)]
struct WireHandshakeMetadata {
    ud_address: [u8; DomainAddress::BYTES],
    memory_regions: Vec<RegisteredMemoryRegion>,
}

impl HandshakeMetadata {
    pub fn to_bytes(&self) -> Vec<u8> {
        let wire = WireHandshakeMetadata {
            ud_address: {
                let mut buf = [0u8; DomainAddress::BYTES];
                buf.copy_from_slice(&self.ud_address.to_bytes());
                buf
            },
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
        let ud_address = DomainAddress::from_bytes(&wire.ud_address).ok_or(
            TransferError::InvalidArgument("invalid UD address in handshake metadata"),
        )?;
        Ok(Self {
            ud_address,
            memory_regions: wire.memory_regions,
        })
    }
}

pub struct TransferEngine {
    backend: SidewayBackend,
}

impl TransferEngine {
    pub fn new() -> Self {
        Self {
            backend: SidewayBackend::new(),
        }
    }

    pub fn initialize(&mut self, nic_name: impl Into<String>) -> Result<()> {
        self.backend.initialize(WorkerConfig {
            nic_name: nic_name.into(),
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

    /// Generate handshake metadata containing this engine's UD address and
    /// a snapshot of all registered memory regions. Call after `register_memory`.
    pub fn handshake_metadata(&self) -> HandshakeMetadata {
        let ud_address = self.backend.session_id();
        let memory_regions = self.backend.registered_memory_regions();
        HandshakeMetadata {
            ud_address,
            memory_regions,
        }
    }

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
        let transferred = match op {
            TransferOp::Read => self.backend.batch_transfer_sync_read(
                &peer.ud_address,
                local_ptrs,
                remote_ptrs,
                lens,
            )?,
            TransferOp::Write => self.backend.batch_transfer_sync_write(
                &peer.ud_address,
                local_ptrs,
                remote_ptrs,
                lens,
            )?,
        };
        let elapsed = started_at.elapsed();
        let elapsed_secs = elapsed.as_secs_f64();
        if elapsed_secs > 0.0 {
            let gbps = (transferred as f64 * 8.0) / elapsed_secs / 1e9;
            let gib_per_sec =
                (transferred as f64) / elapsed_secs / (1024.0 * 1024.0 * 1024.0);
            log::debug!(
                "batch_transfer {:?} e2e: bytes={}, chunks={}, elapsed_ms={:.3}, bw_gbps={:.3}, bw_gibps={:.3}",
                op, transferred, lens.len(), elapsed_secs * 1000.0, gbps, gib_per_sec
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
    use super::{HandshakeMetadata, TransferEngine, TransferOp};
    use crate::control_protocol::RegisteredMemoryRegion;
    use crate::domain_address::DomainAddress;
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
            ud_address: DomainAddress::from_parts([1u8; 16], 2, 3, 4),
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
            ud_address: DomainAddress::from_parts([7u8; 16], 100, 200, 0x1111_1111),
            memory_regions: vec![RegisteredMemoryRegion {
                base_ptr: 0x1000,
                len: 0x2000,
                rkey: 42,
            }],
        };
        let bytes = meta.to_bytes();
        let decoded = HandshakeMetadata::from_bytes(&bytes).expect("decode");
        assert_eq!(decoded.ud_address, meta.ud_address);
        assert_eq!(decoded.memory_regions, meta.memory_regions);
    }

    #[test]
    fn handshake_metadata_rejects_garbage() {
        assert!(HandshakeMetadata::from_bytes(&[1, 2, 3]).is_err());
    }
}
