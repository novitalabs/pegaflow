use crate::{
    api::WorkerConfig,
    domain_address::DomainAddress,
    error::{Result, TransferError},
    sideway_backend::SidewayBackend,
};
use std::time::Instant;

pub struct MooncakeTransferEngine {
    backend: SidewayBackend,
}

impl MooncakeTransferEngine {
    pub fn new() -> Self {
        Self {
            backend: SidewayBackend::new(),
        }
    }

    pub fn initialize(&mut self, nic_name: impl Into<String>, rpc_port: u16) -> Result<()> {
        self.backend.initialize(WorkerConfig {
            nic_name: nic_name.into(),
            rpc_port,
        })?;
        Ok(())
    }

    pub fn register_memory(&self, ptr: u64, len: usize) -> Result<()> {
        self.backend.register_memory(ptr, len)
    }

    pub fn unregister_memory(&self, ptr: u64) -> Result<()> {
        self.backend.unregister_memory(ptr)
    }

    pub fn batch_register_memory(&self, ptrs: &[u64], lens: &[usize]) -> Result<()> {
        if ptrs.len() != lens.len() {
            return Err(TransferError::BatchLengthMismatch {
                ptrs: ptrs.len(),
                lens: lens.len(),
            });
        }
        for (ptr, len) in ptrs.iter().copied().zip(lens.iter().copied()) {
            self.backend.register_memory(ptr, len)?;
        }
        Ok(())
    }

    pub fn transfer_sync_write(
        &self,
        session_id: &DomainAddress,
        local_ptr: u64,
        remote_ptr: u64,
        len: usize,
    ) -> Result<usize> {
        self.backend
            .transfer_sync_write(session_id, local_ptr, remote_ptr, len)
    }

    pub fn batch_transfer_sync_write(
        &self,
        session_id: &DomainAddress,
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
        let transferred =
            self.backend
                .batch_transfer_sync_write(session_id, local_ptrs, remote_ptrs, lens)?;
        let elapsed = started_at.elapsed();
        let elapsed_secs = elapsed.as_secs_f64();
        if elapsed_secs > 0.0 {
            let gbps = (transferred as f64 * 8.0) / elapsed_secs / 1e9;
            let gib_per_sec = (transferred as f64) / elapsed_secs / (1024.0 * 1024.0 * 1024.0);
            log::debug!(
                "batch_transfer_sync_write e2e bandwidth: bytes={}, chunks={}, elapsed_ms={:.3}, bw_gbps={:.3}, bw_gibps={:.3}",
                transferred,
                lens.len(),
                elapsed_secs * 1000.0,
                gbps,
                gib_per_sec
            );
        }
        Ok(transferred)
    }

    pub fn get_rpc_port(&self) -> Result<u16> {
        self.backend.rpc_port()
    }

    pub fn get_session_id(&self) -> DomainAddress {
        self.backend.session_id()
    }
}

impl Default for MooncakeTransferEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::MooncakeTransferEngine;
    use crate::error::TransferError;

    #[test]
    fn batch_len_mismatch_fails() {
        let engine = MooncakeTransferEngine::new();

        let err = engine
            .batch_register_memory(&[0x1000, 0x2000], &[4096])
            .expect_err("must fail for mismatch");
        assert_eq!(err, TransferError::BatchLengthMismatch { ptrs: 2, lens: 1 });
    }

    #[test]
    fn batch_transfer_len_mismatch_fails() {
        let engine = MooncakeTransferEngine::new();
        let session_id = crate::DomainAddress::from_parts([1_u8; 16], 2, 3, 4);

        let err = engine
            .batch_transfer_sync_write(&session_id, &[0x1000, 0x2000], &[0x3000], &[128])
            .expect_err("must fail for mismatch");
        assert_eq!(err, TransferError::BatchLengthMismatch { ptrs: 2, lens: 1 });
    }
}
