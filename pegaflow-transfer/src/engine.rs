use std::sync::Arc;

use crate::{
    api::WorkerConfig,
    backend::RdmaBackend,
    error::{Result, TransferError},
    sideway_backend::SidewayBackend,
};

pub struct MooncakeTransferEngine {
    backend: Arc<dyn RdmaBackend>,
}

impl MooncakeTransferEngine {
    pub fn new() -> Self {
        Self {
            backend: Arc::new(SidewayBackend::new()),
        }
    }

    pub fn with_backend(backend: Arc<dyn RdmaBackend>) -> Self {
        Self { backend }
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

    pub fn batch_unregister_memory(&self, ptrs: &[u64]) -> Result<()> {
        for ptr in ptrs.iter().copied() {
            self.backend.unregister_memory(ptr)?;
        }
        Ok(())
    }

    pub fn transfer_sync_write(
        &self,
        session_id: &str,
        local_ptr: u64,
        remote_ptr: u64,
        len: usize,
    ) -> Result<usize> {
        self.backend
            .transfer_sync_write(session_id, local_ptr, remote_ptr, len)
    }

    pub fn batch_transfer_sync_write(
        &self,
        session_id: &str,
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

        let mut transferred = 0usize;
        for ((local_ptr, remote_ptr), len) in local_ptrs
            .iter()
            .copied()
            .zip(remote_ptrs.iter().copied())
            .zip(lens.iter().copied())
        {
            transferred += self
                .backend
                .transfer_sync_write(session_id, local_ptr, remote_ptr, len)?;
        }
        Ok(transferred)
    }

    pub fn get_rpc_port(&self) -> Result<u16> {
        self.backend.rpc_port()
    }

    pub fn get_session_id(&self) -> Result<String> {
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
    use std::{collections::HashMap, sync::Arc};

    use parking_lot::Mutex;

    use super::MooncakeTransferEngine;
    use crate::{
        api::WorkerConfig,
        backend::RdmaBackend,
        error::{Result, TransferError},
    };

    #[derive(Default)]
    struct MockBackend {
        rpc_port: Mutex<Option<u16>>,
        session_id: Mutex<Option<String>>,
        memory: Mutex<HashMap<u64, usize>>,
    }

    impl RdmaBackend for MockBackend {
        fn initialize(&self, config: WorkerConfig) -> Result<()> {
            *self.rpc_port.lock() = Some(config.rpc_port);
            *self.session_id.lock() = Some(format!("mock:{}", config.rpc_port));
            Ok(())
        }

        fn rpc_port(&self) -> Result<u16> {
            self.rpc_port.lock().ok_or(TransferError::NotInitialized)
        }

        fn session_id(&self) -> Result<String> {
            self.session_id
                .lock()
                .clone()
                .ok_or(TransferError::NotInitialized)
        }

        fn register_memory(&self, ptr: u64, len: usize) -> Result<()> {
            self.memory.lock().insert(ptr, len);
            Ok(())
        }

        fn unregister_memory(&self, ptr: u64) -> Result<()> {
            let removed = self.memory.lock().remove(&ptr);
            if removed.is_none() {
                return Err(TransferError::MemoryNotRegistered { ptr });
            }
            Ok(())
        }

        fn transfer_sync_write(
            &self,
            _session_id: &str,
            local_ptr: u64,
            _remote_ptr: u64,
            len: usize,
        ) -> Result<usize> {
            let memory = self.memory.lock();
            let Some(registered) = memory.get(&local_ptr) else {
                return Err(TransferError::MemoryNotRegistered { ptr: local_ptr });
            };
            if len > *registered {
                return Err(TransferError::InvalidArgument(
                    "len exceeds registered memory",
                ));
            }
            Ok(len)
        }
    }

    #[test]
    fn mooncake_minimal_path_works() {
        let mut engine = MooncakeTransferEngine::with_backend(Arc::new(MockBackend::default()));
        engine
            .initialize("mlx5_0", 50051)
            .expect("init should succeed");
        engine
            .register_memory(0x1000, 4096)
            .expect("register should succeed");

        let written = engine
            .transfer_sync_write("127.0.0.1:50052", 0x1000, 0x2000, 1024)
            .expect("write should succeed");
        assert_eq!(written, 1024);

        let session_id = engine.get_session_id().expect("session id should exist");
        assert_eq!(session_id, "mock:50051");
    }

    #[test]
    fn batch_len_mismatch_fails() {
        let mut engine = MooncakeTransferEngine::with_backend(Arc::new(MockBackend::default()));
        engine
            .initialize("mlx5_0", 50051)
            .expect("init should succeed");

        let err = engine
            .batch_register_memory(&[0x1000, 0x2000], &[4096])
            .expect_err("must fail for mismatch");
        assert_eq!(err.to_string(), "batch length mismatch: ptrs=2, lens=1");
    }
}
