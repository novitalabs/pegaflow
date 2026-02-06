use crate::{api::WorkerConfig, domain_address::DomainAddress, error::Result};

pub trait RdmaBackend: Send + Sync {
    fn initialize(&self, config: WorkerConfig) -> Result<()>;
    fn rpc_port(&self) -> Result<u16>;
    fn session_id(&self) -> DomainAddress;

    fn register_memory(&self, ptr: u64, len: usize) -> Result<()>;
    fn unregister_memory(&self, ptr: u64) -> Result<()>;

    fn transfer_sync_write(
        &self,
        session_id: &DomainAddress,
        local_ptr: u64,
        remote_ptr: u64,
        len: usize,
    ) -> Result<usize>;
}
