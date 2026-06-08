mod engine;
mod error;
mod rc_backend;
pub mod rdma_topo;

mod cuda_lib;
mod cuda_sys;
mod cudart_sys;
pub mod v2;

pub use engine::{
    ConnectionStatus, HandshakeMetadata, MemoryRegion, TransferDesc, TransferEngine, TransferOp,
};
pub use error::{Result, TransferError};

pub fn init_logging() {
    pegaflow_common::logging::init_stderr("info,pegaflow_transfer=debug");
}
