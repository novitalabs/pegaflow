mod engine;
mod error;
mod rc_backend;
pub mod rdma_topo;

pub use engine::{
    HandshakeMetadata, RcEndpoint, RegisteredMemoryRegion, TransferEngine, TransferOp,
};
pub use error::{Result, TransferError};

pub fn init_logging() {
    pegaflow_common::logging::init_stderr("info,pegaflow_transfer=debug");
}
