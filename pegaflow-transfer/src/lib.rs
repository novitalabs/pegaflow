mod api;
mod control_protocol;
pub(crate) mod domain_address;
mod engine;
mod error;
pub mod rdma_topo;
mod sideway_backend;

pub use engine::{HandshakeMetadata, TransferEngine, TransferOp};
pub use error::{Result, TransferError};

pub fn init_logging() {
    pegaflow_common::logging::init_stderr("info,pegaflow_transfer=debug");
}
