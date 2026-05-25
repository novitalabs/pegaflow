mod engine;
mod error;
mod rc_backend;
pub mod rdma_topo;

// v2 RDMA stack (upstream-derived from pplx-garden). FFI bindings + the
// cuda_lib wrapper layer live at crate root so internal `use libibverbs_sys::...`
// etc. resolve naturally; v2/ holds the fabric/worker/transfer logic itself.
#[cfg(feature = "v2-rdma")]
mod cuda_lib;
#[cfg(feature = "v2-rdma")]
mod cuda_sys;
#[cfg(feature = "v2-rdma")]
mod cudart_sys;
#[cfg(feature = "v2-rdma")]
mod libibverbs_sys;
#[cfg(feature = "v2-rdma")]
pub mod v2;

pub use engine::{
    ConnectionStatus, HandshakeMetadata, MemoryRegion, TransferDesc, TransferEngine, TransferOp,
};
pub use error::{Result, TransferError};

pub fn init_logging() {
    pegaflow_common::logging::init_stderr("info,pegaflow_transfer=debug");
}
