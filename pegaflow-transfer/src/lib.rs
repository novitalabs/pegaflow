mod api;
mod control_protocol;
mod domain_address;
mod engine;
mod error;
mod logging;
mod sideway_backend;

pub use domain_address::DomainAddress;
pub use engine::MooncakeTransferEngine;
pub use error::{Result, TransferError};

pub fn init_logging() {
    logging::ensure_initialized();
}
