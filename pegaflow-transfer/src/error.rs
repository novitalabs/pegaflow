use thiserror::Error;

pub type Result<T> = std::result::Result<T, TransferError>;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum TransferError {
    #[error("transfer engine not initialized")]
    NotInitialized,
    #[error("unsupported protocol: {0}")]
    UnsupportedProtocol(String),
    #[error("invalid argument: {0}")]
    InvalidArgument(&'static str),
    #[error("batch length mismatch: ptrs={ptrs}, lens={lens}")]
    BatchLengthMismatch { ptrs: usize, lens: usize },
    #[error("memory is not registered: ptr={ptr:#x}")]
    MemoryNotRegistered { ptr: u64 },
    #[error("rdma device not found: {0}")]
    DeviceNotFound(String),
    #[error("address resolution failed: {0}")]
    AddressResolution(String),
    #[error("backend error: {0}")]
    Backend(String),
    #[error("unimplemented: {0}")]
    Unimplemented(&'static str),
}
