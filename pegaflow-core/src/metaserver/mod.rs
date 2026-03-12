//! MetaServer service for distributed block hash coordination.
//!
//! Provides a gRPC service that tracks which node owns which block hashes,
//! enabling P2P block discovery across a multi-node cluster.

pub mod service;
pub mod store;

pub use service::GrpcMetaService;
pub use store::{BlockHashStore, CrossNodeBlock};
