//! Inter-node communication module for PegaFlow.
//!
//! This module provides functionality for:
//! - Kubernetes service discovery of PegaFlow engine instances
//! - gRPC client for inter-node communication (Query, Health)
//! - Instance registry for managing discovered instances
//!
//! # Module Structure
//!
//! - [`types`]: Common type definitions (configs, errors, result types)
//! - [`registry`]: Thread-safe instance registry
//! - [`service_discovery`]: Kubernetes pod watcher
//! - [`client`]: gRPC client for remote PegaFlow instances
//!
//! # Service Discovery
//!
//! PegaFlow instances are discovered by watching Kubernetes pods with the label
//! `novita.ai/pegaflow: app`. Each discovered instance is registered in the
//! [`InstanceRegistry`] for easy access.
//!
//! # gRPC Client
//!
//! The [`PegaflowClient`] provides a high-level API for communicating with remote
//! PegaFlow instances, particularly useful for P/D disaggregation scenarios where
//! a decode instance needs to query a prefill instance for cached KV blocks.
//!

pub mod client;
pub(crate) mod registrar;
pub mod registry;
pub mod service_discovery;
pub mod types;

// Re-export commonly used types for convenience
pub use client::{PegaflowClient, PegaflowClientPool};
pub use registrar::{MetaServerRegistrar, MetaServerRegistrarConfig};
pub use registry::InstanceRegistry;
pub use service_discovery::start_service_discovery;
pub use types::{
    ClientConfig, ClientError, DEFAULT_CONNECT_TIMEOUT, DEFAULT_GRPC_PORT,
    DEFAULT_PEGAFLOW_LABEL_KEY, DEFAULT_PEGAFLOW_LABEL_VALUE, DEFAULT_REQUEST_TIMEOUT,
    PegaflowInstance, QueryPrefetchStatus, QueryResult, ServiceDiscoveryConfig,
};
