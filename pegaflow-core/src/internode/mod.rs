//! Inter-node communication module for PegaFlow.

pub(crate) mod metaserver_client;

pub use metaserver_client::{
    DEFAULT_METASERVER_QUEUE_DEPTH, MetaServerClient, MetaServerClientConfig,
};
