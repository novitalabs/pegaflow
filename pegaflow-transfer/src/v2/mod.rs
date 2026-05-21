//! RDMA Verbs transfer engine (upstream-derived from `pplx-garden`).
#![allow(
    dead_code,
    unreachable_pub,
    unused_imports,
    clippy::allow_attributes_without_reason,
    clippy::cloned_instead_of_copied,
    clippy::enum_variant_names,
    clippy::explicit_into_iter_loop,
    clippy::explicit_iter_loop,
    clippy::manual_assert,
    clippy::semicolon_if_nothing_returned,
    clippy::unnecessary_wraps,
    clippy::useless_conversion,
    reason = "upstream-derived RDMA fabric keeps its porting shape while PD push integration lands"
)]

mod api;
mod cpu_affinity;
mod domain_group;
mod error;
mod fabric_engine;
mod host_buffer;
mod imm_count;
mod interface;
mod mr;
mod provider;
mod provider_dispatch;
mod rdma_op;
mod topo;
mod transfer_engine;
mod transfer_engine_builder;
mod utils;
mod verbs;
mod worker;

pub use crate::cuda_lib::{CudaDeviceId, Device};
pub use api::{
    BarrierTransferRequest, DomainAddress, DomainGroupRouting, GroupTransferRouting, ImmCounter,
    ImmTransferRequest, MemoryRegionDescriptor, MemoryRegionHandle, MemoryRegionRemoteKey,
    PagedTransferRequest, PeerGroupHandle, ScatterTarget, ScatterTransferRequest,
    SingleTransferRequest, SmallVec, TransferRequest,
};
pub use error::{FabricLibError, Result, VerbsError};
pub use interface::{
    AsyncTransferEngine, BouncingErrorCallback, BouncingRecvCallback, CallbackResult,
    ErrorCallback, RdmaEngine, RecvCallback, SendBuffer, SendCallback, SendRecvEngine,
};
pub use provider_dispatch::DomainInfo;
pub use topo::{TopologyGroup, detect_topology};
pub use transfer_engine::{ImmCallbackFn, ImmCountCallback, TransferCallback, TransferEngine};
pub use transfer_engine_builder::TransferEngineBuilder;

pub(crate) use fabric_engine::FabricEngine;
