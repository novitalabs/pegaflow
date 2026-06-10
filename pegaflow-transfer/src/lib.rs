//! RDMA Verbs transfer engine (upstream-derived from `pplx-garden`).
//!
//! One-sided RDMA WRITE data plane with a UD control plane that handshakes
//! RC connections lazily per peer. Peers exchange [`MemoryRegionDescriptor`]s
//! out of band (e.g. via gRPC); no other connection setup is required.

mod api;
mod cpu_affinity;
mod cuda_lib;
mod cuda_sys;
mod cudart_sys;
mod domain_group;
mod error;
mod fabric_engine;
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

pub use api::{
    BarrierTransferRequest, DomainAddress, DomainGroupRouting, GroupTransferRouting, ImmCounter,
    ImmTransferRequest, MemoryRegionDescriptor, MemoryRegionHandle, MemoryRegionRemoteKey,
    PagedTransferRequest, PeerGroupHandle, ScatterTarget, ScatterTransferRequest,
    SingleTransferRequest, SmallVec, TransferRequest,
};
pub use cuda_lib::{CudaDeviceId, CudaDeviceMemory, Device};
pub use error::{FabricLibError, Result, VerbsError};
pub use interface::{
    AsyncTransferEngine, BouncingErrorCallback, BouncingRecvCallback, CallbackResult,
    ErrorCallback, RdmaEngine, RecvCallback, SendBuffer, SendCallback, SendRecvEngine,
};
pub use provider_dispatch::DomainInfo;
pub use topo::{HostTopologyGroup, TopologyGroup, detect_host_topology, detect_topology};
pub use transfer_engine::{ImmCallbackFn, ImmCountCallback, TransferCallback, TransferEngine};
pub use transfer_engine_builder::TransferEngineBuilder;

pub fn init_logging() {
    pegaflow_common::logging::init_stderr("info,pegaflow_transfer=debug");
}
