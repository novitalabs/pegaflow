//! GPU<->CPU KV-cache transfer backends.
//!
//! A transfer is a batch of independent contiguous copies, each described by a
//! [`CopyDesc`] (`device` address, `host` address, `size`). The same batch can
//! be moved by two backends with different tradeoffs:
//!
//! - [`MemcpyBackend`]: coalesces contiguous copies and issues one
//!   `cuMemcpy{Hto,Dto}DAsync_v2` per merged range. Drives the copy engines
//!   (DMA); best for few/large transfers and overlaps with compute.
//! - [`KernelBackend`]: a single grid-strided kernel reads/writes mapped pinned
//!   host memory directly (zero-copy over PCIe). One launch regardless of the
//!   number of fragments; best when the batch is highly fragmented and the
//!   per-call launch latency of the memcpy path dominates.
//!
//! Both backends only *enqueue* onto the caller's stream and never synchronize,
//! so a caller can batch many layers and synchronize exactly once.

mod kernel;
mod memcpy;

pub use kernel::KernelBackend;
pub use memcpy::MemcpyBackend;

use std::sync::Arc;

use cudarc::driver::CudaStream;

/// One contiguous copy between device memory and mapped, pinned host memory.
///
/// `device` is an absolute device virtual address. `host` is the host virtual
/// address used by CUDA memcpy. `host_device` is the device-visible address for
/// the same mapped pinned memory, queried at allocation time; registered and
/// write-combined host memory must not assume host and device pointer values are
/// identical. The direction is fixed by which [`TransferBackend`] method is
/// called, not stored.
#[derive(Clone, Copy)]
pub struct CopyDesc {
    pub device: u64,
    pub host: *mut u8,
    pub host_device: u64,
    pub size: usize,
}

// SAFETY: `host` points into pinned memory owned by the caller, who guarantees
// it outlives the transfer. The pointer is only dereferenced by CUDA, never by
// the Rust side, so moving the descriptor across threads is sound.
unsafe impl Send for CopyDesc {}

/// Enqueue a batch of copies onto a CUDA stream. Implementations must only
/// submit work; synchronization is the caller's responsibility.
pub trait TransferBackend: Send {
    /// Host -> device.
    fn h2d(&self, copies: &[CopyDesc], stream: &Arc<CudaStream>) -> Result<(), String>;
    /// Device -> host.
    fn d2h(&self, copies: &[CopyDesc], stream: &Arc<CudaStream>) -> Result<(), String>;
    /// Short name for logging (matches the CLI value that selects it).
    fn name(&self) -> &'static str;
}

/// Which [`TransferBackend`] an instance's GPU worker pools use. The connector
/// picks this per model (kernel for MLA, direct otherwise) and sends it with
/// the registration; the pools are spawned per (instance, GPU), so each
/// instance can run a different backend.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TransferMode {
    /// One directional `cuMemcpyAsync` per coalesced range, on the DMA copy
    /// engines. The default: best bandwidth for few/large transfers.
    #[default]
    Direct,
    /// A single grid-strided copy kernel that reads/writes mapped pinned host
    /// memory directly. Wins when the batch is so fragmented that per-call
    /// launch latency on the direct path dominates.
    Kernel,
}
