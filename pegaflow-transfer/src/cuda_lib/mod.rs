//! CUDA wrapper layer (upstream-derived from `pplx-garden`).
//!
//! Exposes safe wrappers over the raw `cuda_sys` / `cudart_sys` FFI modules.
#![allow(
    dead_code,
    non_snake_case,
    unreachable_pub,
    unused_imports,
    clippy::allow_attributes_without_reason,
    clippy::enum_variant_names,
    clippy::manual_assert,
    clippy::mem_forget,
    reason = "upstream-derived CUDA support has staged API surface and FFI naming"
)]

pub mod cumem;
pub mod driver;
pub mod event;
pub mod rt;

mod device;
mod error;
mod mem;

pub use device::{CudaDeviceId, Device};
pub use error::{CudaError, CudaResult};
pub use mem::CudaDeviceMemory;
