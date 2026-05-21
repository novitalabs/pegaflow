//! Raw FFI bindings for the CUDA runtime API.
//!
//! Re-exported from `cudarc` so transfer uses the same CUDA bindings as the
//! rest of the workspace.
#![allow(warnings)]

pub use cudarc::runtime::sys::cudaDeviceAttr::cudaDevAttrMultiProcessorCount;
pub use cudarc::runtime::sys::cudaGetDeviceProperties_v2 as cudaGetDeviceProperties;
pub use cudarc::runtime::sys::cudaMemcpyKind::cudaMemcpyHostToDevice;
pub use cudarc::runtime::sys::cudaMemoryType::cudaMemoryTypeDevice;
pub use cudarc::runtime::sys::*;
