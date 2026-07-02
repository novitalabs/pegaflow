//! Raw FFI bindings for the CUDA driver API.
//!
//! Re-exported from `cudarc` so transfer uses the same CUDA bindings as the
//! rest of the workspace.
#![allow(warnings)]

pub use cudarc::driver::sys::*;
pub const CUDA_SUCCESS: u32 = cudaError_enum::CUDA_SUCCESS as u32;

pub use cudarc::driver::sys::CUmemAccess_flags_enum::CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
pub use cudarc::driver::sys::CUmemAllocationGranularity_flags_enum::CU_MEM_ALLOC_GRANULARITY_MINIMUM;
// `const` re-bindings, not `pub use`: cudarc 0.19.7 regenerated this type as a
// tuple struct with associated constants (0.19.3 had a Rust enum), and
// associated constants cannot be imported with `use`. The `Type::NAME` path
// syntax is identical for both forms, so these bindings compile against
// either cudarc generation.
pub const CU_MEM_HANDLE_TYPE_FABRIC: CUmemAllocationHandleType_enum =
    CUmemAllocationHandleType_enum::CU_MEM_HANDLE_TYPE_FABRIC;
pub const CU_MEM_HANDLE_TYPE_NONE: CUmemAllocationHandleType_enum =
    CUmemAllocationHandleType_enum::CU_MEM_HANDLE_TYPE_NONE;
pub const CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR: CUmemAllocationHandleType_enum =
    CUmemAllocationHandleType_enum::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
pub use cudarc::driver::sys::CUmemAllocationType_enum::CU_MEM_ALLOCATION_TYPE_PINNED;
pub use cudarc::driver::sys::CUmemLocationType_enum::CU_MEM_LOCATION_TYPE_DEVICE;
pub use cudarc::driver::sys::CUmemRangeHandleType_enum::CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD;
pub use cudarc::driver::sys::CUmulticastGranularity_flags_enum::CU_MULTICAST_GRANULARITY_MINIMUM;

pub unsafe fn cuMemAlloc(dptr: *mut u64, bytesize: usize) -> CUresult {
    cuMemAlloc_v2(dptr, bytesize)
}

pub unsafe fn cuMemFree(dptr: u64) -> CUresult {
    cuMemFree_v2(dptr)
}
