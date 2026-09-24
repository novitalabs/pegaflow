use std::any::Any;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::ptr::NonNull;
use std::sync::Arc;

use crate::cuda_lib::driver::{cu_get_address_range, cu_get_dma_buf_fd};
use crate::{Result, TransferError};

/// An owner-exported DMA-BUF mapped through CUDA IPC in this process.
/// The mapping owner must retain the CUDA IPC tensor until all MRs are gone.
pub struct CudaDmaBuf {
    pub ptr: u64,
    pub len: usize,
    fd: OwnedFd,
    _mapping_owner: Arc<dyn Any + Send + Sync>,
}

impl CudaDmaBuf {
    /// Validate that the FD covers the complete imported CUDA allocation.
    pub fn from_import(
        ptr: u64,
        len: usize,
        fd: OwnedFd,
        mapping_owner: Arc<dyn Any + Send + Sync>,
    ) -> Result<Self> {
        let (base, allocation_len) =
            cu_get_address_range(ptr).map_err(|error| TransferError::Backend(error.to_string()))?;
        if base != ptr || allocation_len != len || len == 0 {
            return Err(TransferError::InvalidArgument(
                "DMA-BUF range does not match the imported CUDA allocation",
            ));
        }
        Ok(Self {
            ptr,
            len,
            fd,
            _mapping_owner: mapping_owner,
        })
    }

    pub(crate) fn fd(&self) -> RawFd {
        self.fd.as_raw_fd()
    }
}

impl std::fmt::Debug for CudaDmaBuf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaDmaBuf")
            .field("ptr", &self.ptr)
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

/// Export in the allocation-owning process with its CUDA context current.
/// Returns an owned FD, allocation base and full allocation length.
pub fn export_cuda_dma_buf(ptr: u64, len: usize) -> Result<(OwnedFd, u64, usize)> {
    let (base, allocation_len) =
        cu_get_address_range(ptr).map_err(|error| TransferError::Backend(error.to_string()))?;
    if len == 0
        || ptr < base
        || ptr.checked_add(len as u64).is_none()
        || base.checked_add(allocation_len as u64).is_none()
        || ptr + len as u64 > base + allocation_len as u64
    {
        return Err(TransferError::InvalidArgument(
            "CUDA view exceeds its allocation",
        ));
    }
    let base_ptr = NonNull::new(base as *mut std::ffi::c_void)
        .ok_or(TransferError::InvalidArgument("CUDA allocation is null"))?;
    let fd = cu_get_dma_buf_fd(base_ptr, allocation_len).map_err(|error| {
        TransferError::Backend(format!("CUDA owner DMA-BUF export failed: {error}"))
    })?;
    // CUDA returned a new process-local descriptor, uniquely owned here.
    Ok((unsafe { OwnedFd::from_raw_fd(fd) }, base, allocation_len))
}
