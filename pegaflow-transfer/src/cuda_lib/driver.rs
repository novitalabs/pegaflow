#[allow(unused_imports)]
use crate::{cuda_sys, cudart_sys};
use std::{
    ffi::{CStr, c_char, c_void},
    ptr::{NonNull, null},
};

use crate::cuda_sys::{
    CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, cuGetErrorString, cuMemGetHandleForAddressRange,
};

type Result<T> = std::result::Result<T, CudaDriverError>;

#[derive(Clone, Debug)]
pub struct CudaDriverError {
    pub code: u32,
    pub context: &'static str,
}

impl CudaDriverError {
    pub fn new(code: u32, context: &'static str) -> Self {
        Self { code, context }
    }
}

impl std::fmt::Display for CudaDriverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut errstr: *const c_char = null();
        unsafe {
            let err: cuda_sys::cudaError_enum = std::mem::transmute_copy(&self.code);
            cuGetErrorString(err, &mut errstr)
        };

        write!(
            f,
            "CudaDriverError: code {} ({:?}), context: {}",
            self.code,
            unsafe { CStr::from_ptr(errstr) },
            self.context
        )
    }
}

impl std::error::Error for CudaDriverError {}

pub fn cu_get_dma_buf_fd(ptr: NonNull<c_void>, len: usize) -> Result<i32> {
    let mut dmabuf_fd: i32 = -1;
    let ret = unsafe {
        cuMemGetHandleForAddressRange(
            &raw mut dmabuf_fd as *mut c_void,
            ptr.as_ptr() as u64,
            len,
            CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
            0,
        )
    };
    if ret == cuda_sys::cudaError_enum::CUDA_SUCCESS {
        Ok(dmabuf_fd)
    } else {
        Err(CudaDriverError::new(
            ret as u32,
            "cuMemGetHandleForAddressRange",
        ))
    }
}
