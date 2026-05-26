#[allow(unused_imports)]
use crate::{cuda_sys, cudart_sys};
use std::{
    ffi::{CStr, c_void},
    ptr::NonNull,
};

pub type CudaResult<T> = std::result::Result<T, CudartError>;

#[derive(Clone, Debug)]
pub struct CudartError {
    pub code: u32,
    pub context: &'static str,
}

impl CudartError {
    pub fn new(code: u32, context: &'static str) -> Self {
        Self { code, context }
    }
}

impl std::fmt::Display for CudartError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CudartError: code {} ({:?}), context: {}",
            self.code,
            unsafe {
                let err: cudart_sys::cudaError = std::mem::transmute_copy(&self.code);
                CStr::from_ptr(cudart_sys::cudaGetErrorString(err))
            },
            self.context
        )
    }
}

impl std::error::Error for CudartError {}

pub use crate::cudart_sys::{cudaMemoryTypeDevice, cudaPointerAttributes};
pub fn cudaPointerGetAttributes(ptr: NonNull<c_void>) -> CudaResult<cudaPointerAttributes> {
    let mut attrs: cudaPointerAttributes = unsafe { std::mem::zeroed() };
    let ret = unsafe { cudart_sys::cudaPointerGetAttributes(&raw mut attrs, ptr.as_ptr()) };
    if ret == cudart_sys::cudaError::cudaSuccess {
        Ok(attrs)
    } else {
        Err(CudartError::new(ret as u32, "cudaPointerGetAttributes"))
    }
}

pub use crate::cudart_sys::cudaDeviceProp;
pub fn cudaGetDeviceProperties(device: i32) -> CudaResult<cudaDeviceProp> {
    let mut prop: cudaDeviceProp = unsafe { std::mem::zeroed() };
    let ret = unsafe { cudart_sys::cudaGetDeviceProperties(&raw mut prop, device) };
    if ret == cudart_sys::cudaError::cudaSuccess {
        Ok(prop)
    } else {
        Err(CudartError::new(ret as u32, "cudaGetDeviceProperties"))
    }
}

pub fn cudaGetDeviceCount() -> CudaResult<i32> {
    let mut count = 0;
    let ret = unsafe { cudart_sys::cudaGetDeviceCount(&raw mut count) };
    if ret == cudart_sys::cudaError::cudaSuccess {
        Ok(count)
    } else {
        Err(CudartError::new(ret as u32, "cudaGetDeviceCount"))
    }
}

pub fn cudaSetDevice(device: i32) -> CudaResult<()> {
    let ret = unsafe { cudart_sys::cudaSetDevice(device) };
    if ret == cudart_sys::cudaError::cudaSuccess {
        Ok(())
    } else {
        Err(CudartError::new(ret as u32, "cudaSetDevice"))
    }
}

pub fn cudaHostAlloc(size: usize, flags: u32) -> CudaResult<NonNull<c_void>> {
    let mut ptr = std::ptr::null_mut();
    let ret = unsafe { cudart_sys::cudaHostAlloc(&raw mut ptr, size, flags) };
    if ret == cudart_sys::cudaError::cudaSuccess {
        Ok(NonNull::new(ptr).unwrap())
    } else {
        Err(CudartError::new(ret as u32, "cudaHostAlloc"))
    }
}

pub fn cudaFreeHost(ptr: NonNull<c_void>) -> CudaResult<()> {
    let ret = unsafe { cudart_sys::cudaFreeHost(ptr.as_ptr()) };
    if ret == cudart_sys::cudaError::cudaSuccess {
        Ok(())
    } else {
        Err(CudartError::new(ret as u32, "cudaFreeHost"))
    }
}

pub fn cudaGetNumSMs(device: u8) -> CudaResult<usize> {
    let mut numSMs = 0;
    let ret = unsafe {
        cuda_sys::cuDeviceGetAttribute(
            &mut numSMs,
            cuda_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            device as i32,
        )
    };
    if ret == cuda_sys::cudaError_enum::CUDA_SUCCESS {
        Ok(numSMs as usize)
    } else {
        Err(CudartError::new(ret as u32, "cudaGetNumSMs"))
    }
}
