#[allow(unused_imports)]
use crate::{cuda_sys, cudart_sys, gdrapi_sys};
use std::{ffi::c_void, ptr::NonNull};

use crate::cudart_sys::{cudaHostAllocMapped, cudaHostAllocPortable, cudaMemAttachGlobal};
use libc::memset;

use crate::cuda_lib::rt::CudartError;
use crate::cuda_lib::rt::{cudaFreeHost, cudaHostAlloc, cudaSetDevice};

/// Owned Cuda memory. It will be freed when dropped.
pub struct CudaDeviceMemory {
    ptr: NonNull<c_void>,
    size: usize,
}

impl CudaDeviceMemory {
    /// Allocate a device-only CUDA buffer.
    pub fn device(size: usize) -> Result<Self, CudartError> {
        let mut ptr = std::ptr::null_mut();
        let ret = unsafe { cudart_sys::cudaMalloc(&raw mut ptr, size) };
        let ptr = NonNull::new(ptr).ok_or_else(|| CudartError::new(ret as u32, "cudaMalloc"))?;
        Ok(Self { ptr, size })
    }

    pub fn device_on(size: usize, device: u8) -> Result<Self, CudartError> {
        cudaSetDevice(device as i32)?;
        Self::device(size)
    }

    /// Allocate a CUDA buffer visible to both host and device.
    pub fn alloc(size: usize) -> Result<Self, CudartError> {
        let mut ptr = std::ptr::null_mut();
        let ret = unsafe { cudart_sys::cudaMallocManaged(&raw mut ptr, size, cudaMemAttachGlobal) };
        let ptr =
            NonNull::new(ptr).ok_or_else(|| CudartError::new(ret as u32, "cudaMallocManaged"))?;
        Ok(Self { ptr, size })
    }

    /// Create a device buffer, initialized from some host values.
    pub fn from_vec<T: Sized>(data: &[T]) -> Result<Self, CudartError> {
        let size = std::mem::size_of_val(data);
        let mem = Self::device(size)?;
        unsafe {
            cudart_sys::cudaMemcpy(
                mem.ptr.as_ptr(),
                data.as_ptr() as *const c_void,
                size,
                cudart_sys::cudaMemcpyHostToDevice,
            );
        }
        Ok(mem)
    }

    pub fn ptr(&self) -> NonNull<c_void> {
        self.ptr
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn leak(self) -> NonNull<c_void> {
        let ptr = self.ptr;
        std::mem::forget(self);
        ptr
    }

    pub fn zero(&self) {
        unsafe {
            cudart_sys::cudaMemset(self.ptr.as_ptr(), 0, self.size);
        }
    }

    pub fn fill(&self, value: u8) -> Result<(), CudartError> {
        let ret = unsafe { cudart_sys::cudaMemset(self.ptr.as_ptr(), value as i32, self.size) };
        if ret == cudart_sys::cudaError::cudaSuccess {
            Ok(())
        } else {
            Err(CudartError::new(ret as u32, "cudaMemset"))
        }
    }

    pub fn copy_to_vec(&self) -> Result<Vec<u8>, CudartError> {
        let mut data = vec![0; self.size];
        let ret = unsafe {
            cudart_sys::cudaMemcpy(
                data.as_mut_ptr() as *mut c_void,
                self.ptr.as_ptr(),
                self.size,
                cudart_sys::cudaMemcpyDeviceToHost,
            )
        };
        if ret == cudart_sys::cudaError::cudaSuccess {
            Ok(data)
        } else {
            Err(CudartError::new(ret as u32, "cudaMemcpyDeviceToHost"))
        }
    }

    pub fn get_ptr<T>(&self) -> *const T {
        self.ptr.as_ptr() as *const T
    }

    pub fn get_mut_ptr<T>(&mut self) -> *mut T {
        self.ptr.as_ptr() as *mut T
    }

    pub fn as_mut_slice<T>(&mut self) -> &mut [T] {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.ptr.as_ptr() as *mut T,
                self.size / std::mem::size_of::<T>(),
            )
        }
    }
}

impl Drop for CudaDeviceMemory {
    fn drop(&mut self) {
        unsafe { cudart_sys::cudaFree(self.ptr.as_ptr()) };
    }
}

unsafe impl Send for CudaDeviceMemory {}
unsafe impl Sync for CudaDeviceMemory {}

pub struct CudaHostMemory {
    pub ptr: NonNull<c_void>,
    pub size: usize,
}

impl CudaHostMemory {
    pub fn alloc(size: usize) -> Result<Self, CudartError> {
        let ptr = cudaHostAlloc(size, cudaHostAllocPortable | cudaHostAllocMapped)?;
        unsafe { memset(ptr.as_ptr(), 0, size) };
        Ok(CudaHostMemory { ptr, size })
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn get_ptr<T>(&self) -> *const T {
        self.ptr.as_ptr() as *const T
    }

    pub fn get_mut_ptr<T>(&self) -> *mut T {
        self.ptr.as_ptr() as *mut T
    }

    pub fn get_ref(&self, index: usize) -> &u64 {
        unsafe { &*((self.ptr.as_ptr() as *const u64).add(index)) }
    }

    pub fn get_mut(&mut self, index: usize) -> &mut u64 {
        unsafe { &mut *((self.ptr.as_ptr() as *mut u64).add(index)) }
    }

    pub fn zero(&self) {
        unsafe {
            let slice = std::slice::from_raw_parts_mut(self.ptr.as_ptr() as *mut u8, self.size);
            slice.fill(0);
        }
    }

    pub fn as_slice<T>(&self) -> &[T] {
        let elemsize = std::mem::size_of::<T>();
        assert!(self.size.is_multiple_of(elemsize));
        unsafe { std::slice::from_raw_parts(self.get_ptr::<T>(), self.size / elemsize) }
    }
}

impl Drop for CudaHostMemory {
    fn drop(&mut self) {
        if let Err(error) = cudaFreeHost(self.ptr) {
            panic!("Failed to free UVM memory: {}", error);
        }
    }
}

unsafe impl Send for CudaHostMemory {}
unsafe impl Sync for CudaHostMemory {}
