use crate::cuda_lib::rt::{CudaResult, CudartError};
#[allow(unused_imports)]
use crate::{cuda_sys, cudart_sys, gdrapi_sys};

pub struct CudaEvent {
    pub event: cudart_sys::cudaEvent_t,
}

impl CudaEvent {
    pub fn new() -> CudaResult<Self> {
        let mut event = std::ptr::null_mut();
        let ret = unsafe { cudart_sys::cudaEventCreate(&mut event) };
        if ret != cudart_sys::cudaError::cudaSuccess {
            return Err(CudartError::new(ret as u32, "cudaEventCreate"));
        }
        Ok(CudaEvent { event })
    }

    pub fn record(&self) -> CudaResult<()> {
        let ret = unsafe { cudart_sys::cudaEventRecord(self.event, std::ptr::null_mut()) };
        if ret != cudart_sys::cudaError::cudaSuccess {
            return Err(CudartError::new(ret as u32, "cudaEventRecord"));
        }
        Ok(())
    }

    pub fn synchronize(&self) -> CudaResult<()> {
        let ret = unsafe { cudart_sys::cudaEventSynchronize(self.event) };
        if ret != cudart_sys::cudaError::cudaSuccess {
            return Err(CudartError::new(ret as u32, "cudaEventSynchronize"));
        }
        Ok(())
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        let ret = unsafe { cudart_sys::cudaEventDestroy(self.event) };
        if ret != cudart_sys::cudaError::cudaSuccess {
            panic!("cudaEventDestroy failed: {:?}", ret);
        }
    }
}

unsafe impl Send for CudaEvent {}
unsafe impl Sync for CudaEvent {}
