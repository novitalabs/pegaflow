use std::ffi::c_void;

use cudarc::driver::sys;

pub struct GpuBuffer {
    ptr: sys::CUdeviceptr,
    len: usize,
}

impl GpuBuffer {
    pub fn alloc(len: usize) -> Self {
        assert!(len > 0, "GpuBuffer::alloc: len must be > 0");
        let mut ptr: sys::CUdeviceptr = 0;
        // SAFETY: len > 0 (asserted above). ptr is a valid stack pointer.
        check_cuda(
            unsafe { sys::cuMemAlloc_v2(&raw mut ptr, len) },
            "cuMemAlloc_v2",
        );
        Self { ptr, len }
    }

    pub fn as_u64(&self) -> u64 {
        self.ptr
    }

    pub fn copy_from_host(&self, data: &[u8]) {
        assert_eq!(data.len(), self.len);
        // SAFETY: self.ptr is a valid device pointer from cuMemAlloc_v2.
        // data.len() matches the copy size.
        check_cuda(
            unsafe { sys::cuMemcpyHtoD_v2(self.ptr, data.as_ptr() as *const c_void, self.len) },
            "cuMemcpyHtoD_v2",
        );
    }

    pub fn copy_to_host(&self) -> Vec<u8> {
        let mut output = vec![0u8; self.len];
        // SAFETY: self.ptr is valid, out has sufficient capacity (len bytes).
        check_cuda(
            unsafe { sys::cuMemcpyDtoH_v2(output.as_mut_ptr() as *mut c_void, self.ptr, self.len) },
            "cuMemcpyDtoH_v2",
        );
        output
    }

    pub fn zero(&self) {
        // SAFETY: self.ptr is a valid device pointer, self.len matches the allocation.
        check_cuda(
            unsafe { sys::cuMemsetD8_v2(self.ptr, 0, self.len) },
            "cuMemsetD8_v2",
        );
    }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        if self.ptr != 0 {
            // SAFETY: self.ptr was allocated by cuMemAlloc_v2 and has not been freed.
            check_cuda(unsafe { sys::cuMemFree_v2(self.ptr) }, "cuMemFree_v2");
            self.ptr = 0;
        }
    }
}

fn check_cuda(result: sys::CUresult, op: &str) {
    assert!(
        result == sys::CUresult::CUDA_SUCCESS,
        "{op} failed with {result:?}"
    );
}
