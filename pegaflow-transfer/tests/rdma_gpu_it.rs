use std::{env, ffi::c_void};

use cudarc::driver::{CudaContext, sys};
use pegaflow_transfer::MooncakeTransferEngine;

const ENV_RDMA_NIC: &str = "PEGAFLOW_TRANSFER_IT_NIC";
const ENV_BASE_PORT: &str = "PEGAFLOW_TRANSFER_IT_BASE_PORT";

struct GpuBuffer {
    ptr: sys::CUdeviceptr,
    len: usize,
}

impl GpuBuffer {
    fn alloc(len: usize) -> Self {
        let mut ptr: sys::CUdeviceptr = 0;
        check_cuda(
            unsafe { sys::cuMemAlloc_v2(&raw mut ptr, len) },
            "cuMemAlloc_v2",
        );
        Self { ptr, len }
    }

    fn as_u64(&self) -> u64 {
        self.ptr
    }

    fn copy_from_host(&self, data: &[u8]) {
        assert_eq!(data.len(), self.len);
        check_cuda(
            unsafe { sys::cuMemcpyHtoD_v2(self.ptr, data.as_ptr() as *const c_void, self.len) },
            "cuMemcpyHtoD_v2",
        );
    }

    fn copy_to_host(&self) -> Vec<u8> {
        let mut output = vec![0_u8; self.len];
        check_cuda(
            unsafe { sys::cuMemcpyDtoH_v2(output.as_mut_ptr() as *mut c_void, self.ptr, self.len) },
            "cuMemcpyDtoH_v2",
        );
        output
    }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        if self.ptr != 0 {
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

fn read_env(name: &str) -> Option<String> {
    env::var(name).ok().and_then(|value| {
        let trimmed = value.trim().to_string();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed)
        }
    })
}

#[test]
#[ignore = "requires RDMA + CUDA; set PEGAFLOW_TRANSFER_IT_NIC"]
fn it_rdma_gpu_transfer_sync_write() -> Result<(), Box<dyn std::error::Error>> {
    let Some(nic_name) = read_env(ENV_RDMA_NIC) else {
        eprintln!("skip: {ENV_RDMA_NIC} is not set");
        return Ok(());
    };
    let base_port = read_env(ENV_BASE_PORT)
        .as_deref()
        .unwrap_or("56050")
        .parse::<u16>()?;

    let Ok(_ctx) = CudaContext::new(0) else {
        eprintln!("skip: CUDA context (device 0) is not available");
        return Ok(());
    };

    let bytes = 4096usize;
    let src = GpuBuffer::alloc(bytes);
    let dst = GpuBuffer::alloc(bytes);

    let mut payload = vec![0_u8; bytes];
    for (idx, value) in payload.iter_mut().enumerate() {
        *value = (idx % 251) as u8;
    }
    src.copy_from_host(&payload);
    dst.copy_from_host(&vec![0_u8; bytes]);

    let mut sender = MooncakeTransferEngine::new();
    sender.initialize("worker-a", "rdma", nic_name.clone(), base_port)?;
    sender.register_memory(src.as_u64(), bytes)?;

    let mut receiver = MooncakeTransferEngine::new();
    receiver.initialize("worker-b", "rdma", nic_name, base_port + 10)?;
    receiver.register_memory(dst.as_u64(), bytes)?;

    let receiver_session = receiver.get_session_id()?;

    let written =
        sender.transfer_sync_write(&receiver_session, src.as_u64(), dst.as_u64(), bytes)?;
    assert_eq!(written, bytes);

    let output = dst.copy_to_host();
    assert_eq!(output, payload);

    sender.unregister_memory(src.as_u64())?;
    receiver.unregister_memory(dst.as_u64())?;
    Ok(())
}
