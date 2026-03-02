use std::{env, ffi::c_void, time::Instant};

use cudarc::driver::{CudaContext, sys};
use pegaflow_transfer::MooncakeTransferEngine;

const ENV_RDMA_NIC: &str = "PEGAFLOW_TRANSFER_IT_NIC";
const ENV_BASE_PORT: &str = "PEGAFLOW_TRANSFER_IT_BASE_PORT";
const ENV_TRANSFER_BYTES: &str = "PEGAFLOW_TRANSFER_IT_BYTES";
const ENV_TRANSFER_WARMUP: &str = "PEGAFLOW_TRANSFER_IT_WARMUP";
const ENV_TRANSFER_ITERS: &str = "PEGAFLOW_TRANSFER_IT_ITERS";
const DEFAULT_TRANSFER_BYTES: usize = 1 << 30;
const DEFAULT_TRANSFER_WARMUP: usize = 1;
const DEFAULT_TRANSFER_ITERS: usize = 20;

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

fn gib_per_sec(bytes: usize, elapsed_secs: f64) -> f64 {
    if elapsed_secs <= 0.0 {
        return 0.0;
    }
    bytes as f64 / elapsed_secs / (1024.0 * 1024.0 * 1024.0)
}

fn gbps(bytes: usize, elapsed_secs: f64) -> f64 {
    if elapsed_secs <= 0.0 {
        return 0.0;
    }
    (bytes as f64 * 8.0) / elapsed_secs / 1e9
}

fn percentile_index(len: usize, p: f64) -> usize {
    (((len as f64) * p).ceil() as usize)
        .saturating_sub(1)
        .min(len.saturating_sub(1))
}

#[test]
#[ignore = "requires RDMA + CUDA; set PEGAFLOW_TRANSFER_IT_NIC"]
fn it_rdma_gpu_transfer_sync_write() -> Result<(), Box<dyn std::error::Error>> {
    let total_start = Instant::now();
    let Some(nic_name) = read_env(ENV_RDMA_NIC) else {
        eprintln!("skip: {ENV_RDMA_NIC} is not set");
        return Ok(());
    };
    let base_port = read_env(ENV_BASE_PORT)
        .as_deref()
        .unwrap_or("56050")
        .parse::<u16>()?;
    let bytes = read_env(ENV_TRANSFER_BYTES)
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(DEFAULT_TRANSFER_BYTES);
    let warmup = read_env(ENV_TRANSFER_WARMUP)
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(DEFAULT_TRANSFER_WARMUP);
    let iters = read_env(ENV_TRANSFER_ITERS)
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(DEFAULT_TRANSFER_ITERS);
    if iters == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("{ENV_TRANSFER_ITERS} must be > 0"),
        )
        .into());
    }
    eprintln!(
        "[it] start: nic={} base_port={} cuda_device=0 bytes={} warmup={} iters={}",
        nic_name, base_port, bytes, warmup, iters
    );

    let setup_start = Instant::now();
    let Ok(_ctx) = CudaContext::new(0) else {
        eprintln!("skip: CUDA context (device 0) is not available");
        return Ok(());
    };

    let src = GpuBuffer::alloc(bytes);
    let dst = GpuBuffer::alloc(bytes);

    let mut payload = vec![0_u8; bytes];
    for (idx, value) in payload.iter_mut().enumerate() {
        *value = (idx % 251) as u8;
    }
    src.copy_from_host(&payload);
    dst.copy_from_host(&vec![0_u8; bytes]);

    let mut sender = MooncakeTransferEngine::new();
    sender.initialize(nic_name.clone(), base_port)?;
    sender.register_memory(src.as_u64(), bytes)?;

    let mut receiver = MooncakeTransferEngine::new();
    receiver.initialize(nic_name, base_port + 10)?;
    receiver.register_memory(dst.as_u64(), bytes)?;

    let receiver_session = receiver.get_session_id();
    let setup_elapsed = setup_start.elapsed();
    eprintln!(
        "[it] setup done: bytes={} session={} elapsed_ms={:.3}",
        bytes,
        receiver_session,
        setup_elapsed.as_secs_f64() * 1_000.0
    );

    let total_iters = warmup + iters;
    let mut measured_elapsed_secs = Vec::with_capacity(iters);
    for iter in 0..total_iters {
        let transfer_start = Instant::now();
        let written =
            sender.transfer_sync_write(&receiver_session, src.as_u64(), dst.as_u64(), bytes)?;
        assert_eq!(written, bytes);
        let elapsed = transfer_start.elapsed();
        let elapsed_secs = elapsed.as_secs_f64();

        if iter < warmup {
            eprintln!(
                "[it] warmup iter={}/{} elapsed_us={} throughput_gib_s={:.6} throughput_gbps={:.6}",
                iter + 1,
                warmup,
                elapsed.as_micros(),
                gib_per_sec(written, elapsed_secs),
                gbps(written, elapsed_secs)
            );
            continue;
        }
        measured_elapsed_secs.push(elapsed_secs);
    }

    let sum_elapsed_secs: f64 = measured_elapsed_secs.iter().sum();
    let total_bytes = bytes * measured_elapsed_secs.len();
    let avg_gib_s = gib_per_sec(total_bytes, sum_elapsed_secs);
    let avg_gbps = gbps(total_bytes, sum_elapsed_secs);

    let mut lat_ms: Vec<f64> = measured_elapsed_secs
        .iter()
        .map(|sec| sec * 1_000.0)
        .collect();
    lat_ms.sort_by(f64::total_cmp);
    let p50_lat_ms = lat_ms[percentile_index(lat_ms.len(), 0.50)];
    let p95_lat_ms = lat_ms[percentile_index(lat_ms.len(), 0.95)];

    let mut throughput_gib_s: Vec<f64> = measured_elapsed_secs
        .iter()
        .map(|sec| gib_per_sec(bytes, *sec))
        .collect();
    throughput_gib_s.sort_by(f64::total_cmp);
    let p50_gib_s = throughput_gib_s[percentile_index(throughput_gib_s.len(), 0.50)];
    let p95_gib_s = throughput_gib_s[percentile_index(throughput_gib_s.len(), 0.95)];
    let p50_gbps = p50_gib_s * 8.0 * 1024.0 * 1024.0 * 1024.0 / 1e9;
    let p95_gbps = p95_gib_s * 8.0 * 1024.0 * 1024.0 * 1024.0 / 1e9;

    eprintln!(
        "[it] transfer summary: bytes={} warmup={} measured_iters={} avg_gib_s={:.6} avg_gbps={:.6} p50_lat_ms={:.3} p95_lat_ms={:.3} p50_gib_s={:.6} p95_gib_s={:.6} p50_gbps={:.6} p95_gbps={:.6}",
        bytes,
        warmup,
        measured_elapsed_secs.len(),
        avg_gib_s,
        avg_gbps,
        p50_lat_ms,
        p95_lat_ms,
        p50_gib_s,
        p95_gib_s,
        p50_gbps,
        p95_gbps
    );

    let verify_start = Instant::now();
    let output = dst.copy_to_host();
    assert_eq!(output, payload);

    sender.unregister_memory(src.as_u64())?;
    receiver.unregister_memory(dst.as_u64())?;
    eprintln!(
        "[it] verify+cleanup done: elapsed_ms={:.3}, total_elapsed_ms={:.3}",
        verify_start.elapsed().as_secs_f64() * 1_000.0,
        total_start.elapsed().as_secs_f64() * 1_000.0
    );
    Ok(())
}
