//! Benchmark for batch GPU<>CPU transfer functions.
//!
//! Compares `batch_copy_segments_to_gpu` performance under cuda-12 (merge-contiguous
//! loop) vs cuda-12.8+/13 (`cuMemcpyBatchAsync_v2`).
//!
//! Key question: how much does the batch API help when segments are scattered?
//!
//! Matrix (4 benchmarks):
//!   batch_count: 64, 1024
//!   fragmentation: contiguous, fully scattered
//!   segment_size: 240 KiB (10x enlarged for comparison)
//!   direction: H2D only (D2H path is symmetric)

use std::ffi::c_void;
use std::ptr;
use std::sync::Arc;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use cudarc::driver::sys;
use cudarc::driver::{CudaContext, CudaStream};
use pegaflow_core::KVCacheRegistration;
use pegaflow_core::transfer::batch_copy_segments_to_gpu;

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

struct BatchFixture {
    _ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    host_ptr: *mut u8,
    device_ptr: sys::CUdeviceptr,
    registration: KVCacheRegistration,
}

impl BatchFixture {
    fn new(segment_size: usize, num_segments: usize) -> Self {
        let ctx = CudaContext::new(0).expect("CUDA context");
        ctx.bind_to_thread().expect("bind");
        let stream = ctx.new_stream().expect("non-default stream");

        let total_bytes = segment_size * num_segments;

        let mut host_raw: *mut c_void = ptr::null_mut();
        check_cuda(
            unsafe { sys::cuMemAllocHost_v2(&mut host_raw, total_bytes) },
            "cuMemAllocHost_v2",
        );
        assert!(!host_raw.is_null());

        let host_ptr = host_raw as *mut u8;
        unsafe {
            let slice = std::slice::from_raw_parts_mut(host_ptr, total_bytes);
            for (i, b) in slice.iter_mut().enumerate() {
                *b = (i & 0xFF) as u8;
            }
        }

        let mut device_ptr: sys::CUdeviceptr = 0;
        check_cuda(
            unsafe { sys::cuMemAlloc_v2(&mut device_ptr, total_bytes) },
            "cuMemAlloc_v2",
        );

        let registration = KVCacheRegistration {
            data_ptr: device_ptr,
            size_bytes: total_bytes,
            num_blocks: num_segments,
            bytes_per_block: segment_size,
            block_size_bytes: segment_size,
            kv_stride_bytes: 0,
            segments: 1,
            padded_bytes_per_block: segment_size,
            padded_block_size_bytes: segment_size,
        };

        Self {
            _ctx: ctx,
            stream,
            host_ptr,
            device_ptr,
            registration,
        }
    }

    /// Build H2D transfer list.
    /// `scattered`: if true, reverse-swap all GPU offsets so nothing is contiguous.
    fn h2d_transfers(&self, num_segments: usize, scattered: bool) -> Vec<(usize, *const u8)> {
        let seg = self.registration.bytes_per_block;
        let mut gpu_offsets: Vec<usize> = (0..num_segments).map(|i| i * seg).collect();

        if scattered {
            // Full reverse — no pair is contiguous
            gpu_offsets.reverse();
        }

        gpu_offsets
            .into_iter()
            .enumerate()
            .map(|(i, gpu_off)| {
                let cpu_ptr = unsafe { self.host_ptr.add(i * seg) } as *const u8;
                (gpu_off, cpu_ptr)
            })
            .collect()
    }

    fn sync(&self) {
        check_cuda(
            unsafe { sys::cuStreamSynchronize(self.stream.cu_stream()) },
            "cuStreamSynchronize",
        );
    }
}

impl Drop for BatchFixture {
    fn drop(&mut self) {
        unsafe {
            if self.device_ptr != 0 {
                let _ = sys::cuMemFree_v2(self.device_ptr);
            }
            if !self.host_ptr.is_null() {
                let _ = sys::cuMemFreeHost(self.host_ptr as *mut c_void);
            }
        }
    }
}

fn check_cuda(result: sys::CUresult, op: &str) {
    assert!(
        result == sys::CUresult::CUDA_SUCCESS,
        "{op} failed with {result:?}"
    );
}

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

const SEGMENT_SIZE: usize = 240 * 1024; // 240 KiB, 10x enlarged for comparison
const BATCH_COUNTS: &[usize] = &[64, 1024];

// ---------------------------------------------------------------------------
// Benchmark
// ---------------------------------------------------------------------------

fn h2d_benchmarks(c: &mut Criterion) {
    for &count in BATCH_COUNTS {
        let total_bytes = SEGMENT_SIZE * count;
        let group_name = format!("h2d/240KiB/n{count}");
        let mut group = c.benchmark_group(&group_name);
        group.throughput(Throughput::Bytes(total_bytes as u64));

        for (name, scattered) in [("contiguous", false), ("scattered", true)] {
            group.bench_function(BenchmarkId::new("batch_copy", name), |b| {
                let fixture = BatchFixture::new(SEGMENT_SIZE, count);
                let transfers = fixture.h2d_transfers(count, scattered);

                b.iter(|| {
                    batch_copy_segments_to_gpu(
                        &transfers,
                        SEGMENT_SIZE,
                        &fixture.registration,
                        &fixture.stream,
                    )
                    .expect("h2d failed");
                    fixture.sync();
                });
            });
        }
        group.finish();
    }
}

criterion_group!(benches, h2d_benchmarks);
criterion_main!(benches);
