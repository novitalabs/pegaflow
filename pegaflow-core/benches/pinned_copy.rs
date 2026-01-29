use std::{ffi::c_void, ptr, sync::Arc};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use cudarc::driver::{CudaContext, sys};

const BYTES_PER_BLOCK: usize = 24 * 1024; // 24 KiB segments
const SEGMENTS_PER_BLOCK: usize = 2;
const NUM_BLOCKS_IN_LAYER: usize = 27;

#[derive(Debug)]
struct TransferFixture {
    ctx: Arc<CudaContext>,
    host_ptr: *mut u8,
    device_ptr: sys::CUdeviceptr,
    bytes_per_block: usize,
    segments: usize,
    kv_stride: usize,
}

impl TransferFixture {
    fn new(bytes_per_block: usize, segments: usize, kv_stride: usize) -> Self {
        assert!(segments > 0, "segments must be > 0");
        assert!(bytes_per_block > 0, "bytes_per_block must be > 0");
        assert!(kv_stride >= bytes_per_block, "stride must be >= block size");

        let ctx = CudaContext::new(0).expect("CUDA context");
        ctx.bind_to_thread().expect("bind CUDA context");

        let block_bytes = bytes_per_block * segments;
        let mut host_ptr: *mut c_void = ptr::null_mut();
        check_cuda(
            unsafe { sys::cuMemAllocHost_v2(&mut host_ptr, block_bytes) },
            "cuMemAllocHost_v2",
        );
        let host_ptr = host_ptr as *mut u8;
        unsafe {
            let slice = std::slice::from_raw_parts_mut(host_ptr, block_bytes);
            for (idx, byte) in slice.iter_mut().enumerate() {
                *byte = (idx & 0xFF) as u8;
            }
        }

        let mut device_ptr: sys::CUdeviceptr = 0;
        let device_bytes = kv_stride * (segments - 1) + bytes_per_block;
        check_cuda(
            unsafe { sys::cuMemAlloc_v2(&mut device_ptr, device_bytes) },
            "cuMemAlloc_v2",
        );

        Self {
            ctx,
            host_ptr,
            device_ptr,
            bytes_per_block,
            segments,
            kv_stride,
        }
    }

    fn block_bytes(&self) -> usize {
        self.bytes_per_block * self.segments
    }

    fn ensure_context(&self) {
        self.ctx
            .bind_to_thread()
            .unwrap_or_else(|err| panic!("Failed to bind CUDA context: {err:?}"));
    }

    fn copy_single(&self) {
        self.ensure_context();
        check_cuda(
            unsafe {
                sys::cuMemcpyHtoD_v2(
                    self.device_ptr,
                    self.host_ptr as *const c_void,
                    self.block_bytes(),
                )
            },
            "cuMemcpyHtoD_v2(single)",
        );
    }

    fn copy_segments(&self) {
        self.ensure_context();
        for segment in 0..self.segments {
            let src = unsafe { self.host_ptr.add(segment * self.bytes_per_block) };
            let dst = self.device_ptr + (segment * self.kv_stride) as u64;
            check_cuda(
                unsafe { sys::cuMemcpyHtoD_v2(dst, src as *const c_void, self.bytes_per_block) },
                "cuMemcpyHtoD_v2(segment)",
            );
        }
    }

    fn copy_with_memcpy2d(&self) {
        self.ensure_context();
        let request = sys::CUDA_MEMCPY2D_st {
            srcXInBytes: 0,
            srcY: 0,
            srcMemoryType: sys::CUmemorytype_enum::CU_MEMORYTYPE_HOST,
            srcHost: self.host_ptr as *const c_void,
            srcDevice: 0,
            srcArray: ptr::null_mut(),
            srcPitch: self.bytes_per_block,
            dstXInBytes: 0,
            dstY: 0,
            dstMemoryType: sys::CUmemorytype_enum::CU_MEMORYTYPE_DEVICE,
            dstHost: ptr::null_mut(),
            dstDevice: self.device_ptr,
            dstArray: ptr::null_mut(),
            dstPitch: self.kv_stride,
            WidthInBytes: self.bytes_per_block,
            Height: self.segments,
        };

        check_cuda(unsafe { sys::cuMemcpy2D_v2(&request) }, "cuMemcpy2D_v2");
    }
}

impl Drop for TransferFixture {
    fn drop(&mut self) {
        unsafe {
            if self.device_ptr != 0 {
                sys::cuMemFree_v2(self.device_ptr);
            }
            if !self.host_ptr.is_null() {
                sys::cuMemFreeHost(self.host_ptr as *mut c_void);
            }
        }
    }
}

fn contiguous_benchmarks(c: &mut Criterion) {
    let block_bytes = BYTES_PER_BLOCK * SEGMENTS_PER_BLOCK;
    let mut group = c.benchmark_group("pinned_contiguous");
    group.throughput(Throughput::Bytes(block_bytes as u64));

    group.bench_function(BenchmarkId::new("single_memcpy", block_bytes), |b| {
        let fixture = TransferFixture::new(BYTES_PER_BLOCK, SEGMENTS_PER_BLOCK, BYTES_PER_BLOCK);
        b.iter(|| fixture.copy_single());
    });

    group.bench_function(BenchmarkId::new("loop_segments", block_bytes), |b| {
        let fixture = TransferFixture::new(BYTES_PER_BLOCK, SEGMENTS_PER_BLOCK, BYTES_PER_BLOCK);
        b.iter(|| fixture.copy_segments());
    });

    group.finish();
}

fn strided_benchmarks(c: &mut Criterion) {
    let block_bytes = BYTES_PER_BLOCK * SEGMENTS_PER_BLOCK;
    let kv_stride = BYTES_PER_BLOCK * NUM_BLOCKS_IN_LAYER;
    let mut group = c.benchmark_group("pinned_strided");
    group.throughput(Throughput::Bytes(block_bytes as u64));

    group.bench_function(BenchmarkId::new("loop_segments", block_bytes), |b| {
        let fixture = TransferFixture::new(BYTES_PER_BLOCK, SEGMENTS_PER_BLOCK, kv_stride);
        b.iter(|| fixture.copy_segments());
    });

    group.bench_function(BenchmarkId::new("memcpy2d", block_bytes), |b| {
        let fixture = TransferFixture::new(BYTES_PER_BLOCK, SEGMENTS_PER_BLOCK, kv_stride);
        b.iter(|| fixture.copy_with_memcpy2d());
    });

    group.finish();
}

fn check_cuda(result: sys::CUresult, op: &str) {
    if result != sys::CUresult::CUDA_SUCCESS {
        panic!("{op} failed with {result:?}");
    }
}

criterion_group!(benches, contiguous_benchmarks, strided_benchmarks);
criterion_main!(benches);
