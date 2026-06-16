//! Reproduces the `QueryBlocksForTransfer` CPU cost: building + postcard-encoding
//! the `TransferPlan` from sealed blocks (server side), and decoding + validating
//! it (client side). The wire stage breakdown showed query ~182 ms for 4500
//! blocks, scaling with `blocks * slots`.
//!
//! Run:
//!   cargo bench -p pegaflow-core --no-default-features \
//!       --features bench,cuda-13,rdma --bench transfer_query

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use pegaflow_core::bench_support::QueryFixture;

fn bench_query(c: &mut Criterion) {
    const SLOTS: usize = 288; // layers(36) * tp(8), split K/V
    const SEG_BYTES: usize = 16;

    for stage in ["encode", "decode"] {
        let mut group = c.benchmark_group(format!("query_{stage}"));
        for &blocks in &[70usize, 281, 4500] {
            let fixture = QueryFixture::new(blocks, SLOTS, SEG_BYTES);
            if blocks == 4500 {
                eprintln!("wire bytes (4500 blocks) = {}", fixture.wire_len());
            }
            group.throughput(Throughput::Elements((blocks * SLOTS * 2) as u64));
            group.bench_with_input(
                BenchmarkId::from_parameter(blocks),
                &fixture,
                |b, fixture| match stage {
                    "encode" => b.iter(|| black_box(fixture.encode())),
                    _ => b.iter(|| black_box(fixture.decode())),
                },
            );
        }
        group.finish();
    }
}

criterion_group!(benches, bench_query);
criterion_main!(benches);
