//! Isolated profiling driver for the `QueryBlocksForTransfer` encode path.
//! Builds one vLLM-shaped fixture once, then encodes it in a tight loop so a
//! sampling profiler sees ~only the plan-build + postcard-encode hot path.
//!
//!   cargo build --example encode_profile --release --no-default-features \
//!       --features bench,cuda-13,rdma
//!   samply record -- target/release/examples/encode_profile

use std::hint::black_box;
use std::time::Instant;

use pegaflow_core::bench_support::QueryFixture;

const BLOCKS: usize = 4500;
const SLOTS: usize = 288;
const SEG_BYTES: usize = 16;
const ITERS: usize = 40;

fn main() {
    let fixture = QueryFixture::new(BLOCKS, SLOTS, SEG_BYTES);
    eprintln!(
        "fixture blocks={BLOCKS} slots={SLOTS} wire_bytes={}; looping encode {ITERS}x",
        fixture.wire_len()
    );
    let t0 = Instant::now();
    let mut acc = 0usize;
    for _ in 0..ITERS {
        acc += black_box(fixture.encode());
    }
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / ITERS as f64;
    eprintln!("mean encode = {ms:.2} ms/iter (acc={acc})");
}
