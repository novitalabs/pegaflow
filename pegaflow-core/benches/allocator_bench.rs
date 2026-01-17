/* This is a Rust benchmark for ScaledOffsetAllocator that simulates a steady-state LLM KV-cache workload, 
closely mirroring allocator_bench.cpp and production usage. */

use pegaflow_core::allocator::{Allocation, ScaledOffsetAllocator};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::VecDeque;
use std::time::Instant;

const PAGE_SIZE_BYTES: u64 = 128 * 1024;
const MIN_PAGES: u64 = 20;
const MAX_PAGES: u64 = 100;
const CYCLES: usize = 20_000;

type Allocator = ScaledOffsetAllocator;

pub fn benchmark_kv_cache_fragmentation() {
    let mut rng = StdRng::seed_from_u64(42);
    let mut allocator = Allocator::new(1_000_000_000_000).unwrap();
    let mut live: VecDeque<Allocation> = VecDeque::new();

    let initial_report = allocator.storage_report();
    let capacity = initial_report.total_free_bytes;

    let mut total_allocated = 0u64;
    let mut internal_frag = 0u64;

    let mut oom_count = 0u64;
    let mut oom_free_sum = 0u64;
    let mut oom_largest_sum = 0u64;

    let mut alloc_latency = std::time::Duration::ZERO;
    let mut alloc_calls = 0u64;

    let mut min_util: f64 = 1.0;

    loop {
        let report = allocator.storage_report();
        let used = capacity.saturating_sub(report.total_free_bytes);
        let utilization = used as f64 / capacity as f64;

        if utilization >= 0.85 {
            break;
        }

        let pages = rng.random_range(MIN_PAGES..=MAX_PAGES);
        let req = pages * PAGE_SIZE_BYTES;

        match allocator.allocate(req) {
            Ok(Some(a)) => {
                total_allocated += a.size_bytes.get();
                internal_frag += a.size_bytes.get().saturating_sub(req);
                live.push_back(a);
            }
            _ => break,
        }
    }

    let start = Instant::now();

    for cycle in 0..CYCLES {
        let pages = rng.random_range(MIN_PAGES..=MAX_PAGES);
        let req = pages * PAGE_SIZE_BYTES;

        let t0 = Instant::now();
        let res = allocator.allocate(req);
        alloc_latency += t0.elapsed();
        alloc_calls += 1;

        match res {
            Ok(Some(a)) => {
                total_allocated += a.size_bytes.get();
                internal_frag += a.size_bytes.get().saturating_sub(req);
                live.push_back(a);
            }
            Ok(None) => {
                let r = allocator.storage_report();
                oom_count += 1;
                oom_free_sum += r.total_free_bytes;
                oom_largest_sum += r.largest_free_allocation_bytes;
            }
            Err(e) => panic!("{e}"),
        }

        // churn
        if cycle % 500 == 0 && !live.is_empty() {
            for _ in 0..(live.len() / 50 + 1) {
                if let Some(old) = live.pop_front() {
                    allocator.free(&old);
                    total_allocated -= old.size_bytes.get();
                }
            }
        }

        let r = allocator.storage_report();
        let used = capacity.saturating_sub(r.total_free_bytes);
        let util = used as f64 / capacity as f64;
        min_util = min_util.min(util);

        let ext_frag = if r.total_free_bytes > 0 {
            1.0 - (r.largest_free_allocation_bytes as f64 / r.total_free_bytes as f64)
        } else {
            0.0
        };

        if cycle % 1000 == 0 {
            println!(
                "Cycle {:5} | util={:.2}% | int={:.2}% | ext={:.2}% | OOMs={}",
                cycle,
                util * 100.0,
                (internal_frag as f64 / total_allocated as f64) * 100.0,
                ext_frag * 100.0,
                oom_count
            );
        }
    }

    println!("\n--- FINAL ---");
    println!("Capacity: {:.2} GB", capacity as f64 / 1e9);
    println!("Final allocated: {:.2} GB", total_allocated as f64 / 1e9);
    println!("Min utilization: {:.2}%", min_util * 100.0);
    println!("OOM events: {}", oom_count);

    if oom_count > 0 {
        println!(
            "Avg free bytes at OOM: {:.2} MB",
            (oom_free_sum as f64 / oom_count as f64) / 1e6
        );
        println!(
            "Avg largest free block at OOM: {:.2} MB",
            (oom_largest_sum as f64 / oom_count as f64) / 1e6
        );
        println!(
            "Avg OOM external fragmentation: {:.2}%",
            100.0 * (1.0 - (oom_largest_sum as f64 / oom_free_sum as f64))
        );
    }

    println!(
        "Avg alloc latency: {:?}",
        alloc_latency / alloc_calls as u32
    );

    println!("Elapsed: {:?}", start.elapsed());
}

fn main() {
    benchmark_kv_cache_fragmentation();
}
