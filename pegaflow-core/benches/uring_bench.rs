/*
========================================
io_uring SSD Benchmark (pegaflow)
========================================

This benchmark is designed to approximate fio-style randrw workloads
using the pegaflow io_uring engine.

Example fio command:

fio --name=uring-test \
    --filename=testfile \
    --rw=randrw \
    --rwmixwrite=50 \
    --bs=4k \
    --iodepth=64 \
    --numjobs=1 \
    --size=1G \
    --group_reporting

Parameter mapping:

| Benchmark Param | fio Param      |
|----------------|----------------|
| block_size     | bs             |
| io_depth       | iodepth        |
| threads        | numjobs        |
| concurrency    | iodepth*numjobs|
| write_ratio    | rwmixwrite     |
| total_bytes    | size           |

Notes:
- Workload is random IO (uniform distribution)
- Latency includes userspace + kernel + device time
- Buffers are reused to avoid allocator noise
- File should be preallocated (fallocate) for realistic results
*/

use std::{
    fs::OpenOptions,
    io,
    os::unix::io::AsRawFd,
    time::Instant,
};

use clap::Parser;
use hdrhistogram::Histogram;
use rand::{ RngExt, SeedableRng};
use rand::rngs::StdRng;
use tokio::sync::oneshot;

use pegaflow_core::backing::uring::{UringConfig, UringIoEngine};

/// IO benchmark configuration
#[derive(Parser, Debug)]
#[command(ignore_errors = true)]
struct Config {
    /// File path (must be on SSD)
    #[arg(long)]
    file: String,

    /// Total bytes to process
    #[arg(long, default_value = "1073741824")] // 1GB
    total_bytes: u64,

    /// Block size (bytes)
    #[arg(long, default_value = "4096")]
    block_size: usize,

    /// Number of io_uring shards (threads)
    #[arg(long, default_value = "1")]
    threads: usize,

    /// IO depth per shard
    #[arg(long, default_value = "64")]
    io_depth: usize,

    /// Total concurrency (overrides threads * io_depth)
    #[arg(long)]
    concurrency: Option<usize>,

    /// Write ratio (0.0 = read-only, 1.0 = write-only)
    #[arg(long, default_value = "0.5")]
    write_ratio: f64,

    /// RNG seed (for reproducibility)
    #[arg(long, default_value = "42")]
    seed: u64,
}

/// In-flight request
struct Pending {
    start: Instant,
    rx: oneshot::Receiver<io::Result<usize>>,
    buf: Vec<u8>, // keeps memory alive
}

fn main() -> io::Result<()> {
    let cfg = Config::parse();

    println!("==== uring benchmark ====");
    println!("{:#?}", cfg);

    // Open file
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(&cfg.file)?;

    let fd = file.as_raw_fd();

    // Create engine
    let engine = UringIoEngine::new(
        fd,
        UringConfig {
            threads: cfg.threads,
            io_depth: cfg.io_depth,
            ..Default::default()
        },
    )?;

    let total_ops = cfg.total_bytes / cfg.block_size as u64;

    // Determine concurrency
    let max_inflight = cfg
        .concurrency
        .unwrap_or(cfg.threads * cfg.io_depth);

    let mut rng = StdRng::seed_from_u64(cfg.seed);

    let mut histogram = Histogram::<u64>::new(3).unwrap();

    let mut inflight: Vec<Pending> = Vec::with_capacity(max_inflight);

    // Preallocate buffers (IMPORTANT)
    let mut buffer_pool: Vec<Vec<u8>> = (0..max_inflight)
        .map(|_| vec![0u8; cfg.block_size])
        .collect();

    let mut submitted = 0u64;
    let mut completed = 0u64;

    // NOTE:
    // - Random IO pattern (uniform distribution)
    // - Approximates fio randrw workload

    let start = Instant::now();

    while completed < total_ops {
        // Submit IOs
        while submitted < total_ops && inflight.len() < max_inflight {
            let offset =
                rng.random_range(0..(cfg.total_bytes - cfg.block_size as u64));
            let is_write = rng.random_bool(cfg.write_ratio);

            let mut buf = buffer_pool.pop().expect("buffer pool empty");

            let rx = if is_write {
                engine.writev_at_async(vec![(buf.as_ptr(), buf.len())], offset)?
            } else {
                engine.readv_at_async(vec![(buf.as_mut_ptr(), buf.len())], offset)?
            };

            inflight.push(Pending {
                start: Instant::now(),
                rx,
                buf,
            });

            submitted += 1;
        }

        // Poll completions
        let mut i = 0;
        while i < inflight.len() {
            match inflight[i].rx.try_recv() {
                Ok(res) => {
                    let pending = inflight.swap_remove(i);

                    if res.is_ok() {
                        let latency =
                            pending.start.elapsed().as_micros() as u64;
                        histogram.record(latency).unwrap();
                    }

                    buffer_pool.push(pending.buf);

                    completed += 1;
                }
                Err(oneshot::error::TryRecvError::Empty) => {
                    i += 1;
                }
                Err(oneshot::error::TryRecvError::Closed) => {
                    let pending = inflight.swap_remove(i);
                    buffer_pool.push(pending.buf);
                    completed += 1;
                }
            }
        }

        // Prevent CPU spin
        if inflight.len() == max_inflight {
            std::thread::yield_now();
        }
    }

    let elapsed = start.elapsed().as_secs_f64();

    let iops = completed as f64 / elapsed;
    let throughput_mb =
        (completed * cfg.block_size as u64) as f64 / (1024.0 * 1024.0) / elapsed;

    println!("\n==== Results ====");
    println!("Time: {:.2} s", elapsed);
    println!("Total Ops: {}", completed);
    println!("IOPS: {:.2}", iops);
    println!("Throughput: {:.2} MB/s", throughput_mb);

    println!("\nLatency (µs):");
    println!("p50: {}", histogram.value_at_quantile(0.50));
    println!("p95: {}", histogram.value_at_quantile(0.95));
    println!("p99: {}", histogram.value_at_quantile(0.99));

    Ok(())
}