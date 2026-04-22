/*
========================================
io_uring SSD Benchmark (pegaflow)
========================================

fio equivalent example:

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
- Random IO (uniform distribution, block-aligned)
- Latency = submission → completion (includes queueing)
- Buffers reused (no allocator noise)
- File SHOULD be preallocated for realistic SSD results
*/

use std::{
    fs::OpenOptions,
    io,
    os::unix::{fs::OpenOptionsExt, io::AsRawFd},
    time::{Duration, Instant},
};

use clap::Parser;
use hdrhistogram::Histogram;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use tokio::sync::oneshot;

use pegaflow_core::backing::uring::{UringConfig, UringIoEngine};

#[derive(Parser, Debug)]
struct Config {
    #[arg(long)]
    file: String,

    #[arg(long, default_value = "1073741824")]
    total_bytes: u64,

    #[arg(long, default_value = "4096")]
    block_size: usize,

    #[arg(long, default_value = "1")]
    threads: usize,

    #[arg(long, default_value = "64")]
    io_depth: usize,

    #[arg(long)]
    concurrency: Option<usize>,

    #[arg(long, default_value = "0.5")]
    write_ratio: f64,

    #[arg(long, default_value = "42")]
    seed: u64,

    #[arg(long, default_value = "false")]
    direct: bool,
}

struct Pending {
    start: Instant,
    rx: oneshot::Receiver<io::Result<usize>>,
    buf: Vec<u8>,
}

fn main() -> io::Result<()> {
    let cfg = Config::parse();

    println!("==== uring benchmark ====");
    println!("{:#?}", cfg);

    if cfg.block_size == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "block_size must be > 0",
        ));
    }

    if cfg.total_bytes < cfg.block_size as u64 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "total_bytes must be >= block_size",
        ));
    }

    if cfg.threads == 0 || cfg.io_depth == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "threads and io_depth must be > 0",
        ));
    }

    let engine_capacity = cfg.threads * cfg.io_depth;

    if let Some(c) = cfg.concurrency
        && (c == 0 || c != engine_capacity)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "concurrency must equal threads * io_depth and be > 0",
        ));
    }

    let max_inflight = cfg.concurrency.unwrap_or(engine_capacity);

    let mut opts = OpenOptions::new();
    opts.read(true).write(true).create(true);

    if cfg.direct {
        opts.custom_flags(libc::O_DIRECT);
    }

    let file = opts.open(&cfg.file)?;
    let fd = file.as_raw_fd();

    let file_size = file.metadata()?.len();

    if file_size < cfg.total_bytes {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "file size ({}) < total_bytes ({}). Preallocate file first.",
                file_size, cfg.total_bytes
            ),
        ));
    }

    let total_ops = cfg.total_bytes / cfg.block_size as u64;

    let max_blocks = file_size / cfg.block_size as u64;

    let engine = UringIoEngine::new(
        fd,
        UringConfig {
            threads: cfg.threads,
            io_depth: cfg.io_depth,
            ..Default::default()
        },
    )?;

    let mut rng = StdRng::seed_from_u64(cfg.seed);

    let mut histogram = Histogram::<u64>::new_with_bounds(1, 60_000_000, 3).unwrap();

    let mut inflight: Vec<Pending> = Vec::with_capacity(max_inflight);

    let mut buffer_pool: Vec<Vec<u8>> = (0..max_inflight)
        .map(|_| vec![0u8; cfg.block_size])
        .collect();

    let mut submitted = 0u64;
    let mut completed = 0u64;

    let start = Instant::now();

    while completed < total_ops {
        // Submit
        while submitted < total_ops && inflight.len() < max_inflight {
            let block_idx = rng.random_range(0..max_blocks);
            let offset = block_idx * cfg.block_size as u64;

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

                    match res {
                        Ok(n) if n == cfg.block_size => {
                            let latency = pending.start.elapsed().as_micros() as u64;
                            let latency = latency.max(1);

                            histogram.record(latency).unwrap();
                        }
                        Ok(n) => {
                            return Err(io::Error::other(format!(
                                "Partial IO detected: {} bytes",
                                n
                            )));
                        }
                        Err(e) => {
                            return Err(io::Error::other(format!("IO error: {:?}", e)));
                        }
                    }

                    buffer_pool.push(pending.buf);
                    completed += 1;
                }
                Err(oneshot::error::TryRecvError::Empty) => {
                    i += 1;
                }
                Err(oneshot::error::TryRecvError::Closed) => {
                    return Err(io::Error::other("Completion channel closed unexpectedly"));
                }
            }
        }

        if inflight.len() == max_inflight {
            std::thread::sleep(Duration::from_micros(10));
        }
    }

    let elapsed = start.elapsed().as_secs_f64();

    let iops = completed as f64 / elapsed;
    let throughput_mb = (completed * cfg.block_size as u64) as f64 / (1024.0 * 1024.0) / elapsed;

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
