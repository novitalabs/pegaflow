//! CPU-memory RDMA benchmark for the v2 transfer engine.
//!
//! Models the PD push data path: prefill writes blocks into decode memory, then
//! sends an immediate as the task completion signal.

use std::{
    collections::{BTreeSet, HashSet},
    io, mem,
    num::NonZeroU8,
    ptr::{self, NonNull},
    sync::{
        Arc,
        atomic::{AtomicI64, Ordering},
    },
    thread,
    time::Instant,
};

use clap::{Parser, ValueEnum};
use pegaflow_common::read_cpu_topology_from_sysfs;
use pegaflow_transfer::{
    init_logging,
    v2::{
        CudaDeviceId, CudaDeviceMemory, Device, DomainGroupRouting, ImmTransferRequest,
        MemoryRegionDescriptor, MemoryRegionHandle, RdmaEngine, SingleTransferRequest,
        TopologyGroup, TransferEngine, TransferEngineBuilder, TransferRequest, detect_topology,
    },
};

#[derive(Parser)]
#[command(
    name = "pegaflow-cpu-bench-v2",
    version,
    about = "v2 RDMA CPU-memory PD push benchmark"
)]
struct Cli {
    /// Memory kind to register and transfer.
    #[arg(long, value_enum, default_value_t = MemoryKind::Host)]
    memory: MemoryKind,

    /// Block size, for example "4mb" or "2mb".
    #[arg(long, default_value = "4mb")]
    block_size: String,

    /// Blocks per task: a single number, or an inclusive range like "100-200".
    #[arg(long, default_value = "150")]
    blocks_per_task: String,

    /// Registered memory pool size. Defaults to the largest task transfer size.
    #[arg(long)]
    pool_size: Option<String>,

    /// Allocate the registered memory pool with MAP_HUGETLB huge pages.
    #[arg(long)]
    use_hugepages: bool,

    /// Number of measured tasks.
    #[arg(long, default_value_t = 50)]
    tasks: usize,

    /// Number of warmup tasks.
    #[arg(long, default_value_t = 5)]
    warmup_tasks: usize,

    /// Restrict to one CUDA device.
    #[arg(long)]
    cuda_device: Option<u8>,

    /// Restrict to a single NUMA node.
    #[arg(long)]
    numa: Option<u8>,

    /// Restrict to RDMA domains by name. Accepts comma-separated values or repeated values.
    #[arg(long, value_delimiter = ',', num_args = 1..)]
    domains: Vec<String>,

    /// Exclude RDMA domains by name.
    #[arg(long, value_delimiter = ',', num_args = 1..)]
    exclude_domain: Vec<String>,

    /// Only run the aggregate routing pass.
    #[arg(long)]
    aggregate_only: bool,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum MemoryKind {
    Host,
    Cuda,
}

struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    fn range(&mut self, lo: usize, hi: usize) -> usize {
        if lo == hi {
            return lo;
        }
        let span = (hi - lo + 1) as u64;
        (lo as u64 + self.next_u64() % span) as usize
    }
}

struct NumaBuffer {
    ptr: NonNull<u8>,
    len: usize,
}

unsafe impl Send for NumaBuffer {}
unsafe impl Sync for NumaBuffer {}

impl NumaBuffer {
    fn alloc(numa_node: u32, len: usize, use_hugepages: bool) -> Self {
        assert!(len > 0);

        let cpu_topo =
            read_cpu_topology_from_sysfs().expect("failed to read NUMA CPU topology from sysfs");
        let cpus = cpu_topo
            .get(&numa_node)
            .unwrap_or_else(|| panic!("no CPUs found for NUMA{numa_node}"))
            .clone();

        let mmap_len = if use_hugepages {
            let huge_page_size = read_hugepage_size_from_proc()
                .expect("failed to read Hugepagesize from /proc/meminfo");
            align_up(len, huge_page_size)
        } else {
            len
        };

        let (tx, rx) = std::sync::mpsc::channel();
        let handle = thread::Builder::new()
            .name(format!("numa{numa_node}-alloc"))
            .spawn(move || {
                unsafe {
                    let mut cpu_set: libc::cpu_set_t = mem::zeroed();
                    for &cpu in &cpus {
                        libc::CPU_SET(cpu, &mut cpu_set);
                    }
                    let ret =
                        libc::sched_setaffinity(0, mem::size_of::<libc::cpu_set_t>(), &cpu_set);
                    assert_eq!(
                        ret,
                        0,
                        "sched_setaffinity failed: {}",
                        io::Error::last_os_error()
                    );
                }

                let flags = if use_hugepages {
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_HUGETLB
                } else {
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS
                };
                let p = unsafe {
                    libc::mmap(
                        ptr::null_mut(),
                        mmap_len,
                        libc::PROT_READ | libc::PROT_WRITE,
                        flags,
                        -1,
                        0,
                    )
                };
                assert_ne!(
                    p,
                    libc::MAP_FAILED,
                    "mmap failed: {}",
                    io::Error::last_os_error()
                );
                unsafe {
                    ptr::write_bytes(p as *mut u8, 0xAB, mmap_len);
                }
                tx.send(p as u64).unwrap();
            })
            .expect("failed to spawn NUMA alloc thread");
        handle.join().expect("NUMA alloc thread panicked");

        let ptr = NonNull::new(rx.recv().unwrap() as *mut u8).expect("mmap returned null");
        Self { ptr, len: mmap_len }
    }

    fn fill(&self, pattern: u8) {
        unsafe {
            ptr::write_bytes(self.ptr.as_ptr(), pattern, self.len);
        }
    }
}

impl Drop for NumaBuffer {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr.as_ptr() as *mut libc::c_void, self.len);
        }
    }
}

enum BenchBuffer {
    Host(NumaBuffer),
    Cuda(CudaDeviceMemory),
}

impl BenchBuffer {
    fn alloc(
        memory: MemoryKind,
        cuda_device: u8,
        numa_node: u32,
        len: usize,
        use_hugepages: bool,
    ) -> Self {
        match memory {
            MemoryKind::Host => Self::Host(NumaBuffer::alloc(numa_node, len, use_hugepages)),
            MemoryKind::Cuda => Self::Cuda(
                CudaDeviceMemory::device_on(len, cuda_device).expect("cudaMalloc failed"),
            ),
        }
    }

    fn ptr(&self) -> NonNull<u8> {
        match self {
            BenchBuffer::Host(buf) => buf.ptr,
            BenchBuffer::Cuda(buf) => buf.ptr().cast(),
        }
    }

    fn device(&self, cuda_device: u8) -> Device {
        match self {
            BenchBuffer::Host(_) => Device::Host,
            BenchBuffer::Cuda(_) => Device::Cuda(CudaDeviceId(cuda_device)),
        }
    }

    fn fill(&self, pattern: u8) {
        match self {
            BenchBuffer::Host(buf) => buf.fill(pattern),
            BenchBuffer::Cuda(buf) => buf.fill(pattern).expect("cudaMemset failed"),
        }
    }
}

struct EngineContext {
    prefill: TransferEngine,
    decode: TransferEngine,
    prefill_buf: BenchBuffer,
    decode_buf: BenchBuffer,
    prefill_mr: MemoryRegionHandle,
    decode_mr: MemoryRegionDescriptor,
}

impl Drop for EngineContext {
    fn drop(&mut self) {
        let _ = self
            .prefill
            .unregister_memory(self.prefill_buf.ptr().cast());
        let _ = self.decode.unregister_memory(self.decode_buf.ptr().cast());
        self.prefill.stop();
        self.decode.stop();
    }
}

#[derive(Clone, Copy)]
enum BenchRouting {
    Pinned { domain_idx: u8 },
    Aggregate { num_shards: NonZeroU8 },
}

impl BenchRouting {
    fn label(self) -> String {
        match self {
            BenchRouting::Pinned { domain_idx } => format!("domain-{domain_idx}"),
            BenchRouting::Aggregate { num_shards } => format!("aggregate-{}-domains", num_shards),
        }
    }

    fn transfer_domain(self) -> DomainGroupRouting {
        match self {
            BenchRouting::Pinned { domain_idx } => DomainGroupRouting::Pinned { domain_idx },
            BenchRouting::Aggregate { num_shards } => {
                DomainGroupRouting::RoundRobinSharded { num_shards }
            }
        }
    }

    fn imm_domain(self) -> DomainGroupRouting {
        match self {
            BenchRouting::Pinned { domain_idx } => DomainGroupRouting::Pinned { domain_idx },
            BenchRouting::Aggregate { .. } => DomainGroupRouting::Pinned { domain_idx: 0 },
        }
    }
}

struct TaskResult {
    latency_ms: f64,
    bytes: usize,
}

struct BenchResult {
    label: String,
    tasks: Vec<TaskResult>,
}

fn main() {
    init_logging();
    let cli = Cli::parse();

    let block_size = parse_size(&cli.block_size);
    let block_range = parse_block_range(&cli.blocks_per_task);
    let total_tasks = cli.warmup_tasks + cli.tasks;
    let schedule = generate_task_schedule(total_tasks, block_range, 0x42);
    let max_blocks = *schedule.iter().max().expect("schedule should not be empty");
    let min_pool_size = max_blocks * block_size;
    let pool_size = cli
        .pool_size
        .as_deref()
        .map(parse_size)
        .unwrap_or(min_pool_size);
    assert!(
        pool_size >= block_size,
        "pool-size must be at least one block"
    );
    let pool_blocks = pool_size / block_size;

    println!(
        "pegaflow-cpu-bench-v2: memory={:?} block_size={} blocks_per_task={} tasks={} warmup={} pool_size={} hugepages={}",
        cli.memory,
        cli.block_size,
        cli.blocks_per_task,
        cli.tasks,
        cli.warmup_tasks,
        format_size(pool_size),
        cli.use_hugepages,
    );

    let topology = detect_topology().expect("v2 topology detection failed");
    let detected_domains = topology
        .iter()
        .map(|group| group.domains.len())
        .sum::<usize>();
    let detected_cuda_devices = topology
        .iter()
        .map(|group| group.cuda_device)
        .collect::<HashSet<_>>()
        .len();
    println!(
        "topology detected: groups={} cuda_devices={} domains={}",
        topology.len(),
        detected_cuda_devices,
        detected_domains
    );
    let include_domains: BTreeSet<String> = cli.domains.iter().cloned().collect();
    let exclude_domains: BTreeSet<String> = cli.exclude_domain.iter().cloned().collect();

    let mut selected_any = false;
    for group in topology
        .iter()
        .filter(|group| cli.cuda_device.is_none_or(|id| group.cuda_device == id))
        .filter(|group| cli.numa.is_none_or(|numa| group.numa == numa))
    {
        let domains = group
            .domains
            .iter()
            .filter(|domain| {
                include_domains.is_empty() || include_domains.contains(&*domain.name())
            })
            .filter(|domain| !exclude_domains.contains(&*domain.name()))
            .cloned()
            .collect::<Vec<_>>();
        if domains.is_empty() {
            continue;
        }
        selected_any = true;

        println!();
        println!(
            "--- cuda:{} numa:{} domains=[{}] cpus=[{}] ---",
            group.cuda_device,
            group.numa,
            domains
                .iter()
                .map(|domain| domain.name().into_owned())
                .collect::<Vec<_>>()
                .join(", "),
            group
                .cpus
                .iter()
                .map(u16::to_string)
                .collect::<Vec<_>>()
                .join(", "),
        );

        let ctx = create_engine_context(group, domains, cli.memory, pool_size, cli.use_hugepages);
        println!(
            "  engines ready: domains={} groups={} aggregate_link_speed={} Mbps",
            ctx.prefill.num_domains(),
            ctx.prefill.num_groups(),
            ctx.prefill.aggregated_link_speed(),
        );

        let nets_per_gpu = ctx.prefill.nets_per_gpu();
        let mut routings = Vec::new();
        if !cli.aggregate_only {
            for domain_idx in 0..nets_per_gpu.get() {
                routings.push(BenchRouting::Pinned { domain_idx });
            }
        }
        if nets_per_gpu.get() > 1 {
            routings.push(BenchRouting::Aggregate {
                num_shards: nets_per_gpu,
            });
        }

        for routing in routings {
            let result = run_bench(
                &ctx,
                routing,
                &schedule,
                cli.warmup_tasks,
                block_size,
                pool_blocks,
            );
            print_bench_result(&result, block_size);
        }
    }

    if !selected_any {
        eprintln!("error: no v2 topology groups matched the selected filters");
        std::process::exit(1);
    }

    println!();
    println!("bench complete.");
}

fn create_engine_context(
    group: &TopologyGroup,
    domains: Vec<pegaflow_transfer::v2::DomainInfo>,
    memory: MemoryKind,
    buf_size: usize,
    use_hugepages: bool,
) -> EngineContext {
    let prefill_worker_cpu = group
        .cpus
        .first()
        .copied()
        .expect("topology group has no CPU");
    let decode_worker_cpu = group.cpus.get(1).copied().unwrap_or(prefill_worker_cpu);

    let prefill_buf = BenchBuffer::alloc(
        memory,
        group.cuda_device,
        group.numa as u32,
        buf_size,
        use_hugepages,
    );
    let decode_buf = BenchBuffer::alloc(
        memory,
        group.cuda_device,
        group.numa as u32,
        buf_size,
        use_hugepages,
    );
    prefill_buf.fill(0xBB);
    decode_buf.fill(0xAA);

    let prefill = build_engine(group.cuda_device, domains.clone(), prefill_worker_cpu);
    let decode = build_engine(group.cuda_device, domains, decode_worker_cpu);

    let prefill_mr = prefill
        .register_memory_local(
            prefill_buf.ptr().cast(),
            buf_size,
            prefill_buf.device(group.cuda_device),
        )
        .expect("prefill register_memory_local failed");
    let (_decode_local_mr, decode_mr) = decode
        .register_memory_allow_remote(
            decode_buf.ptr().cast(),
            buf_size,
            decode_buf.device(group.cuda_device),
        )
        .expect("decode register_memory_allow_remote failed");

    EngineContext {
        prefill,
        decode,
        prefill_buf,
        decode_buf,
        prefill_mr,
        decode_mr,
    }
}

fn build_engine(
    cuda_device: u8,
    domains: Vec<pegaflow_transfer::v2::DomainInfo>,
    pin_worker_cpu: u16,
) -> TransferEngine {
    let mut builder = TransferEngineBuilder::default();
    builder.add_gpu_domains(cuda_device, domains, pin_worker_cpu);
    builder.build().expect("v2 transfer engine build failed")
}

fn run_bench(
    ctx: &EngineContext,
    routing: BenchRouting,
    schedule: &[usize],
    warmup: usize,
    block_size: usize,
    pool_blocks: usize,
) -> BenchResult {
    let mut tasks = Vec::with_capacity(schedule.len().saturating_sub(warmup));
    let mut rng = SimpleRng::new(0xfeed_4242);
    let imm_data = 0xC0DE_0000 | (routing_id(routing) as u32);
    let imm_counter = ctx.decode.get_imm_counter(imm_data);

    for (i, &nblocks) in schedule.iter().enumerate() {
        let offsets = build_random_block_offsets(nblocks, block_size, pool_blocks, &mut rng);

        let tx_counter = Arc::new(AtomicI64::new(0));
        let err_counter = Arc::new(AtomicI64::new(0));
        let start = Instant::now();

        for &offset in &offsets {
            let request = TransferRequest::Single(SingleTransferRequest {
                src_mr: ctx.prefill_mr,
                src_offset: offset as u64,
                length: block_size as u64,
                imm_data: None,
                dst_mr: ctx.decode_mr.clone(),
                dst_offset: offset as u64,
                domain: routing.transfer_domain(),
            });
            ctx.prefill
                .submit_transfer_atomic(request, Arc::clone(&tx_counter), Arc::clone(&err_counter))
                .expect("v2 Single transfer submit failed");
        }

        wait_atomic_count(&tx_counter, nblocks as i64);
        assert_eq!(
            err_counter.load(Ordering::Acquire),
            0,
            "v2 Single transfer failed"
        );

        let imm_tx_counter = Arc::new(AtomicI64::new(0));
        let imm_err_counter = Arc::new(AtomicI64::new(0));
        ctx.prefill
            .submit_transfer_atomic(
                TransferRequest::Imm(ImmTransferRequest {
                    imm_data,
                    dst_mr: ctx.decode_mr.clone(),
                    domain: routing.imm_domain(),
                }),
                Arc::clone(&imm_tx_counter),
                Arc::clone(&imm_err_counter),
            )
            .expect("v2 Imm transfer submit failed");
        imm_counter.wait(1);
        wait_atomic_count(&imm_tx_counter, 1);
        assert_eq!(
            imm_err_counter.load(Ordering::Acquire),
            0,
            "v2 Imm transfer failed"
        );

        let elapsed = start.elapsed();
        if i >= warmup {
            tasks.push(TaskResult {
                latency_ms: elapsed.as_secs_f64() * 1000.0,
                bytes: nblocks * block_size,
            });
        }
    }

    BenchResult {
        label: routing.label(),
        tasks,
    }
}

fn build_random_block_offsets(
    nblocks: usize,
    block_size: usize,
    pool_blocks: usize,
    rng: &mut SimpleRng,
) -> Vec<usize> {
    let mut seen = HashSet::with_capacity(nblocks.min(pool_blocks));
    let mut offsets = Vec::with_capacity(nblocks);
    while offsets.len() < nblocks {
        let block_idx = rng.range(0, pool_blocks - 1);
        if nblocks > pool_blocks || seen.insert(block_idx) {
            offsets.push(block_idx * block_size);
        }
    }
    offsets
}

fn wait_atomic_count(counter: &AtomicI64, target: i64) {
    while counter.load(Ordering::Acquire) < target {
        std::hint::spin_loop();
    }
}

fn routing_id(routing: BenchRouting) -> u8 {
    match routing {
        BenchRouting::Pinned { domain_idx } => domain_idx,
        BenchRouting::Aggregate { .. } => 0x7f,
    }
}

fn print_bench_result(result: &BenchResult, block_size: usize) {
    let mut latencies = result
        .tasks
        .iter()
        .map(|task| task.latency_ms)
        .collect::<Vec<_>>();
    latencies.sort_by(f64::total_cmp);

    let avg_bytes =
        result.tasks.iter().map(|task| task.bytes).sum::<usize>() / result.tasks.len().max(1);
    let avg_blocks = avg_bytes / block_size;
    let p50 = percentile(&latencies, 0.50);
    let p95 = percentile(&latencies, 0.95);
    let p99 = percentile(&latencies, 0.99);

    println!();
    println!("=== v2 PD push WRITE+IMM: {} ===", result.label);
    println!(
        "  avg {} blocks x {} ({:.1} MiB/task)",
        avg_blocks,
        format_size(block_size),
        avg_bytes as f64 / (1024.0 * 1024.0),
    );
    println!("  p50={p50:.2}ms  p95={p95:.2}ms  p99={p99:.2}ms");
    println!(
        "  p50 equiv: {:.1} Gbps ({:.2} GiB/s)",
        gbps(avg_bytes, p50 / 1000.0),
        gib_per_sec(avg_bytes, p50 / 1000.0),
    );
}

fn generate_task_schedule(
    total_tasks: usize,
    block_range: (usize, usize),
    seed: u64,
) -> Vec<usize> {
    let mut rng = SimpleRng::new(seed);
    (0..total_tasks)
        .map(|_| rng.range(block_range.0, block_range.1))
        .collect()
}

fn parse_block_range(s: &str) -> (usize, usize) {
    if let Some((lo, hi)) = s.split_once('-') {
        let lo = lo.trim().parse().expect("invalid blocks-per-task low");
        let hi = hi.trim().parse().expect("invalid blocks-per-task high");
        assert!(lo <= hi, "blocks-per-task: low must <= high");
        assert!(lo > 0, "blocks-per-task: must be > 0");
        (lo, hi)
    } else {
        let n = s.trim().parse().expect("invalid blocks-per-task");
        assert!(n > 0, "blocks-per-task: must be > 0");
        (n, n)
    }
}

fn parse_size(s: &str) -> usize {
    let s = s.trim().to_lowercase();
    let (num_str, multiplier) = if s.ends_with("tb") {
        (&s[..s.len() - 2], 1usize << 40)
    } else if s.ends_with("gb") {
        (&s[..s.len() - 2], 1usize << 30)
    } else if s.ends_with("mb") {
        (&s[..s.len() - 2], 1usize << 20)
    } else if s.ends_with("kb") {
        (&s[..s.len() - 2], 1usize << 10)
    } else {
        (s.as_str(), 1usize)
    };
    let num = num_str.parse::<f64>().expect("invalid size number");
    (num * multiplier as f64) as usize
}

fn format_size(bytes: usize) -> String {
    if bytes >= 1 << 30 {
        format!("{:.1}GB", bytes as f64 / (1u64 << 30) as f64)
    } else if bytes >= 1 << 20 {
        format!("{:.0}MB", bytes as f64 / (1u64 << 20) as f64)
    } else if bytes >= 1 << 10 {
        format!("{:.0}KB", bytes as f64 / (1u64 << 10) as f64)
    } else {
        format!("{bytes}B")
    }
}

fn gib_per_sec(bytes: usize, secs: f64) -> f64 {
    if secs <= 0.0 {
        return 0.0;
    }
    bytes as f64 / secs / (1024.0 * 1024.0 * 1024.0)
}

fn gbps(bytes: usize, secs: f64) -> f64 {
    if secs <= 0.0 {
        return 0.0;
    }
    (bytes as f64 * 8.0) / secs / 1e9
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 * p).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    sorted[idx]
}

fn read_hugepage_size_from_proc() -> Option<usize> {
    let content = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in content.lines() {
        if line.starts_with("Hugepagesize:") {
            let parts = line.split_whitespace().collect::<Vec<_>>();
            if parts.len() == 3 && parts[2] == "kB" {
                let kb = parts[1].parse::<usize>().ok()?;
                return Some(kb * 1024);
            }
        }
    }
    None
}

fn align_up(value: usize, align: usize) -> usize {
    assert!(align.is_power_of_two(), "alignment must be a power of two");
    (value + align - 1) & !(align - 1)
}
