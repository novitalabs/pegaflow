use std::io::{self, Read, Write};
use std::mem;
use std::net::TcpStream;
use std::ptr::{self, NonNull};
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};
use std::thread;
use std::{future::IntoFuture, pin::pin};

use mea::oneshot;
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::TransferDesc;

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

pub fn parse_size(s: &str) -> usize {
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
    let num: f64 = num_str.parse().expect("invalid size number");
    (num * multiplier as f64) as usize
}

pub fn parse_block_range(s: &str) -> (usize, usize) {
    if let Some((lo, hi)) = s.split_once('-') {
        let lo: usize = lo.trim().parse().expect("invalid blocks-per-task low");
        let hi: usize = hi.trim().parse().expect("invalid blocks-per-task high");
        assert!(lo <= hi, "blocks-per-task: low must <= high");
        assert!(lo > 0, "blocks-per-task: must be > 0");
        (lo, hi)
    } else {
        let n: usize = s.trim().parse().expect("invalid blocks-per-task");
        assert!(n > 0, "blocks-per-task: must be > 0");
        (n, n)
    }
}

pub fn generate_task_schedule(
    total_tasks: usize,
    block_range: (usize, usize),
    seed: u64,
) -> Vec<usize> {
    let mut rng = SimpleRng::new(seed);
    (0..total_tasks)
        .map(|_| rng.range(block_range.0, block_range.1))
        .collect()
}

pub struct NumaBuffer {
    pub ptr: NonNull<u8>,
    pub len: usize,
}

unsafe impl Send for NumaBuffer {}
unsafe impl Sync for NumaBuffer {}

impl NumaBuffer {
    pub fn alloc(numa_node: u32, len: usize) -> Self {
        assert!(len > 0);

        let cpu_topo = pegaflow_common::read_cpu_topology_from_sysfs()
            .expect("failed to read NUMA CPU topology from sysfs");
        let cpus = cpu_topo
            .get(&numa_node)
            .unwrap_or_else(|| panic!("no CPUs found for NUMA{}", numa_node));

        let cpus = cpus.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        let handle = thread::Builder::new()
            .name(format!("numa{}-alloc", numa_node))
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
                        std::io::Error::last_os_error()
                    );
                }

                let p = unsafe {
                    libc::mmap(
                        ptr::null_mut(),
                        len,
                        libc::PROT_READ | libc::PROT_WRITE,
                        libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                        -1,
                        0,
                    )
                };
                assert_ne!(p, libc::MAP_FAILED, "mmap failed");
                unsafe {
                    ptr::write_bytes(p as *mut u8, 0xAB, len);
                }
                tx.send(p as u64).unwrap();
            })
            .expect("failed to spawn NUMA alloc thread");
        handle.join().expect("NUMA alloc thread panicked");
        let ptr = NonNull::new(rx.recv().unwrap() as *mut u8).expect("mmap returned null");

        Self { ptr, len }
    }

    pub fn fill(&self, pattern: u8) {
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct BufferSpec {
    pub numa_node: u32,
    pub base_ptr: u64,
    pub len: usize,
}

pub fn build_segmented_numa_scatter(
    local_bufs: &[NumaBuffer],
    remote_specs: &[BufferSpec],
    nblocks: usize,
    block_size: usize,
    descs_per_block: usize,
) -> Vec<TransferDesc> {
    assert!(!local_bufs.is_empty(), "local buffers must not be empty");
    assert_eq!(
        local_bufs.len(),
        remote_specs.len(),
        "local/remote NUMA buffer count mismatch"
    );
    assert!(descs_per_block > 0, "descs_per_block must be > 0");
    assert_eq!(
        block_size % descs_per_block,
        0,
        "block_size must be divisible by descs_per_block"
    );

    let seg_size = block_size / descs_per_block;
    let mut block_offsets = vec![0usize; local_bufs.len()];
    let mut descs = Vec::with_capacity(nblocks * descs_per_block);

    for block_idx in 0..nblocks {
        let buf_idx = block_idx % local_bufs.len();
        let block_off = block_offsets[buf_idx];
        block_offsets[buf_idx] += block_size;

        assert!(
            block_off + block_size <= local_bufs[buf_idx].len,
            "local NUMA buffer too small for workload"
        );
        assert!(
            block_off + block_size <= remote_specs[buf_idx].len,
            "remote NUMA buffer too small for workload"
        );

        for seg_idx in 0..descs_per_block {
            let seg_off = block_off + seg_idx * seg_size;
            descs.push(TransferDesc {
                local_ptr: unsafe {
                    NonNull::new_unchecked(local_bufs[buf_idx].ptr.as_ptr().add(seg_off))
                },
                remote_ptr: unsafe {
                    NonNull::new_unchecked((remote_specs[buf_idx].base_ptr as *mut u8).add(seg_off))
                },
                len: seg_size,
            });
        }
    }

    descs
}

pub fn gib_per_sec(bytes: usize, secs: f64) -> f64 {
    if secs <= 0.0 {
        return 0.0;
    }
    bytes as f64 / secs / (1024.0 * 1024.0 * 1024.0)
}

pub fn gbps(bytes: usize, secs: f64) -> f64 {
    if secs <= 0.0 {
        return 0.0;
    }
    (bytes as f64 * 8.0) / secs / 1e9
}

pub fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 * p).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    sorted[idx]
}

pub fn format_size(bytes: usize) -> String {
    if bytes >= 1 << 30 {
        format!("{:.1}GB", bytes as f64 / (1u64 << 30) as f64)
    } else if bytes >= 1 << 20 {
        format!("{:.0}MB", bytes as f64 / (1u64 << 20) as f64)
    } else if bytes >= 1 << 10 {
        format!("{:.0}KB", bytes as f64 / (1u64 << 10) as f64)
    } else {
        format!("{}B", bytes)
    }
}

pub fn send_framed<T: Serialize>(stream: &mut TcpStream, msg: &T) -> io::Result<()> {
    let payload = bincode::serde::encode_to_vec(msg, bincode::config::standard())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
    let len = payload.len() as u32;
    stream.write_all(&len.to_be_bytes())?;
    stream.write_all(&payload)?;
    stream.flush()?;
    Ok(())
}

pub fn recv_framed<T: DeserializeOwned>(stream: &mut TcpStream) -> io::Result<T> {
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf)?;
    let len = u32::from_be_bytes(len_buf) as usize;
    let mut payload = vec![0u8; len];
    stream.read_exact(&mut payload)?;
    let (msg, consumed) = bincode::serde::decode_from_slice(&payload, bincode::config::standard())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
    if consumed != payload.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "trailing bytes in framed payload",
        ));
    }
    Ok(msg)
}

pub fn block_recv<T>(rx: oneshot::Receiver<T>) -> Result<T, oneshot::RecvError> {
    struct ThreadWaker(thread::Thread);
    impl Wake for ThreadWaker {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }

        fn wake_by_ref(self: &Arc<Self>) {
            self.0.unpark();
        }
    }

    let waker = Waker::from(Arc::new(ThreadWaker(thread::current())));
    let mut cx = Context::from_waker(&waker);
    let mut fut = pin!(rx.into_future());

    loop {
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(v) => return v,
            Poll::Pending => thread::park(),
        }
    }
}
