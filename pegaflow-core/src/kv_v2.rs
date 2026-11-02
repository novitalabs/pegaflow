//! Content-addressed KV storage (protocol v2).
//!
//! Objects are opaque byte strings keyed by `(fingerprint, prefix)`. Clients copy
//! strided byte runs between their registered device regions and objects; an
//! object becomes visible to queries once its writes cover every byte. The store
//! never interprets layers, ranks or layouts: the client computes every offset.

use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::Arc;
use std::time::{Duration, Instant};

use cudarc::driver::{CudaContext, CudaStream, sys};
use hashlink::LinkedHashMap;
use log::{debug, info, warn};
use parking_lot::Mutex;

use crate::pinned_pool::{PinnedAllocation, PinnedAllocator};
use crate::transfer::{StridedCopier, StridedCopy, StridedMode};

/// Leases not released within this time are dropped.
const LEASE_TTL: Duration = Duration::from_secs(600);
/// Unsealed objects with no write for this long are dropped.
const UNSEALED_TTL: Duration = Duration::from_secs(300);

/// `(fingerprint, prefix)`, both SHA-256 values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ObjectKey {
    pub fingerprint: [u8; 32],
    pub prefix: [u8; 32],
}

/// `rows` copies of `width` bytes; row `r` is at `region_offset + r * region_stride`
/// of the device region and `object_offset + r * object_stride` of the object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Run {
    pub region: u32,
    pub region_offset: u64,
    pub object_offset: u64,
    pub width: u64,
    pub rows: u32,
    pub region_stride: u64,
    pub object_stride: u64,
}

impl Run {
    /// Strides with `rows == 1` normalized to `width`.
    fn pitches(&self) -> (u64, u64) {
        if self.rows == 1 {
            (self.width, self.width)
        } else {
            (self.region_stride, self.object_stride)
        }
    }

    /// One past the last byte on the region and object side, if the run is well formed.
    fn ends(&self) -> Option<(u64, u64)> {
        if self.width == 0 || self.rows == 0 {
            return None;
        }
        let (rp, op) = self.pitches();
        if rp < self.width || op < self.width {
            return None;
        }
        let last = u64::from(self.rows - 1);
        Some((
            last.checked_mul(rp)?
                .checked_add(self.region_offset)?
                .checked_add(self.width)?,
            last.checked_mul(op)?
                .checked_add(self.object_offset)?
                .checked_add(self.width)?,
        ))
    }
}

#[derive(Debug, Clone)]
pub struct ObjectWrite {
    pub key: ObjectKey,
    pub size: u64,
    pub runs: Vec<Run>,
}

#[derive(Debug, Clone)]
pub struct ObjectRead {
    pub key: ObjectKey,
    pub runs: Vec<Run>,
}

#[derive(Debug, Clone)]
pub struct KeyList {
    pub keys: Vec<ObjectKey>,
    pub stop_at_miss: bool,
}

#[derive(Debug)]
pub enum KvV2Error {
    /// A malformed request: bad run, size mismatch, unknown region.
    Invalid(String),
    /// An unknown device, lease or leased key.
    NotFound(String),
    /// The pinned pool cannot hold a new object.
    Exhausted(String),
    /// A CUDA call failed.
    Cuda(String),
}

impl std::fmt::Display for KvV2Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(m) => write!(f, "invalid request: {m}"),
            Self::NotFound(m) => write!(f, "not found: {m}"),
            Self::Exhausted(m) => write!(f, "pool exhausted: {m}"),
            Self::Cuda(m) => write!(f, "cuda: {m}"),
        }
    }
}

impl std::error::Error for KvV2Error {}

type Result<T> = std::result::Result<T, KvV2Error>;

/// Bytes of one object in pinned host memory.
struct ObjectBytes {
    alloc: Arc<PinnedAllocation>,
    size: u64,
}

impl ObjectBytes {
    fn host(&self, offset: u64) -> *mut u8 {
        // SAFETY: callers validated `offset` against `size`, which the allocation covers.
        unsafe { self.alloc.as_ptr().cast_mut().add(offset as usize) }
    }

    /// The device-visible address of byte `offset` of the mapped allocation.
    fn host_device(&self, offset: u64) -> u64 {
        self.alloc.mapped_ptr().device().as_ptr() as u64 + offset
    }
}

/// Written bytes of one object, one bit per `grain` bytes.
///
/// Every run of a save lands on whole granules: `grain` divides the object
/// size and each run's offset, width and stride. A save whose runs need a
/// finer grain splits every bit first.
struct Coverage {
    grain: u64,
    bits: Vec<u64>,
    set: u64,
    granules: u64,
}

impl Coverage {
    fn new(size: u64, grain: u64) -> Self {
        let granules = size / grain;
        Self {
            grain,
            bits: vec![0; granules.div_ceil(64) as usize],
            set: 0,
            granules,
        }
    }

    /// Split every granule into granules of `grain` bytes, which divides `self.grain`.
    fn refine(&mut self, grain: u64) {
        if grain == self.grain {
            return;
        }
        let k = self.grain / grain;
        let mut finer = Self::new(self.granules * self.grain, grain);
        for g in 0..self.granules {
            if self.bits[(g / 64) as usize] >> (g % 64) & 1 == 1 {
                finer.mark(g * k, (g + 1) * k);
            }
        }
        *self = finer;
    }

    /// Set granules `[lo, hi)`.
    fn mark(&mut self, lo: u64, hi: u64) {
        let mut g = lo;
        while g < hi {
            let (w, b) = ((g / 64) as usize, g % 64);
            let n = (64 - b).min(hi - g);
            let mask = if n == 64 {
                u64::MAX
            } else {
                ((1u64 << n) - 1) << b
            };
            self.set += u64::from((mask & !self.bits[w]).count_ones());
            self.bits[w] |= mask;
            g += n;
        }
    }

    /// Mark every row of `run`, whose bytes lie on whole granules.
    fn add(&mut self, run: &Run) {
        let (_, op) = run.pitches();
        let g = self.grain;
        let (first, step, width) = (run.object_offset / g, op / g, run.width / g);
        if width == 1 {
            for r in 0..u64::from(run.rows) {
                let at = first + r * step;
                let (w, b) = ((at / 64) as usize, at % 64);
                self.set += u64::from(self.bits[w] >> b & 1 == 0);
                self.bits[w] |= 1 << b;
            }
        } else {
            for r in 0..u64::from(run.rows) {
                let lo = first + r * step;
                self.mark(lo, lo + width);
            }
        }
    }

    fn complete(&self) -> bool {
        self.set == self.granules
    }
}

/// The largest grain dividing `size` and every offset, width and stride of `runs`.
fn grain_of(size: u64, runs: &[Run]) -> u64 {
    fn gcd(mut a: u64, mut b: u64) -> u64 {
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a
    }
    runs.iter().fold(size, |g, r| {
        let g = gcd(gcd(g, r.object_offset), r.width);
        if r.rows > 1 {
            gcd(g, r.object_stride)
        } else {
            g
        }
    })
}

struct Entry {
    bytes: Arc<ObjectBytes>,
    /// `None` until the first save and once sealed.
    covered: Option<Coverage>,
    sealed: bool,
    touched: Instant,
}

struct Lease {
    objects: HashMap<ObjectKey, Arc<ObjectBytes>>,
    expires: Instant,
}

/// One registered device: its CUDA context, a copy stream and its IPC-opened regions.
struct Device {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    regions: Vec<(sys::CUdeviceptr, u64)>,
    /// Serializes copies on the stream and owns their descriptor buffer.
    copier: Mutex<StridedCopier>,
}

impl Drop for Device {
    fn drop(&mut self) {
        if self.ctx.bind_to_thread().is_err() {
            return;
        }
        for &(ptr, _) in &self.regions {
            // SAFETY: `ptr` came from cuIpcOpenMemHandle in this context and is closed once.
            let r = unsafe { sys::cuIpcCloseMemHandle(ptr) };
            if r != sys::cudaError_enum::CUDA_SUCCESS {
                warn!("kv_v2: cuIpcCloseMemHandle failed: {r:?}");
            }
        }
    }
}

#[derive(Default)]
struct State {
    /// Least recently used first.
    objects: LinkedHashMap<ObjectKey, Entry>,
    leases: HashMap<u64, Lease>,
    next_lease: u64,
}

impl State {
    /// Drop expired leases and abandoned unsealed objects.
    fn sweep(&mut self, now: Instant) {
        self.leases.retain(|_, l| l.expires > now);
        let stale: Vec<ObjectKey> = self
            .objects
            .iter()
            .filter(|(_, e)| !e.sealed && now.duration_since(e.touched) > UNSEALED_TTL)
            .map(|(k, _)| *k)
            .collect();
        for k in stale {
            self.objects.remove(&k);
        }
    }

    /// Remove the least recently used object no lease or copy holds.
    fn evict_one(&mut self) -> bool {
        let victim = self
            .objects
            .iter()
            .find(|(_, e)| e.sealed && Arc::strong_count(&e.bytes) == 1)
            .map(|(k, _)| *k);
        victim.is_some_and(|k| self.objects.remove(&k).is_some())
    }
}

/// The v2 object store.
pub struct KvV2Store {
    allocator: PinnedAllocator,
    mode: StridedMode,
    devices: Mutex<HashMap<(String, u32), Arc<Device>>>,
    state: Mutex<State>,
}

impl KvV2Store {
    /// A store over `capacity` bytes of pinned host memory, copying in `mode`.
    pub fn new(capacity: usize, use_hugepages: bool, mode: StridedMode) -> Self {
        let allocator = PinnedAllocator::new_global(capacity, 1, use_hugepages, false, None);
        info!("kv_v2: pinned pool of {capacity} bytes, {mode:?} copies");
        Self {
            allocator,
            mode,
            devices: Mutex::new(HashMap::new()),
            state: Mutex::new(State::default()),
        }
    }

    /// Open `regions` (IPC handle, size) of device `device_id` for `(instance, rank)`.
    pub fn register(
        &self,
        instance: &str,
        rank: u32,
        device_id: i32,
        regions: &[(Vec<u8>, u64)],
    ) -> Result<()> {
        let ordinal = usize::try_from(device_id)
            .map_err(|_| KvV2Error::Invalid(format!("device_id {device_id}")))?;
        let ctx = CudaContext::new(ordinal).map_err(|e| KvV2Error::Cuda(format!("{e:?}")))?;
        ctx.bind_to_thread()
            .map_err(|e| KvV2Error::Cuda(format!("{e:?}")))?;
        let stream = ctx
            .new_stream()
            .map_err(|e| KvV2Error::Cuda(format!("{e:?}")))?;
        let copier = StridedCopier::new(&ctx, self.mode).map_err(KvV2Error::Cuda)?;
        let mut device = Device {
            ctx,
            stream,
            regions: Vec::with_capacity(regions.len()),
            copier: Mutex::new(copier),
        };
        for (i, (handle, size)) in regions.iter().enumerate() {
            let ptr = open_ipc(handle).map_err(|e| KvV2Error::Cuda(format!("region {i}: {e}")))?;
            device.regions.push((ptr, *size));
            let mapped = mapped_size(ptr).map_err(KvV2Error::Cuda)?;
            if mapped < *size {
                return Err(KvV2Error::Invalid(format!(
                    "region {i} declares {size} bytes but maps {mapped}"
                )));
            }
        }
        let key = (instance.to_string(), rank);
        info!(
            "kv_v2: registered {instance} rank {rank} on device {device_id} with {} regions",
            device.regions.len()
        );
        self.devices.lock().insert(key, Arc::new(device));
        Ok(())
    }

    /// Drop every device of `instance`; returns how many.
    pub fn unregister(&self, instance: &str) -> usize {
        let mut devices = self.devices.lock();
        let before = devices.len();
        devices.retain(|(i, _), _| i != instance);
        before - devices.len()
    }

    fn device(&self, instance: &str, rank: u32) -> Result<Arc<Device>> {
        self.devices
            .lock()
            .get(&(instance.to_string(), rank))
            .cloned()
            .ok_or_else(|| KvV2Error::NotFound(format!("device {instance} rank {rank}")))
    }

    /// Hit flags per list and a lease holding every hit.
    pub fn query(&self, lists: &[KeyList]) -> (u64, Vec<Vec<bool>>) {
        let now = Instant::now();
        let mut st = self.state.lock();
        st.sweep(now);
        let mut held = HashMap::new();
        let mut out = Vec::with_capacity(lists.len());
        for l in lists {
            let mut live = true;
            let mut hits = Vec::with_capacity(l.keys.len());
            for k in &l.keys {
                let hit = live
                    && match st.objects.to_back(k) {
                        Some(e) if e.sealed => {
                            held.insert(*k, Arc::clone(&e.bytes));
                            true
                        }
                        _ => false,
                    };
                live &= hit || !l.stop_at_miss;
                hits.push(hit);
            }
            out.push(hits);
        }
        st.next_lease += 1;
        let lease = st.next_lease;
        st.leases.insert(
            lease,
            Lease {
                objects: held,
                expires: now + LEASE_TTL,
            },
        );
        (lease, out)
    }

    pub fn release(&self, lease: u64) -> bool {
        self.state.lock().leases.remove(&lease).is_some()
    }

    fn allocate(&self, st: &mut State, size: u64) -> Result<Arc<PinnedAllocation>> {
        let size =
            NonZeroU64::new(size).ok_or_else(|| KvV2Error::Invalid("empty object".into()))?;
        loop {
            if let Some(a) = self.allocator.allocate(size, crate::NumaNode::UNKNOWN) {
                return Ok(a);
            }
            if !st.evict_one() {
                return Err(KvV2Error::Exhausted(format!("{size} bytes")));
            }
        }
    }

    /// Copy `writes` from the device of `(instance, rank)`; returns how many objects it sealed.
    pub fn save(&self, instance: &str, rank: u32, writes: &[ObjectWrite]) -> Result<usize> {
        let start = Instant::now();
        let device = self.device(instance, rank)?;
        let mut copies = Vec::new();
        let mut targets = Vec::new();
        {
            let mut st = self.state.lock();
            for w in writes {
                for run in &w.runs {
                    check(&device, run, w.size)?;
                }
                if let Some(e) = st.objects.get(&w.key)
                    && e.bytes.size != w.size
                {
                    return Err(KvV2Error::Invalid(format!(
                        "object size {} differs from the stored {}",
                        w.size, e.bytes.size
                    )));
                }
            }
            for w in writes {
                if !st.objects.contains_key(&w.key) {
                    let alloc = self.allocate(&mut st, w.size)?;
                    st.objects.insert(
                        w.key,
                        Entry {
                            bytes: Arc::new(ObjectBytes {
                                alloc,
                                size: w.size,
                            }),
                            covered: None,
                            sealed: false,
                            touched: Instant::now(),
                        },
                    );
                }
                let e = st.objects.to_back(&w.key).expect("inserted above");
                if e.sealed {
                    continue;
                }
                for run in &w.runs {
                    copies.push(copy_of(&device, &e.bytes, run));
                }
                targets.push((w.key, Arc::clone(&e.bytes)));
            }
        }
        let copy = Instant::now();
        transfer(&device, &copies, Direction::DeviceToHost)?;
        let copied = copy.elapsed();
        let mut st = self.state.lock();
        let mut sealed = 0;
        for w in writes {
            let Some(e) = st.objects.get_mut(&w.key) else {
                continue;
            };
            let ours = targets
                .iter()
                .any(|(k, b)| *k == w.key && Arc::ptr_eq(b, &e.bytes));
            if !ours || e.sealed {
                continue;
            }
            e.touched = Instant::now();
            let grain = grain_of(w.size, &w.runs);
            let c = e
                .covered
                .get_or_insert_with(|| Coverage::new(w.size, grain));
            c.refine(grain_of(c.grain, &w.runs));
            for run in &w.runs {
                c.add(run);
            }
            if c.complete() {
                e.sealed = true;
                e.covered = None;
                sealed += 1;
            }
        }
        debug!(
            "kv_v2 save {instance}/{rank}: {} copies in {:?} of {:?}",
            copies.len(),
            copied,
            start.elapsed()
        );
        Ok(sealed)
    }

    /// Copy leased objects into the device of `(instance, rank)`.
    pub fn load(&self, instance: &str, rank: u32, lease: u64, reads: &[ObjectRead]) -> Result<()> {
        let start = Instant::now();
        let device = self.device(instance, rank)?;
        let mut copies = Vec::new();
        let mut keep = Vec::new();
        {
            let st = self.state.lock();
            let l = st
                .leases
                .get(&lease)
                .ok_or_else(|| KvV2Error::NotFound(format!("lease {lease}")))?;
            for r in reads {
                let bytes = l.objects.get(&r.key).ok_or_else(|| {
                    KvV2Error::NotFound(format!("lease {lease} does not hold {:?}", r.key))
                })?;
                for run in &r.runs {
                    check(&device, run, bytes.size)?;
                    copies.push(copy_of(&device, bytes, run));
                }
                keep.push(Arc::clone(bytes));
            }
        }
        let copy = Instant::now();
        transfer(&device, &copies, Direction::HostToDevice)?;
        drop(keep);
        debug!(
            "kv_v2 load {instance}/{rank}: {} copies in {:?} of {:?}",
            copies.len(),
            copy.elapsed(),
            start.elapsed()
        );
        Ok(())
    }

    /// `(sealed objects, all objects, live leases)`.
    pub fn stats(&self) -> (usize, usize, usize) {
        let st = self.state.lock();
        let sealed = st.objects.values().filter(|e| e.sealed).count();
        (sealed, st.objects.len(), st.leases.len())
    }
}

fn check(device: &Device, run: &Run, object_size: u64) -> Result<()> {
    let (_, size) = device
        .regions
        .get(run.region as usize)
        .ok_or_else(|| KvV2Error::Invalid(format!("unknown region {}", run.region)))?;
    match run.ends() {
        Some((re, oe)) if re <= *size && oe <= object_size => Ok(()),
        _ => Err(KvV2Error::Invalid(format!(
            "run {run:?} is empty, overlapping or out of bounds"
        ))),
    }
}

fn copy_of(device: &Device, bytes: &ObjectBytes, run: &Run) -> StridedCopy {
    let (rp, op) = run.pitches();
    StridedCopy {
        device: device.regions[run.region as usize].0 + run.region_offset,
        host: bytes.host(run.object_offset),
        host_device: bytes.host_device(run.object_offset),
        width: run.width,
        rows: run.rows,
        device_pitch: rp,
        host_pitch: op,
    }
}

#[derive(Clone, Copy)]
enum Direction {
    DeviceToHost,
    HostToDevice,
}

fn transfer(device: &Device, copies: &[StridedCopy], dir: Direction) -> Result<()> {
    if copies.is_empty() {
        return Ok(());
    }
    let mut copier = device.copier.lock();
    device
        .ctx
        .bind_to_thread()
        .map_err(|e| KvV2Error::Cuda(format!("{e:?}")))?;
    match dir {
        Direction::DeviceToHost => copier.d2h(copies, &device.stream),
        Direction::HostToDevice => copier.h2d(copies, &device.stream),
    }
    .map_err(KvV2Error::Cuda)?;
    device
        .stream
        .synchronize()
        .map_err(|e| KvV2Error::Cuda(format!("{e:?}")))
}

fn open_ipc(handle: &[u8]) -> std::result::Result<sys::CUdeviceptr, String> {
    // SAFETY: CUipcMemHandle is 64 plain bytes.
    let mut h: sys::CUipcMemHandle = unsafe { std::mem::zeroed() };
    if handle.len() != h.reserved.len() {
        return Err(format!("ipc handle has {} bytes", handle.len()));
    }
    for (d, s) in h.reserved.iter_mut().zip(handle) {
        *d = *s as std::ffi::c_char;
    }
    let mut ptr: sys::CUdeviceptr = 0;
    // SAFETY: the current thread is bound to the target context; `ptr` is an out parameter.
    let r = unsafe {
        sys::cuIpcOpenMemHandle_v2(
            &mut ptr,
            h,
            sys::CUipcMem_flags::CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS as u32,
        )
    };
    if r == sys::cudaError_enum::CUDA_SUCCESS {
        Ok(ptr)
    } else {
        Err(format!("cuIpcOpenMemHandle: {r:?}"))
    }
}

/// Bytes mapped from `ptr` to the end of its allocation.
fn mapped_size(ptr: sys::CUdeviceptr) -> std::result::Result<u64, String> {
    let (mut base, mut size): (sys::CUdeviceptr, usize) = (0, 0);
    // SAFETY: out parameters only; `ptr` is a mapped device address.
    let r = unsafe { sys::cuMemGetAddressRange_v2(&mut base, &mut size, ptr) };
    if r != sys::cudaError_enum::CUDA_SUCCESS {
        return Err(format!("cuMemGetAddressRange: {r:?}"));
    }
    Ok(base + size as u64 - ptr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coverage_seals_only_when_complete() {
        let run = |object_offset, width, rows, object_stride| Run {
            region: 0,
            region_offset: 0,
            object_offset,
            width,
            rows,
            region_stride: width,
            object_stride,
        };
        // One save per batch, as the store sees them; the object is 192 bytes.
        let sealed = |saves: &[&[Run]]| {
            let mut c: Option<Coverage> = None;
            for runs in saves {
                let grain = grain_of(192, runs);
                let c = c.get_or_insert_with(|| Coverage::new(192, grain));
                c.refine(grain_of(c.grain, runs));
                runs.iter().for_each(|r| c.add(r));
            }
            c.is_some_and(|c| c.complete())
        };
        // Two interleaved heads of 32 bytes, rows 64 apart in the object.
        let (h0, h1) = (run(0, 32, 3, 64), run(32, 32, 3, 64));
        assert!(sealed(&[&[h0, h1]]));
        assert!(sealed(&[&[h1], &[h0]]));
        assert!(!sealed(&[&[h0], &[h0]]));
        // A later save on a finer grain keeps what the first one covered.
        assert!(sealed(&[
            &[run(0, 96, 1, 0)],
            &[run(96, 8, 1, 0), run(104, 88, 1, 0)]
        ]));
        assert!(!sealed(&[&[run(0, 96, 1, 0)], &[run(100, 92, 1, 0)]]));
        // A byte grain spans several bitmap words.
        assert!(sealed(&[&[run(0, 1, 70, 1), run(70, 122, 1, 0)]]));
    }

    #[test]
    fn runs_reject_overlap_and_overflow() {
        let run = |width, rows, rs, os| Run {
            region: 0,
            region_offset: 0,
            object_offset: 0,
            width,
            rows,
            region_stride: rs,
            object_stride: os,
        };
        assert_eq!(run(8, 3, 16, 8).ends(), Some((40, 24)));
        assert_eq!(run(8, 1, 0, 0).ends(), Some((8, 8)));
        assert!(run(8, 2, 4, 8).ends().is_none());
        assert!(run(0, 1, 0, 0).ends().is_none());
        assert!(run(8, 2, u64::MAX, 8).ends().is_none());
    }
}
