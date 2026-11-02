//! Byte-addressed objects on the block store.
//!
//! An object is an opaque byte string keyed by `(fingerprint, prefix)`. Clients
//! copy strided byte runs between their registered device regions and objects;
//! the engine never interprets layers, ranks or layouts. Several devices may
//! each write part of one object; it seals once every byte is covered and is
//! stored as a one-slot, one-segment [`SealedBlock`] under the [`BlockKey`]
//! `(hex(fingerprint), prefix)`, so objects share the pinned pool, LRU
//! eviction and query leases with KV blocks. Sealed objects stay in pinned
//! memory: nothing reads them back from SSD or a remote node yet, so they are
//! not written to SSD or announced to the MetaServer.

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::num::NonZeroU64;
use std::sync::Arc;
use std::time::{Duration, Instant};

use cudarc::driver::{CudaContext, CudaStream, sys};
use log::{debug, info, warn};
use parking_lot::Mutex;

use crate::block::{BlockKey, RawBlock, SealedBlock, Segment};
use crate::metrics::core_metrics;
use crate::pinned_pool::{MappedPinnedPtr, PinnedAllocation};
use crate::transfer::{StridedCopier, StridedCopy, StridedMode};
use crate::{EngineError, NumaNode, PegaEngine, QueryLeaseId};

/// Unsealed objects with no write for this long are dropped.
const UNSEALED_TTL: Duration = Duration::from_secs(300);

/// `(fingerprint, prefix)`, both SHA-256 values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ObjectKey {
    pub fingerprint: [u8; 32],
    pub prefix: [u8; 32],
}

impl ObjectKey {
    /// Hex namespaces never equal a model namespace a KV connector derives.
    fn block_key(&self) -> BlockKey {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let namespace = self
            .fingerprint
            .iter()
            .flat_map(|b| [HEX[usize::from(b >> 4)], HEX[usize::from(b & 15)]])
            .map(char::from)
            .collect();
        BlockKey::new(namespace, self.prefix.to_vec())
    }
}

/// `rows` copies of `width` bytes; row `r` is at `region_offset + r * region_stride`
/// of the device region and `object_offset + r * object_stride` of the object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObjectRun {
    pub region: u32,
    pub region_offset: u64,
    pub object_offset: u64,
    pub width: u64,
    pub rows: u32,
    pub region_stride: u64,
    pub object_stride: u64,
}

impl ObjectRun {
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
    pub runs: Vec<ObjectRun>,
}

#[derive(Debug, Clone)]
pub struct ObjectRead {
    pub key: ObjectKey,
    pub runs: Vec<ObjectRun>,
}

#[derive(Debug, Clone)]
pub struct ObjectKeyList {
    pub keys: Vec<ObjectKey>,
    pub stop_at_miss: bool,
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
    fn add(&mut self, run: &ObjectRun) {
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
fn grain_of(size: u64, runs: &[ObjectRun]) -> u64 {
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

/// An object still being written.
struct Pending {
    alloc: Arc<PinnedAllocation>,
    size: u64,
    /// `None` until the first save's copies complete.
    covered: Option<Coverage>,
    touched: Instant,
}

/// One registered device: its CUDA context, a copy stream and its IPC-opened regions.
struct Device {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    regions: Vec<(sys::CUdeviceptr, u64)>,
    /// The GPU's NUMA node, recorded on sealed objects.
    numa: NumaNode,
    /// Where object memory is allocated: the GPU's node when the pool is NUMA-aware.
    alloc_numa: Option<NumaNode>,
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
                warn!("objects: cuIpcCloseMemHandle failed: {r:?}");
            }
        }
    }
}

/// Object devices and unsealed objects; sealed objects live in the block store.
pub(crate) struct ObjectStore {
    mode: StridedMode,
    devices: Mutex<HashMap<(String, u32), Arc<Device>>>,
    pending: Mutex<HashMap<BlockKey, Pending>>,
}

impl ObjectStore {
    pub(crate) fn new(mode: StridedMode) -> Self {
        Self {
            mode,
            devices: Mutex::new(HashMap::new()),
            pending: Mutex::new(HashMap::new()),
        }
    }

    pub(crate) fn has_devices(&self) -> bool {
        !self.devices.lock().is_empty()
    }

    fn has_instance(&self, instance: &str) -> bool {
        self.devices.lock().keys().any(|(i, _)| i == instance)
    }

    /// Drop every device of `instance`; returns how many.
    pub(crate) fn remove_instance(&self, instance: &str) -> usize {
        let mut devices = self.devices.lock();
        let before = devices.len();
        devices.retain(|(i, _), _| i != instance);
        before - devices.len()
    }

    /// Drop every device; returns the instances they belonged to.
    pub(crate) fn remove_all(&self) -> Vec<String> {
        let mut ids: Vec<String> = self.devices.lock().drain().map(|((i, _), _)| i).collect();
        ids.sort();
        ids.dedup();
        ids
    }

    fn device(&self, instance: &str, rank: u32) -> Result<Arc<Device>, EngineError> {
        self.devices
            .lock()
            .get(&(instance.to_string(), rank))
            .cloned()
            .ok_or_else(|| {
                EngineError::InstanceMissing(format!("{instance} (object device rank {rank})"))
            })
    }
}

/// A sealed object as a one-slot, one-segment block.
fn seal(p: Pending, numa: NumaNode) -> SealedBlock {
    let segment = Segment::new(p.alloc.as_non_null(), p.size as usize, p.alloc);
    SealedBlock::from_slots(vec![(RawBlock::single_segment(segment), numa)])
}

impl PegaEngine {
    /// Use `mode` for object copies.
    pub fn with_object_copy_mode(mut self, mode: StridedMode) -> Self {
        self.objects = ObjectStore::new(mode);
        self
    }

    /// Open `regions` (IPC handle, size) of device `device_id` as object
    /// device `rank` of `instance_id`, replacing an earlier registration.
    pub fn register_object_device(
        &self,
        instance_id: &str,
        rank: u32,
        device_id: i32,
        regions: &[(Vec<u8>, u64)],
    ) -> Result<(), EngineError> {
        let cuda = |e: cudarc::driver::DriverError| EngineError::CudaInit(format!("{e:?}"));
        let ordinal = usize::try_from(device_id)
            .map_err(|_| EngineError::InvalidArgument(format!("device_id {device_id}")))?;
        let numa = self.topology.numa_for_gpu(device_id);
        let alloc_numa = (self.storage.is_numa_enabled() && !numa.is_unknown()).then_some(numa);
        let ctx = CudaContext::new(ordinal).map_err(cuda)?;
        ctx.bind_to_thread().map_err(cuda)?;
        let stream = ctx.new_stream().map_err(cuda)?;
        let copier = StridedCopier::new(&ctx, self.objects.mode).map_err(EngineError::CudaInit)?;
        let mut device = Device {
            ctx,
            stream,
            regions: Vec::with_capacity(regions.len()),
            numa,
            alloc_numa,
            copier: Mutex::new(copier),
        };
        for (i, (handle, size)) in regions.iter().enumerate() {
            let ptr =
                open_ipc(handle).map_err(|e| EngineError::CudaInit(format!("region {i}: {e}")))?;
            device.regions.push((ptr, *size));
            let mapped = mapped_size(ptr).map_err(EngineError::CudaInit)?;
            if mapped < *size {
                return Err(EngineError::InvalidArgument(format!(
                    "region {i} declares {size} bytes but maps {mapped}"
                )));
            }
        }
        info!(
            "Registered object device: instance={instance_id} rank={rank} device={device_id} \
             regions={} numa={numa:?} copy={:?}",
            device.regions.len(),
            self.objects.mode
        );
        self.objects
            .devices
            .lock()
            .insert((instance_id.to_string(), rank), Arc::new(device));
        self.storage.set_has_instance(true);
        Ok(())
    }

    /// Hit flags per list, and a lease of `instance_id` holding every hit.
    pub fn query_objects(
        &self,
        instance_id: &str,
        lists: &[ObjectKeyList],
    ) -> Result<(QueryLeaseId, Vec<Vec<bool>>), EngineError> {
        if !self.objects.has_instance(instance_id) {
            return Err(EngineError::InstanceMissing(instance_id.to_string()));
        }
        let mut held = HashMap::new();
        let mut out = Vec::with_capacity(lists.len());
        for l in lists {
            let keys: Vec<BlockKey> = l.keys.iter().map(ObjectKey::block_key).collect();
            let found = self.storage.lookup_blocks(&keys, l.stop_at_miss);
            let mut hits = vec![false; keys.len()];
            for ((key, block), hit) in keys.into_iter().zip(found).zip(&mut hits) {
                if let Some(block) = block {
                    *hit = true;
                    held.insert(key, block);
                }
            }
            out.push(hits);
        }
        let asked: usize = out.iter().map(Vec::len).sum();
        let hit = out.iter().flatten().filter(|h| **h).count();
        let metrics = core_metrics();
        metrics.cache_block_hits.add(hit as u64, &[]);
        metrics.cache_block_misses.add((asked - hit) as u64, &[]);
        let lease = self.query_leases.create_objects(instance_id, held);
        Ok((lease, out))
    }

    /// Copy `writes` from object device `rank` of `instance_id`; returns how
    /// many objects this save sealed into the cache. Writes to sealed objects
    /// copy nothing.
    pub fn save_objects(
        &self,
        instance_id: &str,
        rank: u32,
        writes: &[ObjectWrite],
    ) -> Result<usize, EngineError> {
        let start = Instant::now();
        let device = self.objects.device(instance_id, rank)?;
        for w in writes {
            if w.size == 0 {
                return Err(EngineError::InvalidArgument(format!(
                    "object {:?} is empty",
                    w.key
                )));
            }
            for run in &w.runs {
                check(&device, run, w.size)?;
            }
        }
        let keys: Vec<BlockKey> = writes.iter().map(|w| w.key.block_key()).collect();
        let resident = self.storage.lookup_blocks(&keys, false);
        let mut copies = Vec::new();
        let mut targets = Vec::new();
        {
            let now = Instant::now();
            let mut pending = self.objects.pending.lock();
            pending.retain(|_, p| now.duration_since(p.touched) <= UNSEALED_TTL);
            for (i, (w, block)) in writes.iter().zip(&resident).enumerate() {
                if let Some(block) = block {
                    if block.memory_footprint() != w.size {
                        return Err(size_mismatch(w, block.memory_footprint()));
                    }
                    continue;
                }
                let p = match pending.entry(keys[i].clone()) {
                    Entry::Occupied(o) if o.get().size != w.size => {
                        return Err(size_mismatch(w, o.get().size));
                    }
                    Entry::Occupied(o) => o.into_mut(),
                    Entry::Vacant(v) => {
                        let size = NonZeroU64::new(w.size).expect("checked above");
                        let alloc =
                            self.storage
                                .allocate(size, device.alloc_numa)
                                .ok_or_else(|| {
                                    EngineError::Storage(format!(
                                        "pinned pool cannot hold a {} byte object",
                                        w.size
                                    ))
                                })?;
                        v.insert(Pending {
                            alloc,
                            size: w.size,
                            covered: None,
                            touched: now,
                        })
                    }
                };
                let base = p.alloc.mapped_ptr();
                copies.extend(w.runs.iter().map(|run| copy_of(&device, base, run)));
                targets.push((i, Arc::clone(&p.alloc)));
            }
        }
        let copy = Instant::now();
        transfer(&device, &copies, Direction::DeviceToHost)?;
        let copied = copy.elapsed();

        let mut sealed = Vec::new();
        {
            let mut pending = self.objects.pending.lock();
            for (i, alloc) in targets {
                let w = &writes[i];
                let Some(p) = pending.get_mut(&keys[i]) else {
                    continue;
                };
                // Dropped as stale and started again while this save copied.
                if !Arc::ptr_eq(&p.alloc, &alloc) {
                    continue;
                }
                p.touched = Instant::now();
                let c = p
                    .covered
                    .get_or_insert_with(|| Coverage::new(w.size, grain_of(w.size, &w.runs)));
                c.refine(grain_of(c.grain, &w.runs));
                w.runs.iter().for_each(|run| c.add(run));
                if c.complete() {
                    let p = pending.remove(&keys[i]).expect("present above");
                    sealed.push((keys[i].clone(), Arc::new(seal(p, device.numa))));
                }
            }
        }
        let n = self.storage.insert_objects(sealed);
        debug!(
            "save_objects {instance_id}/{rank}: {} copies in {copied:?} of {:?}, sealed {n}",
            copies.len(),
            start.elapsed()
        );
        Ok(n)
    }

    /// Copy objects `lease` holds into object device `rank` of `instance_id`.
    /// The lease stays valid until released or expired.
    pub fn load_objects(
        &self,
        instance_id: &str,
        rank: u32,
        lease: &QueryLeaseId,
        reads: &[ObjectRead],
    ) -> Result<(), EngineError> {
        let start = Instant::now();
        let device = self.objects.device(instance_id, rank)?;
        let keys: Vec<BlockKey> = reads.iter().map(|r| r.key.block_key()).collect();
        let blocks = self
            .query_leases
            .leased_objects(instance_id, lease, &keys)
            .map_err(EngineError::InvalidArgument)?;
        let mut copies = Vec::new();
        for ((r, block), key) in reads.iter().zip(&blocks).zip(&keys) {
            let block = block.as_ref().ok_or_else(|| {
                EngineError::InvalidArgument(format!("{lease:?} does not hold {key:?}"))
            })?;
            let base = block
                .slots()
                .first()
                .and_then(|s| s.segment_mapped_ptr(0))
                .ok_or_else(|| EngineError::Storage(format!("{key:?} is not an object")))?;
            for run in &r.runs {
                check(&device, run, block.memory_footprint())?;
                copies.push(copy_of(&device, base, run));
            }
        }
        let copy = Instant::now();
        transfer(&device, &copies, Direction::HostToDevice)?;
        drop(blocks);
        debug!(
            "load_objects {instance_id}/{rank}: {} copies in {:?} of {:?}",
            copies.len(),
            copy.elapsed(),
            start.elapsed()
        );
        Ok(())
    }
}

fn size_mismatch(w: &ObjectWrite, stored: u64) -> EngineError {
    EngineError::InvalidArgument(format!(
        "object {:?} has size {stored}, write says {}",
        w.key, w.size
    ))
}

fn check(device: &Device, run: &ObjectRun, object_size: u64) -> Result<(), EngineError> {
    let (_, size) = device
        .regions
        .get(run.region as usize)
        .ok_or_else(|| EngineError::InvalidArgument(format!("unknown region {}", run.region)))?;
    match run.ends() {
        Some((re, oe)) if re <= *size && oe <= object_size => Ok(()),
        _ => Err(EngineError::InvalidArgument(format!(
            "run {run:?} is empty, overlapping or out of bounds"
        ))),
    }
}

fn copy_of(device: &Device, base: MappedPinnedPtr, run: &ObjectRun) -> StridedCopy {
    let (rp, op) = run.pitches();
    let at = base.add(run.object_offset as usize);
    StridedCopy {
        device: device.regions[run.region as usize].0 + run.region_offset,
        host: at.host().as_ptr(),
        host_device: at.device().as_ptr() as u64,
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

fn transfer(device: &Device, copies: &[StridedCopy], dir: Direction) -> Result<(), EngineError> {
    if copies.is_empty() {
        return Ok(());
    }
    let failed = |e: String| EngineError::Storage(format!("object copy: {e}"));
    let mut copier = device.copier.lock();
    device
        .ctx
        .bind_to_thread()
        .map_err(|e| failed(format!("{e:?}")))?;
    match dir {
        Direction::DeviceToHost => copier.d2h(copies, &device.stream),
        Direction::HostToDevice => copier.h2d(copies, &device.stream),
    }
    .map_err(failed)?;
    device
        .stream
        .synchronize()
        .map_err(|e| failed(format!("{e:?}")))
}

fn open_ipc(handle: &[u8]) -> Result<sys::CUdeviceptr, String> {
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
fn mapped_size(ptr: sys::CUdeviceptr) -> Result<u64, String> {
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
        let run = |object_offset, width, rows, object_stride| ObjectRun {
            region: 0,
            region_offset: 0,
            object_offset,
            width,
            rows,
            region_stride: width,
            object_stride,
        };
        // One save per batch, as the store sees them; the object is 192 bytes.
        let sealed = |saves: &[&[ObjectRun]]| {
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
        let run = |width, rows, rs, os| ObjectRun {
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
