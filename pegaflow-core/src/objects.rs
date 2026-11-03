//! Byte-addressed objects on the block store.
//!
//! An object is an opaque byte string keyed by `(fingerprint, prefix)`. Clients
//! copy strided byte runs between their registered device regions and objects;
//! the engine never interprets layers, ranks or layouts. The client fixes each
//! object's size, coverage unit and chunks; the engine only checks that every
//! writer agrees and that runs respect them. Several devices may each write
//! part of one object: a save claims the units it copies, skips units another
//! save already claimed, and the object seals once every unit is claimed and
//! no copy is in flight. It is stored as a [`SealedBlock`] with one slot per
//! chunk under the [`BlockKey`] `(hex(fingerprint), prefix)`, so objects share
//! the pinned pool, LRU eviction, query leases, the SSD cache and MetaServer
//! registration with KV blocks, and a query brings an evicted object back.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::num::NonZeroU64;
use std::sync::Arc;
use std::time::{Duration, Instant};

use cudarc::driver::{CudaContext, CudaStream, sys};
use log::{debug, info, warn};
use parking_lot::Mutex;
use tokio::sync::{mpsc, oneshot};

use crate::block::{BlockKey, RawBlock, SealedBlock, Segment};
use crate::gpu_worker::{LoadCompletion, spawn_pinned_worker, wait_worker_ready};
use crate::metrics::core_metrics;
use crate::pinned_pool::{MappedPinnedPtr, PinnedAllocation};
use crate::seal_offload::io_len;
use crate::storage::{FetchPart, PartsPoll};
use crate::transfer::{Dim, Direction, MAX_DIMS, StridedCopier, StridedCopy};
use crate::{EngineError, NumaNode, PegaEngine, QueryLeaseId};

/// Most units one object may have; bounds its coverage bitmap.
const MAX_UNITS: u64 = 1 << 22;
/// Most chunks one object may have.
const MAX_CHUNKS: usize = 1 << 12;
/// Most keys and points one query may carry.
const MAX_QUERY_KEYS: usize = 1 << 16;
const MAX_QUERY_POINTS: usize = 1 << 12;

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

/// One dimension of an [`ObjectRun`]: `count` steps of `region_stride` bytes
/// in the device region and `object_stride` bytes in the object; the strides
/// are ignored when `count == 1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObjectDim {
    pub count: u32,
    pub region_stride: u64,
    pub object_stride: u64,
}

/// Rows of `width` bytes over up to [`MAX_DIMS`] dimensions, the first
/// fastest: row `(i_0, i_1, ..)` is at `region_offset + sum(i_d * region_stride_d)`
/// of the device region and `object_offset + sum(i_d * object_stride_d)` of
/// the object. No dimensions is one row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectRun {
    pub region: u32,
    pub region_offset: u64,
    pub object_offset: u64,
    pub width: u64,
    pub dims: Vec<ObjectDim>,
}

impl ObjectRun {
    /// Rows the run copies.
    fn rows(&self) -> u64 {
        self.dims
            .iter()
            .fold(1u64, |n, d| n.saturating_mul(u64::from(d.count)))
    }

    fn bytes(&self) -> u64 {
        self.width.saturating_mul(self.rows())
    }

    /// Region and object offset of every row, the first dimension fastest.
    fn row_offsets(&self) -> impl Iterator<Item = (u64, u64)> + '_ {
        offsets((self.region_offset, self.object_offset), &self.dims)
    }

    /// Object bytes per block and the dimensions over blocks: with the
    /// dimensions ordered by object stride, a block is the rows of those that
    /// are back to back in the object. Only valid for a well formed run.
    fn object_blocks(&self) -> (u64, Vec<ObjectDim>) {
        let mut dims: Vec<ObjectDim> = self.dims.iter().filter(|d| d.count > 1).copied().collect();
        dims.sort_unstable_by_key(|d| d.object_stride);
        let mut len = self.width;
        let mut k = 0;
        while let Some(d) = dims.get(k)
            && d.object_stride == len
        {
            len *= u64::from(d.count);
            k += 1;
        }
        dims.drain(..k);
        (len, dims)
    }

    /// One past the last byte on the region and object side, if the run is
    /// well formed: nonempty, at most [`MAX_DIMS`] dimensions, and on each
    /// side, with its dimensions ordered by stride, every stride at least the
    /// extent of the rows below it, so no two rows overlap.
    fn ends(&self) -> Option<(u64, u64)> {
        if self.width == 0 || self.dims.len() > MAX_DIMS || self.dims.iter().any(|d| d.count == 0) {
            return None;
        }
        let end = |base: u64, stride: fn(&ObjectDim) -> u64| {
            let mut dims: Vec<(u64, u64)> = self
                .dims
                .iter()
                .filter(|d| d.count > 1)
                .map(|d| (stride(d), u64::from(d.count)))
                .collect();
            dims.sort_unstable();
            let mut extent = self.width;
            for (s, n) in dims {
                if s < extent {
                    return None;
                }
                extent = (n - 1).checked_mul(s)?.checked_add(extent)?;
            }
            base.checked_add(extent)
        };
        Some((
            end(self.region_offset, |d| d.region_stride)?,
            end(self.object_offset, |d| d.object_stride)?,
        ))
    }
}

#[derive(Debug, Clone)]
pub struct ObjectWrite {
    pub key: ObjectKey,
    pub size: u64,
    /// Coverage granule: divides every chunk and every run's object offset,
    /// width and object stride.
    pub unit: u64,
    /// Chunk sizes in order, summing to `size`; empty for one chunk.
    pub chunks: Vec<u64>,
    pub runs: Vec<ObjectRun>,
}

#[derive(Debug, Clone)]
pub struct ObjectRead {
    pub key: ObjectKey,
    pub runs: Vec<ObjectRun>,
}

/// Reads of objects one lease holds.
#[derive(Debug, Clone)]
pub struct ObjectLoad {
    pub lease: QueryLeaseId,
    pub reads: Vec<ObjectRead>,
}

/// A resume point: the first `blocks` keys of every prefix chain of its
/// query, and `keys`.
#[derive(Debug, Clone)]
pub struct ObjectPoint {
    pub blocks: u32,
    pub keys: Vec<ObjectKey>,
}

/// Prefix chains, each of one fingerprint, and the points a query picks
/// from, in ascending order of preference.
#[derive(Debug, Clone)]
pub struct ObjectQuery {
    pub prefixes: Vec<Vec<ObjectKey>>,
    pub points: Vec<ObjectPoint>,
}

#[derive(Debug)]
pub enum ObjectQueryStatus {
    /// Missing objects are being fetched; ask again with the same `req_id`.
    Loading,
    /// The point found, if any, and how many leading blocks of every chain
    /// the lease holds: those of the point, or with no points the longest
    /// prefix every chain holds. The lease also holds the point's keys; none
    /// is issued when it would hold nothing.
    Ready {
        point: Option<usize>,
        blocks: u32,
        lease: Option<QueryLeaseId>,
    },
}

/// A validated [`ObjectQuery`] over block keys.
struct Plan {
    prefixes: Vec<Vec<BlockKey>>,
    points: Vec<(usize, Vec<BlockKey>)>,
}

impl Plan {
    fn new(q: &ObjectQuery) -> Result<Self, EngineError> {
        let bad = |why: String| Err(EngineError::InvalidArgument(why));
        let keys: usize = q.prefixes.iter().map(Vec::len).sum::<usize>()
            + q.points.iter().map(|p| p.keys.len()).sum::<usize>();
        if keys > MAX_QUERY_KEYS || q.points.len() > MAX_QUERY_POINTS {
            return bad(format!(
                "{keys} keys and {} points exceed {MAX_QUERY_KEYS} and {MAX_QUERY_POINTS}",
                q.points.len()
            ));
        }
        if let Some(c) = q
            .prefixes
            .iter()
            .position(|c| c.iter().any(|k| k.fingerprint != c[0].fingerprint))
        {
            return bad(format!("prefix chain {c} mixes fingerprints"));
        }
        let shortest = q.prefixes.iter().map(Vec::len).min().unwrap_or(0);
        if let Some(i) = q.points.iter().position(|p| p.blocks as usize > shortest) {
            return bad(format!(
                "point {i} needs {} blocks of chains as short as {shortest}",
                q.points[i].blocks
            ));
        }
        let keys = |l: &[ObjectKey]| l.iter().map(ObjectKey::block_key).collect();
        Ok(Self {
            prefixes: q.prefixes.iter().map(|c| keys(c)).collect(),
            points: q
                .points
                .iter()
                .map(|p| (p.blocks as usize, keys(&p.keys)))
                .collect(),
        })
    }

    /// With `chain(c)` leading blocks of chain `c` and the point keys `has`
    /// marks, the last point that holds, and the blocks that answer leases.
    fn best(
        &self,
        chain: impl Fn(usize) -> usize,
        has: impl Fn(&BlockKey) -> bool,
    ) -> (Option<usize>, usize) {
        let common = (0..self.prefixes.len()).map(chain).min().unwrap_or(0);
        if self.points.is_empty() {
            return (None, common);
        }
        self.points
            .iter()
            .rposition(|(blocks, keys)| *blocks <= common && keys.iter().all(&has))
            .map_or((None, 0), |p| (Some(p), self.points[p].0))
    }
}

/// Set bits `[lo, hi)` as `(word, mask)` pairs.
fn masks(lo: u64, hi: u64) -> impl Iterator<Item = (usize, u64)> {
    let mut g = lo;
    std::iter::from_fn(move || {
        if g >= hi {
            return None;
        }
        let (w, b) = ((g / 64) as usize, g % 64);
        let n = (64 - b).min(hi - g);
        let mask = if n == 64 {
            u64::MAX
        } else {
            ((1u64 << n) - 1) << b
        };
        g += n;
        Some((w, mask))
    })
}

/// One bit per unit.
struct Bits {
    words: Vec<u64>,
    set: u64,
    len: u64,
}

impl Bits {
    fn new(len: u64) -> Self {
        Self {
            words: vec![0; len.div_ceil(64) as usize],
            set: 0,
            len,
        }
    }

    fn get(&self, i: u64) -> bool {
        self.words[(i / 64) as usize] >> (i % 64) & 1 == 1
    }

    fn none(&self, lo: u64, hi: u64) -> bool {
        masks(lo, hi).all(|(w, m)| self.words[w] & m == 0)
    }

    fn mark(&mut self, lo: u64, hi: u64) {
        for (w, m) in masks(lo, hi) {
            self.set += u64::from((m & !self.words[w]).count_ones());
            self.words[w] |= m;
        }
    }

    fn complete(&self) -> bool {
        self.set == self.len
    }
}

/// Size, unit and chunk sizes of an object; every writer sends the same ones.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Shape {
    size: u64,
    unit: u64,
    chunks: Vec<u64>,
    /// Object offset of each chunk.
    starts: Vec<u64>,
}

impl Shape {
    fn of(w: &ObjectWrite) -> Result<Self, EngineError> {
        let bad = |why: String| {
            Err(EngineError::InvalidArgument(format!(
                "object {:?}: {why}",
                w.key
            )))
        };
        let chunks = if w.chunks.is_empty() {
            vec![w.size]
        } else {
            w.chunks.clone()
        };
        if w.size == 0 {
            return bad("size is zero".into());
        }
        if w.unit == 0 || !w.size.is_multiple_of(w.unit) {
            return bad(format!("unit {} does not divide size {}", w.unit, w.size));
        }
        if w.size / w.unit > MAX_UNITS {
            return bad(format!(
                "{} units of {} bytes exceed {MAX_UNITS}",
                w.size / w.unit,
                w.unit
            ));
        }
        if chunks.len() > MAX_CHUNKS {
            return bad(format!("{} chunks exceed {MAX_CHUNKS}", chunks.len()));
        }
        if chunks.iter().any(|&c| c == 0 || !c.is_multiple_of(w.unit)) {
            return bad(format!(
                "chunks {chunks:?} are not whole units of {}",
                w.unit
            ));
        }
        let starts = starts(&chunks);
        if chunks.iter().try_fold(0u64, |a, &c| a.checked_add(c)) != Some(w.size) {
            return bad(format!("chunks {chunks:?} do not sum to size {}", w.size));
        }
        Ok(Self {
            size: w.size,
            unit: w.unit,
            chunks,
            starts,
        })
    }

    /// The run's units are whole and lie in one chunk; returns that chunk.
    fn check(&self, run: &ObjectRun) -> Result<usize, EngineError> {
        let u = self.unit;
        let aligned = run.object_offset.is_multiple_of(u)
            && run.width.is_multiple_of(u)
            && run
                .dims
                .iter()
                .all(|d| d.count == 1 || d.object_stride.is_multiple_of(u));
        if !aligned {
            return Err(EngineError::InvalidArgument(format!(
                "run {run:?} is not on whole units of {u} bytes"
            )));
        }
        run.ends()
            .and_then(|(_, oe)| chunk_of(&self.starts, &self.chunks, run.object_offset, oe))
            .ok_or_else(|| {
                EngineError::InvalidArgument(format!(
                    "run {run:?} is out of bounds or crosses a chunk of {:?}",
                    self.chunks
                ))
            })
    }
}

/// `base` advanced to every index of `dims`, the first fastest, on the region
/// and the object side.
fn offsets(base: (u64, u64), dims: &[ObjectDim]) -> impl Iterator<Item = (u64, u64)> + '_ {
    let total: u64 = dims.iter().map(|d| u64::from(d.count)).product();
    let mut index = [0u32; MAX_DIMS];
    let mut at = base;
    (0..total).map(move |_| {
        let cur = at;
        // Wrapping: a single-step dimension's strides are arbitrary, and
        // stepping past the last row may leave u64; both undo exactly.
        for (i, d) in dims.iter().enumerate() {
            index[i] += 1;
            at = (
                at.0.wrapping_add(d.region_stride),
                at.1.wrapping_add(d.object_stride),
            );
            if index[i] < d.count {
                break;
            }
            index[i] = 0;
            let n = u64::from(d.count);
            at = (
                at.0.wrapping_sub(n.wrapping_mul(d.region_stride)),
                at.1.wrapping_sub(n.wrapping_mul(d.object_stride)),
            );
        }
        cur
    })
}

/// Object offset of each of `chunks`.
fn starts(chunks: &[u64]) -> Vec<u64> {
    chunks
        .iter()
        .scan(0u64, |at, &c| {
            let s = *at;
            *at = at.saturating_add(c);
            Some(s)
        })
        .collect()
}

/// The chunk holding object bytes `[lo, hi)`, if one does.
fn chunk_of(starts: &[u64], sizes: &[u64], lo: u64, hi: u64) -> Option<usize> {
    let i = starts.partition_point(|&s| s <= lo).checked_sub(1)?;
    (hi <= starts[i] + sizes[i]).then_some(i)
}

/// One allocation per chunk of an object, padded for SSD I/O, and its NUMA node.
type Chunks = Vec<(Arc<PinnedAllocation>, NumaNode)>;

/// An object still being written.
struct Pending {
    /// Tells this object from one under the same key started after it was dropped.
    id: u64,
    shape: Shape,
    /// The copy that is cached, written to SSD and announced to peers.
    chunks: Chunks,
    /// Node-local copies on further sockets, by socket, for content devices
    /// read whole across sockets; every save writes them too.
    replicas: Vec<(u32, Chunks)>,
    /// Units a save copies or copied.
    claimed: Bits,
    /// Saves whose copies into the object are in flight.
    copying: usize,
    touched: Instant,
}

impl Pending {
    /// Pinned bytes of every copy.
    fn bytes(&self) -> u64 {
        self.shape.size * (1 + self.replicas.len() as u64)
    }

    /// Every allocation, kept alive while a save copies into them.
    fn allocations(&self) -> Vec<Arc<PinnedAllocation>> {
        std::iter::once(&self.chunks)
            .chain(self.replicas.iter().map(|(_, c)| c))
            .flat_map(|c| c.iter().map(|(a, _)| Arc::clone(a)))
            .collect()
    }

    /// Queue `run`, inside `chunk`, into every copy, each tagged with its node.
    fn queue(
        &self,
        device: &Device,
        run: &ObjectRun,
        chunk: usize,
        out: &mut Vec<(NumaNode, StridedCopy)>,
    ) {
        let offset = (run.object_offset - self.shape.starts[chunk]) as usize;
        for chunks in std::iter::once(&self.chunks).chain(self.replicas.iter().map(|(_, c)| c)) {
            let (alloc, numa) = &chunks[chunk];
            out.push((*numa, copy_of(device, alloc.mapped_ptr().add(offset), run)));
        }
    }

    /// Claim the units of `run` (in `chunk`) no save claimed yet and queue
    /// their copies; returns the bytes queued.
    fn claim(
        &mut self,
        device: &Device,
        run: &ObjectRun,
        chunk: usize,
        out: &mut Vec<(NumaNode, StridedCopy)>,
    ) -> u64 {
        let copies = 1 + self.replicas.len() as u64;
        let u = self.shape.unit;
        // Blocks in units: the run's object offset, width and strides are
        // whole units (checked).
        let (len, mut outer) = run.object_blocks();
        for d in &mut outer {
            d.object_stride /= u;
        }
        let (first, len) = (run.object_offset / u, len / u);
        let blocks = || offsets((0, first), &outer).map(|(_, lo)| (lo, lo + len));
        let whole = blocks().all(|(lo, hi)| self.claimed.none(lo, hi));
        if whole {
            for (lo, hi) in blocks() {
                self.claimed.mark(lo, hi);
            }
            self.queue(device, run, chunk, out);
            return run.bytes() * copies;
        }
        // Another save claimed some of these units: copy the rest piece by piece.
        let mut bytes = 0;
        for (at, object_at) in run.row_offsets() {
            let (lo, hi) = (object_at / u, (object_at + run.width) / u);
            let mut g = lo;
            while g < hi {
                if self.claimed.get(g) {
                    g += 1;
                    continue;
                }
                let start = g;
                while g < hi && !self.claimed.get(g) {
                    g += 1;
                }
                self.claimed.mark(start, g);
                let piece = ObjectRun {
                    region: run.region,
                    region_offset: at + (start - lo) * u,
                    object_offset: start * u,
                    width: (g - start) * u,
                    dims: Vec::new(),
                };
                self.queue(device, &piece, chunk, out);
                bytes += piece.width * copies;
            }
        }
        bytes
    }

    /// The primary and every replica by socket, one slot per chunk, each a
    /// single segment of the chunk's bytes.
    fn seal(self) -> (SealedBlock, Vec<(u32, SealedBlock)>) {
        let sizes = self.shape.chunks;
        let block = |chunks: Chunks| {
            SealedBlock::from_slots(
                chunks
                    .into_iter()
                    .zip(&sizes)
                    .map(|((alloc, numa), &size)| {
                        let segment = Segment::new(alloc.as_non_null(), size as usize, alloc);
                        (RawBlock::single_segment(segment), numa)
                    })
                    .collect(),
            )
        };
        let replicas = self
            .replicas
            .into_iter()
            .map(|(socket, c)| (socket, block(c)))
            .collect();
        (block(self.chunks), replicas)
    }
}

#[derive(Default)]
struct PendingObjects {
    objects: HashMap<BlockKey, Pending>,
    next_id: u64,
}

impl PendingObjects {
    fn insert(&mut self, key: BlockKey, p: Pending) {
        core_metrics().inflight_bytes.add(p.bytes() as i64, &[]);
        self.objects.insert(key, p);
    }

    fn take(&mut self, key: &BlockKey) -> Option<Pending> {
        let p = self.objects.remove(key)?;
        core_metrics().inflight_bytes.add(-(p.bytes() as i64), &[]);
        Some(p)
    }

    /// Drop objects no save touched for `max_age`; returns how many. A copy
    /// still in flight keeps its memory alive, and its save then finds the
    /// object gone.
    fn expire(&mut self, max_age: Duration) -> usize {
        let stale: Vec<BlockKey> = self
            .objects
            .iter()
            .filter(|(_, p)| p.touched.elapsed() > max_age)
            .map(|(k, _)| k.clone())
            .collect();
        for key in &stale {
            warn!("objects: dropping unsealed {key:?}, idle for over {max_age:?}");
            self.take(key);
        }
        stale.len()
    }
}

/// A device's IPC-opened regions; closed when the last holder drops them.
struct Regions {
    ctx: Arc<CudaContext>,
    list: Vec<(sys::CUdeviceptr, u64)>,
}

impl Drop for Regions {
    fn drop(&mut self) {
        if self.ctx.bind_to_thread().is_err() {
            return;
        }
        for &(ptr, _) in &self.list {
            // SAFETY: `ptr` came from cuIpcOpenMemHandle in this context and is closed once.
            let r = unsafe { sys::cuIpcCloseMemHandle(ptr) };
            if r != sys::cudaError_enum::CUDA_SUCCESS {
                warn!("objects: cuIpcCloseMemHandle failed: {r:?}");
            }
        }
    }
}

/// A copy stream of one device and the copier that owns its descriptor buffers.
struct Lane {
    ctx: Arc<CudaContext>,
    /// The device's NUMA node.
    numa: NumaNode,
    stream: Arc<CudaStream>,
    copier: StridedCopier,
}

impl Lane {
    fn new(ctx: &Arc<CudaContext>, numa: NumaNode) -> Result<Self, EngineError> {
        ctx.bind_to_thread()
            .map_err(|e| EngineError::CudaInit(format!("{e:?}")))?;
        let stream = ctx
            .new_stream()
            .map_err(|e| EngineError::CudaInit(format!("{e:?}")))?;
        let copier = StridedCopier::new(ctx).map_err(EngineError::CudaInit)?;
        Ok(Self {
            ctx: Arc::clone(ctx),
            numa,
            stream,
            copier,
        })
    }

    /// Run `copies`, each tagged with the node of its pinned memory.
    ///
    /// Writes into host memory go one launch per node, the lane's own node
    /// first: a launch whose writes alternate between nodes moves a fraction
    /// of what one writing a single node does, and ranks of one instance
    /// then start on different nodes. Reads from several nodes overlap well,
    /// so a load is one launch.
    fn transfer(
        &mut self,
        copies: &[(NumaNode, StridedCopy)],
        dir: Direction,
    ) -> Result<(), EngineError> {
        if copies.is_empty() {
            return Ok(());
        }
        let failed = |e: String| EngineError::Storage(format!("object copy: {e}"));
        self.ctx
            .bind_to_thread()
            .map_err(|e| failed(format!("{e:?}")))?;
        let mut sorted = copies.to_vec();
        if dir == Direction::DeviceToHost {
            // Stable, so each node's copies keep their order.
            sorted.sort_by_key(|(numa, _)| (*numa != self.numa, *numa));
        }
        let flat: Vec<StridedCopy> = sorted.iter().map(|(_, c)| *c).collect();
        let mut groups = Vec::new();
        match dir {
            Direction::HostToDevice => groups.push(&flat[..]),
            Direction::DeviceToHost => {
                let mut at = 0;
                for group in sorted.chunk_by(|a, b| a.0 == b.0) {
                    groups.push(&flat[at..at + group.len()]);
                    at += group.len();
                }
            }
        }
        self.copier
            .copy(&groups, dir, &self.stream)
            .map_err(failed)?;
        self.stream
            .synchronize()
            .map_err(|e| failed(format!("{e:?}")))
    }
}

struct CopyJob {
    copies: Vec<(NumaNode, StridedCopy)>,
    /// Pinned memory the copies touch, kept alive until they finish.
    keep: Box<dyn Send>,
    bytes: u64,
    done: LoadCompletion,
}

/// Run a device's copies of one direction in order until the device is dropped.
fn copy_worker(
    mut lane: Lane,
    dir: Direction,
    _regions: Arc<Regions>,
    mut jobs: mpsc::UnboundedReceiver<CopyJob>,
) {
    while let Some(job) = jobs.blocking_recv() {
        let start = Instant::now();
        let result = lane.transfer(&job.copies, dir);
        let elapsed = start.elapsed();
        let m = core_metrics();
        match (&result, dir) {
            (Ok(()), Direction::DeviceToHost) => {
                m.save_bytes.add(job.bytes, &[]);
                m.save_duration_seconds.record(elapsed.as_secs_f64(), &[]);
            }
            (Ok(()), Direction::HostToDevice) => {
                m.load_bytes.add(job.bytes, &[]);
                m.load_duration_seconds.record(elapsed.as_secs_f64(), &[]);
            }
            (Err(e), Direction::HostToDevice) => {
                m.load_failures.add(1, &[]);
                warn!("load_objects failed: {e}");
            }
            (Err(e), Direction::DeviceToHost) => warn!("save_objects failed: {e}"),
        }
        debug!(
            "object copy {}: {} copies, {} bytes in {elapsed:?}",
            if dir == Direction::DeviceToHost {
                "d2h"
            } else {
                "h2d"
            },
            job.copies.len(),
            job.bytes,
        );
        drop(job.keep);
        job.done.signal(result);
    }
}

/// One registered device: its regions and a save and a load worker, each
/// pinned to the GPU's node with its own stream, so a load never queues
/// behind a save.
struct Device {
    regions: Arc<Regions>,
    /// The GPU's NUMA node and its socket.
    numa: NumaNode,
    socket: u32,
    /// Whether object memory is placed by node: the pool is NUMA-aware and
    /// the GPU's node is known.
    placed: bool,
    save: mpsc::UnboundedSender<CopyJob>,
    load: mpsc::UnboundedSender<CopyJob>,
}

impl Device {
    /// The run is well formed and inside its region.
    fn check(&self, run: &ObjectRun) -> Result<(), EngineError> {
        let (_, size) = self.regions.list.get(run.region as usize).ok_or_else(|| {
            EngineError::InvalidArgument(format!("unknown region {}", run.region))
        })?;
        match run.ends() {
            Some((re, _)) if re <= *size => Ok(()),
            _ => Err(EngineError::InvalidArgument(format!(
                "run {run:?} is empty, overlapping or out of bounds"
            ))),
        }
    }

    /// Queue `job` on the save or the load worker.
    fn submit(&self, dir: Direction, job: CopyJob) -> Result<(), EngineError> {
        let worker = match dir {
            Direction::DeviceToHost => &self.save,
            Direction::HostToDevice => &self.load,
        };
        worker
            .send(job)
            .map_err(|_| EngineError::Storage("object copy worker exited".into()))
    }

    /// Run `copies` on the worker of `dir` and wait for them.
    async fn run(
        &self,
        dir: Direction,
        copies: Vec<(NumaNode, StridedCopy)>,
        keep: Box<dyn Send>,
        bytes: u64,
    ) -> Result<(), EngineError> {
        let (reply, done) = oneshot::channel();
        self.submit(
            dir,
            CopyJob {
                copies,
                keep,
                bytes,
                done: LoadCompletion::Channel(reply),
            },
        )?;
        done.await
            .map_err(|_| EngineError::Storage("object copy worker exited".into()))?
    }
}

/// Object devices, unsealed objects, and which contents to copy per socket;
/// sealed objects live in the block store.
#[derive(Default)]
pub(crate) struct ObjectStore {
    devices: Mutex<HashMap<(String, u32), Arc<Device>>>,
    pending: Mutex<PendingObjects>,
    /// Per fingerprint namespace and socket, the ranks there that read its
    /// objects whole from another socket: content devices hold replicated,
    /// such as a latent every rank keeps. New objects of it get a copy on
    /// every socket two such ranks share: one rank alone reads across
    /// sockets as fast as locally, so a copy would only slow saves down.
    replicated: Mutex<HashMap<String, BTreeMap<u32, BTreeSet<u32>>>>,
}

impl ObjectStore {
    pub(crate) fn has_devices(&self) -> bool {
        !self.devices.lock().is_empty()
    }

    /// Drop every device of `instance`; returns how many.
    pub(crate) fn remove_instance(&self, instance: &str) -> usize {
        let mut devices = self.devices.lock();
        let before = devices.len();
        devices.retain(|(i, _), _| i != instance);
        before - devices.len()
    }

    /// Drop every device; returns the instance of each.
    pub(crate) fn remove_all(&self) -> Vec<String> {
        self.devices.lock().drain().map(|((i, _), _)| i).collect()
    }

    /// Drop unsealed objects no save touched for `max_age`; returns how many.
    pub(crate) fn expire_unsealed(&self, max_age: Duration) -> usize {
        self.pending.lock().expire(max_age)
    }

    /// The nodes of `instance`'s placed devices, by socket.
    fn instance_nodes(&self, instance: &str) -> BTreeMap<u32, Vec<NumaNode>> {
        let mut by_socket: BTreeMap<u32, Vec<NumaNode>> = BTreeMap::new();
        for ((i, _), d) in self.devices.lock().iter() {
            if i == instance && d.placed {
                by_socket.entry(d.socket).or_default().push(d.numa);
            }
        }
        for nodes in by_socket.values_mut() {
            nodes.sort_unstable();
            nodes.dedup();
        }
        by_socket
    }

    /// Where a new object of `namespace` that `device` writes keeps its
    /// copies: the primary first, each a socket and the nodes its chunks
    /// stripe over (none: wherever the pool allocates).
    ///
    /// Striping over the nodes of the instance's devices lets its ranks
    /// share every node's read bandwidth instead of all reading the node of
    /// whichever rank saved first. Contents read whole across sockets keep
    /// one copy per socket instead, each striped within it, so no rank reads
    /// across sockets.
    fn placement(
        &self,
        namespace: &str,
        device: &Device,
        nodes: &BTreeMap<u32, Vec<NumaNode>>,
    ) -> Vec<(u32, Vec<NumaNode>)> {
        if !device.placed {
            return vec![(device.socket, Vec::new())];
        }
        let readers: BTreeSet<u32> = self
            .replicated
            .lock()
            .get(namespace)
            .map(|by_socket| {
                by_socket
                    .iter()
                    .filter(|(_, ranks)| ranks.len() >= 2)
                    .map(|(s, _)| *s)
                    .collect()
            })
            .unwrap_or_default();
        if readers.is_empty() {
            return vec![(device.socket, nodes.values().flatten().copied().collect())];
        }
        let local = nodes.get(&device.socket).cloned().unwrap_or_default();
        std::iter::once((device.socket, local))
            .chain(
                nodes
                    .iter()
                    .filter(|(s, _)| **s != device.socket && readers.contains(s))
                    .map(|(s, n)| (*s, n.clone())),
            )
            .collect()
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

    fn require_instance(&self, instance: &str) -> Result<(), EngineError> {
        if self.devices.lock().keys().any(|(i, _)| i == instance) {
            Ok(())
        } else {
            Err(EngineError::InstanceMissing(instance.to_string()))
        }
    }
}

/// The key of `key`'s node-local copy on `socket`. Hex namespaces hold no
/// `@`, so it never names a primary object.
fn replica_key(key: &BlockKey, socket: u32) -> BlockKey {
    BlockKey::new(format!("{}@{socket}", key.namespace), key.hash.clone())
}

/// What a save does with one write.
enum Target {
    /// Sealed already: nothing to copy.
    Resident,
    Pending,
    /// A new object of this many padded bytes per chunk, with these copies.
    New(Vec<NonZeroU64>, Vec<(u32, Vec<NumaNode>)>),
}

impl PegaEngine {
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
        let ctx = CudaContext::new(ordinal).map_err(cuda)?;
        ctx.bind_to_thread().map_err(cuda)?;
        let mut opened = Regions {
            ctx: Arc::clone(&ctx),
            list: Vec::with_capacity(regions.len()),
        };
        for (i, (handle, size)) in regions.iter().enumerate() {
            let ptr =
                open_ipc(handle).map_err(|e| EngineError::CudaInit(format!("region {i}: {e}")))?;
            opened.list.push((ptr, *size));
            let mapped = mapped_size(ptr).map_err(EngineError::CudaInit)?;
            if mapped < *size {
                return Err(EngineError::InvalidArgument(format!(
                    "region {i} declares {size} bytes but maps {mapped}"
                )));
            }
        }
        let regions = Arc::new(opened);
        let spawn = |dir: Direction, name: &str| {
            let (tx, rx) = mpsc::unbounded_channel();
            let (ctx, held) = (Arc::clone(&ctx), Arc::clone(&regions));
            let ready = spawn_pinned_worker(
                format!("gpu{device_id}-obj{name}"),
                numa,
                move || Lane::new(&ctx, numa),
                move |lane| copy_worker(lane, dir, held, rx),
            )?;
            wait_worker_ready(&format!("object {name}"), device_id, ready)?;
            Ok::<_, EngineError>(tx)
        };
        let save = spawn(Direction::DeviceToHost, "save")?;
        let load = spawn(Direction::HostToDevice, "load")?;
        let socket = self.topology.socket_of(numa);
        info!(
            "Registered object device: instance={instance_id} rank={rank} device={device_id} \
             regions={} numa={numa:?} socket={socket}",
            regions.list.len(),
        );
        let device = Device {
            regions,
            numa,
            socket,
            placed: self.storage.is_numa_enabled() && numa.is_valid(),
            save,
            load,
        };
        self.objects
            .devices
            .lock()
            .insert((instance_id.to_string(), rank), Arc::new(device));
        self.storage.set_has_instance(true);
        Ok(())
    }

    /// Answer `query` for request `req_id` of `instance_id`, as `QueryPrefetch`
    /// does for KV blocks.
    ///
    /// The first call finds the last point every chain and its keys can reach
    /// in memory, from peers or on this node's SSD; with no points, the
    /// longest prefix every chain reaches. If that needs objects not in
    /// memory it fetches them, peers first and then SSD, and answers
    /// `Loading`; later calls with the same `req_id` poll the fetch. `Ready`
    /// answers from what is then held, a shorter point if the fetch fell short.
    pub async fn query_objects(
        &self,
        instance_id: &str,
        req_id: &str,
        query: &ObjectQuery,
    ) -> Result<ObjectQueryStatus, EngineError> {
        self.objects.require_instance(instance_id)?;
        let plan = Plan::new(query)?;
        let tag = format!("objects:{instance_id}:{req_id}");
        match self.storage.poll_fetch(&tag).await {
            PartsPoll::Loading => return Ok(ObjectQueryStatus::Loading),
            PartsPoll::Done(fetched) => {
                return Ok(self.ready_objects(instance_id, &plan, fetched));
            }
            PartsPoll::Idle => {}
        }
        let chains: Vec<Vec<Option<Arc<SealedBlock>>>> = plan
            .prefixes
            .iter()
            .map(|c| self.storage.get_blocks_aligned(c))
            .collect();
        let resident: Vec<usize> = chains
            .iter()
            .map(|c| c.iter().take_while(|b| b.is_some()).count())
            .collect();
        let point_keys: Vec<BlockKey> = plan
            .points
            .iter()
            .flat_map(|(_, keys)| keys.iter().cloned())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        let held: std::collections::HashSet<BlockKey> = point_keys
            .iter()
            .zip(self.storage.get_blocks_aligned(&point_keys))
            .filter_map(|(k, b)| b.map(|_| k.clone()))
            .collect();
        let in_memory = plan.best(|c| resident[c], |k| held.contains(k));
        let full = match plan.points.len() {
            0 => plan
                .prefixes
                .iter()
                .zip(&resident)
                .all(|(c, &r)| r == c.len()),
            n => in_memory.0 == Some(n - 1),
        };
        if full {
            return Ok(self.ready_objects(instance_id, &plan, Vec::new()));
        }

        // What peers and SSD add: each chain past its resident prefix, and
        // every point key not in memory, each on its own.
        let missing: Vec<BlockKey> = point_keys
            .into_iter()
            .filter(|k| !held.contains(k))
            .collect();
        let mut sets: Vec<Vec<BlockKey>> = plan
            .prefixes
            .iter()
            .zip(&resident)
            .map(|(c, &r)| c[r..].to_vec())
            .collect();
        sets.extend(missing.iter().map(|k| vec![k.clone()]));
        let reach = self.storage.reach(&sets).await;
        let chain_reach: Vec<usize> = resident.iter().zip(&reach).map(|(r, n)| r + n).collect();
        let reachable: std::collections::HashSet<&BlockKey> = missing
            .iter()
            .zip(&reach[plan.prefixes.len()..])
            .filter(|(_, n)| **n == 1)
            .map(|(k, _)| k)
            .collect();
        let target = plan.best(
            |c| chain_reach[c],
            |k| held.contains(k) || reachable.contains(k),
        );
        if target == in_memory {
            return Ok(self.ready_objects(instance_id, &plan, Vec::new()));
        }

        // A point is of no use partly fetched; a plain prefix is.
        let whole = target.0.is_some();
        let mut parts: Vec<FetchPart> = plan
            .prefixes
            .iter()
            .zip(&resident)
            .filter(|(_, r)| **r < target.1)
            .map(|(c, &r)| FetchPart {
                keys: c[r..target.1].to_vec(),
                whole,
            })
            .collect();
        if let Some(p) = target.0 {
            let mut seen = std::collections::HashSet::new();
            parts.extend(
                plan.points[p]
                    .1
                    .iter()
                    .filter(|k| !held.contains(*k) && seen.insert(*k))
                    .map(|k| FetchPart {
                        keys: vec![k.clone()],
                        whole: true,
                    }),
            );
        }
        debug!(
            "query_objects {instance_id}/{req_id}: point {:?} at {} blocks fetches {} keys",
            target.0,
            target.1,
            parts.iter().map(|p| p.keys.len()).sum::<usize>()
        );
        if !self.storage.start_fetch(&tag, parts) {
            return Ok(self.ready_objects(instance_id, &plan, Vec::new()));
        }
        Ok(ObjectQueryStatus::Loading)
    }

    /// Lease the best point, or prefix, of what is resident or in `fetched`.
    fn ready_objects(
        &self,
        instance_id: &str,
        plan: &Plan,
        fetched: Vec<(BlockKey, Arc<SealedBlock>)>,
    ) -> ObjectQueryStatus {
        let fetched: HashMap<BlockKey, Arc<SealedBlock>> = fetched.into_iter().collect();
        // Touch what is resident; fetched blocks evicted since are still held here.
        let lookup = |keys: &[BlockKey]| -> Vec<Option<Arc<SealedBlock>>> {
            self.storage
                .get_blocks_aligned(keys)
                .into_iter()
                .zip(keys)
                .map(|(b, k)| b.or_else(|| fetched.get(k).cloned()))
                .collect()
        };
        let chains: Vec<Vec<Option<Arc<SealedBlock>>>> =
            plan.prefixes.iter().map(|c| lookup(c)).collect();
        let point_keys: Vec<BlockKey> = plan
            .points
            .iter()
            .flat_map(|(_, keys)| keys.iter().cloned())
            .collect();
        let points: HashMap<BlockKey, Arc<SealedBlock>> = point_keys
            .iter()
            .cloned()
            .zip(lookup(&point_keys))
            .filter_map(|(k, b)| Some((k, b?)))
            .collect();
        let (point, blocks) = plan.best(
            |c| chains[c].iter().take_while(|b| b.is_some()).count(),
            |k| points.contains_key(k),
        );
        let mut objects: HashMap<BlockKey, Arc<SealedBlock>> = HashMap::new();
        for (keys, held) in plan.prefixes.iter().zip(&chains) {
            for (k, b) in keys[..blocks].iter().zip(held) {
                objects.insert(
                    k.clone(),
                    Arc::clone(b.as_ref().expect("leading blocks held")),
                );
            }
        }
        if let Some(p) = point {
            for k in &plan.points[p].1 {
                objects.insert(k.clone(), Arc::clone(&points[k]));
            }
        }
        let metrics = core_metrics();
        let asked = plan.prefixes.iter().map(Vec::len).sum::<usize>();
        let hit = blocks * plan.prefixes.len();
        metrics.cache_block_hits.add(hit as u64, &[]);
        metrics.cache_block_misses.add((asked - hit) as u64, &[]);
        ObjectQueryStatus::Ready {
            point,
            blocks: blocks as u32,
            lease: (!objects.is_empty())
                .then(|| self.query_leases.create_objects(instance_id, objects)),
        }
    }

    /// Copy `writes` from object device `rank` of `instance_id`; returns how
    /// many objects this save sealed into the cache. Units already claimed by
    /// another save, and writes to sealed objects, copy nothing.
    ///
    /// Dropping the future before it finishes leaves the objects it copies
    /// into unsealed until [`Self::gc_stale_inflight`] expires them.
    pub async fn save_objects(
        &self,
        instance_id: &str,
        rank: u32,
        writes: &[ObjectWrite],
    ) -> Result<usize, EngineError> {
        let start = Instant::now();
        let device = self.objects.device(instance_id, rank)?;
        let mut shapes = Vec::with_capacity(writes.len());
        let mut chunks = Vec::with_capacity(writes.len());
        for w in writes {
            let shape = Shape::of(w)?;
            let mut of_runs = Vec::with_capacity(w.runs.len());
            for run in &w.runs {
                device.check(run)?;
                of_runs.push(shape.check(run)?);
            }
            shapes.push(shape);
            chunks.push(of_runs);
        }
        let keys: Vec<BlockKey> = writes.iter().map(|w| w.key.block_key()).collect();
        let resident = self.storage.get_blocks_aligned(&keys);
        let nodes = self.objects.instance_nodes(instance_id);
        let mut copies = Vec::new();
        let mut bytes = 0;
        // (write, pending id) of every object this save copies into.
        let mut targets = Vec::new();
        let mut keep: Vec<Arc<PinnedAllocation>> = Vec::new();
        {
            let now = Instant::now();
            let mut pending = self.objects.pending.lock();
            // Check every write before allocating or claiming anything.
            let mut plan = Vec::with_capacity(writes.len());
            for (i, block) in resident.iter().enumerate() {
                let shape = &shapes[i];
                let target = if let Some(block) = block {
                    let stored: Vec<u64> = block
                        .slots()
                        .iter()
                        .map(RawBlock::memory_footprint)
                        .collect();
                    if stored != shape.chunks {
                        return Err(mismatch(&writes[i], format!("chunks {stored:?}")));
                    }
                    Target::Resident
                } else if let Some(p) = pending.objects.get(&keys[i]) {
                    if p.shape != *shape {
                        return Err(mismatch(&writes[i], format!("{:?}", p.shape)));
                    }
                    Target::Pending
                } else if let Some((j, _)) = plan
                    .iter()
                    .find(|(j, t)| matches!(t, Target::New(..)) && keys[*j] == keys[i])
                {
                    // A second write of the same new object in this save.
                    if shapes[*j] != *shape {
                        return Err(mismatch(&writes[i], format!("{:?}", shapes[*j])));
                    }
                    Target::Pending
                } else {
                    let copies = self.objects.placement(&keys[i].namespace, &device, &nodes);
                    Target::New(
                        shape
                            .chunks
                            .iter()
                            .map(|&c| NonZeroU64::new(io_len(c)).expect("chunks are nonzero"))
                            .collect(),
                        copies,
                    )
                };
                plan.push((i, target));
            }
            for (i, target) in plan {
                let key = &keys[i];
                match target {
                    Target::Resident => continue,
                    Target::Pending => {}
                    Target::New(sizes, placement) => {
                        let seed = stripe_seed(&writes[i].key);
                        let mut copies = Vec::with_capacity(placement.len());
                        for (socket, nodes) in placement {
                            let mut allocs = Vec::with_capacity(sizes.len());
                            for (c, &size) in sizes.iter().enumerate() {
                                let numa = match nodes.as_slice() {
                                    [] => None,
                                    n => Some(n[(seed + c) % n.len()]),
                                };
                                let alloc = self.storage.allocate(size, numa).ok_or_else(|| {
                                    EngineError::Storage(format!(
                                        "pinned pool cannot hold a {} byte object",
                                        shapes[i].size
                                    ))
                                })?;
                                allocs.push((alloc, numa.unwrap_or(device.numa)));
                            }
                            copies.push((socket, allocs));
                        }
                        let (_, chunks) = copies.remove(0);
                        pending.next_id += 1;
                        let p = Pending {
                            id: pending.next_id,
                            claimed: Bits::new(shapes[i].size / shapes[i].unit),
                            shape: shapes[i].clone(),
                            chunks,
                            replicas: copies,
                            copying: 0,
                            touched: now,
                        };
                        pending.insert(key.clone(), p);
                    }
                }
                let p = pending.objects.get_mut(key).expect("present above");
                let before = copies.len();
                for (run, &chunk) in writes[i].runs.iter().zip(&chunks[i]) {
                    bytes += p.claim(&device, run, chunk, &mut copies);
                }
                if copies.len() > before {
                    p.copying += 1;
                    p.touched = now;
                    targets.push((i, p.id));
                    keep.extend(p.allocations());
                }
            }
        }
        let copy = Instant::now();
        let prep = copy.duration_since(start);
        let n_copies = copies.len();
        let result = device
            .run(Direction::DeviceToHost, copies, Box::new(keep), bytes)
            .await;
        let copied = copy.elapsed();

        let mut sealed = Vec::new();
        let mut replicas = Vec::new();
        {
            let mut pending = self.objects.pending.lock();
            for &(i, id) in &targets {
                let Some(p) = pending.objects.get_mut(&keys[i]) else {
                    continue;
                };
                // Dropped as stale or failed, and started again while this save copied.
                if p.id != id {
                    continue;
                }
                if result.is_err() {
                    // Its claims no longer say which units hold data.
                    pending.take(&keys[i]);
                    continue;
                }
                p.copying -= 1;
                p.touched = Instant::now();
                if p.copying == 0 && p.claimed.complete() {
                    let (block, copies) = pending.take(&keys[i]).expect("present above").seal();
                    replicas.extend(
                        copies
                            .into_iter()
                            .map(|(socket, b)| (replica_key(&keys[i], socket), Arc::new(b))),
                    );
                    sealed.push((keys[i].clone(), Arc::new(block)));
                }
            }
        }
        result?;
        let n = self.storage.insert_sealed(&sealed, &replicas);
        debug!(
            "save_objects {instance_id}/{rank}: {n_copies} copies, {bytes} bytes: prep {prep:?} \
             copy {copied:?} finish {:?}, sealed {n} with {} socket copies",
            copy.elapsed() - copied,
            replicas.len()
        );
        Ok(n)
    }

    /// Copy objects held by the leases of `loads` into object device `rank`
    /// of `instance_id`. With `load_state_shm`, returns once the request is
    /// checked and sets that `LoadState` when the copies finish; otherwise
    /// returns after them. Loads of a device run in order on their own stream.
    /// Leases stay valid until released or expired.
    ///
    /// A read takes the copy on the device's socket when there is one. Ranks
    /// reading a whole chunk from another socket mark the object's content
    /// as one to copy on their socket from then on.
    pub async fn load_objects(
        &self,
        instance_id: &str,
        rank: u32,
        loads: &[ObjectLoad],
        load_state_shm: Option<&str>,
    ) -> Result<(), EngineError> {
        let device = self.objects.device(instance_id, rank)?;
        let mut copies = Vec::new();
        let mut objects = Vec::new();
        let mut bytes = 0;
        let mut learned: Vec<String> = Vec::new();
        for l in loads {
            let keys: Vec<BlockKey> = l.reads.iter().map(|r| r.key.block_key()).collect();
            let leased = self
                .query_leases
                .leased_objects(instance_id, &l.lease, &keys)
                .map_err(EngineError::InvalidArgument)?;
            let local: Vec<BlockKey> = keys.iter().map(|k| replica_key(k, device.socket)).collect();
            let local = self.storage.get_blocks_aligned(&local);
            for (((r, leased), local), key) in l.reads.iter().zip(leased).zip(local).zip(&keys) {
                let leased = leased.ok_or_else(|| {
                    EngineError::InvalidArgument(format!("{:?} does not hold {key:?}", l.lease))
                })?;
                let block = local.unwrap_or(leased);
                let sizes: Vec<u64> = block
                    .slots()
                    .iter()
                    .map(RawBlock::memory_footprint)
                    .collect();
                let starts = starts(&sizes);
                let mut read = vec![0u64; sizes.len()];
                for run in &r.runs {
                    device.check(run)?;
                    let (c, numa, at) = run
                        .ends()
                        .and_then(|(_, oe)| chunk_of(&starts, &sizes, run.object_offset, oe))
                        .and_then(|c| {
                            let base = block.slots()[c].segment_mapped_ptr(0)?;
                            let at = base.add((run.object_offset - starts[c]) as usize);
                            Some((c, block.slot_numas()[c], at))
                        })
                        .ok_or_else(|| {
                            EngineError::InvalidArgument(format!(
                                "run {run:?} is out of bounds or crosses a chunk of {key:?}"
                            ))
                        })?;
                    copies.push((numa, copy_of(&device, at, run)));
                    let n = run.bytes();
                    read[c] += n;
                    bytes += n;
                }
                let remote_whole = device.placed
                    && block.slot_numas().iter().zip(&read).zip(&sizes).any(
                        |((&numa, &got), &size)| {
                            got >= size && self.topology.socket_of(numa) != device.socket
                        },
                    );
                if remote_whole {
                    learned.push(key.namespace.clone());
                }
                objects.push(block);
            }
        }
        if !learned.is_empty() {
            let mut replicated = self.objects.replicated.lock();
            for namespace in learned {
                let ranks = replicated
                    .entry(namespace.clone())
                    .or_default()
                    .entry(device.socket)
                    .or_default();
                if ranks.insert(rank) && ranks.len() == 2 {
                    info!(
                        "objects: ranks on socket {} read {namespace} whole from another \
                         socket; copying it per socket",
                        device.socket
                    );
                }
            }
        }
        let keep = Box::new(objects);
        match load_state_shm {
            Some(name) => device.submit(
                Direction::HostToDevice,
                CopyJob {
                    copies,
                    keep,
                    bytes,
                    done: LoadCompletion::Shm(name.to_string()),
                },
            ),
            None => {
                device
                    .run(Direction::HostToDevice, copies, keep, bytes)
                    .await
            }
        }
    }
}

/// Where an object's chunks start striping, so one-chunk objects alternate nodes.
fn stripe_seed(key: &ObjectKey) -> usize {
    let mut b = [0; 8];
    b.copy_from_slice(&key.prefix[..8]);
    u64::from_le_bytes(b) as usize
}

fn mismatch(w: &ObjectWrite, stored: String) -> EngineError {
    EngineError::InvalidArgument(format!(
        "object {:?} is {stored}, write says size {} unit {} chunks {:?}",
        w.key, w.size, w.unit, w.chunks
    ))
}

/// The copy of `run` between its device region and pinned memory at `at`.
fn copy_of(device: &Device, at: MappedPinnedPtr, run: &ObjectRun) -> StridedCopy {
    let dims: Vec<Dim> = run
        .dims
        .iter()
        .map(|d| Dim {
            count: u64::from(d.count),
            device_pitch: d.region_stride,
            host_pitch: d.object_stride,
        })
        .collect();
    StridedCopy::new(
        device.regions.list[run.region as usize].0 + run.region_offset,
        at.device().as_ptr() as u64,
        run.width,
        &dims,
    )
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

    fn dim(count: u32, region_stride: u64, object_stride: u64) -> ObjectDim {
        ObjectDim {
            count,
            region_stride,
            object_stride,
        }
    }

    fn run(object_offset: u64, width: u64, rows: u32, object_stride: u64) -> ObjectRun {
        ObjectRun {
            region: 0,
            region_offset: 0,
            object_offset,
            width,
            dims: vec![dim(rows, width, object_stride)],
        }
    }

    fn write(size: u64, unit: u64, chunks: &[u64]) -> ObjectWrite {
        ObjectWrite {
            key: ObjectKey {
                fingerprint: [0; 32],
                prefix: [0; 32],
            },
            size,
            unit,
            chunks: chunks.to_vec(),
            runs: Vec::new(),
        }
    }

    #[test]
    fn shapes_check_units_and_chunks() {
        let s = Shape::of(&write(192, 32, &[64, 128])).expect("valid");
        // Two interleaved heads of 32 bytes, rows 64 apart, inside chunk 1.
        assert_eq!(s.check(&run(64, 32, 2, 64)).ok(), Some(1));
        assert_eq!(s.check(&run(0, 64, 1, 0)).ok(), Some(0));
        for (case, r) in [
            ("off unit", run(16, 32, 1, 0)),
            ("crosses chunks", run(32, 64, 1, 0)),
            ("past the end", run(160, 32, 2, 64)),
            ("overlapping rows", run(64, 64, 2, 32)),
        ] {
            assert!(s.check(&r).is_err(), "{case}");
        }
        for (case, w) in [
            ("unit does not divide", write(192, 40, &[])),
            ("chunk off unit", write(192, 32, &[48, 144])),
            ("chunks short", write(192, 32, &[64, 64])),
            ("zero unit", write(192, 0, &[])),
            ("too many units", write(MAX_UNITS + 1, 1, &[])),
        ] {
            assert!(Shape::of(&w).is_err(), "{case}");
        }
        assert_eq!(Shape::of(&write(96, 32, &[])).unwrap().chunks, vec![96]);
    }

    #[test]
    fn plans_pick_the_last_point_every_chain_holds() {
        let key = |f: u8, n: u8| ObjectKey {
            fingerprint: [f; 32],
            prefix: [n; 32],
        };
        let point = |blocks, keys: &[u8]| ObjectPoint {
            blocks,
            keys: keys.iter().map(|&n| key(9, n)).collect(),
        };
        let q = ObjectQuery {
            prefixes: vec![(0..8).map(|n| key(0, n)).collect()],
            points: vec![point(6, &[96]), point(6, &[107]), point(8, &[140])],
        };
        let plan = Plan::new(&q).expect("valid");
        let held = |ns: &'static [u8]| move |k: &BlockKey| ns.iter().any(|&n| k.hash[0] == n);
        // 140 needs 8 blocks, the chain holds 7: 107 wins over 96.
        assert_eq!(plan.best(|_| 7, held(&[96, 107, 140])), (Some(1), 6));
        assert_eq!(plan.best(|_| 8, held(&[96, 140])), (Some(2), 8));
        assert_eq!(plan.best(|_| 5, held(&[96, 107, 140])), (None, 0));
        let prefix_only = Plan::new(&ObjectQuery {
            prefixes: q.prefixes.clone(),
            points: vec![],
        })
        .unwrap();
        assert_eq!(prefix_only.best(|_| 5, held(&[])), (None, 5));
        for (case, q) in [
            (
                "past the chain",
                ObjectQuery {
                    points: vec![point(9, &[])],
                    ..q.clone()
                },
            ),
            (
                "mixed fingerprints",
                ObjectQuery {
                    prefixes: vec![vec![key(0, 0), key(1, 1)]],
                    points: vec![],
                },
            ),
        ] {
            assert!(Plan::new(&q).is_err(), "{case}");
        }
    }

    #[test]
    fn bits_track_ranges_across_words() {
        let mut b = Bits::new(130);
        assert!(b.none(0, 130));
        b.mark(60, 70);
        b.mark(65, 128);
        assert_eq!(b.set, 68);
        assert!(b.get(60) && b.get(127) && !b.get(59) && !b.get(128));
        assert!(!b.none(0, 61) && b.none(0, 60) && b.none(128, 130));
        b.mark(0, 60);
        b.mark(128, 130);
        assert!(b.complete());
    }

    #[test]
    fn runs_reject_overlap_and_overflow() {
        let run = |width, dims: &[ObjectDim]| ObjectRun {
            region: 0,
            region_offset: 0,
            object_offset: 0,
            width,
            dims: dims.to_vec(),
        };
        assert_eq!(run(8, &[dim(3, 16, 8)]).ends(), Some((40, 24)));
        assert_eq!(run(8, &[]).ends(), Some((8, 8)));
        // A single step ignores its strides.
        let one = run(8, &[dim(1, u64::MAX, u64::MAX), dim(2, 8, 8)]);
        assert_eq!(one.ends(), Some((16, 16)));
        assert_eq!(one.row_offsets().collect::<Vec<_>>(), [(0, 0), (8, 8)]);
        // Heads 64 apart in the region, interleaved with rows in the object.
        let hnd = run(8, &[dim(4, 8, 16), dim(2, 64, 8)]);
        assert_eq!(hnd.ends(), Some((96, 64)));
        let rows: Vec<_> = hnd.row_offsets().collect();
        let naive: Vec<_> = (0..8u64)
            .map(|r| (r % 4 * 8 + r / 4 * 64, r % 4 * 16 + r / 4 * 8))
            .collect();
        assert_eq!(rows, naive);
        // Heads interleave with rows in the object: one block in object order.
        assert_eq!(hnd.object_blocks(), (64, vec![]));
        // Rows back to back in the object form one block per outer step.
        let nhd = run(8, &[dim(4, 64, 8), dim(2, 8, 64)]);
        assert_eq!(nhd.object_blocks(), (32, nhd.dims[1..].to_vec()));
        for (case, r) in [
            ("overlapping rows", run(8, &[dim(2, 4, 8)])),
            ("empty", run(0, &[])),
            ("zero count", run(8, &[dim(0, 8, 8)])),
            ("overflow", run(8, &[dim(2, u64::MAX, 8)])),
            (
                "outer inside inner",
                run(8, &[dim(4, 8, 8), dim(2, 16, 64)]),
            ),
            ("too many dims", run(8, &[dim(2, 8, 8); MAX_DIMS + 1])),
        ] {
            assert!(r.ends().is_none(), "{case}");
        }
    }
}
