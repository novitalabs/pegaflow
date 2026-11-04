//! Byte-addressed objects on the block store.
//!
//! An object is an opaque byte string keyed by `(fingerprint, prefix)` and cut
//! into slots, as a KV block is cut into `layers x ranks` slots. Each save
//! copies strided runs of one device's registered regions into whole slots of
//! pinned memory on that device's NUMA node, and hands them to the insert
//! worker like a KV save; the object seals once all of its slots arrived. The
//! engine never interprets layers, ranks or layouts: the client decides how
//! many slots an object has, which device writes each, and what bytes go where.
//!
//! Sealed objects are [`SealedBlock`]s under the [`BlockKey`]
//! `(hex(fingerprint), prefix)`, one single-segment slot per object slot, so
//! they share the pinned pool, LRU eviction, query leases, the SSD cache and
//! MetaServer registration with KV blocks, and a query brings an evicted
//! object back from a peer or SSD.

use std::collections::{BTreeMap, HashMap, HashSet, hash_map::Entry};
use std::num::NonZeroU64;
use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{CudaContext, CudaStream, sys};
use log::{debug, error, info, warn};
use parking_lot::Mutex;
use tokio::sync::{mpsc, oneshot};

use crate::backing::SSD_ALIGNMENT;
use crate::block::{BlockKey, RawBlock, SealedBlock, Segment};
use crate::gpu_worker::{LoadCompletion, spawn_pinned_worker, wait_worker_ready};
use crate::metrics::core_metrics;
use crate::offload::{RawSaveBatch, RawSaveLayer};
use crate::pinned_pool::{MappedPinnedPtr, PinnedAllocation};
use crate::storage::{FetchPart, PartsPoll};
use crate::transfer::{Dim, Direction, MAX_DIMS, StridedCopier, StridedCopy};
use crate::{EngineError, NumaNode, PegaEngine, QueryLeaseId};

/// Most slots one object may have.
const MAX_SLOTS: u32 = 1 << 12;
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
/// in the device region and `slot_stride` bytes in the slot; the strides are
/// ignored when `count == 1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObjectDim {
    pub count: u32,
    pub region_stride: u64,
    pub slot_stride: u64,
}

/// Rows of `width` bytes over up to [`MAX_DIMS`] dimensions, the first
/// fastest: row `(i_0, i_1, ..)` is at `region_offset + sum(i_d * region_stride_d)`
/// of the device region and `slot_offset + sum(i_d * slot_stride_d)` of the
/// slot. No dimensions is one row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectRun {
    pub region: u32,
    pub region_offset: u64,
    pub slot_offset: u64,
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

    /// One past the last byte on the region and slot side, if the run is
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
            end(self.slot_offset, |d| d.slot_stride)?,
        ))
    }
}

/// One slot of an object, written whole by one save: its runs copy every one
/// of its `size` bytes once.
#[derive(Debug, Clone)]
pub struct SlotWrite {
    pub slot: u32,
    pub size: u64,
    pub runs: Vec<ObjectRun>,
}

/// Slots of an object of `slots` slots one device writes.
#[derive(Debug, Clone)]
pub struct ObjectWrite {
    pub key: ObjectKey,
    pub slots: u32,
    pub writes: Vec<SlotWrite>,
}

#[derive(Debug, Clone)]
pub struct SlotRead {
    pub slot: u32,
    pub runs: Vec<ObjectRun>,
}

#[derive(Debug, Clone)]
pub struct ObjectRead {
    pub key: ObjectKey,
    pub reads: Vec<SlotRead>,
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
    /// is issued when it would hold nothing. `slots` has the slot count of
    /// every object leased: the leading `blocks` of each chain in turn, then
    /// the point's keys.
    Ready {
        point: Option<usize>,
        blocks: u32,
        lease: Option<QueryLeaseId>,
        slots: Vec<u32>,
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
    stream: Arc<CudaStream>,
    copier: StridedCopier,
}

impl Lane {
    fn new(ctx: &Arc<CudaContext>) -> Result<Self, EngineError> {
        ctx.bind_to_thread()
            .map_err(|e| EngineError::CudaInit(format!("{e:?}")))?;
        let stream = ctx
            .new_stream()
            .map_err(|e| EngineError::CudaInit(format!("{e:?}")))?;
        let copier = StridedCopier::new(ctx).map_err(EngineError::CudaInit)?;
        Ok(Self {
            ctx: Arc::clone(ctx),
            stream,
            copier,
        })
    }

    fn transfer(&mut self, copies: &[StridedCopy], dir: Direction) -> Result<(), EngineError> {
        if copies.is_empty() {
            return Ok(());
        }
        let failed = |e: String| EngineError::Storage(format!("object copy: {e}"));
        self.ctx
            .bind_to_thread()
            .map_err(|e| failed(format!("{e:?}")))?;
        self.copier
            .copy(copies, dir, &self.stream)
            .map_err(failed)?;
        self.stream
            .synchronize()
            .map_err(|e| failed(format!("{e:?}")))
    }
}

struct CopyJob {
    copies: Vec<StridedCopy>,
    /// Pinned memory the copies touch, kept alive until they finish.
    keep: Box<dyn Send>,
    bytes: u64,
    done: LoadCompletion,
}

/// Run a device's copies of one direction in order until the device is
/// dropped. Records load metrics as the KV load worker does; saves record
/// theirs on the RPC path.
fn copy_worker(
    mut lane: Lane,
    device_id: i32,
    dir: Direction,
    _regions: Arc<Regions>,
    mut jobs: mpsc::UnboundedReceiver<CopyJob>,
) {
    while let Some(job) = jobs.blocking_recv() {
        let start = Instant::now();
        let result = lane.transfer(&job.copies, dir);
        let elapsed = start.elapsed();
        if dir == Direction::HostToDevice {
            let m = core_metrics();
            match &result {
                Ok(()) if job.bytes > 0 => {
                    m.load_bytes.add(job.bytes, &[]);
                    m.load_duration_seconds.record(elapsed.as_secs_f64(), &[]);
                }
                Ok(()) => {}
                Err(e) => {
                    error!("Object load task failed: device={device_id} error={e:?}");
                    m.load_failures.add(1, &[]);
                }
            }
        }
        debug!(
            "Object copy completed: device={device_id} dir={dir:?} copies={} bytes={} elapsed_ms={:.2}",
            job.copies.len(),
            job.bytes,
            elapsed.as_secs_f64() * 1000.0,
        );
        drop(job.keep);
        job.done.signal(result);
    }
    info!("Object {dir:?} worker shutting down: device={device_id}");
}

/// One registered device: its regions and a save and a load worker, each
/// pinned to the GPU's node with its own stream, so a load never queues
/// behind a save.
struct Device {
    regions: Arc<Regions>,
    /// The GPU's NUMA node; saves allocate there.
    numa: NumaNode,
    save: mpsc::UnboundedSender<CopyJob>,
    load: mpsc::UnboundedSender<CopyJob>,
}

impl Device {
    /// The run is well formed, inside its region, and inside a slot of `size` bytes.
    fn check(&self, run: &ObjectRun, size: u64) -> Result<(), EngineError> {
        let (_, region) = self.regions.list.get(run.region as usize).ok_or_else(|| {
            EngineError::InvalidArgument(format!("unknown region {}", run.region))
        })?;
        match run.ends() {
            Some((re, se)) if re <= *region && se <= size => Ok(()),
            _ => Err(EngineError::InvalidArgument(format!(
                "run {run:?} is empty, overlapping or out of bounds of a {size} byte slot"
            ))),
        }
    }

    fn submit(&self, dir: Direction, job: CopyJob) -> Result<(), EngineError> {
        let worker = match dir {
            Direction::DeviceToHost => &self.save,
            Direction::HostToDevice => &self.load,
        };
        worker
            .send(job)
            .map_err(|_| EngineError::Storage("object copy worker exited".into()))
    }

    /// The copy of `run` between its device region and pinned memory at `at`.
    fn copy_of(&self, at: MappedPinnedPtr, run: &ObjectRun) -> StridedCopy {
        let dims: Vec<Dim> = run
            .dims
            .iter()
            .map(|d| Dim {
                count: u64::from(d.count),
                device_pitch: d.region_stride,
                host_pitch: d.slot_stride,
            })
            .collect();
        StridedCopy::new(
            self.regions.list[run.region as usize].0 + run.region_offset,
            at.device().as_ptr() as u64,
            run.width,
            &dims,
        )
    }
}

/// Registered object devices by `(instance, rank)`; objects live in the block store.
#[derive(Default)]
pub(crate) struct ObjectStore {
    devices: Mutex<HashMap<(String, u32), Arc<Device>>>,
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

fn already_registered(instance: &str, rank: u32) -> EngineError {
    EngineError::InvalidArgument(format!(
        "object device rank {rank} of {instance} is already registered"
    ))
}

impl PegaEngine {
    /// Open `regions` (IPC handle, size) of device `device_id` as object
    /// device `rank` of `instance_id`. A rank registers once until its
    /// instance is unregistered or its session ends. Returns the GPU's NUMA
    /// node, where saves of the device allocate, and its CPU socket.
    pub fn register_object_device(
        &self,
        instance_id: &str,
        rank: u32,
        device_id: i32,
        regions: &[(Vec<u8>, u64)],
    ) -> Result<(NumaNode, u32), EngineError> {
        let key = (instance_id.to_string(), rank);
        if self.objects.devices.lock().contains_key(&key) {
            return Err(already_registered(instance_id, rank));
        }
        let cuda = |e: cudarc::driver::DriverError| EngineError::CudaInit(format!("{e:?}"));
        let ordinal = usize::try_from(device_id)
            .map_err(|_| EngineError::InvalidArgument(format!("device_id {device_id}")))?;
        let numa = self.gpu_numa(device_id)?;
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
                move || Lane::new(&ctx),
                move |lane| copy_worker(lane, device_id, dir, held, rx),
            )?;
            wait_worker_ready(&format!("object {name}"), device_id, ready)?;
            Ok::<_, EngineError>(tx)
        };
        let save = spawn(Direction::DeviceToHost, "save")?;
        let load = spawn(Direction::HostToDevice, "load")?;
        let region_count = regions.list.len();
        let device = Device {
            regions,
            numa,
            save,
            load,
        };
        match self.objects.devices.lock().entry(key) {
            Entry::Occupied(_) => return Err(already_registered(instance_id, rank)),
            Entry::Vacant(v) => {
                v.insert(Arc::new(device));
            }
        }
        self.storage.set_has_instance(true);
        let socket = self.topology.socket_of(numa);
        info!(
            "Registered object device: instance={instance_id} rank={rank} device={device_id} \
             regions={region_count} numa={numa} socket={socket}"
        );
        Ok((numa, socket))
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
        let resident: Vec<usize> = plan
            .prefixes
            .iter()
            .map(|c| {
                self.storage
                    .contains_keys(c)
                    .iter()
                    .take_while(|&&b| b)
                    .count()
            })
            .collect();
        let point_keys: Vec<BlockKey> = plan
            .points
            .iter()
            .flat_map(|(_, keys)| keys.iter().cloned())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        let held: HashSet<BlockKey> = point_keys
            .iter()
            .zip(self.storage.contains_keys(&point_keys))
            .filter(|&(_, b)| b)
            .map(|(k, _)| k.clone())
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
        let reachable: HashSet<&BlockKey> = missing
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
            let mut seen = HashSet::new();
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
        let mut slots = Vec::new();
        let mut lease = |k: &BlockKey, b: &Arc<SealedBlock>| {
            slots.push(b.slots().len() as u32);
            objects.insert(k.clone(), Arc::clone(b));
        };
        for (keys, held) in plan.prefixes.iter().zip(&chains) {
            for (k, b) in keys[..blocks].iter().zip(held) {
                lease(k, b.as_ref().expect("leading blocks held"));
            }
        }
        if let Some(p) = point {
            for k in &plan.points[p].1 {
                lease(k, &points[k]);
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
            slots,
        }
    }

    /// Copy `writes` from object device `rank` of `instance_id` into pinned
    /// memory on the device's NUMA node and queue them for the insert worker,
    /// as a KV save does; returns once the copies finish. Writes to objects
    /// already cached copy nothing. An object seals once all of its slots
    /// arrived; one whose writers disagree on its slot count keeps the first
    /// count, and unsealed objects expire with KV inflight blocks.
    pub async fn save_objects(
        &self,
        instance_id: &str,
        rank: u32,
        writes: &[ObjectWrite],
    ) -> Result<(), EngineError> {
        self.query_leases.sweep_expired();
        let start = Instant::now();
        let device = self.objects.device(instance_id, rank)?;
        for w in writes {
            check_write(&device, w)?;
        }
        let keys: Vec<BlockKey> = writes.iter().map(|w| w.key.block_key()).collect();
        let resident = self.storage.contains_keys(&keys);
        let todo: Vec<usize> = (0..writes.len()).filter(|&i| !resident[i]).collect();
        if todo.is_empty() {
            return Ok(());
        }

        // Like KV saves: SSD-aligned strides when SSD is on, and one
        // allocation per (namespace, slot) across this save's objects unless
        // blockwise, as a KV save allocates once per layer.
        let pad = self.storage.is_ssd_enabled();
        let stride = |size: u64| {
            if pad {
                size.next_multiple_of(SSD_ALIGNMENT as u64)
            } else {
                size
            }
        };
        let mut by_slot: BTreeMap<(&str, u32), Vec<(usize, &SlotWrite)>> = BTreeMap::new();
        for &i in &todo {
            for s in &writes[i].writes {
                by_slot
                    .entry((keys[i].namespace.as_str(), s.slot))
                    .or_default()
                    .push((i, s));
            }
        }
        let numa = Some(device.numa);
        let blockwise = self.storage.blockwise_alloc();
        let mut copies = Vec::new();
        let mut keep: Vec<Arc<PinnedAllocation>> = Vec::new();
        // (write, slot, block, stride) of every slot copied.
        let mut saved: Vec<(usize, u32, RawBlock, u64)> = Vec::new();
        let (mut bytes, mut padded) = (0u64, 0u64);
        for (&(_, slot), list) in &by_slot {
            let groups: Vec<&[(usize, &SlotWrite)]> = if blockwise {
                list.chunks(1).collect()
            } else {
                vec![&list[..]]
            };
            for group in groups {
                let total: u64 = group.iter().map(|(_, s)| stride(s.size)).sum();
                let alloc = NonZeroU64::new(total)
                    .and_then(|n| self.storage.allocate(n, numa))
                    .ok_or_else(|| {
                        EngineError::Storage(format!(
                            "pinned pool exhausted while allocating {total} bytes of object slot {slot}"
                        ))
                    })?;
                let mut at = 0usize;
                for &(i, s) in group {
                    let base = alloc.mapped_ptr().add(at);
                    copies.extend(
                        s.runs
                            .iter()
                            .map(|r| device.copy_of(base.add(r.slot_offset as usize), r)),
                    );
                    let len = stride(s.size);
                    let segment = Segment::new(base.host(), len as usize, Arc::clone(&alloc));
                    saved.push((i, slot, RawBlock::single_segment(segment), len));
                    bytes += s.size;
                    padded += len;
                    at += len as usize;
                }
                keep.push(alloc);
            }
        }
        let prep = start.elapsed();
        let (reply, done) = oneshot::channel();
        device.submit(
            Direction::DeviceToHost,
            CopyJob {
                copies,
                keep: Box::new(keep),
                bytes,
                done: LoadCompletion::Channel(reply),
            },
        )?;
        done.await
            .map_err(|_| EngineError::Storage("object copy worker exited".into()))??;

        let metrics = core_metrics();
        metrics.save_bytes.add(padded, &[]);
        metrics
            .save_duration_seconds
            .record(start.elapsed().as_secs_f64(), &[]);
        debug!(
            "save_objects completed: instance_id={instance_id} rank={rank} objects={} slots={} \
             bytes={padded} numa={} prep_ms={:.2} total_ms={:.2}",
            todo.len(),
            saved.len(),
            device.numa,
            prep.as_secs_f64() * 1000.0,
            start.elapsed().as_secs_f64() * 1000.0,
        );

        // One batch per (namespace, slot count), one layer per slot, as KV
        // saves send one batch per group.
        type Layers = BTreeMap<u32, RawSaveLayer>;
        let mut batches: BTreeMap<(&str, u32), Layers> = BTreeMap::new();
        for (i, slot, block, len) in saved {
            let layer = batches
                .entry((keys[i].namespace.as_str(), writes[i].slots))
                .or_default()
                .entry(slot)
                .or_insert_with(|| RawSaveLayer {
                    slot_id: slot as usize,
                    padded_block_size: len as usize,
                    blocks: Vec::new(),
                    block_hashes: Vec::new(),
                });
            layer.blocks.push(block);
            layer.block_hashes.push(keys[i].hash.clone());
        }
        for ((namespace, slots), layers) in batches {
            self.storage.send_raw_insert(RawSaveBatch {
                namespace: namespace.to_string(),
                total_slots: slots as usize,
                numa_node: device.numa,
                layers: layers.into_values().collect(),
            });
        }
        Ok(())
    }

    /// Copy objects held by the leases of `loads` into object device `rank`
    /// of `instance_id`, as a KV load: returns once the request is checked
    /// and sets the `LoadState` at `load_state_shm` when the copies finish,
    /// or right away if checking fails. Loads of a device run in order on
    /// their own stream. Leases stay valid until released or expired.
    pub fn load_objects(
        &self,
        instance_id: &str,
        rank: u32,
        loads: &[ObjectLoad],
        load_state_shm: &str,
    ) -> Result<(), EngineError> {
        crate::submit_with_load_state(load_state_shm, "load_objects", |done| {
            self.submit_object_loads(instance_id, rank, loads, done)
        })
    }

    fn submit_object_loads(
        &self,
        instance_id: &str,
        rank: u32,
        loads: &[ObjectLoad],
        done: LoadCompletion,
    ) -> Result<(), EngineError> {
        let device = self.objects.device(instance_id, rank)?;
        let mut copies = Vec::new();
        let mut objects = Vec::new();
        let mut bytes = 0;
        for l in loads {
            let keys: Vec<BlockKey> = l.reads.iter().map(|r| r.key.block_key()).collect();
            let leased = self
                .query_leases
                .leased_objects(instance_id, &l.lease, &keys)
                .map_err(EngineError::InvalidArgument)?;
            for ((r, block), key) in l.reads.iter().zip(leased).zip(&keys) {
                let block = block.ok_or_else(|| {
                    EngineError::InvalidArgument(format!("{:?} does not hold {key:?}", l.lease))
                })?;
                for s in &r.reads {
                    let slot = block
                        .slots()
                        .get(s.slot as usize)
                        .and_then(|b| Some((b.segment_mapped_ptr(0)?, b.memory_footprint())));
                    let Some((base, size)) = slot else {
                        return Err(EngineError::InvalidArgument(format!(
                            "{key:?} has {} slots, read asks for slot {}",
                            block.slots().len(),
                            s.slot
                        )));
                    };
                    for run in &s.runs {
                        device.check(run, size)?;
                        copies.push(device.copy_of(base.add(run.slot_offset as usize), run));
                        bytes += run.bytes();
                    }
                }
                objects.push(block);
            }
        }
        device.submit(
            Direction::HostToDevice,
            CopyJob {
                copies,
                keep: Box::new(objects),
                bytes,
                done,
            },
        )
    }
}

/// `w` names valid slots once each, and every slot's runs lie in the device's
/// regions and the slot and copy as many bytes as it has.
fn check_write(device: &Device, w: &ObjectWrite) -> Result<(), EngineError> {
    let bad = |why: String| {
        Err(EngineError::InvalidArgument(format!(
            "object {:?}: {why}",
            w.key
        )))
    };
    if w.slots == 0 || w.slots > MAX_SLOTS {
        return bad(format!("{} slots, expected 1 to {MAX_SLOTS}", w.slots));
    }
    let mut seen = vec![false; w.slots as usize];
    for s in &w.writes {
        if s.slot >= w.slots || std::mem::replace(&mut seen[s.slot as usize], true) {
            return bad(format!("slot {} is out of range or written twice", s.slot));
        }
        if s.size == 0 {
            return bad(format!("slot {} is empty", s.slot));
        }
        let mut copied = 0u64;
        for run in &s.runs {
            device.check(run, s.size)?;
            copied = copied.saturating_add(run.bytes());
        }
        if copied != s.size {
            return bad(format!(
                "slot {} has {} bytes but its runs copy {copied}",
                s.slot, s.size
            ));
        }
    }
    Ok(())
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

    fn dim(count: u32, region_stride: u64, slot_stride: u64) -> ObjectDim {
        ObjectDim {
            count,
            region_stride,
            slot_stride,
        }
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
    fn runs_reject_overlap_and_overflow() {
        let run = |width, dims: &[ObjectDim]| ObjectRun {
            region: 0,
            region_offset: 0,
            slot_offset: 0,
            width,
            dims: dims.to_vec(),
        };
        assert_eq!(run(8, &[dim(3, 16, 8)]).ends(), Some((40, 24)));
        assert_eq!(run(8, &[]).ends(), Some((8, 8)));
        // A single step ignores its strides.
        let one = run(8, &[dim(1, u64::MAX, u64::MAX), dim(2, 8, 8)]);
        assert_eq!(one.ends(), Some((16, 16)));
        // Heads 64 apart in the region, interleaved with rows in the slot.
        assert_eq!(
            run(8, &[dim(4, 8, 16), dim(2, 64, 8)]).ends(),
            Some((96, 64))
        );
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
