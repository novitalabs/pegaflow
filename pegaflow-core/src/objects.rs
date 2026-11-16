//! RPC V2 objects on the block store.
//!
//! An object is keyed by `(namespace, prefix)`, two 32-byte digests. A
//! namespace is one KV kind with all its layers; every device registers the
//! namespaces it holds and its regions once, then saves and loads all layers
//! of an object in one span list of `layers x slice_bytes` bytes. An object is
//! `num_groups x layers` slots of `slice_bytes`; slot `layer x G + group`
//! belongs to group `group`.
//!
//! Each device names, per namespace, the group it loads and whether it saves
//! it; at most one device saves a group. A save copies the device's bytes into
//! one slot per layer, in pinned memory on its NUMA node, and hands them to
//! the insert worker like a KV save; the object seals once every slot arrived.
//! A load reads its group's slots. The engine knows devices only: which bytes
//! a group holds, and which device reads which group, is the client's choice.
//!
//! Sealed objects are [`SealedBlock`]s under the [`BlockKey`]
//! `(hex(namespace), prefix)`, so they share the pinned pool, LRU eviction,
//! the SSD cache, RDMA fetch and MetaServer registration with KV blocks. The
//! engine never interprets a namespace beyond comparing its bytes.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::num::NonZeroU64;
use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{CudaContext, CudaStream, sys};
use log::{debug, error, info, warn};
use parking_lot::Mutex;
use tokio::sync::{mpsc, oneshot};

use crate::backing::SSD_ALIGNMENT;
use crate::block::{BlockKey, PrefetchStatus, RawBlock, SealedBlock, Segment};
use crate::gpu_worker::{LoadCompletion, build_backend, spawn_pinned_worker, wait_worker_ready};
use crate::metrics::core_metrics;
use crate::offload::{RawSaveBatch, RawSaveLayer};
use crate::pinned_pool::{MappedPinnedPtr, PinnedAllocation};
use crate::storage::MembersPoll;
use crate::transfer::{CopyDesc, TransferBackend, TransferMode};
use crate::{EngineError, NumaNode, PegaEngine, QueryLeaseId};

/// `(namespace, prefix)`, both 32-byte digests.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ObjectKey {
    pub namespace: [u8; 32],
    pub prefix: [u8; 32],
}

impl ObjectKey {
    /// Hex namespaces never equal a model namespace a KV connector derives.
    fn block_key(&self) -> BlockKey {
        BlockKey::new(hex(&self.namespace), self.prefix.to_vec())
    }
}

/// Keys log as the first 8 bytes of each digest.
impl fmt::Debug for ObjectKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}/{}",
            hex(&self.namespace[..8]),
            hex(&self.prefix[..8])
        )
    }
}

fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    bytes
        .iter()
        .flat_map(|b| [HEX[usize::from(b >> 4)], HEX[usize::from(b & 15)]])
        .map(char::from)
        .collect()
}

/// One `(offset, len)` pair inside a region, in bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    pub offset: u64,
    pub len: u64,
}

impl Span {
    /// Wire size of one `ObjectSpans.span` element.
    const WIRE_BYTES: usize = 16;

    /// Decode one element: `offset u64 LE` then `len u64 LE`, exactly 16 bytes.
    pub fn decode(raw: &[u8]) -> Result<Self, String> {
        let words: &[u8; Self::WIRE_BYTES] = raw.try_into().map_err(|_| {
            format!(
                "span element is {} bytes, expected {}",
                raw.len(),
                Self::WIRE_BYTES
            )
        })?;
        let (offset, len) = words.split_at(8);
        Ok(Self {
            offset: u64::from_le_bytes(offset.try_into().expect("8-byte half")),
            len: u64::from_le_bytes(len.try_into().expect("8-byte half")),
        })
    }
}

/// Where one object sits in a device's region: spans over every layer of its
/// namespace, in `layer_names` order.
#[derive(Debug, Clone)]
pub struct ObjectSpans {
    pub key: ObjectKey,
    pub region: u32,
    pub spans: Vec<Span>,
}

/// One namespace; every device of an instance registers the same.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NamespaceMeta {
    pub namespace: [u8; 32],
    pub layer_names: Vec<String>,
    /// Bytes of one slot: one group's part of one layer.
    pub slice_bytes: u64,
    pub num_groups: u32,
    /// Loads per object; a lease is released after this many.
    pub world_size: u32,
    pub interleave_token: u64,
}

/// What one device does in one namespace: the group it loads, and whether it
/// saves that group.
#[derive(Debug, Clone)]
pub struct NamespaceRole {
    pub meta: NamespaceMeta,
    pub group: u32,
    pub save: bool,
}

#[derive(Debug)]
pub enum ObjectQueryStatus {
    /// Missing objects are being fetched; ask again with the same query id.
    Loading,
    /// Prefix queries: `hits` leading keys. Membership queries: the keys at
    /// `hit_positions`. The lease holds exactly those; none on 0 hits.
    Ready {
        hits: usize,
        lease: Option<QueryLeaseId>,
        hit_positions: Vec<u32>,
    },
}

/// The slot layout of a namespace: one slot per `(group, layer)`.
#[derive(Debug)]
struct Shape {
    world_size: u32,
    layers: usize,
    slice_bytes: u64,
    groups: u32,
}

impl Shape {
    fn new(meta: &NamespaceMeta) -> Result<Self, String> {
        if meta.world_size == 0 {
            return Err("world_size is 0".into());
        }
        if meta.layer_names.is_empty() {
            return Err("no layers".into());
        }
        if meta.slice_bytes == 0 || meta.num_groups == 0 {
            return Err("slice_bytes and num_groups must be positive".into());
        }
        let layers = meta.layer_names.len();
        (layers as u64)
            .checked_mul(meta.slice_bytes)
            .and_then(|_| layers.checked_mul(meta.num_groups as usize))
            .ok_or("layers x slice_bytes or layers x num_groups overflows")?;
        Ok(Self {
            world_size: meta.world_size,
            layers,
            slice_bytes: meta.slice_bytes,
            groups: meta.num_groups,
        })
    }

    fn total_slots(&self) -> usize {
        self.groups as usize * self.layers
    }

    fn slot(&self, layer: usize, group: u32) -> usize {
        layer * self.groups as usize + group as usize
    }

    /// Bytes one device saves or loads of an object.
    fn object_bytes(&self) -> u64 {
        self.layers as u64 * self.slice_bytes
    }

    /// Pinned base and allocation of `group`'s slot in each layer of `block`.
    fn read_slots(
        &self,
        group: u32,
        block: &SealedBlock,
    ) -> Result<Vec<(MappedPinnedPtr, usize)>, String> {
        if block.slots().len() != self.total_slots() {
            return Err(format!(
                "object has {} slots, its namespace {}",
                block.slots().len(),
                self.total_slots()
            ));
        }
        (0..self.layers)
            .map(|l| {
                let raw = &block.slots()[self.slot(l, group)];
                // Slots may be longer: SSD padding.
                match (
                    raw.segment_mapped_ptr(0),
                    raw.segment_allocation_id(0),
                    raw.segment_size(0),
                ) {
                    (Some(at), Some(alloc), Some(n)) if n as u64 >= self.slice_bytes => {
                        Ok((at, alloc))
                    }
                    _ => Err(format!(
                        "slot of layer {l} is shorter than {}",
                        self.slice_bytes
                    )),
                }
            })
            .collect()
    }
}

/// One span placed in its layer's slot.
#[derive(Debug, PartialEq, Eq)]
struct Piece {
    layer: usize,
    slot_offset: u64,
    region_offset: u64,
    len: u64,
}

/// Place the spans of `o` in their layers: they lie inside a region of
/// `region_size` bytes, none crosses a layer, and together they fill every
/// layer.
fn pieces(
    o: &ObjectSpans,
    region_size: Option<u64>,
    shape: &Shape,
) -> Result<Vec<Piece>, EngineError> {
    let bad = |why: String| EngineError::InvalidArgument(format!("object {:?}: {why}", o.key));
    let Some(size) = region_size else {
        return Err(bad(format!("unknown region {}", o.region)));
    };
    let slice = shape.slice_bytes;
    let mut out = Vec::with_capacity(o.spans.len());
    let mut at = 0u64;
    for s in o.spans.iter().filter(|s| s.len > 0) {
        if !s.offset.checked_add(s.len).is_some_and(|end| end <= size) {
            return Err(bad(format!(
                "span {s:?} is out of region {} of {size} bytes",
                o.region
            )));
        }
        let (layer, slot_offset) = (at / slice, at % slice);
        if layer >= shape.layers as u64 {
            return Err(bad(format!(
                "spans cover more than {} layers x {slice}",
                shape.layers
            )));
        }
        if s.len > slice - slot_offset {
            return Err(bad(format!(
                "span {s:?} crosses layer {layer} at {slot_offset} of {slice}"
            )));
        }
        out.push(Piece {
            layer: layer as usize,
            slot_offset,
            region_offset: s.offset,
            len: s.len,
        });
        at += s.len;
    }
    if at != shape.object_bytes() {
        return Err(bad(format!(
            "spans cover {at} bytes, expected {} layers x {slice}",
            shape.layers
        )));
    }
    Ok(out)
}

/// A device's IPC-opened regions; closed when the device and its last
/// queued copy drop them.
struct Regions {
    ctx: Arc<CudaContext>,
    list: Vec<(sys::CUdeviceptr, u64)>,
}

impl Drop for Regions {
    fn drop(&mut self) {
        if let Err(e) = self.ctx.bind_to_thread() {
            warn!(
                "objects: leaking {} IPC regions, cannot bind context: {e:?}",
                self.list.len()
            );
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Direction {
    Save,
    Load,
}

/// A copy stream of one device and the backend it registered with.
struct Lane {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    backend: Box<dyn TransferBackend>,
}

impl Lane {
    fn new(ctx: &Arc<CudaContext>, mode: TransferMode) -> Result<Self, EngineError> {
        ctx.bind_to_thread()
            .map_err(|e| EngineError::CudaInit(format!("{e:?}")))?;
        let stream = ctx
            .new_stream()
            .map_err(|e| EngineError::CudaInit(format!("{e:?}")))?;
        Ok(Self {
            ctx: Arc::clone(ctx),
            stream,
            backend: build_backend(mode, ctx)?,
        })
    }

    fn transfer(&self, copies: &[CopyDesc], dir: Direction) -> Result<(), EngineError> {
        if copies.is_empty() {
            return Ok(());
        }
        let failed = |e: String| EngineError::Storage(format!("object copy: {e}"));
        self.ctx
            .bind_to_thread()
            .map_err(|e| failed(format!("{e:?}")))?;
        match dir {
            Direction::Save => self.backend.d2h(copies, &self.stream),
            Direction::Load => self.backend.h2d(copies, &self.stream),
        }
        .map_err(failed)?;
        self.stream
            .synchronize()
            .map_err(|e| failed(format!("{e:?}")))
    }
}

struct CopyJob {
    copies: Vec<CopyDesc>,
    /// Pinned memory the copies touch, kept alive until they finish.
    keep: Box<dyn Send>,
    bytes: u64,
    done: LoadCompletion,
}

/// Run a device's copies of one direction in order until the device is
/// dropped. Records load metrics as the KV load worker does; saves record
/// theirs on the RPC path.
fn copy_worker(
    lane: Lane,
    device_id: i32,
    dir: Direction,
    mut jobs: mpsc::UnboundedReceiver<CopyJob>,
) {
    while let Some(job) = jobs.blocking_recv() {
        let start = Instant::now();
        let result = lane.transfer(&job.copies, dir);
        let elapsed = start.elapsed();
        if dir == Direction::Load {
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

/// What a device does in one namespace.
struct Role {
    shape: Arc<Shape>,
    group: u32,
    save: bool,
}

/// One registered device: its roles, its regions and a save and a load
/// worker, each pinned to the GPU's node with its own stream, so a load never
/// queues behind a save.
pub(crate) struct Device {
    roles: HashMap<[u8; 32], Role>,
    regions: Arc<Regions>,
    /// The GPU's NUMA node; saves allocate there.
    numa: NumaNode,
    save: mpsc::UnboundedSender<CopyJob>,
    load: mpsc::UnboundedSender<CopyJob>,
}

impl Device {
    fn role(&self, namespace: &[u8; 32]) -> Result<&Role, EngineError> {
        self.roles.get(namespace).ok_or_else(|| {
            EngineError::InvalidArgument(format!(
                "namespace {} is not registered on this device",
                hex(&namespace[..8])
            ))
        })
    }

    fn region_size(&self, region: u32) -> Option<u64> {
        self.regions.list.get(region as usize).map(|r| r.1)
    }

    /// The copy of `p` of an object in `region` to or from pinned memory at
    /// `slot` of allocation `alloc`.
    fn copy(&self, region: u32, p: &Piece, slot: MappedPinnedPtr, alloc: usize) -> CopyDesc {
        let at = slot.add(p.slot_offset as usize);
        CopyDesc {
            device: self.regions.list[region as usize].0 + p.region_offset,
            host: at.host().as_ptr(),
            host_device: at.device().as_ptr() as u64,
            size: p.len as usize,
            device_allocation: region as usize,
            host_allocation: alloc,
        }
    }

    fn submit(&self, dir: Direction, job: CopyJob) -> Result<(), EngineError> {
        // A queued copy keeps the regions mapped; unregistering an idle
        // device closes them before it returns.
        let job = CopyJob {
            keep: Box::new((job.keep, Arc::clone(&self.regions))),
            ..job
        };
        let worker = match dir {
            Direction::Save => &self.save,
            Direction::Load => &self.load,
        };
        worker
            .send(job)
            .map_err(|_| EngineError::Storage("object copy worker exited".into()))
    }
}

struct Namespace {
    meta: NamespaceMeta,
    shape: Arc<Shape>,
}

/// The namespaces and devices of one instance.
#[derive(Default)]
pub(crate) struct ObjectInstance {
    namespaces: HashMap<[u8; 32], Namespace>,
    devices: HashMap<i32, Arc<Device>>,
    /// The device saving each `(namespace, group)`.
    savers: HashMap<([u8; 32], u32), i32>,
}

impl ObjectInstance {
    fn shape(&self, namespace: &[u8; 32]) -> Result<Arc<Shape>, EngineError> {
        self.namespaces
            .get(namespace)
            .map(|n| Arc::clone(&n.shape))
            .ok_or_else(|| {
                EngineError::InvalidArgument(format!(
                    "namespace {} is not registered",
                    hex(&namespace[..8])
                ))
            })
    }

    /// Device `device_id` with `roles` may join: the device is new, the
    /// namespaces other devices registered match, and no group it saves has
    /// a saver yet.
    fn check_join(
        &self,
        device_id: i32,
        roles: &HashMap<[u8; 32], (Namespace, u32, bool)>,
    ) -> Result<(), EngineError> {
        if self.devices.contains_key(&device_id) {
            return Err(EngineError::InvalidArgument(format!(
                "device {device_id} is already registered"
            )));
        }
        for (ns, (n, group, save)) in roles {
            if self
                .namespaces
                .get(ns)
                .is_some_and(|known| known.meta != n.meta)
            {
                return Err(EngineError::InvalidArgument(format!(
                    "namespace {} differs from what another device registered",
                    hex(&ns[..8])
                )));
            }
            if let Some(d) = self.savers.get(&(*ns, *group)).filter(|_| *save) {
                return Err(EngineError::InvalidArgument(format!(
                    "group {group} of namespace {} already has saver device {d}",
                    hex(&ns[..8])
                )));
            }
        }
        Ok(())
    }
}

/// Registered RPC V2 instances; objects live in the block store.
#[derive(Default)]
pub(crate) struct ObjectStore {
    instances: Mutex<HashMap<String, ObjectInstance>>,
}

impl ObjectStore {
    pub(crate) fn has_instances(&self) -> bool {
        !self.instances.lock().is_empty()
    }

    /// Take `instance` out; dropping it closes its regions once no copy
    /// holds them, so callers drop it outside locks.
    pub(crate) fn remove_instance(&self, instance: &str) -> Option<ObjectInstance> {
        self.instances.lock().remove(instance)
    }

    /// Take every instance out; as [`Self::remove_instance`].
    pub(crate) fn remove_all(&self) -> HashMap<String, ObjectInstance> {
        std::mem::take(&mut *self.instances.lock())
    }

    /// Device `device_id` of `instance`.
    fn device(&self, instance: &str, device_id: i32) -> Result<Arc<Device>, EngineError> {
        let all = self.instances.lock();
        let inst = all
            .get(instance)
            .ok_or_else(|| EngineError::InstanceMissing(instance.to_string()))?;
        inst.devices
            .get(&device_id)
            .cloned()
            .ok_or_else(|| EngineError::InstanceMissing(format!("{instance} (device {device_id})")))
    }

    fn shape(&self, instance: &str, namespace: &[u8; 32]) -> Result<Arc<Shape>, EngineError> {
        self.instances
            .lock()
            .get(instance)
            .ok_or_else(|| EngineError::InstanceMissing(instance.to_string()))?
            .shape(namespace)
    }
}

impl PegaEngine {
    /// Register device `device_id` of `instance_id` with its `roles` and
    /// `regions` (IPC handle, size). A device registers once until its
    /// instance is unregistered or its session ends.
    pub fn register_context_v2(
        &self,
        instance_id: &str,
        device_id: i32,
        roles: Vec<NamespaceRole>,
        regions: &[(Vec<u8>, u64)],
        transfer_mode: TransferMode,
    ) -> Result<(), EngineError> {
        let mut table = HashMap::with_capacity(roles.len());
        for NamespaceRole { meta, group, save } in roles {
            let ns = meta.namespace;
            let bad = |e: String| {
                EngineError::InvalidArgument(format!("namespace {}: {e}", hex(&ns[..8])))
            };
            let shape = Shape::new(&meta).map_err(bad)?;
            if group >= shape.groups {
                return Err(bad(format!(
                    "group {group} is outside num_groups {}",
                    shape.groups
                )));
            }
            let n = Namespace {
                meta,
                shape: Arc::new(shape),
            };
            if table.insert(ns, (n, group, save)).is_some() {
                return Err(bad("listed twice".into()));
            }
        }
        // Fail before opening IPC handles; checked again on insert.
        if let Some(inst) = self.objects.instances.lock().get(instance_id) {
            inst.check_join(device_id, &table)?;
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
            let ctx = Arc::clone(&ctx);
            let ready = spawn_pinned_worker(
                format!("gpu{device_id}-obj{name}"),
                numa,
                move || Lane::new(&ctx, transfer_mode),
                move |lane| copy_worker(lane, device_id, dir, rx),
            )?;
            wait_worker_ready(&format!("object {name}"), device_id, ready)?;
            Ok::<_, EngineError>(tx)
        };
        let save = spawn(Direction::Save, "save")?;
        let load = spawn(Direction::Load, "load")?;
        let region_count = regions.list.len();
        let namespace_count = table.len();
        let roles = table
            .iter()
            .map(|(ns, (n, group, save))| {
                let role = Role {
                    shape: Arc::clone(&n.shape),
                    group: *group,
                    save: *save,
                };
                (*ns, role)
            })
            .collect();
        let device = Arc::new(Device {
            roles,
            regions,
            numa,
            save,
            load,
        });
        {
            // Under the instances lock, as unregistering checks for object
            // instances and clears `has_instance` there.
            let _instances = self
                .instances
                .write()
                .expect("instances write lock poisoned");
            let mut all = self.objects.instances.lock();
            let inst = all.entry(instance_id.to_string()).or_default();
            inst.check_join(device_id, &table)?;
            inst.devices.insert(device_id, device);
            for (ns, (n, group, save)) in table {
                if save {
                    inst.savers.insert((ns, group), device_id);
                }
                inst.namespaces.entry(ns).or_insert(n);
            }
            self.storage.set_has_instance(true);
        }
        info!(
            "Registered context v2: instance={instance_id} device={device_id} \
             namespaces={namespace_count} regions={region_count} numa={numa} mode={transfer_mode:?}"
        );
        Ok(())
    }

    /// Answer a query over `keys` of one namespace of `instance_id`.
    ///
    /// With `prefix_search` it is v1 `QueryPrefetch` without
    /// `wait_for_full_prefix`: the leading keys in memory, or `Loading` while
    /// SSD and peers bring more in. Otherwise every key stands on its own:
    /// keys not in memory are fetched from peers and SSD, and the answer is
    /// `Loading` until that ends. Poll with the same `query_id` and keys; a
    /// `None` id gets a fresh one. Returns the id and the answer.
    pub async fn query_object(
        &self,
        instance_id: &str,
        query_id: Option<&str>,
        keys: &[ObjectKey],
        prefix_search: bool,
    ) -> Result<(String, ObjectQueryStatus), EngineError> {
        let query_id = query_id.map_or_else(|| uuid::Uuid::new_v4().to_string(), str::to_string);
        let Some(first) = keys.first() else {
            if !self.objects.instances.lock().contains_key(instance_id) {
                return Err(EngineError::InstanceMissing(instance_id.to_string()));
            }
            let empty = ObjectQueryStatus::Ready {
                hits: 0,
                lease: None,
                hit_positions: Vec::new(),
            };
            return Ok((query_id, empty));
        };
        if keys.iter().any(|k| k.namespace != first.namespace) {
            return Err(EngineError::InvalidArgument(
                "query keys span more than one namespace".into(),
            ));
        }
        let shape = self.objects.shape(instance_id, &first.namespace)?;
        let consumers = shape.world_size as usize;
        let tag = format!("objects:{instance_id}:{query_id}");
        let block_keys: Vec<BlockKey> = keys.iter().map(ObjectKey::block_key).collect();
        let status = if prefix_search {
            self.query_object_prefix(instance_id, &tag, &block_keys, consumers)
                .await
        } else {
            self.query_object_members(instance_id, &tag, &block_keys, consumers)
                .await
        };
        Ok((query_id, status))
    }

    async fn query_object_prefix(
        &self,
        instance_id: &str,
        tag: &str,
        keys: &[BlockKey],
        consumers: usize,
    ) -> ObjectQueryStatus {
        let hashes: Vec<Vec<u8>> = keys.iter().map(|k| k.hash.clone()).collect();
        let status = self
            .storage
            .check_prefix_and_prefetch(tag, &keys[0].namespace, &hashes, false)
            .await;
        let PrefetchStatus::Ready { blocks, missing } = status else {
            return ObjectQueryStatus::Loading;
        };
        let metrics = core_metrics();
        metrics.cache_block_hits.add(blocks.len() as u64, &[]);
        metrics.cache_block_misses.add(missing as u64, &[]);
        let hits = blocks.len();
        let held: HashMap<BlockKey, Arc<SealedBlock>> = keys.iter().cloned().zip(blocks).collect();
        ObjectQueryStatus::Ready {
            hits,
            lease: (!held.is_empty()).then(|| {
                self.query_leases
                    .create_objects(instance_id, held, consumers)
            }),
            hit_positions: Vec::new(),
        }
    }

    async fn query_object_members(
        &self,
        instance_id: &str,
        tag: &str,
        keys: &[BlockKey],
        consumers: usize,
    ) -> ObjectQueryStatus {
        // A finished fetch answers the query even when it found nothing:
        // starting another one for the same keys would loop forever.
        let (polled, fetched): (bool, HashMap<BlockKey, Arc<SealedBlock>>) =
            match self.storage.poll_members(tag).await {
                MembersPoll::Loading => return ObjectQueryStatus::Loading,
                MembersPoll::Done(fetched) => (true, fetched.into_iter().collect()),
                MembersPoll::Idle => (false, HashMap::new()),
            };
        // Touch what is resident; fetched blocks evicted since are still held here.
        let found: Vec<Option<Arc<SealedBlock>>> = self
            .storage
            .get_blocks_aligned(keys)
            .into_iter()
            .zip(keys)
            .map(|(b, k)| b.or_else(|| fetched.get(k).cloned()))
            .collect();
        if !polled {
            let mut seen = HashSet::new();
            let missing: Vec<BlockKey> = keys
                .iter()
                .zip(&found)
                .filter(|(k, b)| b.is_none() && seen.insert(*k))
                .map(|(k, _)| k.clone())
                .collect();
            if self.storage.start_members(tag, missing) {
                return ObjectQueryStatus::Loading;
            }
        }
        let mut held = HashMap::new();
        let mut hit_positions = Vec::new();
        for (i, (k, b)) in keys.iter().zip(found).enumerate() {
            if let Some(b) = b {
                hit_positions.push(i as u32);
                held.insert(k.clone(), b);
            }
        }
        let metrics = core_metrics();
        metrics
            .cache_block_hits
            .add(hit_positions.len() as u64, &[]);
        metrics
            .cache_block_misses
            .add((keys.len() - hit_positions.len()) as u64, &[]);
        ObjectQueryStatus::Ready {
            hits: hit_positions.len(),
            lease: (!held.is_empty()).then(|| {
                self.query_leases
                    .create_objects(instance_id, held, consumers)
            }),
            hit_positions,
        }
    }

    /// Copy `writes` from device `device_id` of `instance_id` into pinned memory on
    /// the device's NUMA node and queue them for the insert worker, as a KV
    /// save does; returns once the copies finish. Writes to objects already
    /// cached copy nothing. An object seals once every group's saver wrote
    /// every layer; unsealed objects expire with KV inflight blocks.
    pub async fn save_objects(
        &self,
        instance_id: &str,
        device_id: i32,
        writes: &[ObjectSpans],
    ) -> Result<(), EngineError> {
        let start = Instant::now();
        let device = self.objects.device(instance_id, device_id)?;
        let mut roles = Vec::with_capacity(writes.len());
        let mut placed = Vec::with_capacity(writes.len());
        for w in writes {
            let role = device.role(&w.key.namespace)?;
            if !role.save {
                return Err(EngineError::InvalidArgument(format!(
                    "device {device_id} does not save namespace {}",
                    hex(&w.key.namespace[..8])
                )));
            }
            placed.push(pieces(w, device.region_size(w.region), &role.shape)?);
            roles.push(role);
        }
        let shapes: Vec<&Shape> = roles.iter().map(|r| &*r.shape).collect();
        let keys: Vec<BlockKey> = writes.iter().map(|w| w.key.block_key()).collect();
        let todo: Vec<usize> = self
            .storage
            .peek_blocks(&keys)
            .iter()
            .enumerate()
            .filter_map(|(i, b)| b.is_none().then_some(i))
            .collect();
        if todo.is_empty() {
            return Ok(());
        }

        // Like KV saves: SSD-aligned slots when SSD is on, and one allocation
        // per (namespace, layer) across this save's objects unless blockwise.
        let pad = self.storage.is_ssd_enabled();
        let stride = |n: u64| {
            if pad {
                n.next_multiple_of(SSD_ALIGNMENT as u64)
            } else {
                n
            }
        };
        let mut by_layer: BTreeMap<([u8; 32], usize), Vec<usize>> = BTreeMap::new();
        for &i in &todo {
            for l in 0..shapes[i].layers {
                by_layer
                    .entry((writes[i].key.namespace, l))
                    .or_default()
                    .push(i);
            }
        }
        let numa = Some(device.numa);
        let blockwise = self.storage.blockwise_alloc();
        let mut keep: Vec<Arc<PinnedAllocation>> = Vec::new();
        // Pinned base and allocation of each (write, layer) slot.
        let mut slots: HashMap<(usize, usize), (MappedPinnedPtr, usize)> = HashMap::new();
        type Layers = BTreeMap<usize, RawSaveLayer>;
        let mut batches: BTreeMap<[u8; 32], (usize, Layers)> = BTreeMap::new();
        let mut padded = 0u64;
        for ((ns, layer), list) in &by_layer {
            let len = shapes[list[0]].slice_bytes;
            let slot_len = stride(len);
            let groups: Vec<&[usize]> = if blockwise {
                list.chunks(1).collect()
            } else {
                vec![&list[..]]
            };
            for group in groups {
                let total = slot_len * group.len() as u64;
                let alloc = NonZeroU64::new(total)
                    .and_then(|n| self.storage.allocate(n, numa))
                    .ok_or_else(|| {
                        EngineError::Storage(format!(
                            "pinned pool exhausted while allocating {total} bytes of object layer {layer}"
                        ))
                    })?;
                for (j, &i) in group.iter().enumerate() {
                    let base = alloc.mapped_ptr().add(j * slot_len as usize);
                    if slot_len > len {
                        // SAFETY: the allocation holds `slot_len` bytes at
                        // `base`, and nothing else uses them yet. Zero the SSD
                        // padding so loads and SSD writes never see a previous
                        // user's bytes.
                        unsafe {
                            std::ptr::write_bytes(
                                base.host().as_ptr().add(len as usize),
                                0,
                                (slot_len - len) as usize,
                            );
                        }
                    }
                    slots.insert((i, *layer), (base, Arc::as_ptr(&alloc) as usize));
                    let shape = &shapes[i];
                    let saved = batches
                        .entry(*ns)
                        .or_insert_with(|| (shape.total_slots(), Layers::new()))
                        .1
                        .entry(*layer)
                        .or_insert_with(|| RawSaveLayer {
                            slot_id: shape.slot(*layer, roles[i].group),
                            padded_block_size: slot_len as usize,
                            blocks: Vec::new(),
                            block_hashes: Vec::new(),
                        });
                    saved.blocks.push(RawBlock::single_segment(Segment::new(
                        base.host(),
                        slot_len as usize,
                        Arc::clone(&alloc),
                    )));
                    saved.block_hashes.push(keys[i].hash.clone());
                    padded += slot_len;
                }
                keep.push(alloc);
            }
        }
        let mut copies = Vec::new();
        let mut bytes = 0u64;
        for &i in &todo {
            let w = &writes[i];
            for p in &placed[i] {
                let (base, alloc) = slots[&(i, p.layer)];
                copies.push(device.copy(w.region, p, base, alloc));
            }
            bytes += shapes[i].object_bytes();
        }
        let prep = start.elapsed();
        let (reply, done) = oneshot::channel();
        device.submit(
            Direction::Save,
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
            "save_objects completed: instance_id={instance_id} device={device_id} objects={} \
             bytes={padded} numa={} prep_ms={:.2} total_ms={:.2}",
            todo.len(),
            device.numa,
            prep.as_secs_f64() * 1000.0,
            start.elapsed().as_secs_f64() * 1000.0,
        );
        for (ns, (total_slots, layers)) in batches {
            self.storage.send_raw_insert(RawSaveBatch {
                namespace: hex(&ns),
                total_slots,
                numa_node: device.numa,
                layers: layers.into_values().collect(),
            });
        }
        Ok(())
    }

    /// Copy the objects `spans` name, held by `leases`, into device
    /// `device_id` of `instance_id`, as a KV load: returns once the request is
    /// checked and sets the `LoadState` at `load_state_shm` when the copies
    /// finish, or right away if checking fails. A checked request consumes
    /// each of its leases once; one that fails consumes none.
    pub fn load_objects(
        &self,
        instance_id: &str,
        device_id: i32,
        spans: &[ObjectSpans],
        leases: &[QueryLeaseId],
        load_state_shm: &str,
    ) -> Result<(), EngineError> {
        crate::submit_with_load_state(load_state_shm, "load_objects", |done| {
            self.submit_object_loads(instance_id, device_id, spans, leases, done)
        })
    }

    fn submit_object_loads(
        &self,
        instance_id: &str,
        device_id: i32,
        spans: &[ObjectSpans],
        leases: &[QueryLeaseId],
        done: LoadCompletion,
    ) -> Result<(), EngineError> {
        let device = self.objects.device(instance_id, device_id)?;
        let mut placed = Vec::with_capacity(spans.len());
        for s in spans {
            let role = device.role(&s.key.namespace)?;
            placed.push((pieces(s, device.region_size(s.region), &role.shape)?, role));
        }
        let held = self
            .query_leases
            .leased_objects(instance_id, leases)
            .map_err(EngineError::InvalidArgument)?;
        let mut copies = Vec::new();
        let mut objects = Vec::with_capacity(spans.len());
        let mut bytes = 0;
        for (s, (list, role)) in spans.iter().zip(&placed) {
            let block = held.get(&s.key.block_key()).ok_or_else(|| {
                EngineError::InvalidArgument(format!("no lease of this load holds {:?}", s.key))
            })?;
            let slots = role
                .shape
                .read_slots(role.group, block)
                .map_err(|e| EngineError::InvalidArgument(format!("object {:?}: {e}", s.key)))?;
            for p in list {
                let (base, alloc) = slots[p.layer];
                copies.push(device.copy(s.region, p, base, alloc));
            }
            bytes += role.shape.object_bytes();
            objects.push(Arc::clone(block));
        }
        device.submit(
            Direction::Load,
            CopyJob {
                copies,
                keep: Box::new(objects),
                bytes,
                done,
            },
        )?;
        // The copy job holds the objects; the leases may go now.
        for lease in leases {
            self.query_leases.consume_objects(lease);
        }
        Ok(())
    }
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

    fn meta(slice_bytes: u64, num_groups: u32) -> NamespaceMeta {
        NamespaceMeta {
            namespace: [7; 32],
            layer_names: vec!["l0".into(), "l1".into()],
            slice_bytes,
            num_groups,
            world_size: 4,
            interleave_token: 0,
        }
    }

    #[test]
    fn shapes_place_groups_per_layer() {
        let gqa = Shape::new(&meta(16384, 4)).unwrap();
        assert_eq!(
            (gqa.total_slots(), gqa.slot(1, 2), gqa.object_bytes()),
            (8, 6, 32768)
        );
        for (case, m) in [
            ("zero slice", meta(0, 4)),
            ("zero groups", meta(16384, 0)),
            (
                "no layers",
                NamespaceMeta {
                    layer_names: vec![],
                    ..meta(16384, 4)
                },
            ),
            (
                "zero world",
                NamespaceMeta {
                    world_size: 0,
                    ..meta(16384, 4)
                },
            ),
        ] {
            assert!(Shape::new(&m).is_err(), "{case}");
        }
    }

    #[test]
    fn spans_fill_layers_without_crossing() {
        let shape = Shape::new(&meta(16, 4)).unwrap();
        let span = |offset, len| Span { offset, len };
        let obj = |spans: Vec<Span>| ObjectSpans {
            key: ObjectKey {
                namespace: [7; 32],
                prefix: [1; 32],
            },
            region: 0,
            spans,
        };
        let piece = |layer, slot_offset, region_offset, len| Piece {
            layer,
            slot_offset,
            region_offset,
            len,
        };
        // Layer-first K then V: two spans per layer of 8 bytes each.
        assert_eq!(
            pieces(
                &obj(vec![span(0, 8), span(100, 8), span(16, 8), span(116, 8)]),
                Some(200),
                &shape
            )
            .unwrap(),
            [
                piece(0, 0, 0, 8),
                piece(0, 8, 100, 8),
                piece(1, 0, 16, 8),
                piece(1, 8, 116, 8)
            ]
        );
        assert!(pieces(&obj(vec![span(0, 16), span(16, 16)]), Some(32), &shape).is_ok());
        for (case, o, size) in [
            ("unknown region", obj(vec![span(0, 16), span(16, 16)]), None),
            (
                "past the region",
                obj(vec![span(1, 16), span(17, 16)]),
                Some(32),
            ),
            (
                "overflow",
                obj(vec![span(u64::MAX, 2), span(0, 14), span(0, 16)]),
                Some(32),
            ),
            ("crosses a layer", obj(vec![span(0, 32)]), Some(32)),
            (
                "crosses mid-layer",
                obj(vec![span(0, 12), span(12, 20)]),
                Some(32),
            ),
            ("short", obj(vec![span(0, 16), span(16, 15)]), Some(32)),
            (
                "long",
                obj(vec![span(0, 16), span(16, 16), span(0, 1)]),
                Some(32),
            ),
        ] {
            assert!(pieces(&o, size, &shape).is_err(), "{case}");
        }
    }

    #[test]
    fn span_elements_are_16_bytes() {
        let mut raw = 5u64.to_le_bytes().to_vec();
        raw.extend(9u64.to_le_bytes());
        assert_eq!(Span::decode(&raw), Ok(Span { offset: 5, len: 9 }));
        assert!(Span::decode(&raw[..15]).is_err());
        raw.push(0);
        assert!(Span::decode(&raw).is_err());
    }
}
