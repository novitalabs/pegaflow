use std::{
    collections::HashMap,
    io,
    mem::MaybeUninit,
    ptr::{NonNull, null_mut},
    sync::{
        Arc, Weak,
        atomic::{AtomicU64, Ordering},
        mpsc as std_mpsc,
    },
    thread,
    time::{Duration, Instant},
};

use log::{debug, info, warn};
use parking_lot::{Condvar, Mutex};
use rdma_mummy_sys::{
    ibv_ah, ibv_ah_attr, ibv_create_ah, ibv_destroy_ah, ibv_global_route, ibv_modify_qp,
    ibv_modify_qp as raw_ibv_modify_qp, ibv_port_attr, ibv_post_send, ibv_qp_attr,
    ibv_qp_attr_mask, ibv_qp_state, ibv_query_port, ibv_send_flags, ibv_send_wr, ibv_sge,
    ibv_wr_opcode,
};
use sideway::ibverbs::{
    AccessFlags,
    address::{AddressHandleAttribute, Gid},
    completion::{
        GenericCompletionQueue, PollCompletionQueueError, WorkCompletionOperationType,
        WorkCompletionStatus,
    },
    device::{DeviceInfo, DeviceList},
    device_context::{DeviceContext, LinkLayer, Mtu, PortState},
    memory_region::MemoryRegion,
    protection_domain::ProtectionDomain,
    queue_pair::{
        GenericQueuePair, QueuePair, QueuePairAttribute, QueuePairState, QueuePairType,
        SendOperationFlags, SetScatterGatherEntry,
    },
};

use crate::{
    api::WorkerConfig,
    control_protocol::{ControlMessage, RcEndpoint, decode_message, encode_message},
    domain_address::DomainAddress,
    error::{Result, TransferError},
    logging,
};

const UD_QKEY: u32 = 0x1111_1111;
const UD_RECV_SLOTS: usize = 64;
const UD_BUFFER_BYTES: usize = 512;
const UD_GRH_BYTES: usize = 40;
const CONTROL_TIMEOUT: Duration = Duration::from_secs(3);
const MAX_INFLIGHT_WRITES: usize = 96;
const MAX_WR_CHAIN_WRITES: usize = 4;
const DATA_CQ_POLL_SLEEP: Duration = Duration::from_micros(50);

#[derive(Default)]
struct ControlPlane {
    next_request_id: AtomicU64,
    pending_replies: Mutex<HashMap<u64, Option<ControlMessage>>>,
    reply_cv: Condvar,
}

impl ControlPlane {
    fn begin_request(&self) -> u64 {
        let request_id = self.next_request_id.fetch_add(1, Ordering::Relaxed) + 1;
        self.pending_replies.lock().insert(request_id, None);
        request_id
    }

    fn complete_request(&self, request_id: u64, timeout: Duration) -> Option<ControlMessage> {
        let deadline = Instant::now() + timeout;
        let mut guard = self.pending_replies.lock();
        loop {
            if let Some(slot) = guard.get_mut(&request_id) {
                if let Some(message) = slot.take() {
                    guard.remove(&request_id);
                    return Some(message);
                }
            } else {
                return None;
            }

            let now = Instant::now();
            if now >= deadline {
                guard.remove(&request_id);
                return None;
            }
            self.reply_cv.wait_for(&mut guard, deadline - now);
        }
    }

    fn deliver_reply(&self, message: ControlMessage) {
        let request_id = message.request_id();
        let mut guard = self.pending_replies.lock();
        if let Some(slot) = guard.get_mut(&request_id) {
            *slot = Some(message);
            self.reply_cv.notify_all();
        }
    }
}

struct AddressHandle {
    ah: NonNull<ibv_ah>,
}

impl Drop for AddressHandle {
    fn drop(&mut self) {
        unsafe {
            ibv_destroy_ah(self.ah.as_ptr());
        }
    }
}

unsafe impl Send for AddressHandle {}
unsafe impl Sync for AddressHandle {}

struct UdRecvSlot {
    bytes: Box<[u8]>,
    mr: Arc<MemoryRegion>,
}

struct UdSendSlot {
    bytes: Box<[u8]>,
    mr: Arc<MemoryRegion>,
    next_wr_id: u64,
}

struct SidewayRuntime {
    _device_ctx: Arc<DeviceContext>,
    pd: Arc<ProtectionDomain>,
    port_num: u8,
    gid_index: u8,
    link_layer: LinkLayer,
    mtu: Mtu,
    local_gid: Gid,
    local_lid: u16,
    local_ud: DomainAddress,
    ud_qp: Arc<Mutex<GenericQueuePair>>,
    ud_cq: GenericCompletionQueue,
    recv_slots: Vec<UdRecvSlot>,
    send_slot: Mutex<UdSendSlot>,
    ah_cache: Mutex<HashMap<DomainAddress, Arc<AddressHandle>>>,
    control: Arc<ControlPlane>,
}

struct RegisteredMemoryEntry {
    base_ptr: u64,
    len: usize,
    mr: Arc<MemoryRegion>,
}

#[derive(Clone, Copy)]
struct RemoteMemoryEntry {
    base_ptr: u64,
    end_ptr: u64,
    rkey: u32,
}

struct ActiveSession {
    qp: Mutex<GenericQueuePair>,
    send_cq: GenericCompletionQueue,
    _recv_cq: GenericCompletionQueue,
    local_rc: RcEndpoint,
    cmd_tx: std_mpsc::Sender<SessionCommand>,
}

struct WriteOp {
    local_mr: Arc<MemoryRegion>,
    local_ptr: u64,
    remote_ptr: u64,
    len: usize,
    remote_rkey: u32,
}

enum SessionCommand {
    SubmitBatch {
        ops: Vec<WriteOp>,
        done_tx: std_mpsc::Sender<Result<usize>>,
    },
}

#[derive(Default)]
struct SidewayState {
    config: Option<WorkerConfig>,
    runtime: Option<Arc<SidewayRuntime>>,
    registered: HashMap<u64, RegisteredMemoryEntry>,
    sessions: HashMap<DomainAddress, Arc<ActiveSession>>,
    remote_memory_cache: HashMap<DomainAddress, Vec<RemoteMemoryEntry>>,
}

#[derive(Default)]
pub(crate) struct SidewayBackend {
    state: Arc<Mutex<SidewayState>>,
}

impl SidewayBackend {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    fn ensure_initialized(state: &SidewayState) -> Result<&WorkerConfig> {
        state.config.as_ref().ok_or(TransferError::NotInitialized)
    }

    fn find_registered_entry(
        state: &SidewayState,
        ptr: u64,
        len: usize,
    ) -> Option<(u64, usize, Arc<MemoryRegion>)> {
        let end = ptr.checked_add(len as u64)?;
        state.registered.values().find_map(|entry| {
            let entry_end = entry.base_ptr.checked_add(entry.len as u64)?;
            if ptr >= entry.base_ptr && end <= entry_end {
                Some((entry.base_ptr, entry.len, Arc::clone(&entry.mr)))
            } else {
                None
            }
        })
    }

    fn find_cached_remote_memory(
        state: &SidewayState,
        peer_ud: &DomainAddress,
        remote_ptr: u64,
        len: usize,
    ) -> Option<(u32, usize)> {
        let end = remote_ptr.checked_add(len as u64)?;
        let entries = state.remote_memory_cache.get(peer_ud)?;
        for entry in entries.iter() {
            if remote_ptr >= entry.base_ptr && end <= entry.end_ptr {
                let available = usize::try_from(entry.end_ptr - remote_ptr).ok()?;
                return Some((entry.rkey, available));
            }
        }
        None
    }

    fn cache_remote_memory(
        state: &mut SidewayState,
        peer_ud: &DomainAddress,
        remote_ptr: u64,
        available_len: usize,
        rkey: u32,
    ) {
        let Ok(available_len_u64) = u64::try_from(available_len) else {
            return;
        };
        let Some(end_ptr) = remote_ptr.checked_add(available_len_u64) else {
            return;
        };
        if end_ptr <= remote_ptr {
            return;
        }

        let entries = state
            .remote_memory_cache
            .entry(peer_ud.clone())
            .or_default();
        if entries.iter().any(|entry| {
            entry.rkey == rkey && remote_ptr >= entry.base_ptr && end_ptr <= entry.end_ptr
        }) {
            return;
        }

        entries.push(RemoteMemoryEntry {
            base_ptr: remote_ptr,
            end_ptr,
            rkey,
        });
    }

    fn choose_port_and_gid(
        device_ctx: &Arc<DeviceContext>,
    ) -> Result<(u8, u8, LinkLayer, Mtu, Gid, u16)> {
        let dev_attr = device_ctx
            .query_device()
            .map_err(|error| TransferError::Backend(error.to_string()))?;
        let gid_entries = device_ctx
            .query_gid_table()
            .map_err(|error| TransferError::Backend(error.to_string()))?;

        for port_num in 1..=dev_attr.phys_port_cnt() {
            let port_attr = device_ctx
                .query_port(port_num)
                .map_err(|error| TransferError::Backend(error.to_string()))?;
            if port_attr.port_state() != PortState::Active {
                continue;
            }

            let mut picked: Option<(u8, Gid)> = None;
            for entry in gid_entries
                .iter()
                .filter(|entry| entry.port_num() == port_num as u32)
            {
                let gid = entry.gid();
                if gid.is_zero() {
                    continue;
                }
                if !gid.is_unicast_link_local() {
                    picked = Some((entry.gid_index() as u8, gid));
                    break;
                }
                if picked.is_none() {
                    picked = Some((entry.gid_index() as u8, gid));
                }
            }

            let (gid_index, gid) = if let Some(picked) = picked {
                picked
            } else {
                let gid = device_ctx
                    .query_gid(port_num, 0)
                    .map_err(|error| TransferError::Backend(error.to_string()))?;
                (0, gid)
            };

            let mut raw_port = unsafe { MaybeUninit::<ibv_port_attr>::zeroed().assume_init() };
            let query_port_ret = unsafe {
                ibv_query_port(device_ctx.context().as_ptr(), port_num, &raw mut raw_port)
            };
            if query_port_ret != 0 {
                return Err(TransferError::Backend(format!(
                    "query raw port attr failed: {}",
                    io::Error::from_raw_os_error(query_port_ret)
                )));
            }

            return Ok((
                port_num,
                gid_index,
                port_attr.link_layer(),
                port_attr.active_mtu(),
                gid,
                raw_port.lid,
            ));
        }

        warn!(
            "no active port found on NIC {}; cannot initialize transfer runtime",
            device_ctx.name()
        );
        Err(TransferError::Backend(
            "no active port found on selected NIC".to_string(),
        ))
    }

    fn setup_ud_qp(qp: &mut GenericQueuePair, port_num: u8) -> Result<()> {
        let qp_ptr = unsafe { qp.qp().as_ptr() };

        let mut init_attr = unsafe { MaybeUninit::<ibv_qp_attr>::zeroed().assume_init() };
        init_attr.qp_state = ibv_qp_state::IBV_QPS_INIT;
        init_attr.pkey_index = 0;
        init_attr.port_num = port_num;
        init_attr.qkey = UD_QKEY;
        let init_mask = (ibv_qp_attr_mask::IBV_QP_STATE.0
            | ibv_qp_attr_mask::IBV_QP_PKEY_INDEX.0
            | ibv_qp_attr_mask::IBV_QP_PORT.0
            | ibv_qp_attr_mask::IBV_QP_QKEY.0) as i32;
        let ret = unsafe { ibv_modify_qp(qp_ptr, &raw mut init_attr, init_mask) };
        if ret != 0 {
            return Err(TransferError::Backend(format!(
                "UD QP RESET->INIT failed: {}",
                io::Error::from_raw_os_error(ret)
            )));
        }

        let mut rtr_attr = unsafe { MaybeUninit::<ibv_qp_attr>::zeroed().assume_init() };
        rtr_attr.qp_state = ibv_qp_state::IBV_QPS_RTR;
        let ret = unsafe {
            raw_ibv_modify_qp(
                qp_ptr,
                &raw mut rtr_attr,
                ibv_qp_attr_mask::IBV_QP_STATE.0 as i32,
            )
        };
        if ret != 0 {
            return Err(TransferError::Backend(format!(
                "UD QP INIT->RTR failed: {}",
                io::Error::from_raw_os_error(ret)
            )));
        }

        let mut rts_attr = unsafe { MaybeUninit::<ibv_qp_attr>::zeroed().assume_init() };
        rts_attr.qp_state = ibv_qp_state::IBV_QPS_RTS;
        rts_attr.sq_psn = 0;
        let rts_mask =
            (ibv_qp_attr_mask::IBV_QP_STATE.0 | ibv_qp_attr_mask::IBV_QP_SQ_PSN.0) as i32;
        let ret = unsafe { raw_ibv_modify_qp(qp_ptr, &raw mut rts_attr, rts_mask) };
        if ret != 0 {
            return Err(TransferError::Backend(format!(
                "UD QP RTR->RTS failed: {}",
                io::Error::from_raw_os_error(ret)
            )));
        }
        Ok(())
    }

    fn create_runtime(
        config: &WorkerConfig,
        state: Arc<Mutex<SidewayState>>,
    ) -> Result<Arc<SidewayRuntime>> {
        info!(
            "transfer runtime create start: nic={}, rpc_port={}",
            config.nic_name, config.rpc_port
        );
        let device_list =
            DeviceList::new().map_err(|error| TransferError::Backend(error.to_string()))?;
        let device = device_list
            .iter()
            .find(|device| device.name() == config.nic_name)
            .ok_or_else(|| TransferError::DeviceNotFound(config.nic_name.clone()))?;

        let device_ctx = device
            .open()
            .map_err(|error| TransferError::Backend(error.to_string()))?;
        let pd = device_ctx
            .alloc_pd()
            .map_err(|error| TransferError::Backend(error.to_string()))?;

        let (port_num, gid_index, link_layer, mtu, local_gid, local_lid) =
            Self::choose_port_and_gid(&device_ctx)?;
        info!(
            "transfer runtime selected port: nic={}, port={}, gid_index={}, link_layer={:?}, mtu={:?}",
            config.nic_name, port_num, gid_index, link_layer, mtu
        );

        let mut cq_builder = device_ctx.create_cq_builder();
        cq_builder.setup_cqe(256);
        let ud_cq: GenericCompletionQueue = cq_builder
            .build()
            .map_err(|error| TransferError::Backend(error.to_string()))?
            .into();

        let mut qp_builder = pd.create_qp_builder();
        qp_builder
            .setup_qp_type(QueuePairType::UnreliableDatagram)
            .setup_send_ops_flags(SendOperationFlags::Send)
            .setup_send_cq(ud_cq.clone())
            .setup_recv_cq(ud_cq.clone())
            .setup_max_send_wr(256)
            .setup_max_recv_wr(256)
            .setup_max_send_sge(1)
            .setup_max_recv_sge(1);
        let mut ud_qp: GenericQueuePair = qp_builder
            .build_ex()
            .map_err(|error| TransferError::Backend(error.to_string()))?
            .into();
        Self::setup_ud_qp(&mut ud_qp, port_num)?;

        let local_ud =
            DomainAddress::from_parts(local_gid.raw, local_lid, ud_qp.qp_number(), UD_QKEY);

        let mut recv_slots = Vec::with_capacity(UD_RECV_SLOTS);
        for _ in 0..UD_RECV_SLOTS {
            let bytes = vec![0_u8; UD_GRH_BYTES + UD_BUFFER_BYTES].into_boxed_slice();
            let mr = unsafe {
                pd.reg_mr(
                    bytes.as_ptr() as usize,
                    bytes.len(),
                    AccessFlags::LocalWrite,
                )
            }
            .map_err(|error| TransferError::Backend(error.to_string()))?;
            recv_slots.push(UdRecvSlot { bytes, mr });
        }

        let send_bytes = vec![0_u8; UD_BUFFER_BYTES].into_boxed_slice();
        let send_mr = unsafe {
            pd.reg_mr(
                send_bytes.as_ptr() as usize,
                send_bytes.len(),
                AccessFlags::LocalWrite,
            )
        }
        .map_err(|error| TransferError::Backend(error.to_string()))?;

        let runtime = Arc::new(SidewayRuntime {
            _device_ctx: Arc::clone(&device_ctx),
            pd,
            port_num,
            gid_index,
            link_layer,
            mtu,
            local_gid,
            local_lid,
            local_ud,
            ud_qp: Arc::new(Mutex::new(ud_qp)),
            ud_cq,
            recv_slots,
            send_slot: Mutex::new(UdSendSlot {
                bytes: send_bytes,
                mr: send_mr,
                next_wr_id: 1_u64 << 63,
            }),
            ah_cache: Mutex::new(HashMap::new()),
            control: Arc::new(ControlPlane::default()),
        });

        for idx in 0..runtime.recv_slots.len() {
            Self::post_ud_recv(&runtime, idx)?;
        }

        Self::spawn_control_loop(Arc::downgrade(&runtime), Arc::downgrade(&state));
        info!(
            "transfer runtime ready: nic={}, session_id={}",
            config.nic_name, runtime.local_ud
        );
        Ok(runtime)
    }

    fn spawn_control_loop(runtime: Weak<SidewayRuntime>, state: Weak<Mutex<SidewayState>>) {
        let _ = thread::Builder::new()
            .name("pegaflow-sideway-control".to_string())
            .spawn(move || {
                debug!("transfer control loop started");
                loop {
                    let Some(runtime) = runtime.upgrade() else {
                        debug!("transfer control loop stopping: runtime dropped");
                        break;
                    };
                    let Some(state) = state.upgrade() else {
                        debug!("transfer control loop stopping: state dropped");
                        break;
                    };

                    match runtime.ud_cq.start_poll() {
                        Ok(mut poller) => {
                            let mut did_work = false;
                            for wc in &mut poller {
                                did_work = true;
                                Self::handle_ud_completion(&runtime, &state, &wc);
                            }
                            if !did_work {
                                thread::sleep(Duration::from_micros(50));
                            }
                        }
                        Err(PollCompletionQueueError::CompletionQueueEmpty) => {
                            thread::sleep(Duration::from_micros(100));
                        }
                        Err(error) => {
                            warn!("transfer control loop cq poll error: {error}");
                            thread::sleep(Duration::from_millis(1));
                        }
                    }
                }
            });
    }

    fn post_ud_recv(runtime: &SidewayRuntime, slot_idx: usize) -> Result<()> {
        let Some(slot) = runtime.recv_slots.get(slot_idx) else {
            return Err(TransferError::Backend(
                "invalid UD recv slot index".to_string(),
            ));
        };

        let mut qp = runtime.ud_qp.lock();
        let mut guard = qp.start_post_recv();
        let wr = guard.construct_wr(slot_idx as u64);
        unsafe {
            wr.setup_sge(
                slot.mr.lkey(),
                slot.bytes.as_ptr() as u64,
                slot.bytes.len() as u32,
            );
        }
        guard
            .post()
            .map_err(|error| TransferError::Backend(error.to_string()))
    }

    fn get_or_create_ah(
        runtime: &SidewayRuntime,
        peer_ud: &DomainAddress,
    ) -> Result<Arc<AddressHandle>> {
        if let Some(existing) = runtime.ah_cache.lock().get(peer_ud).cloned() {
            return Ok(existing);
        }

        let mut ah_attr = unsafe { MaybeUninit::<ibv_ah_attr>::zeroed().assume_init() };
        ah_attr.grh = ibv_global_route {
            dgid: Gid { raw: peer_ud.gid() }.into(),
            sgid_index: runtime.gid_index,
            hop_limit: 64,
            ..unsafe { MaybeUninit::<ibv_global_route>::zeroed().assume_init() }
        };
        ah_attr.dlid = peer_ud.lid();
        ah_attr.is_global = if runtime.link_layer == LinkLayer::InfiniBand {
            0
        } else {
            1
        };
        ah_attr.port_num = runtime.port_num;
        let raw_ah = unsafe { ibv_create_ah(runtime.pd.pd().as_ptr(), &raw mut ah_attr) };
        let ah = NonNull::new(raw_ah).ok_or_else(|| {
            TransferError::Backend(format!(
                "ibv_create_ah failed: {}",
                io::Error::last_os_error()
            ))
        })?;
        let wrapped = Arc::new(AddressHandle { ah });
        runtime
            .ah_cache
            .lock()
            .entry(peer_ud.clone())
            .or_insert_with(|| Arc::clone(&wrapped));
        Ok(wrapped)
    }

    fn send_control_message(
        runtime: &SidewayRuntime,
        peer_ud: &DomainAddress,
        message: &ControlMessage,
    ) -> Result<()> {
        debug!(
            "control send: kind={}, request_id={}, peer={}",
            message.kind(),
            message.request_id(),
            peer_ud
        );
        let payload = encode_message(message);
        let ah = Self::get_or_create_ah(runtime, peer_ud)?;

        let mut send_slot = runtime.send_slot.lock();
        if payload.len() > send_slot.bytes.len() {
            return Err(TransferError::Backend(
                "control message too large for UD send buffer".to_string(),
            ));
        }
        send_slot.bytes[..payload.len()].copy_from_slice(&payload);
        let wr_id = send_slot.next_wr_id;
        send_slot.next_wr_id = send_slot.next_wr_id.wrapping_add(1);

        let mut sge = ibv_sge {
            addr: send_slot.bytes.as_ptr() as u64,
            length: payload.len() as u32,
            lkey: send_slot.mr.lkey(),
        };
        let mut wr = unsafe { MaybeUninit::<ibv_send_wr>::zeroed().assume_init() };
        wr.wr_id = wr_id;
        wr.next = null_mut();
        wr.sg_list = &raw mut sge;
        wr.num_sge = 1;
        wr.opcode = ibv_wr_opcode::IBV_WR_SEND;
        wr.send_flags = ibv_send_flags::IBV_SEND_SIGNALED.0;
        wr.wr.ud.ah = ah.ah.as_ptr();
        wr.wr.ud.remote_qpn = peer_ud.qp_num();
        wr.wr.ud.remote_qkey = peer_ud.qkey();

        let qp = runtime.ud_qp.lock();
        let ret = unsafe { ibv_post_send(qp.qp().as_ptr(), &raw mut wr, null_mut()) };
        if ret != 0 {
            return Err(TransferError::Backend(format!(
                "ibv_post_send(UD) failed: {}",
                io::Error::from_raw_os_error(ret)
            )));
        }
        Ok(())
    }

    fn handle_ud_completion(
        runtime: &Arc<SidewayRuntime>,
        state: &Arc<Mutex<SidewayState>>,
        wc: &sideway::ibverbs::completion::GenericWorkCompletion,
    ) {
        if wc.status() != WorkCompletionStatus::Success as u32 {
            warn!(
                "ud completion failed: status={}, opcode={}, vendor_err={}",
                wc.status(),
                wc.opcode(),
                wc.vendor_err()
            );
            if wc.opcode() == WorkCompletionOperationType::Receive as u32 {
                let slot_idx = wc.wr_id() as usize;
                let _ = Self::post_ud_recv(runtime, slot_idx);
            }
            return;
        }

        match wc.opcode() {
            x if x == WorkCompletionOperationType::Receive as u32 => {
                let slot_idx = wc.wr_id() as usize;
                if let Some(slot) = runtime.recv_slots.get(slot_idx) {
                    let byte_len = wc.byte_len() as usize;
                    if byte_len > UD_GRH_BYTES && byte_len <= slot.bytes.len() {
                        let payload = &slot.bytes[UD_GRH_BYTES..byte_len];
                        if let Some(message) = decode_message(payload) {
                            debug!(
                                "control recv: kind={}, request_id={}, bytes={}",
                                message.kind(),
                                message.request_id(),
                                byte_len - UD_GRH_BYTES
                            );
                            Self::handle_control_message(runtime, state, message);
                        }
                    }
                }
                let _ = Self::post_ud_recv(runtime, slot_idx);
            }
            x if x == WorkCompletionOperationType::Send as u32 => {}
            _ => {}
        }
    }

    fn handle_control_message(
        runtime: &Arc<SidewayRuntime>,
        state: &Arc<Mutex<SidewayState>>,
        message: ControlMessage,
    ) {
        match message {
            msg @ ControlMessage::ConnectResp { .. } | msg @ ControlMessage::MrQueryResp { .. } => {
                runtime.control.deliver_reply(msg);
            }
            ControlMessage::ConnectReq {
                request_id,
                src_ud,
                rc,
            } => {
                debug!(
                    "control handle connect_req: request_id={}, peer={}",
                    request_id, src_ud
                );
                if let Ok(local_rc) = Self::ensure_passive_session(runtime, state, &src_ud, rc) {
                    let _ = Self::send_control_message(
                        runtime,
                        &src_ud,
                        &ControlMessage::ConnectResp {
                            request_id,
                            src_ud: runtime.local_ud.clone(),
                            rc: local_rc,
                        },
                    );
                } else {
                    warn!(
                        "control handle connect_req failed: request_id={}, peer={}",
                        request_id, src_ud
                    );
                }
            }
            ControlMessage::MrQueryReq {
                request_id,
                src_ud,
                ptr,
                len,
            } => {
                debug!(
                    "control handle mr_query_req: request_id={}, peer={}, ptr={:#x}, len={}",
                    request_id, src_ud, ptr, len
                );
                let response = {
                    let state = state.lock();
                    let requested_len = usize::try_from(len).ok();
                    if let Some(requested_len) = requested_len {
                        if let Some((base_ptr, entry_len, mr)) =
                            Self::find_registered_entry(&state, ptr, requested_len)
                        {
                            let entry_end = base_ptr.saturating_add(entry_len as u64);
                            let available_len = entry_end.saturating_sub(ptr);
                            ControlMessage::MrQueryResp {
                                request_id,
                                src_ud: runtime.local_ud.clone(),
                                found: true,
                                rkey: mr.rkey(),
                                available_len,
                            }
                        } else {
                            ControlMessage::MrQueryResp {
                                request_id,
                                src_ud: runtime.local_ud.clone(),
                                found: false,
                                rkey: 0,
                                available_len: 0,
                            }
                        }
                    } else {
                        ControlMessage::MrQueryResp {
                            request_id,
                            src_ud: runtime.local_ud.clone(),
                            found: false,
                            rkey: 0,
                            available_len: 0,
                        }
                    }
                };
                let _ = Self::send_control_message(runtime, &src_ud, &response);
            }
        }
    }

    fn create_rc_qp(
        runtime: &SidewayRuntime,
        psn_seed: u64,
    ) -> Result<(
        GenericQueuePair,
        GenericCompletionQueue,
        GenericCompletionQueue,
        RcEndpoint,
    )> {
        let mut cq_builder = runtime._device_ctx.create_cq_builder();
        cq_builder.setup_cqe(128);
        let send_cq: GenericCompletionQueue = cq_builder
            .build()
            .map_err(|error| TransferError::Backend(error.to_string()))?
            .into();
        let recv_cq: GenericCompletionQueue = cq_builder
            .build()
            .map_err(|error| TransferError::Backend(error.to_string()))?
            .into();

        let mut qp_builder = runtime.pd.create_qp_builder();
        qp_builder
            .setup_qp_type(QueuePairType::ReliableConnection)
            .setup_send_cq(send_cq.clone())
            .setup_recv_cq(recv_cq.clone())
            .setup_max_send_wr(128)
            .setup_max_recv_wr(16)
            .setup_max_send_sge(1)
            .setup_max_recv_sge(1);
        let mut qp: GenericQueuePair = qp_builder
            .build()
            .map_err(|error| TransferError::Backend(error.to_string()))?
            .into();

        let mut init_attr = QueuePairAttribute::new();
        init_attr
            .setup_state(QueuePairState::Init)
            .setup_pkey_index(0)
            .setup_port(runtime.port_num)
            .setup_access_flags(
                AccessFlags::LocalWrite | AccessFlags::RemoteWrite | AccessFlags::RemoteRead,
            );
        qp.modify(&init_attr)
            .map_err(|error| TransferError::Backend(error.to_string()))?;

        let local_psn = (psn_seed as u32) & 0x00ff_ffff;
        let local_rc = RcEndpoint {
            gid: runtime.local_gid.raw,
            lid: runtime.local_lid,
            qp_num: qp.qp_number(),
            psn: local_psn,
        };
        Ok((qp, send_cq, recv_cq, local_rc))
    }

    fn connect_rc_qp(
        runtime: &SidewayRuntime,
        session: &ActiveSession,
        remote_rc: RcEndpoint,
    ) -> Result<()> {
        debug!(
            "rc connect start: local_qpn={}, remote_qpn={}, remote_lid={}, remote_gid={:?}",
            session.local_rc.qp_num, remote_rc.qp_num, remote_rc.lid, remote_rc.gid
        );
        let mut ah_attr = AddressHandleAttribute::new();
        ah_attr
            .setup_dest_lid(remote_rc.lid)
            .setup_port(runtime.port_num)
            .setup_grh_dest_gid(&Gid { raw: remote_rc.gid })
            .setup_grh_src_gid_index(runtime.gid_index)
            .setup_grh_hop_limit(64);

        let mut qp = session.qp.lock();
        let mut rtr_attr = QueuePairAttribute::new();
        rtr_attr
            .setup_state(QueuePairState::ReadyToReceive)
            .setup_path_mtu(runtime.mtu)
            .setup_dest_qp_num(remote_rc.qp_num)
            .setup_rq_psn(remote_rc.psn)
            .setup_max_dest_read_atomic(1)
            .setup_min_rnr_timer(12)
            .setup_address_vector(&ah_attr);
        qp.modify(&rtr_attr)
            .map_err(|error| TransferError::Backend(error.to_string()))?;

        let mut rts_attr = QueuePairAttribute::new();
        rts_attr
            .setup_state(QueuePairState::ReadyToSend)
            .setup_sq_psn(session.local_rc.psn)
            .setup_timeout(14)
            .setup_retry_cnt(7)
            .setup_rnr_retry(7)
            .setup_max_read_atomic(1);
        qp.modify(&rts_attr)
            .map_err(|error| TransferError::Backend(error.to_string()))?;
        debug!(
            "rc connect ready: local_qpn={}, remote_qpn={}",
            session.local_rc.qp_num, remote_rc.qp_num
        );
        Ok(())
    }

    fn ensure_passive_session(
        runtime: &Arc<SidewayRuntime>,
        state: &Arc<Mutex<SidewayState>>,
        peer_ud: &DomainAddress,
        remote_rc: RcEndpoint,
    ) -> Result<RcEndpoint> {
        let peer_key = peer_ud.clone();
        if let Some(existing) = state.lock().sessions.get(&peer_key).cloned() {
            debug!("passive session reused: peer={}", peer_key);
            return Ok(existing.local_rc);
        }
        info!("passive session create: peer={}", peer_key);

        let seed = runtime
            .control
            .next_request_id
            .fetch_add(1, Ordering::Relaxed);
        let (qp, send_cq, recv_cq, local_rc) = Self::create_rc_qp(runtime, seed)?;
        let (cmd_tx, cmd_rx) = std_mpsc::channel();
        let session = Arc::new(ActiveSession {
            qp: Mutex::new(qp),
            send_cq,
            _recv_cq: recv_cq,
            local_rc,
            cmd_tx,
        });
        Self::connect_rc_qp(runtime, &session, remote_rc)?;
        Self::spawn_session_worker(Arc::clone(&session), cmd_rx);

        let mut guard = state.lock();
        let retained = guard
            .sessions
            .entry(peer_key)
            .or_insert_with(|| Arc::clone(&session))
            .clone();
        info!(
            "passive session ready: local_qpn={}, peer={}",
            retained.local_rc.qp_num, peer_ud
        );
        Ok(retained.local_rc)
    }

    fn ensure_active_session(
        runtime: &Arc<SidewayRuntime>,
        state: &Arc<Mutex<SidewayState>>,
        peer_ud: &DomainAddress,
    ) -> Result<Arc<ActiveSession>> {
        let peer_key = peer_ud.clone();
        if let Some(existing) = state.lock().sessions.get(&peer_key).cloned() {
            debug!("active session reused: peer={peer_key}");
            return Ok(existing);
        }
        info!("active session create: peer={peer_key}");

        let request_id = runtime.control.begin_request();
        let seed = request_id;
        let (qp, send_cq, recv_cq, local_rc) = Self::create_rc_qp(runtime, seed)?;
        let (cmd_tx, cmd_rx) = std_mpsc::channel();
        let session = Arc::new(ActiveSession {
            qp: Mutex::new(qp),
            send_cq,
            _recv_cq: recv_cq,
            local_rc,
            cmd_tx,
        });

        Self::send_control_message(
            runtime,
            peer_ud,
            &ControlMessage::ConnectReq {
                request_id,
                src_ud: runtime.local_ud.clone(),
                rc: local_rc,
            },
        )?;

        let reply = runtime
            .control
            .complete_request(request_id, CONTROL_TIMEOUT)
            .ok_or_else(|| TransferError::Backend("connect response timeout".to_string()))?;

        let remote_rc = match reply {
            ControlMessage::ConnectResp { src_ud, rc, .. } if &src_ud == peer_ud => rc,
            other => {
                return Err(TransferError::Backend(format!(
                    "unexpected connect response: {other:?}"
                )));
            }
        };
        Self::connect_rc_qp(runtime, &session, remote_rc)?;
        Self::spawn_session_worker(Arc::clone(&session), cmd_rx);

        let mut guard = state.lock();
        let retained = guard
            .sessions
            .entry(peer_key)
            .or_insert_with(|| Arc::clone(&session))
            .clone();
        info!(
            "active session ready: local_qpn={}, peer={}",
            retained.local_rc.qp_num, peer_ud
        );
        Ok(retained)
    }

    fn query_remote_memory(
        runtime: &SidewayRuntime,
        peer_ud: &DomainAddress,
        remote_ptr: u64,
        len: usize,
    ) -> Result<(u32, usize)> {
        let request_id = runtime.control.begin_request();
        debug!(
            "mr query send: request_id={}, peer={}, ptr={:#x}, len={}",
            request_id, peer_ud, remote_ptr, len
        );
        Self::send_control_message(
            runtime,
            peer_ud,
            &ControlMessage::MrQueryReq {
                request_id,
                src_ud: runtime.local_ud.clone(),
                ptr: remote_ptr,
                len: len as u64,
            },
        )?;

        let reply = runtime
            .control
            .complete_request(request_id, CONTROL_TIMEOUT)
            .ok_or_else(|| TransferError::Backend("remote MR query timeout".to_string()))?;
        match reply {
            ControlMessage::MrQueryResp {
                src_ud,
                found,
                rkey,
                available_len,
                ..
            } if &src_ud == peer_ud => {
                if !found {
                    return Err(TransferError::InvalidArgument(
                        "remote memory not registered for target ptr",
                    ));
                }
                let available_len = usize::try_from(available_len).map_err(|_| {
                    TransferError::Backend("remote memory available length overflow".to_string())
                })?;
                debug!(
                    "mr query ok: request_id={}, peer={}, rkey={}, available_len={}",
                    request_id, peer_ud, rkey, available_len
                );
                Ok((rkey, available_len))
            }
            other => Err(TransferError::Backend(format!(
                "unexpected MR query response: {other:?}"
            ))),
        }
    }

    fn post_write_wr_chain(
        session: &ActiveSession,
        ops: &[WriteOp],
        first_wr_id: u64,
    ) -> Result<usize> {
        if ops.is_empty() {
            return Ok(0);
        }

        let mut sges: Vec<ibv_sge> = (0..ops.len())
            .map(|_| unsafe { MaybeUninit::<ibv_sge>::zeroed().assume_init() })
            .collect();
        let mut wrs: Vec<ibv_send_wr> = (0..ops.len())
            .map(|_| unsafe { MaybeUninit::<ibv_send_wr>::zeroed().assume_init() })
            .collect();

        for (idx, op) in ops.iter().enumerate() {
            if op.len > u32::MAX as usize {
                return Err(TransferError::InvalidArgument(
                    "len exceeds RDMA SGE length limit",
                ));
            }

            sges[idx] = ibv_sge {
                addr: op.local_ptr,
                length: op.len as u32,
                lkey: op.local_mr.lkey(),
            };

            wrs[idx].wr_id = first_wr_id.wrapping_add(idx as u64);
            wrs[idx].sg_list = &raw mut sges[idx];
            wrs[idx].num_sge = 1;
            wrs[idx].opcode = ibv_wr_opcode::IBV_WR_RDMA_WRITE;
            wrs[idx].send_flags = ibv_send_flags::IBV_SEND_SIGNALED.0;
            wrs[idx].wr.rdma.remote_addr = op.remote_ptr;
            wrs[idx].wr.rdma.rkey = op.remote_rkey;
            wrs[idx].next = if idx + 1 < wrs.len() {
                &raw mut wrs[idx + 1]
            } else {
                null_mut()
            };
        }

        let mut bad_wr = null_mut();
        {
            let qp = session.qp.lock();
            let ret = unsafe {
                ibv_post_send(qp.qp().as_ptr(), wrs.as_mut_ptr(), &raw mut bad_wr)
            };
            if ret != 0 {
                return Err(TransferError::Backend(format!(
                    "ibv_post_send(RDMA_WRITE chain) failed: {}",
                    io::Error::from_raw_os_error(ret)
                )));
            }
        }
        Ok(ops.len())
    }

    fn execute_batch_writes(session: &ActiveSession, ops: Vec<WriteOp>) -> Result<usize> {
        let total_ops = ops.len();
        if total_ops == 0 {
            return Ok(0);
        }

        let mut next_idx = 0usize;
        let mut next_wr_id = 1_u64;
        let mut inflight: HashMap<u64, usize> = HashMap::new();
        let mut transferred = 0usize;

        while next_idx < total_ops || !inflight.is_empty() {
            while next_idx < total_ops && inflight.len() < MAX_INFLIGHT_WRITES {
                let available = MAX_INFLIGHT_WRITES - inflight.len();
                let remaining = total_ops - next_idx;
                let chain_len = MAX_WR_CHAIN_WRITES.min(available).min(remaining);
                let posted = Self::post_write_wr_chain(
                    session,
                    &ops[next_idx..next_idx + chain_len],
                    next_wr_id,
                )?;
                if posted == 0 {
                    break;
                }
                for op in &ops[next_idx..next_idx + posted] {
                    inflight.insert(next_wr_id, op.len);
                    next_wr_id = next_wr_id.wrapping_add(1);
                }
                next_idx += posted;
            }

            match session.send_cq.start_poll() {
                Ok(mut poller) => {
                    let mut did_work = false;
                    for wc in &mut poller {
                        did_work = true;
                        let Some(bytes) = inflight.remove(&wc.wr_id()) else {
                            continue;
                        };
                        if wc.status() != WorkCompletionStatus::Success as u32 {
                            return Err(TransferError::Backend(format!(
                                "send completion failed: status={}, opcode={}, vendor_err={}",
                                wc.status(),
                                wc.opcode(),
                                wc.vendor_err()
                            )));
                        }
                        transferred = transferred.saturating_add(bytes);
                    }
                    if !did_work {
                        thread::sleep(DATA_CQ_POLL_SLEEP);
                    }
                }
                Err(PollCompletionQueueError::CompletionQueueEmpty) => {
                    thread::sleep(DATA_CQ_POLL_SLEEP);
                }
                Err(error) => {
                    return Err(TransferError::Backend(format!(
                        "poll send CQ failed: {error}"
                    )));
                }
            }
        }

        Ok(transferred)
    }

    fn spawn_session_worker(
        session: Arc<ActiveSession>,
        cmd_rx: std_mpsc::Receiver<SessionCommand>,
    ) {
        let _ = thread::Builder::new()
            .name("pegaflow-sideway-session".to_string())
            .spawn(move || {
                debug!(
                    "session worker started: local_qpn={}",
                    session.local_rc.qp_num
                );
                while let Ok(command) = cmd_rx.recv() {
                    match command {
                        SessionCommand::SubmitBatch { ops, done_tx } => {
                            let result = Self::execute_batch_writes(&session, ops);
                            if done_tx.send(result).is_err() {
                                debug!(
                                    "session worker reply receiver dropped: local_qpn={}",
                                    session.local_rc.qp_num
                                );
                            }
                        }
                    }
                }
                debug!(
                    "session worker stopped: local_qpn={}",
                    session.local_rc.qp_num
                );
            });
    }

    fn submit_session_batch(session: &ActiveSession, ops: Vec<WriteOp>) -> Result<usize> {
        let (done_tx, done_rx) = std_mpsc::channel();
        session
            .cmd_tx
            .send(SessionCommand::SubmitBatch { ops, done_tx })
            .map_err(|_| {
                TransferError::Backend("session worker channel disconnected".to_string())
            })?;
        done_rx
            .recv()
            .map_err(|_| TransferError::Backend("session worker dropped completion".to_string()))?
    }
}

impl SidewayBackend {
    pub(crate) fn initialize(&self, config: WorkerConfig) -> Result<()> {
        logging::ensure_initialized();
        if config.nic_name.trim().is_empty() {
            return Err(TransferError::InvalidArgument("nic_name is empty"));
        }
        if config.rpc_port == 0 {
            return Err(TransferError::InvalidArgument("rpc_port must be non-zero"));
        }
        info!(
            "transfer backend initialize: nic={}, rpc_port={}",
            config.nic_name, config.rpc_port
        );

        let runtime = Self::create_runtime(&config, Arc::clone(&self.state))?;
        let mut state = self.state.lock();
        state.config = Some(config);
        state.runtime = Some(Arc::clone(&runtime));
        state.registered.clear();
        state.sessions.clear();
        state.remote_memory_cache.clear();
        info!(
            "transfer backend initialized: session_id={}",
            runtime.local_ud
        );
        Ok(())
    }

    pub(crate) fn rpc_port(&self) -> Result<u16> {
        let state = self.state.lock();
        Ok(Self::ensure_initialized(&state)?.rpc_port)
    }

    pub(crate) fn session_id(&self) -> DomainAddress {
        let state = self.state.lock();
        let runtime = state
            .runtime
            .as_ref()
            .expect("transfer backend not initialized");
        runtime.local_ud.clone()
    }

    pub(crate) fn register_memory(&self, ptr: u64, len: usize) -> Result<()> {
        if ptr == 0 {
            return Err(TransferError::InvalidArgument("ptr must be non-zero"));
        }
        if len == 0 {
            return Err(TransferError::InvalidArgument("len must be non-zero"));
        }

        let mut state = self.state.lock();
        Self::ensure_initialized(&state)?;
        let runtime = state
            .runtime
            .as_ref()
            .ok_or(TransferError::NotInitialized)?;
        let mr = unsafe {
            runtime.pd.reg_mr(
                ptr as usize,
                len,
                AccessFlags::LocalWrite | AccessFlags::RemoteWrite | AccessFlags::RemoteRead,
            )
        }
        .map_err(|error| TransferError::Backend(error.to_string()))?;

        state.registered.insert(
            ptr,
            RegisteredMemoryEntry {
                base_ptr: ptr,
                len,
                mr,
            },
        );
        debug!("memory registered: ptr={:#x}, len={}", ptr, len);
        Ok(())
    }

    pub(crate) fn unregister_memory(&self, ptr: u64) -> Result<()> {
        if ptr == 0 {
            return Err(TransferError::InvalidArgument("ptr must be non-zero"));
        }
        let mut state = self.state.lock();
        Self::ensure_initialized(&state)?;
        let removed = state.registered.remove(&ptr);
        if removed.is_none() {
            return Err(TransferError::MemoryNotRegistered { ptr });
        }
        debug!("memory unregistered: ptr={:#x}", ptr);
        Ok(())
    }

    pub(crate) fn transfer_sync_write(
        &self,
        session_id: &DomainAddress,
        local_ptr: u64,
        remote_ptr: u64,
        len: usize,
    ) -> Result<usize> {
        self.batch_transfer_sync_write(session_id, &[local_ptr], &[remote_ptr], &[len])
    }

    pub(crate) fn batch_transfer_sync_write(
        &self,
        session_id: &DomainAddress,
        local_ptrs: &[u64],
        remote_ptrs: &[u64],
        lens: &[usize],
    ) -> Result<usize> {
        if session_id.to_bytes().len() != DomainAddress::BYTES {
            return Err(TransferError::InvalidArgument(
                "session_id has invalid DomainAddress bytes",
            ));
        }
        if local_ptrs.len() != remote_ptrs.len() {
            return Err(TransferError::BatchLengthMismatch {
                ptrs: local_ptrs.len(),
                lens: remote_ptrs.len(),
            });
        }
        if local_ptrs.len() != lens.len() {
            return Err(TransferError::BatchLengthMismatch {
                ptrs: local_ptrs.len(),
                lens: lens.len(),
            });
        }
        if lens.is_empty() {
            return Ok(0);
        }

        let peer_ud = session_id;
        let local_lookup_start = Instant::now();
        let (runtime, mut prepared_ops): (
            Arc<SidewayRuntime>,
            Vec<(WriteOp, Option<(u32, usize)>)>,
        ) = {
            let state = self.state.lock();
            Self::ensure_initialized(&state)?;
            let runtime = state
                .runtime
                .as_ref()
                .ok_or(TransferError::NotInitialized)?;

            let mut prepared_ops = Vec::with_capacity(lens.len());
            for ((local_ptr, remote_ptr), len) in local_ptrs
                .iter()
                .copied()
                .zip(remote_ptrs.iter().copied())
                .zip(lens.iter().copied())
            {
                if local_ptr == 0 {
                    return Err(TransferError::InvalidArgument("local_ptr must be non-zero"));
                }
                if remote_ptr == 0 {
                    return Err(TransferError::InvalidArgument(
                        "remote_ptr must be non-zero",
                    ));
                }
                if len == 0 {
                    return Err(TransferError::InvalidArgument("len must be non-zero"));
                }
                let Some((_, _, local_mr)) = Self::find_registered_entry(&state, local_ptr, len)
                else {
                    return Err(TransferError::MemoryNotRegistered { ptr: local_ptr });
                };
                let cached_remote =
                    Self::find_cached_remote_memory(&state, peer_ud, remote_ptr, len);
                prepared_ops.push((
                    WriteOp {
                        local_mr,
                        local_ptr,
                        remote_ptr,
                        len,
                        remote_rkey: 0,
                    },
                    cached_remote,
                ));
            }
            (Arc::clone(runtime), prepared_ops)
        };
        let local_lookup_dur = local_lookup_start.elapsed();

        let ensure_session_start = Instant::now();
        let session = Self::ensure_active_session(&runtime, &self.state, peer_ud)?;
        let ensure_session_dur = ensure_session_start.elapsed();

        let mr_query_start = Instant::now();
        let mut mr_cache_hits = 0usize;
        for (op, cached_remote) in prepared_ops.iter_mut() {
            let (remote_rkey, remote_available) =
                if let Some((cached_rkey, cached_available)) = *cached_remote {
                    mr_cache_hits += 1;
                    (cached_rkey, cached_available)
                } else {
                    let (queried_rkey, queried_available) =
                        Self::query_remote_memory(&runtime, peer_ud, op.remote_ptr, op.len)?;
                    {
                        let mut state = self.state.lock();
                        Self::cache_remote_memory(
                            &mut state,
                            peer_ud,
                            op.remote_ptr,
                            queried_available,
                            queried_rkey,
                        );
                    }
                    (queried_rkey, queried_available)
                };
            if op.len > remote_available {
                return Err(TransferError::InvalidArgument(
                    "len exceeds remote registered memory",
                ));
            }
            op.remote_rkey = remote_rkey;
        }
        let mr_query_dur = mr_query_start.elapsed();

        let ops: Vec<WriteOp> = prepared_ops.into_iter().map(|(op, _)| op).collect();
        let submit_start = Instant::now();
        let transferred = Self::submit_session_batch(&session, ops)?;
        let submit_dur = submit_start.elapsed();

        let total_dur = local_lookup_dur + ensure_session_dur + mr_query_dur + submit_dur;
        info!(
            "batch_transfer_sync_write profile: peer={}, bytes={}, chunks={}, local_lookup_ms={:.3}, ensure_session_ms={:.3}, mr_query_ms={:.3}, mr_cache_hits={}, submit_wait_ms={:.3}, total_ms={:.3}",
            peer_ud,
            transferred,
            lens.len(),
            local_lookup_dur.as_secs_f64() * 1000.0,
            ensure_session_dur.as_secs_f64() * 1000.0,
            mr_query_dur.as_secs_f64() * 1000.0,
            mr_cache_hits,
            submit_dur.as_secs_f64() * 1000.0,
            total_dur.as_secs_f64() * 1000.0
        );
        Ok(transferred)
    }
}

#[cfg(test)]
mod tests {
    use super::SidewayBackend;
    use crate::{api::WorkerConfig, error::TransferError};

    #[test]
    fn initialize_rejects_invalid_input() {
        let backend = SidewayBackend::new();
        let error = backend
            .initialize(WorkerConfig {
                nic_name: "".to_string(),
                rpc_port: 50055,
            })
            .expect_err("must fail");
        assert_eq!(error, TransferError::InvalidArgument("nic_name is empty"));
    }
}
