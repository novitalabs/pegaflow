use std::{
    collections::HashMap,
    io,
    mem::MaybeUninit,
    ptr::{NonNull, null_mut},
    sync::{
        Arc, Weak,
        atomic::{AtomicU64, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

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
        GenericQueuePair, PostSendGuard, QueuePair, QueuePairAttribute, QueuePairState,
        QueuePairType, SendOperationFlags, SetScatterGatherEntry, WorkRequestFlags,
    },
};

use crate::{
    api::WorkerConfig,
    backend::RdmaBackend,
    error::{Result, TransferError},
};

const UD_QKEY: u32 = 0x1111_1111;
const UD_RECV_SLOTS: usize = 64;
const UD_BUFFER_BYTES: usize = 512;
const UD_GRH_BYTES: usize = 40;
const CONTROL_TIMEOUT: Duration = Duration::from_secs(3);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct DomainAddressRaw {
    gid: [u8; 16],
    lid: u16,
    qp_num: u32,
    qkey: u32,
}

impl DomainAddressRaw {
    const BYTES: usize = 26;

    fn to_bytes(self) -> [u8; Self::BYTES] {
        let mut bytes = [0_u8; Self::BYTES];
        bytes[..16].copy_from_slice(&self.gid);
        bytes[16..18].copy_from_slice(&self.lid.to_le_bytes());
        bytes[18..22].copy_from_slice(&self.qp_num.to_le_bytes());
        bytes[22..26].copy_from_slice(&self.qkey.to_le_bytes());
        bytes
    }

    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != Self::BYTES {
            return None;
        }
        let mut gid = [0_u8; 16];
        gid.copy_from_slice(&bytes[..16]);
        Some(Self {
            gid,
            lid: u16::from_le_bytes(bytes[16..18].try_into().ok()?),
            qp_num: u32::from_le_bytes(bytes[18..22].try_into().ok()?),
            qkey: u32::from_le_bytes(bytes[22..26].try_into().ok()?),
        })
    }

    fn to_hex(self) -> String {
        bytes_to_hex(&self.to_bytes())
    }

    fn from_hex(s: &str) -> Option<Self> {
        let bytes = hex_to_bytes(s)?;
        Self::from_bytes(&bytes)
    }
}

#[derive(Clone, Copy, Debug)]
struct RcEndpoint {
    gid: [u8; 16],
    lid: u16,
    qp_num: u32,
    psn: u32,
}

impl RcEndpoint {
    const BYTES: usize = 26;

    fn to_bytes(self) -> [u8; Self::BYTES] {
        let mut bytes = [0_u8; Self::BYTES];
        bytes[..16].copy_from_slice(&self.gid);
        bytes[16..18].copy_from_slice(&self.lid.to_le_bytes());
        bytes[18..22].copy_from_slice(&self.qp_num.to_le_bytes());
        bytes[22..26].copy_from_slice(&self.psn.to_le_bytes());
        bytes
    }

    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != Self::BYTES {
            return None;
        }
        let mut gid = [0_u8; 16];
        gid.copy_from_slice(&bytes[..16]);
        Some(Self {
            gid,
            lid: u16::from_le_bytes(bytes[16..18].try_into().ok()?),
            qp_num: u32::from_le_bytes(bytes[18..22].try_into().ok()?),
            psn: u32::from_le_bytes(bytes[22..26].try_into().ok()?),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum MessageType {
    ConnectReq = 1,
    ConnectResp = 2,
    MrQueryReq = 3,
    MrQueryResp = 4,
}

impl MessageType {
    fn from_u8(raw: u8) -> Option<Self> {
        match raw {
            1 => Some(Self::ConnectReq),
            2 => Some(Self::ConnectResp),
            3 => Some(Self::MrQueryReq),
            4 => Some(Self::MrQueryResp),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
enum ControlMessage {
    ConnectReq {
        request_id: u64,
        src_ud: DomainAddressRaw,
        rc: RcEndpoint,
    },
    ConnectResp {
        request_id: u64,
        src_ud: DomainAddressRaw,
        rc: RcEndpoint,
    },
    MrQueryReq {
        request_id: u64,
        src_ud: DomainAddressRaw,
        ptr: u64,
        len: u64,
    },
    MrQueryResp {
        request_id: u64,
        src_ud: DomainAddressRaw,
        found: bool,
        rkey: u32,
        available_len: u64,
    },
}

impl ControlMessage {
    fn request_id(&self) -> u64 {
        match self {
            ControlMessage::ConnectReq { request_id, .. }
            | ControlMessage::ConnectResp { request_id, .. }
            | ControlMessage::MrQueryReq { request_id, .. }
            | ControlMessage::MrQueryResp { request_id, .. } => *request_id,
        }
    }
}

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
    local_ud: DomainAddressRaw,
    ud_qp: Arc<Mutex<GenericQueuePair>>,
    ud_cq: GenericCompletionQueue,
    recv_slots: Vec<UdRecvSlot>,
    send_slot: Mutex<UdSendSlot>,
    ah_cache: Mutex<HashMap<DomainAddressRaw, Arc<AddressHandle>>>,
    control: Arc<ControlPlane>,
}

struct RegisteredMemoryEntry {
    base_ptr: u64,
    len: usize,
    mr: Arc<MemoryRegion>,
}

struct ActiveSession {
    qp: Mutex<GenericQueuePair>,
    send_cq: GenericCompletionQueue,
    _recv_cq: GenericCompletionQueue,
    local_rc: RcEndpoint,
}

#[derive(Default)]
struct SidewayState {
    config: Option<WorkerConfig>,
    runtime: Option<Arc<SidewayRuntime>>,
    registered: HashMap<u64, RegisteredMemoryEntry>,
    sessions: HashMap<String, Arc<ActiveSession>>,
}

#[derive(Default)]
pub struct SidewayBackend {
    state: Arc<Mutex<SidewayState>>,
}

impl SidewayBackend {
    pub fn new() -> Self {
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

    fn encode_message(message: &ControlMessage) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(96);
        match message {
            ControlMessage::ConnectReq {
                request_id,
                src_ud,
                rc,
            } => {
                bytes.push(MessageType::ConnectReq as u8);
                bytes.extend_from_slice(&request_id.to_le_bytes());
                bytes.extend_from_slice(&src_ud.to_bytes());
                bytes.extend_from_slice(&rc.to_bytes());
            }
            ControlMessage::ConnectResp {
                request_id,
                src_ud,
                rc,
            } => {
                bytes.push(MessageType::ConnectResp as u8);
                bytes.extend_from_slice(&request_id.to_le_bytes());
                bytes.extend_from_slice(&src_ud.to_bytes());
                bytes.extend_from_slice(&rc.to_bytes());
            }
            ControlMessage::MrQueryReq {
                request_id,
                src_ud,
                ptr,
                len,
            } => {
                bytes.push(MessageType::MrQueryReq as u8);
                bytes.extend_from_slice(&request_id.to_le_bytes());
                bytes.extend_from_slice(&src_ud.to_bytes());
                bytes.extend_from_slice(&ptr.to_le_bytes());
                bytes.extend_from_slice(&len.to_le_bytes());
            }
            ControlMessage::MrQueryResp {
                request_id,
                src_ud,
                found,
                rkey,
                available_len,
            } => {
                bytes.push(MessageType::MrQueryResp as u8);
                bytes.extend_from_slice(&request_id.to_le_bytes());
                bytes.extend_from_slice(&src_ud.to_bytes());
                bytes.push(u8::from(*found));
                bytes.extend_from_slice(&rkey.to_le_bytes());
                bytes.extend_from_slice(&available_len.to_le_bytes());
            }
        }
        bytes
    }

    fn decode_message(bytes: &[u8]) -> Option<ControlMessage> {
        if bytes.len() < 1 + 8 + DomainAddressRaw::BYTES {
            return None;
        }
        let kind = MessageType::from_u8(bytes[0])?;
        let request_id = u64::from_le_bytes(bytes[1..9].try_into().ok()?);
        let src_ud = DomainAddressRaw::from_bytes(&bytes[9..(9 + DomainAddressRaw::BYTES)])?;
        let payload = &bytes[(9 + DomainAddressRaw::BYTES)..];

        match kind {
            MessageType::ConnectReq => Some(ControlMessage::ConnectReq {
                request_id,
                src_ud,
                rc: RcEndpoint::from_bytes(payload)?,
            }),
            MessageType::ConnectResp => Some(ControlMessage::ConnectResp {
                request_id,
                src_ud,
                rc: RcEndpoint::from_bytes(payload)?,
            }),
            MessageType::MrQueryReq => {
                if payload.len() != 16 {
                    return None;
                }
                Some(ControlMessage::MrQueryReq {
                    request_id,
                    src_ud,
                    ptr: u64::from_le_bytes(payload[..8].try_into().ok()?),
                    len: u64::from_le_bytes(payload[8..16].try_into().ok()?),
                })
            }
            MessageType::MrQueryResp => {
                if payload.len() != 13 {
                    return None;
                }
                Some(ControlMessage::MrQueryResp {
                    request_id,
                    src_ud,
                    found: payload[0] != 0,
                    rkey: u32::from_le_bytes(payload[1..5].try_into().ok()?),
                    available_len: u64::from_le_bytes(payload[5..13].try_into().ok()?),
                })
            }
        }
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

        let mut cq_builder = device_ctx.create_cq_builder();
        cq_builder.setup_cqe(256);
        let ud_cq: GenericCompletionQueue = cq_builder
            .build()
            .map_err(|error| TransferError::Backend(error.to_string()))?
            .into();

        let mut qp_builder = pd.create_qp_builder();
        qp_builder
            .setup_qp_type(QueuePairType::UnreliableDatagram)
            .setup_send_ops_flags(
                SendOperationFlags::Send | SendOperationFlags::SendWithImmediate,
            )
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

        let local_ud = DomainAddressRaw {
            gid: local_gid.raw,
            lid: local_lid,
            qp_num: ud_qp.qp_number(),
            qkey: UD_QKEY,
        };

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
        Ok(runtime)
    }

    fn spawn_control_loop(runtime: Weak<SidewayRuntime>, state: Weak<Mutex<SidewayState>>) {
        let _ = thread::Builder::new()
            .name("pegaflow-sideway-control".to_string())
            .spawn(move || {
                loop {
                    let Some(runtime) = runtime.upgrade() else {
                        break;
                    };
                    let Some(state) = state.upgrade() else {
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
                        Err(_) => {
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
        peer_ud: DomainAddressRaw,
    ) -> Result<Arc<AddressHandle>> {
        if let Some(existing) = runtime.ah_cache.lock().get(&peer_ud).cloned() {
            return Ok(existing);
        }

        let mut ah_attr = unsafe { MaybeUninit::<ibv_ah_attr>::zeroed().assume_init() };
        ah_attr.grh = ibv_global_route {
            dgid: Gid { raw: peer_ud.gid }.into(),
            sgid_index: runtime.gid_index,
            hop_limit: 64,
            ..unsafe { MaybeUninit::<ibv_global_route>::zeroed().assume_init() }
        };
        ah_attr.dlid = peer_ud.lid;
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
            .entry(peer_ud)
            .or_insert_with(|| Arc::clone(&wrapped));
        Ok(wrapped)
    }

    fn send_control_message(
        runtime: &SidewayRuntime,
        peer_ud: DomainAddressRaw,
        message: &ControlMessage,
    ) -> Result<()> {
        let payload = Self::encode_message(message);
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
        wr.wr.ud.remote_qpn = peer_ud.qp_num;
        wr.wr.ud.remote_qkey = peer_ud.qkey;

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
                        if let Some(message) = Self::decode_message(payload) {
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
                if let Ok(local_rc) = Self::ensure_passive_session(runtime, state, src_ud, rc) {
                    let _ = Self::send_control_message(
                        runtime,
                        src_ud,
                        &ControlMessage::ConnectResp {
                            request_id,
                            src_ud: runtime.local_ud,
                            rc: local_rc,
                        },
                    );
                }
            }
            ControlMessage::MrQueryReq {
                request_id,
                src_ud,
                ptr,
                len,
            } => {
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
                                src_ud: runtime.local_ud,
                                found: true,
                                rkey: mr.rkey(),
                                available_len,
                            }
                        } else {
                            ControlMessage::MrQueryResp {
                                request_id,
                                src_ud: runtime.local_ud,
                                found: false,
                                rkey: 0,
                                available_len: 0,
                            }
                        }
                    } else {
                        ControlMessage::MrQueryResp {
                            request_id,
                            src_ud: runtime.local_ud,
                            found: false,
                            rkey: 0,
                            available_len: 0,
                        }
                    }
                };
                let _ = Self::send_control_message(runtime, src_ud, &response);
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
        Ok(())
    }

    fn ensure_passive_session(
        runtime: &Arc<SidewayRuntime>,
        state: &Arc<Mutex<SidewayState>>,
        peer_ud: DomainAddressRaw,
        remote_rc: RcEndpoint,
    ) -> Result<RcEndpoint> {
        let peer_key = peer_ud.to_hex();
        if let Some(existing) = state.lock().sessions.get(&peer_key).cloned() {
            return Ok(existing.local_rc);
        }

        let seed = runtime
            .control
            .next_request_id
            .fetch_add(1, Ordering::Relaxed);
        let (qp, send_cq, recv_cq, local_rc) = Self::create_rc_qp(runtime, seed)?;
        let session = Arc::new(ActiveSession {
            qp: Mutex::new(qp),
            send_cq,
            _recv_cq: recv_cq,
            local_rc,
        });
        Self::connect_rc_qp(runtime, &session, remote_rc)?;

        let mut guard = state.lock();
        let retained = guard
            .sessions
            .entry(peer_key)
            .or_insert_with(|| Arc::clone(&session))
            .clone();
        Ok(retained.local_rc)
    }

    fn ensure_active_session(
        runtime: &Arc<SidewayRuntime>,
        state: &Arc<Mutex<SidewayState>>,
        peer_ud: DomainAddressRaw,
    ) -> Result<Arc<ActiveSession>> {
        let peer_key = peer_ud.to_hex();
        if let Some(existing) = state.lock().sessions.get(&peer_key).cloned() {
            return Ok(existing);
        }

        let request_id = runtime.control.begin_request();
        let seed = request_id;
        let (qp, send_cq, recv_cq, local_rc) = Self::create_rc_qp(runtime, seed)?;
        let session = Arc::new(ActiveSession {
            qp: Mutex::new(qp),
            send_cq,
            _recv_cq: recv_cq,
            local_rc,
        });

        Self::send_control_message(
            runtime,
            peer_ud,
            &ControlMessage::ConnectReq {
                request_id,
                src_ud: runtime.local_ud,
                rc: local_rc,
            },
        )?;

        let reply = runtime
            .control
            .complete_request(request_id, CONTROL_TIMEOUT)
            .ok_or_else(|| TransferError::Backend("connect response timeout".to_string()))?;

        let remote_rc = match reply {
            ControlMessage::ConnectResp { src_ud, rc, .. } if src_ud == peer_ud => rc,
            other => {
                return Err(TransferError::Backend(format!(
                    "unexpected connect response: {other:?}"
                )));
            }
        };
        Self::connect_rc_qp(runtime, &session, remote_rc)?;

        let mut guard = state.lock();
        let retained = guard
            .sessions
            .entry(peer_key)
            .or_insert_with(|| Arc::clone(&session))
            .clone();
        Ok(retained)
    }

    fn query_remote_memory(
        runtime: &SidewayRuntime,
        peer_ud: DomainAddressRaw,
        remote_ptr: u64,
        len: usize,
    ) -> Result<(u32, usize)> {
        let request_id = runtime.control.begin_request();
        Self::send_control_message(
            runtime,
            peer_ud,
            &ControlMessage::MrQueryReq {
                request_id,
                src_ud: runtime.local_ud,
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
            } if src_ud == peer_ud => {
                if !found {
                    return Err(TransferError::InvalidArgument(
                        "remote memory not registered for target ptr",
                    ));
                }
                let available_len = usize::try_from(available_len).map_err(|_| {
                    TransferError::Backend("remote memory available length overflow".to_string())
                })?;
                Ok((rkey, available_len))
            }
            other => Err(TransferError::Backend(format!(
                "unexpected MR query response: {other:?}"
            ))),
        }
    }

    fn wait_send_completion(
        send_cq: &GenericCompletionQueue,
        wr_id: u64,
        timeout: Duration,
    ) -> Result<()> {
        let deadline = Instant::now() + timeout;
        loop {
            match send_cq.start_poll() {
                Ok(mut poller) => {
                    for wc in &mut poller {
                        if wc.wr_id() != wr_id {
                            continue;
                        }
                        if wc.status() != WorkCompletionStatus::Success as u32 {
                            return Err(TransferError::Backend(format!(
                                "send completion failed: status={}, opcode={}, vendor_err={}",
                                wc.status(),
                                wc.opcode(),
                                wc.vendor_err()
                            )));
                        }
                        return Ok(());
                    }
                }
                Err(PollCompletionQueueError::CompletionQueueEmpty) => {}
                Err(error) => {
                    return Err(TransferError::Backend(format!(
                        "poll send CQ failed: {error}"
                    )));
                }
            }

            if Instant::now() >= deadline {
                return Err(TransferError::Backend(
                    "send completion timeout".to_string(),
                ));
            }
            thread::sleep(Duration::from_micros(50));
        }
    }

    fn post_write(
        session: &ActiveSession,
        local_mr: Arc<MemoryRegion>,
        local_ptr: u64,
        remote_ptr: u64,
        len: usize,
        remote_rkey: u32,
    ) -> Result<()> {
        if len > u32::MAX as usize {
            return Err(TransferError::InvalidArgument(
                "len exceeds RDMA SGE length limit",
            ));
        }

        let wr_id = 1_u64;
        {
            let mut qp = session.qp.lock();
            let mut guard = qp.start_post_send();
            let wr = guard
                .construct_wr(wr_id, WorkRequestFlags::Signaled)
                .setup_write(remote_rkey, remote_ptr);
            unsafe {
                wr.setup_sge(local_mr.lkey(), local_ptr, len as u32);
            }
            guard
                .post()
                .map_err(|error| TransferError::Backend(error.to_string()))?;
        }
        Self::wait_send_completion(&session.send_cq, wr_id, Duration::from_secs(2))?;
        Ok(())
    }
}

impl RdmaBackend for SidewayBackend {
    fn initialize(&self, config: WorkerConfig) -> Result<()> {
        if config.nic_name.trim().is_empty() {
            return Err(TransferError::InvalidArgument("nic_name is empty"));
        }
        if config.rpc_port == 0 {
            return Err(TransferError::InvalidArgument("rpc_port must be non-zero"));
        }

        let runtime = Self::create_runtime(&config, Arc::clone(&self.state))?;
        let mut state = self.state.lock();
        state.config = Some(config);
        state.runtime = Some(runtime);
        state.registered.clear();
        state.sessions.clear();
        Ok(())
    }

    fn rpc_port(&self) -> Result<u16> {
        let state = self.state.lock();
        Ok(Self::ensure_initialized(&state)?.rpc_port)
    }

    fn session_id(&self) -> Result<String> {
        let state = self.state.lock();
        let runtime = state
            .runtime
            .as_ref()
            .ok_or(TransferError::NotInitialized)?;
        Ok(runtime.local_ud.to_hex())
    }

    fn register_memory(&self, ptr: u64, len: usize) -> Result<()> {
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
        Ok(())
    }

    fn unregister_memory(&self, ptr: u64) -> Result<()> {
        if ptr == 0 {
            return Err(TransferError::InvalidArgument("ptr must be non-zero"));
        }
        let mut state = self.state.lock();
        Self::ensure_initialized(&state)?;
        let removed = state.registered.remove(&ptr);
        if removed.is_none() {
            return Err(TransferError::MemoryNotRegistered { ptr });
        }
        Ok(())
    }

    fn transfer_sync_write(
        &self,
        session_id: &str,
        local_ptr: u64,
        remote_ptr: u64,
        len: usize,
    ) -> Result<usize> {
        if session_id.trim().is_empty() {
            return Err(TransferError::InvalidArgument("session_id is empty"));
        }
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
        let peer_ud = DomainAddressRaw::from_hex(session_id).ok_or(
            TransferError::InvalidArgument("session_id is not a valid DomainAddress hex"),
        )?;

        let (runtime, local_mr) = {
            let state = self.state.lock();
            Self::ensure_initialized(&state)?;
            let runtime = state
                .runtime
                .as_ref()
                .ok_or(TransferError::NotInitialized)?;
            let Some((_, _, local_mr)) = Self::find_registered_entry(&state, local_ptr, len) else {
                return Err(TransferError::MemoryNotRegistered { ptr: local_ptr });
            };
            (Arc::clone(runtime), local_mr)
        };

        let session = Self::ensure_active_session(&runtime, &self.state, peer_ud)?;
        let (remote_rkey, remote_available) =
            Self::query_remote_memory(&runtime, peer_ud, remote_ptr, len)?;
        if len > remote_available {
            return Err(TransferError::InvalidArgument(
                "len exceeds remote registered memory",
            ));
        }

        Self::post_write(&session, local_mr, local_ptr, remote_ptr, len, remote_rkey)?;
        Ok(len)
    }
}

fn bytes_to_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(char::from_digit((byte >> 4) as u32, 16).unwrap());
        out.push(char::from_digit((byte & 0x0f) as u32, 16).unwrap());
    }
    out
}

fn hex_to_bytes(s: &str) -> Option<Vec<u8>> {
    if s.is_empty() || s.len() % 2 != 0 {
        return None;
    }
    let mut out = Vec::with_capacity(s.len() / 2);
    let chars = s.as_bytes();
    let mut idx = 0;
    while idx < chars.len() {
        let hi = (chars[idx] as char).to_digit(16)? as u8;
        let lo = (chars[idx + 1] as char).to_digit(16)? as u8;
        out.push((hi << 4) | lo);
        idx += 2;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::{DomainAddressRaw, SidewayBackend};
    use crate::{api::WorkerConfig, backend::RdmaBackend, error::TransferError};

    #[test]
    fn domain_address_roundtrip() {
        let addr = DomainAddressRaw {
            gid: [7_u8; 16],
            lid: 123,
            qp_num: 456,
            qkey: 789,
        };
        let hex = addr.to_hex();
        let decoded = DomainAddressRaw::from_hex(&hex).expect("hex decode");
        assert_eq!(decoded, addr);
    }

    #[test]
    fn initialize_rejects_invalid_input() {
        let backend = SidewayBackend::new();
        let error = backend
            .initialize(WorkerConfig {
                bind_addr: "".to_string(),
                nic_name: "".to_string(),
                rpc_port: 50055,
            })
            .expect_err("must fail");
        assert_eq!(error, TransferError::InvalidArgument("nic_name is empty"));
    }
}
