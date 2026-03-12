use std::{
    io,
    mem::MaybeUninit,
    ptr::{NonNull, null_mut},
    sync::{Arc, Weak, atomic::Ordering},
    thread,
    time::{Duration, Instant},
};

use log::{debug, warn};
use rdma_mummy_sys::{
    ibv_ah_attr, ibv_create_ah, ibv_global_route, ibv_post_send, ibv_send_flags, ibv_send_wr,
    ibv_sge, ibv_wr_opcode,
};
use sideway::ibverbs::{
    address::Gid,
    completion::{PollCompletionQueueError, WorkCompletionOperationType, WorkCompletionStatus},
    queue_pair::{QueuePair, SetScatterGatherEntry},
};

use super::{
    AddressHandle, ControlPlane, SidewayBackend, SidewayRuntime, SidewayState, UD_BUFFER_BYTES,
    UD_GRH_BYTES,
};
use crate::{
    control_protocol::{ConnectRespError, ControlMessage, decode_message, encode_message},
    domain_address::DomainAddress,
    error::{Result, TransferError},
};

// ---------------------------------------------------------------------------
// ControlPlane — request/reply matching for UD control messages
// ---------------------------------------------------------------------------

impl ControlPlane {
    pub(super) fn begin_request(&self) -> u64 {
        let request_id = self.next_request_id.fetch_add(1, Ordering::Relaxed) + 1;
        self.pending_replies.lock().insert(request_id, None);
        request_id
    }

    pub(super) fn complete_request(
        &self,
        request_id: u64,
        timeout: Duration,
    ) -> Option<ControlMessage> {
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

    pub(super) fn deliver_reply(&self, message: ControlMessage) {
        let request_id = message.request_id();
        let mut guard = self.pending_replies.lock();
        if let Some(slot) = guard.get_mut(&request_id) {
            *slot = Some(message);
            self.reply_cv.notify_all();
        }
    }
}

// ---------------------------------------------------------------------------
// UD messaging — send, recv, address handle, control loop
// ---------------------------------------------------------------------------

impl SidewayBackend {
    pub(super) fn post_ud_recv(runtime: &SidewayRuntime, slot_idx: usize) -> Result<()> {
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

    pub(super) fn send_control_message(
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
        state: &Arc<parking_lot::Mutex<SidewayState>>,
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
        state: &Arc<parking_lot::Mutex<SidewayState>>,
        message: ControlMessage,
    ) {
        match message {
            msg @ ControlMessage::ConnectResp { .. }
            | msg @ ControlMessage::ConnectRespErr { .. } => {
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
                let local_rc = match Self::ensure_passive_session(runtime, state, &src_ud, rc) {
                    Ok(local_rc) => local_rc,
                    Err(error) => {
                        warn!(
                            "control handle connect_req failed: request_id={}, peer={}, error={error}",
                            request_id, src_ud
                        );
                        return;
                    }
                };

                let remote_memory_regions = {
                    let state = state.lock();
                    Self::snapshot_registered_memory(&state)
                };
                let response = ControlMessage::ConnectResp {
                    request_id,
                    src_ud: runtime.local_ud.clone(),
                    rc: local_rc,
                    remote_memory_regions,
                };
                if encode_message(&response).len() > UD_BUFFER_BYTES {
                    warn!(
                        "connect_resp too large: request_id={}, peer={}, region_count={}, limit={}",
                        request_id,
                        src_ud,
                        match &response {
                            ControlMessage::ConnectResp {
                                remote_memory_regions,
                                ..
                            } => remote_memory_regions.len(),
                            _ => 0,
                        },
                        UD_BUFFER_BYTES
                    );
                    let _ = Self::send_control_message(
                        runtime,
                        &src_ud,
                        &ControlMessage::ConnectRespErr {
                            request_id,
                            src_ud: runtime.local_ud.clone(),
                            error: ConnectRespError::TooManyRegisteredMemoryRegions,
                        },
                    );
                    return;
                }
                if let Err(error) = Self::send_control_message(runtime, &src_ud, &response) {
                    warn!(
                        "control handle connect_req send failed: request_id={}, peer={}, error={error}",
                        request_id, src_ud
                    );
                }
            }
        }
    }

    pub(super) fn spawn_control_loop(
        runtime: Weak<SidewayRuntime>,
        state: Weak<parking_lot::Mutex<SidewayState>>,
    ) {
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
}

use sideway::ibverbs::device_context::LinkLayer;

// handle_ud_completion and handle_control_message are private — only called from spawn_control_loop
