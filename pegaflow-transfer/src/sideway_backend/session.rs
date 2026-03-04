use std::{
    sync::{Arc, mpsc as std_mpsc},
    thread,
};

use log::{debug, info};
use parking_lot::Mutex;
use sideway::ibverbs::queue_pair::WorkRequestOperationType;

use super::{
    ActiveSession, CONTROL_TIMEOUT, RdmaOp, SessionCommand, SidewayBackend, SidewayRuntime,
    SidewayState,
};
use crate::{
    control_protocol::{ControlMessage, RcEndpoint},
    domain_address::DomainAddress,
    error::{Result, TransferError},
};

impl SidewayBackend {
    pub(super) fn ensure_passive_session(
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
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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

    pub(super) fn ensure_active_session(
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

        let (remote_rc, remote_memory_regions) = match reply {
            ControlMessage::ConnectResp {
                src_ud,
                rc,
                remote_memory_regions,
                ..
            } if &src_ud == peer_ud => (rc, remote_memory_regions),
            ControlMessage::ConnectRespErr { src_ud, error, .. } if &src_ud == peer_ud => {
                return Err(TransferError::Backend(format!(
                    "peer rejected connect request: {error:?}"
                )));
            }
            other => {
                return Err(TransferError::Backend(format!(
                    "unexpected connect response: {other:?}"
                )));
            }
        };
        Self::connect_rc_qp(runtime, &session, remote_rc)?;
        Self::spawn_session_worker(Arc::clone(&session), cmd_rx);

        let mut guard = state.lock();
        Self::cache_remote_memory_snapshot(&mut guard, peer_ud, &remote_memory_regions)?;
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
                        SessionCommand::SubmitBatch {
                            ops,
                            opcode,
                            done_tx,
                        } => {
                            let result = Self::execute_batch_rdma(&session, ops, opcode);
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

    pub(super) fn submit_session_batch(
        session: &ActiveSession,
        ops: Vec<RdmaOp>,
        opcode: WorkRequestOperationType,
    ) -> Result<usize> {
        let (done_tx, done_rx) = std_mpsc::channel();
        session
            .cmd_tx
            .send(SessionCommand::SubmitBatch {
                ops,
                opcode,
                done_tx,
            })
            .map_err(|_| {
                TransferError::Backend("session worker channel disconnected".to_string())
            })?;
        done_rx
            .recv()
            .map_err(|_| TransferError::Backend("session worker dropped completion".to_string()))?
    }
}
