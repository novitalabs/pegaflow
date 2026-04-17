use std::collections::HashMap;
use std::sync::Arc;
use std::sync::mpsc as std_mpsc;

use mea::oneshot;
use std::thread;

use log::{debug, warn};
use parking_lot::Mutex;
use pegaflow_common::NumaNode;
use sideway::ibverbs::AccessFlags;
use sideway::ibverbs::address::{AddressHandleAttribute, Gid};
use sideway::ibverbs::completion::{
    GenericCompletionQueue, PollCompletionQueueError, WorkCompletionStatus,
};
use sideway::ibverbs::memory_region::MemoryRegion;
use sideway::ibverbs::queue_pair::{
    GenericQueuePair, PostSendGuard, QueuePair, QueuePairAttribute, QueuePairState, QueuePairType,
    SetScatterGatherEntry, WorkRequestFlags,
};

use super::runtime::RcRuntime;
use crate::engine::{RcEndpoint, TransferOp};
use crate::error::{Result, TransferError};

const MAX_WR_CHAIN_OPS: usize = 4;
const MAX_SEND_WR: u32 = 128;
// One QP per CQ; all WRs are signaled, so CQ depth = SQ depth suffices.
const SEND_CQ_SIZE: u32 = MAX_SEND_WR;
// Recv queue unused (one-sided RDMA only), keep minimal for driver compatibility.
const RECV_CQ_SIZE: u32 = 2;
const MAX_RECV_WR: u32 = 1;
const MAX_RD_ATOMIC: u8 = 16;
const PSN_MASK: u32 = 0x00ff_ffff;

pub(crate) struct RdmaOp {
    pub(crate) local_mr: Arc<MemoryRegion>,
    pub(crate) local_ptr: u64,
    pub(crate) remote_ptr: u64,
    pub(crate) len: usize,
    pub(crate) remote_rkey: u32,
}

enum SessionCommand {
    Transfer {
        ops: Vec<RdmaOp>,
        op: TransferOp,
        done_tx: oneshot::Sender<Result<usize>>,
    },
}

pub(crate) struct RcSession {
    qp: Mutex<GenericQueuePair>,
    send_cq: GenericCompletionQueue,
    _recv_cq: GenericCompletionQueue,
    pub(crate) local_endpoint: RcEndpoint,
    cmd_tx: std_mpsc::Sender<SessionCommand>,
}

impl RcSession {
    /// Create an RC QP in INIT state and return the session (not yet connected).
    pub(crate) fn create(runtime: &RcRuntime, psn_seed: u64) -> Result<Arc<Self>> {
        let mut cq_builder = runtime.device_ctx.create_cq_builder();
        cq_builder.setup_cqe(SEND_CQ_SIZE);
        let send_cq: GenericCompletionQueue = cq_builder
            .build()
            .map_err(|error| TransferError::Backend(error.to_string()))?
            .into();
        cq_builder.setup_cqe(RECV_CQ_SIZE);
        let recv_cq: GenericCompletionQueue = cq_builder
            .build()
            .map_err(|error| TransferError::Backend(error.to_string()))?
            .into();

        let mut qp_builder = runtime.pd.create_qp_builder();
        qp_builder
            .setup_qp_type(QueuePairType::ReliableConnection)
            .setup_send_cq(send_cq.clone())
            .setup_recv_cq(recv_cq.clone())
            .setup_max_send_wr(MAX_SEND_WR)
            .setup_max_recv_wr(MAX_RECV_WR)
            .setup_max_send_sge(1)
            .setup_max_recv_sge(0);
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

        let local_psn = (psn_seed as u32) & PSN_MASK;
        let local_endpoint = RcEndpoint {
            gid: runtime.local_gid.raw,
            lid: 0, // RoCE v2 doesn't use LID
            qp_num: qp.qp_number(),
            psn: local_psn,
        };

        let (cmd_tx, cmd_rx) = std_mpsc::channel();
        let session = Arc::new(Self {
            qp: Mutex::new(qp),
            send_cq,
            _recv_cq: recv_cq,
            local_endpoint,
            cmd_tx,
        });

        Self::spawn_worker(Arc::clone(&session), cmd_rx, runtime.numa_node)?;
        Ok(session)
    }

    /// Connect this QP to the remote peer (INIT → RTR → RTS).
    pub(crate) fn connect(&self, runtime: &RcRuntime, remote: &RcEndpoint) -> Result<()> {
        debug!(
            "rc connect start: local_qpn={}, remote_qpn={}, remote_gid={:?}",
            self.local_endpoint.qp_num, remote.qp_num, remote.gid
        );
        let mut ah_attr = AddressHandleAttribute::new();
        ah_attr
            .setup_dest_lid(remote.lid)
            .setup_port(runtime.port_num)
            .setup_grh_dest_gid(&Gid { raw: remote.gid })
            .setup_grh_src_gid_index(runtime.gid_index)
            .setup_grh_hop_limit(64);

        let mut qp = self.qp.lock();
        let mut rtr_attr = QueuePairAttribute::new();
        rtr_attr
            .setup_state(QueuePairState::ReadyToReceive)
            .setup_path_mtu(runtime.mtu)
            .setup_dest_qp_num(remote.qp_num)
            .setup_rq_psn(remote.psn)
            .setup_max_dest_read_atomic(MAX_RD_ATOMIC)
            .setup_min_rnr_timer(12)
            .setup_address_vector(&ah_attr);
        qp.modify(&rtr_attr)
            .map_err(|error| TransferError::Backend(error.to_string()))?;

        let mut rts_attr = QueuePairAttribute::new();
        rts_attr
            .setup_state(QueuePairState::ReadyToSend)
            .setup_sq_psn(self.local_endpoint.psn)
            .setup_timeout(14)
            .setup_retry_cnt(7)
            .setup_rnr_retry(7)
            .setup_max_read_atomic(MAX_RD_ATOMIC);
        qp.modify(&rts_attr)
            .map_err(|error| TransferError::Backend(error.to_string()))?;
        debug!(
            "rc connect ready: local_qpn={}, remote_qpn={}",
            self.local_endpoint.qp_num, remote.qp_num
        );
        Ok(())
    }

    /// Submit a batch of RDMA ops to the session worker and return a receiver for the result.
    pub(crate) fn transfer_batch_async(
        &self,
        ops: Vec<RdmaOp>,
        op: TransferOp,
    ) -> Result<oneshot::Receiver<Result<usize>>> {
        let (done_tx, done_rx) = oneshot::channel();
        self.cmd_tx
            .send(SessionCommand::Transfer { ops, op, done_tx })
            .map_err(|_| {
                TransferError::Backend("session worker channel disconnected".to_string())
            })?;
        Ok(done_rx)
    }

    fn spawn_worker(
        session: Arc<Self>,
        cmd_rx: std_mpsc::Receiver<SessionCommand>,
        numa_node: NumaNode,
    ) -> Result<()> {
        thread::Builder::new()
            .name("pegaflow-rc-session".to_string())
            .spawn(move || {
                if numa_node.is_valid()
                    && let Err(e) = pegaflow_common::pin_thread_to_numa_node(numa_node)
                {
                    warn!("Failed to pin rc session worker to {}: {}", numa_node, e);
                }
                debug!(
                    "session worker started: local_qpn={}, numa={}",
                    session.local_endpoint.qp_num, numa_node
                );
                while let Ok(command) = cmd_rx.recv() {
                    match command {
                        SessionCommand::Transfer { ops, op, done_tx } => {
                            let result = Self::execute_batch(&session, ops, op);
                            if done_tx.send(result).is_err() {
                                debug!(
                                    "session worker reply receiver dropped: local_qpn={}",
                                    session.local_endpoint.qp_num
                                );
                            }
                        }
                    }
                }
                debug!(
                    "session worker stopped: local_qpn={}",
                    session.local_endpoint.qp_num
                );
            })
            .map_err(|e| TransferError::Backend(format!("failed to spawn session worker: {e}")))?;
        Ok(())
    }

    fn post_rdma_wr_chain(
        qp: &mut GenericQueuePair,
        ops: &[RdmaOp],
        first_wr_id: u64,
        op: TransferOp,
    ) -> Result<usize> {
        if ops.is_empty() {
            return Ok(0);
        }
        let mut guard = qp.start_post_send();
        for (idx, rdma_op) in ops.iter().enumerate() {
            if rdma_op.len > u32::MAX as usize {
                return Err(TransferError::InvalidArgument(
                    "len exceeds RDMA SGE length limit",
                ));
            }
            let wr = guard.construct_wr(
                first_wr_id.wrapping_add(idx as u64),
                WorkRequestFlags::Signaled,
            );
            let handle = match op {
                TransferOp::Write => wr.setup_write(rdma_op.remote_rkey, rdma_op.remote_ptr),
                TransferOp::Read => wr.setup_read(rdma_op.remote_rkey, rdma_op.remote_ptr),
            };
            unsafe {
                handle.setup_sge(
                    rdma_op.local_mr.lkey(),
                    rdma_op.local_ptr,
                    rdma_op.len as u32,
                );
            }
        }
        guard
            .post()
            .map_err(|e| TransferError::Backend(e.to_string()))?;
        Ok(ops.len())
    }

    fn execute_batch(session: &Self, ops: Vec<RdmaOp>, op: TransferOp) -> Result<usize> {
        let total_ops = ops.len();
        if total_ops == 0 {
            return Ok(0);
        }

        let mut next_idx = 0usize;
        let mut next_wr_id = 1_u64;
        let mut inflight: HashMap<u64, usize> = HashMap::new();
        let mut transferred = 0usize;

        while next_idx < total_ops || !inflight.is_empty() {
            while next_idx < total_ops && inflight.len() < MAX_SEND_WR as usize {
                let available = MAX_SEND_WR as usize - inflight.len();
                let remaining = total_ops - next_idx;
                let chain_len = MAX_WR_CHAIN_OPS.min(available).min(remaining);
                let mut qp = session.qp.lock();
                let posted = Self::post_rdma_wr_chain(
                    &mut qp,
                    &ops[next_idx..next_idx + chain_len],
                    next_wr_id,
                    op,
                )?;
                drop(qp);
                if posted == 0 {
                    break;
                }
                for rdma_op in &ops[next_idx..next_idx + posted] {
                    inflight.insert(next_wr_id, rdma_op.len);
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
                                "send completion failed: local_qpn={}, status={}, opcode={}, vendor_err={}",
                                session.local_endpoint.qp_num,
                                wc.status(),
                                wc.opcode(),
                                wc.vendor_err()
                            )));
                        }
                        transferred = transferred.saturating_add(bytes);
                    }
                    if !did_work {
                        std::hint::spin_loop();
                    }
                }
                Err(PollCompletionQueueError::CompletionQueueEmpty) => {
                    std::hint::spin_loop();
                }
                Err(error) => {
                    return Err(TransferError::Backend(format!(
                        "poll send CQ failed: local_qpn={}, {error}",
                        session.local_endpoint.qp_num
                    )));
                }
            }
        }

        Ok(transferred)
    }
}
