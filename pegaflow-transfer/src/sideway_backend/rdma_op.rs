use std::collections::HashMap;

use sideway::ibverbs::{
    completion::{PollCompletionQueueError, WorkCompletionStatus},
    queue_pair::{
        PostSendGuard, QueuePair, SetScatterGatherEntry, WorkRequestFlags, WorkRequestOperationType,
    },
};

use super::{ActiveSession, MAX_INFLIGHT_OPS, RdmaOp, SidewayBackend};
use crate::error::{Result, TransferError};

impl SidewayBackend {
    fn post_single_rdma(
        session: &ActiveSession,
        op: &RdmaOp,
        wr_id: u64,
        opcode: WorkRequestOperationType,
    ) -> Result<()> {
        if op.len > u32::MAX as usize {
            return Err(TransferError::InvalidArgument(
                "len exceeds RDMA SGE length limit",
            ));
        }

        let mut qp = session.qp.lock();
        let mut guard = qp.start_post_send();
        let wr = guard.construct_wr(wr_id, WorkRequestFlags::Signaled);
        let buf = match opcode {
            WorkRequestOperationType::Write => wr.setup_write(op.remote_rkey, op.remote_ptr),
            WorkRequestOperationType::Read => wr.setup_read(op.remote_rkey, op.remote_ptr),
            _ => unreachable!("only Write and Read opcodes are used for RDMA transfers"),
        };
        unsafe {
            buf.setup_sge(op.local_mr.lkey(), op.local_ptr, op.len as u32);
        }
        guard
            .post()
            .map_err(|error| TransferError::Backend(error.to_string()))
    }

    pub(super) fn execute_batch_rdma(
        session: &ActiveSession,
        ops: Vec<RdmaOp>,
        opcode: WorkRequestOperationType,
    ) -> Result<usize> {
        let total_ops = ops.len();
        if total_ops == 0 {
            return Ok(0);
        }

        let mut next_idx = 0usize;
        let mut next_wr_id = 1_u64;
        let mut inflight: HashMap<u64, usize> = HashMap::new();
        let mut transferred = 0usize;

        while next_idx < total_ops || !inflight.is_empty() {
            while next_idx < total_ops && inflight.len() < MAX_INFLIGHT_OPS {
                Self::post_single_rdma(session, &ops[next_idx], next_wr_id, opcode)?;
                inflight.insert(next_wr_id, ops[next_idx].len);
                next_wr_id = next_wr_id.wrapping_add(1);
                next_idx += 1;
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
                        std::hint::spin_loop();
                    }
                }
                Err(PollCompletionQueueError::CompletionQueueEmpty) => {
                    std::hint::spin_loop();
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
}
