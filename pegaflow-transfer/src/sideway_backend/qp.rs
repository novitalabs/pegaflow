use std::{io, mem::MaybeUninit};

use log::debug;
use rdma_mummy_sys::{ibv_modify_qp, ibv_qp_attr, ibv_qp_attr_mask, ibv_qp_state};
use sideway::ibverbs::{
    AccessFlags,
    address::{AddressHandleAttribute, Gid},
    completion::GenericCompletionQueue,
    queue_pair::{GenericQueuePair, QueuePair, QueuePairAttribute, QueuePairState, QueuePairType},
};

use super::{ActiveSession, SidewayBackend, SidewayRuntime, UD_QKEY};
use crate::{
    control_protocol::RcEndpoint,
    error::{Result, TransferError},
};

impl SidewayBackend {
    pub(super) fn setup_ud_qp(qp: &mut GenericQueuePair, port_num: u8) -> Result<()> {
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
            ibv_modify_qp(
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
        let ret = unsafe { ibv_modify_qp(qp_ptr, &raw mut rts_attr, rts_mask) };
        if ret != 0 {
            return Err(TransferError::Backend(format!(
                "UD QP RTR->RTS failed: {}",
                io::Error::from_raw_os_error(ret)
            )));
        }
        Ok(())
    }

    pub(super) fn create_rc_qp(
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

    pub(super) fn connect_rc_qp(
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
            .setup_max_dest_read_atomic(runtime.max_rd_atomic)
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
            .setup_max_read_atomic(runtime.max_rd_atomic);
        qp.modify(&rts_attr)
            .map_err(|error| TransferError::Backend(error.to_string()))?;
        debug!(
            "rc connect ready: local_qpn={}, remote_qpn={}",
            session.local_rc.qp_num, remote_rc.qp_num
        );
        Ok(())
    }
}
