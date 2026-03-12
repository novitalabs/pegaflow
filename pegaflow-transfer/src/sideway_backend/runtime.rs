use std::{collections::HashMap, io, mem::MaybeUninit, sync::Arc};

use log::{info, warn};
use parking_lot::Mutex;
use rdma_mummy_sys::{ibv_port_attr, ibv_query_port};
use sideway::ibverbs::{
    AccessFlags,
    address::Gid,
    completion::GenericCompletionQueue,
    device::{DeviceInfo, DeviceList},
    device_context::{DeviceContext, LinkLayer, Mtu, PortState},
    queue_pair::{GenericQueuePair, QueuePair, QueuePairType, SendOperationFlags},
};

use super::{
    ControlPlane, SidewayBackend, SidewayRuntime, SidewayState, UD_BUFFER_BYTES, UD_GRH_BYTES,
    UD_QKEY, UD_RECV_SLOTS, UdRecvSlot, UdSendSlot,
};
use crate::{
    api::WorkerConfig,
    domain_address::DomainAddress,
    error::{Result, TransferError},
};

impl SidewayBackend {
    pub(super) fn choose_port_and_gid(
        device_ctx: &Arc<DeviceContext>,
    ) -> Result<(u8, u8, LinkLayer, Mtu, Gid, u16, u8)> {
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

            let max_rd_atomic = (dev_attr.attr.orig_attr.max_qp_rd_atom as u8).max(1);
            return Ok((
                port_num,
                gid_index,
                port_attr.link_layer(),
                port_attr.active_mtu(),
                gid,
                raw_port.lid,
                max_rd_atomic,
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

    pub(super) fn create_runtime(
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

        let (port_num, gid_index, link_layer, mtu, local_gid, local_lid, max_rd_atomic) =
            Self::choose_port_and_gid(&device_ctx)?;
        info!(
            "transfer runtime selected port: nic={}, port={}, gid_index={}, link_layer={:?}, mtu={:?}, max_rd_atomic={}",
            config.nic_name, port_num, gid_index, link_layer, mtu, max_rd_atomic
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
            max_rd_atomic,
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
}
