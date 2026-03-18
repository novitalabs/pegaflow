use std::sync::Arc;

use log::warn;
use sideway::ibverbs::address::Gid;
use sideway::ibverbs::device::{DeviceInfo, DeviceList};
use sideway::ibverbs::device_context::{DeviceContext, Mtu, PortState};
use sideway::ibverbs::protection_domain::ProtectionDomain;

use crate::error::{Result, TransferError};

pub(crate) struct RcRuntime {
    pub(crate) device_ctx: Arc<DeviceContext>,
    pub(crate) pd: Arc<ProtectionDomain>,
    pub(crate) port_num: u8,
    pub(crate) gid_index: u8,
    pub(crate) mtu: Mtu,
    pub(crate) local_gid: Gid,
    pub(crate) max_rd_atomic: u8,
}

impl RcRuntime {
    pub(crate) fn open(nic_name: &str) -> Result<Arc<Self>> {
        log::info!("rc runtime open: nic={}", nic_name);
        let device_list =
            DeviceList::new().map_err(|error| TransferError::Backend(error.to_string()))?;
        let device = device_list
            .iter()
            .find(|device| device.name() == nic_name)
            .ok_or_else(|| TransferError::DeviceNotFound(nic_name.to_owned()))?;

        let device_ctx = device
            .open()
            .map_err(|error| TransferError::Backend(error.to_string()))?;
        let pd = device_ctx
            .alloc_pd()
            .map_err(|error| TransferError::Backend(error.to_string()))?;

        let (port_num, gid_index, mtu, local_gid, max_rd_atomic) =
            Self::choose_port_and_gid(&device_ctx)?;
        log::info!(
            "rc runtime ready: nic={}, port={}, gid_index={}, mtu={:?}, max_rd_atomic={}",
            nic_name,
            port_num,
            gid_index,
            mtu,
            max_rd_atomic
        );

        Ok(Arc::new(Self {
            device_ctx,
            pd,
            port_num,
            gid_index,
            mtu,
            local_gid,
            max_rd_atomic,
        }))
    }

    fn choose_port_and_gid(device_ctx: &Arc<DeviceContext>) -> Result<(u8, u8, Mtu, Gid, u8)> {
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

            // Public field through sideway's DeviceAttr (no direct rdma-mummy-sys dep needed).
            let max_rd_atomic = (dev_attr.attr.orig_attr.max_qp_rd_atom as u8).max(1);
            return Ok((
                port_num,
                gid_index,
                port_attr.active_mtu(),
                gid,
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
}
