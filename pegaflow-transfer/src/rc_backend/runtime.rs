use std::fs;
use std::sync::Arc;

use log::error;
use pegaflow_common::NumaNode;
use sideway::ibverbs::address::{Gid, GidType};
use sideway::ibverbs::device::{DeviceInfo, DeviceList};
use sideway::ibverbs::device_context::{DeviceContext, LinkLayer, Mtu, PortState};
use sideway::ibverbs::protection_domain::ProtectionDomain;

use crate::error::{Result, TransferError};
use crate::rdma_topo::nic_numa_node;

pub(crate) struct RcRuntime {
    pub(crate) device_ctx: Arc<DeviceContext>,
    pub(crate) pd: Arc<ProtectionDomain>,
    pub(crate) port_num: u8,
    pub(crate) link_layer: LinkLayer,
    pub(crate) local_lid: u16,
    pub(crate) gid_index: u8,
    pub(crate) mtu: Mtu,
    pub(crate) local_gid: Gid,
    pub(crate) numa_node: NumaNode,
    pub(crate) nic_name: String,
}

impl RcRuntime {
    pub(crate) fn open(nic_name: &str) -> Result<Arc<Self>> {
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

        let (port_num, link_layer, gid_index, mtu, local_gid) =
            Self::choose_port_and_gid(&device_ctx)?;
        let local_lid = match link_layer {
            LinkLayer::InfiniBand => Self::read_port_lid(nic_name, port_num)?,
            LinkLayer::Ethernet => 0,
            LinkLayer::Unspecified => {
                return Err(TransferError::Backend(format!(
                    "unsupported link layer on nic={nic_name}, port={port_num}: {link_layer:?}"
                )));
            }
        };
        let numa_node = nic_numa_node(nic_name);
        log::info!(
            "RDMA NIC ready: nic={}, port={}, link_layer={:?}, local_lid={}, gid_index={}, mtu={:?}, numa={}",
            nic_name,
            port_num,
            link_layer,
            local_lid,
            gid_index,
            mtu,
            numa_node,
        );

        Ok(Arc::new(Self {
            device_ctx,
            pd,
            port_num,
            link_layer,
            local_lid,
            gid_index,
            mtu,
            local_gid,
            numa_node,
            nic_name: nic_name.to_owned(),
        }))
    }

    fn choose_port_and_gid(
        device_ctx: &Arc<DeviceContext>,
    ) -> Result<(u8, LinkLayer, u8, Mtu, Gid)> {
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
            let link_layer = port_attr.link_layer();

            // Pick the best GID: RoCEv2 + IPv4-mapped > IB > any non-link-local.
            // RoCEv1 cannot be routed across L3 — only RoCEv2 works cross-machine.
            let port_entries: Vec<_> = gid_entries
                .iter()
                .filter(|e| e.port_num() == port_num as u32 && !e.gid().is_zero())
                .collect();

            let picked = port_entries
                .iter()
                .find(|e| e.gid_type() == GidType::RoceV2 && !e.gid().is_unicast_link_local())
                .or_else(|| {
                    port_entries
                        .iter()
                        .find(|e| e.gid_type() == GidType::InfiniBand)
                })
                .or_else(|| {
                    port_entries
                        .iter()
                        .find(|e| !e.gid().is_unicast_link_local())
                })
                .map(|e| (e.gid_index() as u8, e.gid()));

            let (gid_index, gid) = if let Some(picked) = picked {
                picked
            } else {
                let gid = device_ctx
                    .query_gid(port_num, 0)
                    .map_err(|error| TransferError::Backend(error.to_string()))?;
                (0, gid)
            };

            return Ok((port_num, link_layer, gid_index, port_attr.active_mtu(), gid));
        }

        error!(
            "no active port found on NIC {}; cannot initialize transfer runtime",
            device_ctx.name()
        );
        Err(TransferError::Backend(
            "no active port found on selected NIC".to_string(),
        ))
    }

    fn read_port_lid(nic_name: &str, port_num: u8) -> Result<u16> {
        let path = format!("/sys/class/infiniband/{nic_name}/ports/{port_num}/lid");
        let raw = fs::read_to_string(&path)
            .map_err(|error| TransferError::Backend(format!("failed to read {path}: {error}")))?;
        let text = raw.trim();
        let lid = if let Some(hex) = text.strip_prefix("0x") {
            u16::from_str_radix(hex, 16)
        } else {
            text.parse()
        }
        .map_err(|error| TransferError::Backend(format!("invalid LID in {path}: {error}")))?;
        if lid == 0 {
            return Err(TransferError::Backend(format!(
                "invalid zero LID for InfiniBand nic={nic_name}, port={port_num}"
            )));
        }
        Ok(lid)
    }
}
