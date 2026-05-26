use std::{borrow::Cow, ffi::CStr, sync::Arc};

use crate::v2::{
    error::{Result, VerbsError},
    provider::RdmaDomainInfo,
};

use crate::libibverbs_sys::{ibv_device, ibv_free_device_list, ibv_get_device_list};
use log::{info, warn};
use sideway::ibverbs::{
    address::GidType,
    device::{DeviceInfo, DeviceList},
    device_context::PortState,
};

pub struct VerbsDeviceList {
    pub list: *mut *mut ibv_device,
    pub num_devices: usize,
}

unsafe impl Send for VerbsDeviceList {}
unsafe impl Sync for VerbsDeviceList {}

impl VerbsDeviceList {
    pub fn get_all_devices() -> Result<Arc<Self>> {
        let mut num_devices = 0;
        let list = unsafe { ibv_get_device_list(&raw mut num_devices) };
        if list.is_null() {
            Err(VerbsError::with_last_os_error("ibv_get_device_list").into())
        } else {
            Ok(Arc::new(Self {
                list,
                num_devices: num_devices as usize,
            }))
        }
    }
}

impl Drop for VerbsDeviceList {
    fn drop(&mut self) {
        unsafe { ibv_free_device_list(self.list) };
    }
}

#[derive(Clone)]
pub struct VerbsDeviceInfo {
    pub device_list: Arc<VerbsDeviceList>,
    pub device_index: usize,
    pub port_num: u8,
    pub gid_index: u8,
}

impl VerbsDeviceInfo {
    pub fn new(device_list: Arc<VerbsDeviceList>, device_index: usize) -> Self {
        let device_name =
            unsafe { CStr::from_ptr((*(*device_list.list.add(device_index))).name.as_ptr()) }
                .to_string_lossy()
                .into_owned();
        let (port_num, gid_index) = choose_port_and_gid(&device_name).unwrap_or_else(|| {
            warn!(
                "Falling back to RDMA default port/gid for nic={device_name}: port=1 gid_index=0"
            );
            (1, 0)
        });
        Self {
            device_list,
            device_index,
            port_num,
            gid_index,
        }
    }

    pub fn device(&self) -> *mut ibv_device {
        unsafe { *self.device_list.list.add(self.device_index) }
    }
}

fn choose_port_and_gid(device_name: &str) -> Option<(u8, u8)> {
    let device_list = DeviceList::new()
        .map_err(|error| {
            warn!("Failed to query sideway RDMA device list for nic={device_name}: {error}");
        })
        .ok()?;
    let device = device_list
        .iter()
        .find(|device| device.name() == device_name)?;
    let device_ctx = device
        .open()
        .map_err(|error| {
            warn!("Failed to open RDMA device for gid selection nic={device_name}: {error}");
        })
        .ok()?;
    let dev_attr = device_ctx
        .query_device()
        .map_err(|error| {
            warn!("Failed to query RDMA device attributes nic={device_name}: {error}");
        })
        .ok()?;
    let gid_entries = device_ctx
        .query_gid_table()
        .map_err(|error| {
            warn!("Failed to query RDMA gid table nic={device_name}: {error}");
        })
        .ok()?;

    for port_num in 1..=dev_attr.phys_port_cnt() {
        let port_attr = match device_ctx.query_port(port_num) {
            Ok(port_attr) => port_attr,
            Err(error) => {
                warn!("Failed to query RDMA port nic={device_name} port={port_num}: {error}");
                continue;
            }
        };
        if port_attr.port_state() != PortState::Active {
            continue;
        }

        let port_entries = gid_entries
            .iter()
            .filter(|entry| entry.port_num() == port_num as u32 && !entry.gid().is_zero())
            .collect::<Vec<_>>();
        let picked = port_entries
            .iter()
            .find(|entry| {
                entry.gid_type() == GidType::RoceV2 && !entry.gid().is_unicast_link_local()
            })
            .or_else(|| {
                port_entries
                    .iter()
                    .find(|entry| entry.gid_type() == GidType::InfiniBand)
            })
            .or_else(|| {
                port_entries
                    .iter()
                    .find(|entry| !entry.gid().is_unicast_link_local())
            })
            .or_else(|| port_entries.first());

        if let Some(entry) = picked {
            let gid_index = entry.gid_index() as u8;
            info!(
                "Selected RDMA GID for nic={device_name}: port={port_num} link_layer={:?} gid_index={} gid_type={:?} gid={}",
                port_attr.link_layer(),
                gid_index,
                entry.gid_type(),
                entry.gid(),
            );
            return Some((port_num, gid_index));
        }

        warn!("Active RDMA port has no usable GID entries: nic={device_name} port={port_num}");
    }

    warn!("No active RDMA port with usable GID found for nic={device_name}");
    None
}

impl RdmaDomainInfo for VerbsDeviceInfo {
    fn name(&self) -> Cow<'_, str> {
        unsafe { CStr::from_ptr((*self.device()).name.as_ptr()).to_string_lossy() }
    }

    fn link_speed(&self) -> u64 {
        let path = format!(
            "{}/ports/{}/rate",
            unsafe { CStr::from_ptr((*self.device()).ibdev_path.as_ptr()).to_string_lossy() },
            self.port_num
        );
        match std::fs::read_to_string(path) {
            Ok(content) => {
                let trimmed = content.trim();
                let end_pos = trimmed.find(' ').unwrap_or(trimmed.len());
                let gbps: f64 = trimmed[..end_pos].parse().unwrap_or(0.0);
                (gbps * 1e9) as u64
            }
            Err(_) => 0,
        }
    }
}
