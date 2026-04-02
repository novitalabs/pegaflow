// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! RDMA NIC topology detection
//!
//! Enumerates RDMA NICs via `/sys/class/infiniband`, detects GPU PCI addresses,
//! and maps everything to NUMA nodes with PCIe hierarchy information.
//!
//! This is the foundation for RDMA NIC selection: given a GPU, pick the NIC(s)
//! on the same NUMA node for lowest-latency RDMA transfers.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;
use std::process::Command;

use pegaflow_common::{NumaNode, format_cpu_list, read_cpu_topology_from_sysfs};

/// GPU device information.
#[derive(Debug, Clone)]
pub struct GpuInfo {
    /// CUDA device ID (e.g. 0)
    pub device_id: u32,
    /// PCI address, e.g. "0000:19:00.0"
    pub pci_addr: String,
    /// NUMA node this GPU is attached to
    pub numa_node: NumaNode,
}

/// Single RDMA NIC information.
#[derive(Debug, Clone)]
pub struct RdmaNicInfo {
    /// Device name, e.g. "mlx5_0"
    pub name: String,
    /// PCI address, e.g. "0000:19:00.0"
    pub pci_addr: String,
    /// NUMA node this NIC is attached to
    pub numa_node: NumaNode,
}

/// NUMA-grouped topology: GPUs, NICs, and CPUs that share a NUMA node.
#[derive(Debug)]
pub struct NumaGroup {
    pub node: NumaNode,
    pub gpus: Vec<GpuInfo>,
    pub nics: Vec<RdmaNicInfo>,
    pub cpus: Vec<usize>,
}

/// Complete system topology: GPUs + NICs + CPUs grouped by NUMA node.
pub struct SystemTopology {
    groups: Vec<NumaGroup>,
}

impl SystemTopology {
    /// Detect full system topology from sysfs and nvidia-smi.
    pub fn detect() -> Self {
        let gpus = detect_gpus();
        let nics = enumerate_rdma_nics();
        let cpu_topo = read_cpu_topology_from_sysfs().unwrap_or_default();

        // Collect all known NUMA node IDs
        let mut node_ids = BTreeSet::new();
        for gpu in &gpus {
            if gpu.numa_node.is_valid() {
                node_ids.insert(gpu.numa_node.0);
            }
        }
        for nic in &nics {
            if nic.numa_node.is_valid() {
                node_ids.insert(nic.numa_node.0);
            }
        }
        for &node_id in cpu_topo.keys() {
            node_ids.insert(node_id);
        }

        // Build per-NUMA groups
        let groups = node_ids
            .into_iter()
            .map(|node_id| {
                let node = NumaNode(node_id);

                let mut group_gpus: Vec<GpuInfo> = gpus
                    .iter()
                    .filter(|g| g.numa_node == node)
                    .cloned()
                    .collect();
                group_gpus.sort_by_key(|g| g.device_id);

                let mut group_nics: Vec<RdmaNicInfo> = nics
                    .iter()
                    .filter(|nic| nic.numa_node == node)
                    .cloned()
                    .collect();
                group_nics.sort_by(|a, b| a.name.cmp(&b.name));

                let cpus = cpu_topo.get(&node_id).cloned().unwrap_or_default();

                NumaGroup {
                    node,
                    gpus: group_gpus,
                    nics: group_nics,
                    cpus,
                }
            })
            .collect();

        SystemTopology { groups }
    }

    /// Get NICs on the same NUMA node as the given GPU.
    ///
    /// Returns an empty slice if the GPU is not found or has no co-located NICs.
    pub fn nics_for_gpu(&self, device_id: u32) -> &[RdmaNicInfo] {
        for group in &self.groups {
            if group.gpus.iter().any(|g| g.device_id == device_id) {
                return &group.nics;
            }
        }
        &[]
    }

    /// Get all NUMA groups.
    pub fn groups(&self) -> &[NumaGroup] {
        &self.groups
    }

    /// Log a human-readable summary of the full system topology.
    pub fn log_summary(&self) {
        log::info!("=== PegaFlow System Topology ===");
        log::info!("");

        for group in &self.groups {
            log::info!("{}:", group.node);

            // GPUs
            if group.gpus.is_empty() {
                log::info!("  GPUs: (none)");
            } else {
                let list: Vec<String> =
                    group.gpus.iter().map(|g| g.device_id.to_string()).collect();
                log::info!("  GPUs: {}", list.join(", "));
            }

            // NICs
            if group.nics.is_empty() {
                log::info!("  NICs: (none)");
            } else {
                let list: Vec<String> = group
                    .nics
                    .iter()
                    .map(|n| format!("{} ({})", n.name, n.pci_addr))
                    .collect();
                log::info!("  NICs: {}", list.join(", "));
            }

            // CPUs
            if group.cpus.is_empty() {
                log::info!("  CPUs: (none)");
            } else {
                log::info!("  CPUs: {}", format_cpu_list(&group.cpus));
            }

            // PCIe: group devices by root port
            self.log_pcie_tree(group);

            log::info!("");
        }
    }

    fn log_pcie_tree(&self, group: &NumaGroup) {
        let mut by_root: BTreeMap<String, Vec<String>> = BTreeMap::new();

        for gpu in &group.gpus {
            if gpu.pci_addr.is_empty() {
                continue;
            }
            let path = get_pcie_path(&gpu.pci_addr);
            let root = path.first().cloned().unwrap_or_else(|| "?".into());
            let label = format!("GPU {}", gpu.device_id);
            by_root
                .entry(root)
                .or_default()
                .push(format_pci_device(&label, &gpu.pci_addr));
        }

        for nic in &group.nics {
            let path = get_pcie_path(&nic.pci_addr);
            let root = path.first().cloned().unwrap_or_else(|| "?".into());
            by_root
                .entry(root)
                .or_default()
                .push(format_pci_device(&nic.name, &nic.pci_addr));
        }

        if !by_root.is_empty() {
            log::info!("  PCIe:");
            for (root, devices) in &by_root {
                log::info!("    [{}] {}", root, devices.join(", "));
            }
        }
    }
}

// ============================================================================
// GPU detection
// ============================================================================

/// Detect all GPUs with PCI address and NUMA affinity.
///
/// Uses a single `nvidia-smi` call to get device index and PCI bus ID,
/// then reads NUMA node from sysfs.
fn detect_gpus() -> Vec<GpuInfo> {
    let output = match Command::new("nvidia-smi")
        .args(["--query-gpu=index,pci.bus_id", "--format=csv,noheader"])
        .output()
    {
        Ok(out) if out.status.success() => out,
        _ => return Vec::new(),
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut gpus = Vec::new();

    for line in stdout.lines() {
        let mut parts = line.splitn(2, ',');
        let (Some(idx_str), Some(addr_str)) = (parts.next(), parts.next()) else {
            continue;
        };

        let device_id: u32 = match idx_str.trim().parse() {
            Ok(id) => id,
            Err(_) => continue,
        };

        let pci_addr = normalize_pci_addr(addr_str);
        let numa_node = read_numa_node_sysfs(&pci_addr);

        gpus.push(GpuInfo {
            device_id,
            pci_addr,
            numa_node,
        });
    }

    gpus.sort_by_key(|g| g.device_id);
    gpus
}

// ============================================================================
// RDMA NIC detection
// ============================================================================

/// Enumerate all RDMA NICs from `/sys/class/infiniband`.
///
/// For each device:
/// - Reads the `device` symlink to extract the PCI address (basename of target)
/// - Reads `device/numa_node` for NUMA affinity
///
/// Skips devices with "bond" in the name.
fn enumerate_rdma_nics() -> Vec<RdmaNicInfo> {
    let ib_dir = Path::new("/sys/class/infiniband");
    if !ib_dir.exists() {
        log::debug!("/sys/class/infiniband not found, no RDMA NICs detected");
        return Vec::new();
    }

    let entries = match fs::read_dir(ib_dir) {
        Ok(entries) => entries,
        Err(e) => {
            log::warn!("Failed to read /sys/class/infiniband: {}", e);
            return Vec::new();
        }
    };

    let mut nics = Vec::new();

    for entry in entries.flatten() {
        let name = match entry.file_name().into_string() {
            Ok(n) => n,
            Err(_) => continue,
        };

        // Skip bond devices
        if name.contains("bond") {
            continue;
        }

        let device_link = entry.path().join("device");

        // Read PCI address from symlink target's basename
        let pci_addr = match fs::read_link(&device_link) {
            Ok(target) => target
                .file_name()
                .and_then(|f| f.to_str())
                .unwrap_or("")
                .to_string(),
            Err(_) => continue,
        };

        if pci_addr.is_empty() {
            continue;
        }

        // Read NUMA node (-1 or parse failure → UNKNOWN)
        let numa_node = read_numa_node_sysfs(&pci_addr);

        nics.push(RdmaNicInfo {
            name,
            pci_addr,
            numa_node,
        });
    }

    nics.sort_by(|a, b| a.name.cmp(&b.name));
    nics
}

// ============================================================================
// PCIe topology helpers
// ============================================================================

/// Get the full PCIe device path from root port to device.
///
/// Reads the canonical sysfs path of `/sys/bus/pci/devices/{addr}` and extracts
/// all PCI address components. The first element is the root port, the last is
/// the device itself, and intermediate elements are PCIe switch ports.
///
/// Returns an empty vec if the sysfs path cannot be resolved.
pub fn get_pcie_path(pci_addr: &str) -> Vec<String> {
    let device_path = format!("/sys/bus/pci/devices/{}", pci_addr);
    let canonical = match fs::canonicalize(&device_path) {
        Ok(p) => p,
        Err(_) => return Vec::new(),
    };

    let path_str = canonical.to_string_lossy();
    path_str
        .split('/')
        .filter(|s| {
            // PCI addresses look like "0000:19:00.0" — contain ':' and '.'
            // but skip "pci0000:00" domain root entries
            s.contains(':') && s.contains('.') && !s.starts_with("pci")
        })
        .map(|s| s.to_string())
        .collect()
}

/// Resolve the NUMA node for an RDMA NIC by name (e.g. "mlx5_0").
pub(crate) fn nic_numa_node(nic_name: &str) -> NumaNode {
    let device_link = format!("/sys/class/infiniband/{}/device", nic_name);
    let pci_addr = match fs::read_link(&device_link) {
        Ok(target) => target
            .file_name()
            .and_then(|f| f.to_str())
            .unwrap_or("")
            .to_string(),
        Err(_) => return NumaNode::UNKNOWN,
    };
    if pci_addr.is_empty() {
        return NumaNode::UNKNOWN;
    }
    read_numa_node_sysfs(&pci_addr)
}

/// Read NUMA node for a PCI device from sysfs.
fn read_numa_node_sysfs(pci_addr: &str) -> NumaNode {
    let numa_path = format!("/sys/bus/pci/devices/{}/numa_node", pci_addr);
    match fs::read_to_string(&numa_path) {
        Ok(s) => match s.trim().parse::<i32>() {
            Ok(n) if n >= 0 => NumaNode(n as u32),
            _ => NumaNode::UNKNOWN,
        },
        Err(_) => NumaNode::UNKNOWN,
    }
}

/// Normalize PCI address to sysfs 4-digit domain format.
///
/// nvidia-smi may return `00000000:19:00.0` (8-digit domain) or uppercase hex.
/// sysfs uses `0000:19:00.0` (4-digit, lowercase).
fn normalize_pci_addr(addr: &str) -> String {
    let addr = addr.trim().to_lowercase();
    let colon_count = addr.chars().filter(|&c| c == ':').count();
    match colon_count {
        2 => {
            // Has domain: DDDD:BB:DD.F or DDDDDDDD:BB:DD.F
            if let Some((domain, rest)) = addr.split_once(':') {
                if domain.len() > 4 {
                    format!("{}:{}", &domain[domain.len() - 4..], rest)
                } else {
                    addr
                }
            } else {
                addr
            }
        }
        1 => {
            // No domain: BB:DD.F → 0000:BB:DD.F
            format!("0000:{}", addr)
        }
        _ => addr,
    }
}

/// Format a PCI device label with address and optional PCIe link info.
///
/// Returns `"name (addr, PCIe X.0 xN)"` when link info is available,
/// or `"name (addr)"` otherwise.
fn format_pci_device(name: &str, pci_addr: &str) -> String {
    match read_pcie_link_info(pci_addr) {
        Some(link) => format!("{} ({}, {})", name, pci_addr, link),
        None => format!("{} ({})", name, pci_addr),
    }
}

/// GT/s prefix to PCIe generation mapping.
///
/// sysfs `current_link_speed` reports values like "16.0 GT/s PCIe".
/// We match the leading numeric token as a string — no floating-point needed.
const PCIE_GEN_TABLE: &[(&str, &str)] = &[
    ("2.5", "1.0"),
    ("5.0", "2.0"),
    ("8.0", "3.0"),
    ("16.0", "4.0"),
    ("32.0", "5.0"),
    ("64.0", "6.0"),
];

/// Map a sysfs `current_link_speed` string (e.g. "16.0 GT/s PCIe") to a PCIe
/// generation string (e.g. "4.0"). Returns "?" for unrecognised values.
fn gts_to_pcie_gen(speed: &str) -> &'static str {
    let token = speed.split_whitespace().next().unwrap_or("");
    for &(gts, pcie_gen) in PCIE_GEN_TABLE {
        if token == gts {
            return pcie_gen;
        }
    }
    "?"
}

/// Read PCIe link speed and width from sysfs, returning a formatted string
/// like "PCIe 4.0 x16". Returns `None` if sysfs files are missing or
/// unreadable — never panics.
fn read_pcie_link_info(pci_addr: &str) -> Option<String> {
    let base = format!("/sys/bus/pci/devices/{}", pci_addr);
    // Use max_link_speed/max_link_width: current_* reflects power-managed state
    // and reports PCIe 1.0 for idle GPUs, which is misleading.
    let speed_raw = fs::read_to_string(format!("{}/max_link_speed", base)).ok()?;
    let width_raw = fs::read_to_string(format!("{}/max_link_width", base)).ok()?;

    let pcie_gen = gts_to_pcie_gen(speed_raw.trim());
    let width = width_raw.trim();

    // Skip if generation is unrecognised or width is empty
    if pcie_gen == "?" || width.is_empty() {
        return None;
    }

    Some(format!("PCIe {} x{}", pcie_gen, width))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_does_not_panic() {
        let topo = SystemTopology::detect();
        let _ = topo.groups();
    }

    #[test]
    fn nics_for_nonexistent_gpu_returns_empty() {
        let topo = SystemTopology::detect();
        assert!(topo.nics_for_gpu(9999).is_empty());
    }

    #[test]
    fn normalize_8digit_domain() {
        assert_eq!(normalize_pci_addr("00000000:19:00.0"), "0000:19:00.0");
    }

    #[test]
    fn normalize_4digit_domain() {
        assert_eq!(normalize_pci_addr("0000:19:00.0"), "0000:19:00.0");
    }

    #[test]
    fn normalize_no_domain() {
        assert_eq!(normalize_pci_addr("19:00.0"), "0000:19:00.0");
    }

    #[test]
    fn normalize_uppercase() {
        assert_eq!(normalize_pci_addr("00000000:2A:00.0"), "0000:2a:00.0");
    }

    #[test]
    fn normalize_whitespace() {
        assert_eq!(normalize_pci_addr("  00000000:19:00.0 "), "0000:19:00.0");
    }

    #[test]
    fn gts_to_pcie_gen_known_speeds() {
        assert_eq!(gts_to_pcie_gen("2.5 GT/s PCIe"), "1.0");
        assert_eq!(gts_to_pcie_gen("5.0 GT/s PCIe"), "2.0");
        assert_eq!(gts_to_pcie_gen("8.0 GT/s PCIe"), "3.0");
        assert_eq!(gts_to_pcie_gen("16.0 GT/s PCIe"), "4.0");
        assert_eq!(gts_to_pcie_gen("32.0 GT/s PCIe"), "5.0");
        assert_eq!(gts_to_pcie_gen("64.0 GT/s PCIe"), "6.0");
    }

    #[test]
    fn gts_to_pcie_gen_unknown_speed() {
        assert_eq!(gts_to_pcie_gen("unknown"), "?");
        assert_eq!(gts_to_pcie_gen(""), "?");
    }

    #[test]
    fn read_pcie_link_info_nonexistent_device() {
        // Non-existent PCI address should return None, not panic
        assert!(read_pcie_link_info("ffff:ff:ff.f").is_none());
    }
}
