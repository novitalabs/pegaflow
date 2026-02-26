// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! RDMA NIC topology detection
//!
//! Enumerates RDMA NICs via `/sys/class/infiniband` and maps them to NUMA nodes,
//! then combines with GPU and CPU topology to form a unified system view.
//!
//! This is the foundation for RDMA NIC selection: given a GPU, pick the NIC(s)
//! on the same NUMA node for lowest-latency RDMA transfers.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use pegaflow_core::NumaNode;
use pegaflow_core::numa::{format_cpu_list, get_gpu_numa_affinity, read_cpu_topology_from_sysfs};

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
    pub gpus: Vec<u32>,
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
        let nics = enumerate_rdma_nics();
        let gpu_affinity = get_gpu_numa_affinity();
        let cpu_topo = read_cpu_topology_from_sysfs().unwrap_or_default();

        // Collect all known NUMA node IDs
        let mut node_ids = BTreeSet::new();
        for (_, node) in &gpu_affinity {
            if node.is_valid() {
                node_ids.insert(node.0);
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

                let mut gpus: Vec<u32> = gpu_affinity
                    .iter()
                    .filter(|(_, n)| *n == node)
                    .map(|(dev, _)| *dev)
                    .collect();
                gpus.sort_unstable();

                let mut nics_on_node: Vec<RdmaNicInfo> = nics
                    .iter()
                    .filter(|nic| nic.numa_node == node)
                    .cloned()
                    .collect();
                nics_on_node.sort_by(|a, b| a.name.cmp(&b.name));

                let cpus = cpu_topo.get(&node_id).cloned().unwrap_or_default();

                NumaGroup {
                    node,
                    gpus,
                    nics: nics_on_node,
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
            if group.gpus.contains(&device_id) {
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
                let list: Vec<String> = group.gpus.iter().map(|g| g.to_string()).collect();
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

            log::info!("");
        }
    }
}

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
        let numa_path = device_link.join("numa_node");
        let numa_node = match fs::read_to_string(&numa_path) {
            Ok(s) => match s.trim().parse::<i32>() {
                Ok(n) if n >= 0 => NumaNode(n as u32),
                _ => NumaNode::UNKNOWN,
            },
            Err(_) => NumaNode::UNKNOWN,
        };

        nics.push(RdmaNicInfo {
            name,
            pci_addr,
            numa_node,
        });
    }

    nics.sort_by(|a, b| a.name.cmp(&b.name));
    nics
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_does_not_panic() {
        // Should work on any machine (may find 0 NICs / 0 GPUs)
        let topo = SystemTopology::detect();
        let _ = topo.groups();
    }

    #[test]
    fn nics_for_nonexistent_gpu_returns_empty() {
        let topo = SystemTopology::detect();
        assert!(topo.nics_for_gpu(9999).is_empty());
    }
}
