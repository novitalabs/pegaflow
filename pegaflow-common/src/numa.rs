// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NUMA (Non-Uniform Memory Access) utilities
//!
//! This module provides:
//! - NUMA node abstraction (`NumaNode`)
//! - System NUMA topology detection (CPU-to-node mapping)
//! - GPU to NUMA node affinity detection
//! - Thread-to-NUMA-node pinning for first-touch allocation policy

use std::collections::HashMap;
use std::fs;
use std::mem;
use std::process::Command;

/// Represents a NUMA node identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NumaNode(pub u32);

impl NumaNode {
    /// Represents an unknown or invalid NUMA node
    pub const UNKNOWN: NumaNode = NumaNode(u32::MAX);

    /// Check if this is the unknown node
    pub fn is_unknown(&self) -> bool {
        self.0 == u32::MAX
    }

    /// Check if this is a valid NUMA node
    pub fn is_valid(&self) -> bool {
        self.0 != u32::MAX
    }
}

impl std::fmt::Display for NumaNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_unknown() {
            write!(f, "UNKNOWN")
        } else {
            write!(f, "NUMA{}", self.0)
        }
    }
}

/// Format a list of CPUs into a compact range representation
///
/// Example: [0, 1, 2, 3, 8, 9, 10] -> "0-3,8-10"
pub fn format_cpu_list(cpus: &[usize]) -> String {
    if cpus.is_empty() {
        return String::new();
    }

    let mut result = Vec::new();
    let mut start = cpus[0];
    let mut prev = cpus[0];

    for &cpu in &cpus[1..] {
        if cpu == prev + 1 {
            prev = cpu;
        } else {
            if start == prev {
                result.push(format!("{}", start));
            } else {
                result.push(format!("{}-{}", start, prev));
            }
            start = cpu;
            prev = cpu;
        }
    }

    if start == prev {
        result.push(format!("{}", start));
    } else {
        result.push(format!("{}-{}", start, prev));
    }

    result.join(",")
}

// ============================================================================
// CPU Topology from sysfs
// ============================================================================

/// Read CPU-to-NUMA mapping from sysfs
///
/// Returns a map of NUMA node ID -> list of CPU IDs.
pub fn read_cpu_topology_from_sysfs() -> Result<HashMap<u32, Vec<usize>>, String> {
    let mut node_to_cpus: HashMap<u32, Vec<usize>> = HashMap::new();

    let node_dir = std::path::Path::new("/sys/devices/system/node");
    if !node_dir.exists() {
        return Err("NUMA not supported: /sys/devices/system/node not found".to_string());
    }

    let entries =
        fs::read_dir(node_dir).map_err(|e| format!("Failed to read node directory: {}", e))?;

    for entry in entries.flatten() {
        let path = entry.path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

        if !name.starts_with("node") {
            continue;
        }

        let node_id: u32 = name[4..]
            .parse()
            .map_err(|_| format!("Invalid node directory name: {}", name))?;

        let cpulist_path = path.join("cpulist");
        if !cpulist_path.exists() {
            continue;
        }

        let cpulist = fs::read_to_string(&cpulist_path)
            .map_err(|e| format!("Failed to read {}: {}", cpulist_path.display(), e))?;

        let cpus = parse_cpulist(cpulist.trim())?;
        node_to_cpus.insert(node_id, cpus);
    }

    if node_to_cpus.is_empty() {
        return Err("No NUMA nodes found".to_string());
    }

    Ok(node_to_cpus)
}

/// Parse Linux cpulist format (e.g. "0-3,8-11" -> [0,1,2,3,8,9,10,11])
fn parse_cpulist(cpulist: &str) -> Result<Vec<usize>, String> {
    let mut cpus = Vec::new();

    if cpulist.is_empty() {
        return Ok(cpus);
    }

    for part in cpulist.split(',') {
        if part.contains('-') {
            let range: Vec<&str> = part.split('-').collect();
            if range.len() != 2 {
                return Err(format!("Invalid CPU range format: {}", part));
            }

            let start: usize = range[0]
                .parse()
                .map_err(|_| format!("Invalid CPU ID: {}", range[0]))?;
            let end: usize = range[1]
                .parse()
                .map_err(|_| format!("Invalid CPU ID: {}", range[1]))?;

            for cpu in start..=end {
                cpus.push(cpu);
            }
        } else {
            let cpu: usize = part
                .parse()
                .map_err(|_| format!("Invalid CPU ID: {}", part))?;
            cpus.push(cpu);
        }
    }

    cpus.sort_unstable();
    cpus.dedup();

    Ok(cpus)
}

// ============================================================================
// Thread pinning
// ============================================================================

/// Pin the current thread to CPUs on a specific NUMA node.
///
/// Sets the CPU affinity of the calling thread to only run on CPUs
/// belonging to the specified NUMA node. Critical for ensuring
/// first-touch memory allocations land on the correct node.
pub fn pin_thread_to_numa_node(node: NumaNode) -> Result<(), String> {
    if node.is_unknown() {
        return Err("Cannot pin to unknown NUMA node".to_string());
    }

    let node_to_cpus = read_cpu_topology_from_sysfs()
        .map_err(|e| format!("Failed to get NUMA topology: {}", e))?;

    let cpus = node_to_cpus
        .get(&node.0)
        .ok_or_else(|| format!("No CPUs found for NUMA node {}", node.0))?;

    if cpus.is_empty() {
        return Err(format!("CPU list is empty for NUMA node {}", node.0));
    }

    // SAFETY: cpu_set_t is a plain C struct safe to zero-initialize. CPU_SET
    // writes to valid indices. sched_setaffinity(0, ...) targets the calling
    // thread; the cpu_set is correctly sized.
    unsafe {
        let mut cpu_set: libc::cpu_set_t = mem::zeroed();

        for cpu in cpus {
            libc::CPU_SET(*cpu, &mut cpu_set);
        }

        let result = libc::sched_setaffinity(
            0, // current thread
            mem::size_of::<libc::cpu_set_t>(),
            &cpu_set,
        );

        if result != 0 {
            let err = std::io::Error::last_os_error();
            return Err(format!("sched_setaffinity failed: {}", err));
        }
    }

    Ok(())
}

/// Run a closure on a thread pinned to a specific NUMA node.
///
/// Spawns a temporary thread, pins it to the specified NUMA node,
/// runs the closure, and returns the result. Useful for first-touch
/// memory allocation policy.
pub fn run_on_numa<T, F>(node: NumaNode, f: F) -> Result<T, String>
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    if node.is_unknown() {
        return Err("Cannot run on unknown NUMA node".to_string());
    }

    let (tx, rx) = std::sync::mpsc::channel();

    let handle = std::thread::Builder::new()
        .name(format!("numa{}-init", node.0))
        .spawn(move || {
            if let Err(e) = pin_thread_to_numa_node(node) {
                let _ = tx.send(Err(e));
                return;
            }

            let result = f();
            let _ = tx.send(Ok(result));
        })
        .map_err(|e| format!("Failed to spawn NUMA thread: {}", e))?;

    let result = rx
        .recv()
        .map_err(|_| "NUMA thread panicked or closed channel".to_string())?;

    handle
        .join()
        .map_err(|_| "NUMA thread panicked".to_string())?;

    result
}

// ============================================================================
// GPU NUMA affinity
// ============================================================================

/// Get the NUMA node for a GPU device via nvidia-smi.
///
/// Returns `NumaNode::UNKNOWN` if nvidia-smi is unavailable or fails.
fn get_device_numa_node(device_id: u32) -> NumaNode {
    let output = match Command::new("nvidia-smi")
        .args([
            "topo",
            "--get-numa-id-of-nearby-cpu",
            "-i",
            &device_id.to_string(),
        ])
        .output()
    {
        Ok(out) if out.status.success() => out,
        _ => {
            return NumaNode::UNKNOWN;
        }
    };

    if let Ok(stdout) = std::str::from_utf8(&output.stdout)
        && let Some(line) = stdout.lines().next()
        && let Some(numa_str) = line.split(':').nth(1)
        && let Ok(node) = numa_str.trim().parse::<u32>()
    {
        return NumaNode(node);
    }

    NumaNode::UNKNOWN
}

/// Get NUMA affinity for all available GPUs.
///
/// Returns (device_id, numa_node) pairs. Empty if nvidia-smi is unavailable.
fn get_gpu_numa_affinity() -> Vec<(u32, NumaNode)> {
    let output = match Command::new("nvidia-smi")
        .args(["--query-gpu=count", "--format=csv,noheader"])
        .output()
    {
        Ok(out) if out.status.success() => out,
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            log::warn!("nvidia-smi failed: {}", stderr);
            return Vec::new();
        }
        Err(e) => {
            log::warn!("nvidia-smi not found or failed to execute: {}", e);
            return Vec::new();
        }
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    let count_str = stdout.lines().next().map(|s| s.trim()).unwrap_or("");
    let count: u32 = match count_str.parse::<u32>() {
        Ok(n) => n,
        Err(e) => {
            log::warn!("Failed to parse GPU count '{}': {}", count_str, e);
            return Vec::new();
        }
    };

    (0..count)
        .map(|device_id| (device_id, get_device_numa_node(device_id)))
        .collect()
}

// ============================================================================
// NumaTopology
// ============================================================================

/// GPU-to-NUMA topology for the system.
///
/// Built once during engine initialization. Provides efficient lookup
/// of NUMA affinity for GPU devices.
#[derive(Debug, Clone)]
pub struct NumaTopology {
    gpu_numa_map: HashMap<i32, NumaNode>,
    numa_nodes: Vec<NumaNode>,
}

impl NumaTopology {
    /// Detect and build the GPU-NUMA topology.
    ///
    /// Queries nvidia-smi for GPU NUMA affinity and reads system NUMA topology.
    pub fn detect() -> Self {
        let gpu_affinity = get_gpu_numa_affinity();
        let gpu_numa_map: HashMap<i32, NumaNode> = gpu_affinity
            .into_iter()
            .map(|(dev, node)| (dev as i32, node))
            .collect();

        let numa_nodes = match read_cpu_topology_from_sysfs() {
            Ok(node_to_cpus) => {
                let mut node_ids: Vec<u32> = node_to_cpus.keys().copied().collect();
                node_ids.sort_unstable();
                node_ids.into_iter().map(NumaNode).collect()
            }
            Err(_) => {
                vec![NumaNode(0)]
            }
        };

        Self {
            gpu_numa_map,
            numa_nodes,
        }
    }

    /// Get the preferred NUMA node for a GPU device.
    ///
    /// Returns `NumaNode::UNKNOWN` if the device is not found.
    pub fn numa_for_gpu(&self, device_id: i32) -> NumaNode {
        self.gpu_numa_map
            .get(&device_id)
            .copied()
            .unwrap_or(NumaNode::UNKNOWN)
    }

    /// Get all NUMA nodes in the system.
    pub fn numa_nodes(&self) -> &[NumaNode] {
        &self.numa_nodes
    }

    /// Get the number of NUMA nodes.
    pub fn num_nodes(&self) -> usize {
        self.numa_nodes.len()
    }

    /// Check if this is a multi-NUMA system.
    pub fn is_multi_numa(&self) -> bool {
        self.numa_nodes.len() > 1
    }

    /// Log the detected topology.
    pub fn log_summary(&self) {
        log::info!("=== GPU-NUMA Topology ===");
        log::info!("NUMA nodes: {}", self.num_nodes());

        if self.gpu_numa_map.is_empty() {
            log::warn!("No GPU NUMA affinity detected (nvidia-smi unavailable?)");
        } else {
            let mut devices: Vec<_> = self.gpu_numa_map.iter().collect();
            devices.sort_by_key(|(dev, _)| *dev);
            for (dev, node) in devices {
                log::info!("  GPU {} -> {}", dev, node);
            }
        }
    }
}

impl Default for NumaTopology {
    fn default() -> Self {
        Self::detect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numa_node_display() {
        assert_eq!(format!("{}", NumaNode(0)), "NUMA0");
        assert_eq!(format!("{}", NumaNode(7)), "NUMA7");
        assert_eq!(format!("{}", NumaNode::UNKNOWN), "UNKNOWN");
    }

    #[test]
    fn test_pin_unknown_node_fails() {
        let result = pin_thread_to_numa_node(NumaNode::UNKNOWN);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("unknown"));
    }

    #[test]
    fn test_format_cpu_list() {
        assert_eq!(format_cpu_list(&[]), "");
        assert_eq!(format_cpu_list(&[0]), "0");
        assert_eq!(format_cpu_list(&[0, 1, 2, 3]), "0-3");
        assert_eq!(format_cpu_list(&[0, 2, 4]), "0,2,4");
        assert_eq!(format_cpu_list(&[0, 1, 2, 4, 5]), "0-2,4-5");
        assert_eq!(format_cpu_list(&[0, 1, 2, 4, 6, 7, 8]), "0-2,4,6-8");
    }

    #[test]
    fn test_parse_cpulist_range() {
        let cpus = parse_cpulist("0-3").unwrap();
        assert_eq!(cpus, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_parse_cpulist_list() {
        let cpus = parse_cpulist("0,4,8").unwrap();
        assert_eq!(cpus, vec![0, 4, 8]);
    }

    #[test]
    fn test_parse_cpulist_mixed() {
        let cpus = parse_cpulist("0-2,8,16-17").unwrap();
        assert_eq!(cpus, vec![0, 1, 2, 8, 16, 17]);
    }

    #[test]
    fn test_parse_cpulist_hyperthreading() {
        let cpus = parse_cpulist("0-15,32-47").unwrap();
        assert_eq!(cpus.len(), 32);
        assert_eq!(cpus[0], 0);
        assert_eq!(cpus[15], 15);
        assert_eq!(cpus[16], 32);
        assert_eq!(cpus[31], 47);
    }

    #[test]
    fn test_parse_cpulist_empty() {
        let cpus = parse_cpulist("").unwrap();
        assert!(cpus.is_empty());
    }

    #[test]
    fn test_parse_cpulist_single_cpu() {
        let cpus = parse_cpulist("5").unwrap();
        assert_eq!(cpus, vec![5]);
    }
}
