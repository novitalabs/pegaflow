// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NUMA (Non-Uniform Memory Access) utilities
//!
//! This module provides:
//! - NUMA node abstraction (`NumaNode`)
//! - Current CPU NUMA node detection
//! - GPU to NUMA node affinity detection
//! - Thread-to-NUMA-node pinning for first-touch allocation policy

use std::mem;
use std::process::Command;

pub mod topology;

use topology::NumaTopology;

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

/// Get the current CPU's NUMA node using the `getcpu` syscall
///
/// Returns `NumaNode::UNKNOWN` if the syscall fails (e.g., on non-Linux systems
/// or in restricted containers).
pub fn get_current_cpu_numa_node() -> NumaNode {
    unsafe {
        let mut cpu: libc::c_uint = 0;
        let mut node: libc::c_uint = 0;

        // getcpu syscall: int getcpu(unsigned *cpu, unsigned *node, struct getcpu_cache *tcache);
        let result = libc::syscall(
            libc::SYS_getcpu,
            &mut cpu,
            &mut node,
            std::ptr::null_mut::<libc::c_void>(),
        );

        if result == 0 {
            NumaNode(node)
        } else {
            NumaNode::UNKNOWN
        }
    }
}

/// Get the NUMA node for a GPU device
///
/// Uses `nvidia-smi topo --get-numa-id-of-nearby-cpu` to query the NUMA affinity
/// of the specified GPU. This returns the NUMA node closest to the GPU's PCIe bus.
///
/// If nvidia-smi is not available or fails, returns `NumaNode::UNKNOWN`.
///
/// # Arguments
/// * `device_id` - The CUDA device ID (e.g., 0 for GPU 0)
pub fn get_device_numa_node(device_id: u32) -> NumaNode {
    // Use nvidia-smi topo to get NUMA ID of nearest CPU
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

/// Pin the current thread to CPUs on a specific NUMA node
///
/// This sets the CPU affinity of the calling thread to only run on CPUs
/// belonging to the specified NUMA node. This is critical for ensuring
/// that memory allocations follow the first-touch policy on the correct node.
///
/// # Arguments
/// * `node` - The target NUMA node
///
/// # Errors
/// Returns an error if:
/// - The NUMA topology cannot be read
/// - The node ID is invalid
/// - The sched_setaffinity syscall fails
pub fn pin_thread_to_numa_node(node: NumaNode) -> Result<(), String> {
    if node.is_unknown() {
        return Err("Cannot pin to unknown NUMA node".to_string());
    }

    let topology =
        NumaTopology::from_sysfs().map_err(|e| format!("Failed to get NUMA topology: {}", e))?;

    let cpus = topology
        .cpus_for_node(node.0)
        .ok_or_else(|| format!("No CPUs found for NUMA node {}", node.0))?;

    if cpus.is_empty() {
        return Err(format!("CPU list is empty for NUMA node {}", node.0));
    }

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

/// Get NUMA affinity information for all available GPUs
///
/// Returns a vector of (device_id, numa_node) pairs for all GPUs
/// that can be detected. If nvidia-smi is not available, returns
/// an empty vector.
pub fn get_gpu_numa_affinity() -> Vec<(u32, NumaNode)> {
    // First, try to get the number of GPUs
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
    // nvidia-smi may return multiple lines, take the first non-empty line
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

/// Log a summary of the system NUMA topology
///
/// This is useful for debugging and diagnostics. Uses `log::info!` and `log::warn!`
/// for output. Make sure to initialize logging before calling this function.
///
/// # Example
/// ```
/// use pegaflow_core::logging;
/// use pegaflow_core::numa::log_numa_summary;
///
/// logging::init_stdout_colored("info");
/// log_numa_summary();
/// ```
pub fn log_numa_summary() {
    log::info!("=== PegaFlow NUMA Topology ===");

    // Current process NUMA node
    let current_node = get_current_cpu_numa_node();
    log::info!("Current Process: {}", current_node);

    // System NUMA topology
    match NumaTopology::from_sysfs() {
        Ok(topology) => {
            log::info!("System NUMA Topology:");
            log::info!("  Total NUMA nodes: {}", topology.num_nodes());
            log::info!("  Total CPUs: {}", topology.total_cpus());

            for node_id in topology.node_ids() {
                if let Some(cpus) = topology.cpus_for_node(node_id) {
                    let cpu_list = format_cpu_list(cpus);
                    log::info!(
                        "  {}: {} CPUs ({})",
                        NumaNode(node_id),
                        cpus.len(),
                        cpu_list
                    );
                }
            }
        }
        Err(e) => {
            log::warn!("Failed to detect NUMA topology: {}", e);
            log::warn!("  (This is normal for single-node systems or containers)");
        }
    }

    // GPU NUMA affinity
    let gpu_affinity = get_gpu_numa_affinity();
    if !gpu_affinity.is_empty() {
        log::info!("GPU NUMA Affinity:");
        for (device_id, node) in gpu_affinity {
            log::info!("  GPU {} -> {}", device_id, node);
        }
    } else {
        log::warn!("GPU NUMA Affinity: Not available (nvidia-smi not found)");
    }
}

/// Format a list of CPUs into a compact range representation
///
/// Example: [0, 1, 2, 3, 8, 9, 10] -> "0-3,8-10"
fn format_cpu_list(cpus: &[usize]) -> String {
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
            // End current range
            if start == prev {
                result.push(format!("{}", start));
            } else {
                result.push(format!("{}-{}", start, prev));
            }
            start = cpu;
            prev = cpu;
        }
    }

    // Add final range
    if start == prev {
        result.push(format!("{}", start));
    } else {
        result.push(format!("{}-{}", start, prev));
    }

    result.join(",")
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
    fn test_get_current_cpu_numa_node() {
        // Should either return a valid node or UNKNOWN
        let node = get_current_cpu_numa_node();

        // If not unknown, should be a reasonable NUMA node number
        if !node.is_unknown() {
            assert!(node.0 < 16, "NUMA node {} seems unreasonably high", node.0);
        }
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
}
