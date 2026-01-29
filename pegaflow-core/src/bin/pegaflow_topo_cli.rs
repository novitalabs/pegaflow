// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! PegaFlow NUMA Topology CLI
//!
//! A simple command-line tool to display the NUMA topology as perceived by PegaFlow.
//!
//! Usage:
//!   cargo run --bin pegaflow_topo_cli
//!
//! Output includes:
//! - Current process NUMA node
//! - System NUMA topology (nodes, CPUs per node)
//! - GPU to NUMA node affinity

use pegaflow_core::logging;
use pegaflow_core::numa::log_numa_summary;

fn main() {
    // Initialize logging with colored stdout output
    logging::init_stdout_colored("info");

    // Log the complete NUMA topology summary
    log_numa_summary();
}
