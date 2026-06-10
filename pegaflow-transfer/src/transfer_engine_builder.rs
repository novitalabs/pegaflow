use std::collections::HashSet;

use anyhow::{Error, Result};

use crate::{
    provider_dispatch::DomainInfo, topo::detect_topology, transfer_engine::TransferEngine,
    worker::Worker,
};

struct GpuDomainSpec {
    cuda_device: u8,
    domains: Vec<DomainInfo>,
    pin_worker_cpu: u16,
}

#[derive(Default)]
pub struct TransferEngineBuilder {
    gpus: Vec<GpuDomainSpec>,
}

impl TransferEngineBuilder {
    /// Build a host-memory transfer engine: one worker driving `domains`,
    /// with its polling thread restricted to `pin_worker_cpus` (typically the
    /// CPUs of one NUMA node; empty = no affinity). Unlike [`Self::build`],
    /// this does not consult the GPU topology; callers group NICs per NUMA
    /// node themselves (see `detect_host_topology`).
    pub fn build_host(
        domains: Vec<DomainInfo>,
        pin_worker_cpus: Vec<u16>,
    ) -> Result<TransferEngine> {
        if domains.is_empty() {
            return Err(Error::msg("No domains for host transfer engine"));
        }
        let num_unique = domains
            .iter()
            .map(|d| d.name())
            .collect::<HashSet<_>>()
            .len();
        if num_unique != domains.len() {
            return Err(Error::msg("Duplicated domains for host transfer engine"));
        }
        let worker = Worker {
            domain_list: domains,
            pin_worker_cpus,
        };
        Ok(TransferEngine::new(vec![(0, worker)])?)
    }

    pub fn add_gpu_domains(
        &mut self,
        cuda_device: u8,
        domains: Vec<DomainInfo>,
        pin_worker_cpu: u16,
    ) {
        self.gpus.push(GpuDomainSpec {
            cuda_device,
            domains,
            pin_worker_cpu,
        });
    }

    pub fn build(&self) -> Result<TransferEngine> {
        let system_topo = detect_topology()?;

        // Validate that there's no duplicated GPUs
        let num_gpus = self
            .gpus
            .iter()
            .map(|s| s.cuda_device)
            .collect::<HashSet<_>>()
            .len();
        if num_gpus != self.gpus.len() {
            return Err(Error::msg("Duplicated GPUs in the builder"));
        }
        if num_gpus == 0 {
            return Err(Error::msg("No GPUs in the builder"));
        }

        // Validate builder params and prepare workers
        let mut workers = Vec::with_capacity(self.gpus.len());
        for spec in &self.gpus {
            let Some(topo) = system_topo
                .iter()
                .find(|t| t.cuda_device == spec.cuda_device)
            else {
                return Err(Error::msg(format!(
                    "cuda:{} not found in system topology",
                    spec.cuda_device
                )));
            };

            let num_domains = spec
                .domains
                .iter()
                .map(|d| d.name())
                .collect::<HashSet<_>>()
                .len();
            if num_domains != spec.domains.len() {
                return Err(Error::msg(format!(
                    "Duplicated domains in cuda:{}",
                    spec.cuda_device
                )));
            }

            for d in &spec.domains {
                if !topo.domains.iter().any(|t| t.name() == d.name()) {
                    return Err(Error::msg(format!(
                        "Domain {} not found in the topology group of cuda:{}",
                        d.name(),
                        spec.cuda_device
                    )));
                }
            }

            if !topo.cpus.contains(&spec.pin_worker_cpu) {
                return Err(Error::msg(format!(
                    "CPU {} not found in the topology group of cuda:{}",
                    spec.pin_worker_cpu, spec.cuda_device
                )));
            }

            let domain_list: Vec<_> = spec.domains.to_vec();
            let worker = Worker {
                domain_list,
                pin_worker_cpus: vec![spec.pin_worker_cpu],
            };
            workers.push((spec.cuda_device, worker));
        }

        // Create the transfer engine.
        Ok(TransferEngine::new(workers)?)
    }
}
