use std::{
    ffi::c_void,
    ptr::NonNull,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering::SeqCst},
    },
};

use super::cpu_affinity::pin_cpu;
use crossbeam_channel::TryRecvError;
use log::debug;

use crate::{
    api::{
        DomainAddress, MemoryRegionDescriptor, MemoryRegionHandle, PeerGroupHandle, SmallVec,
        TransferCompletionEntry, TransferCounter, TransferId, TransferRequest,
    },
    domain_group::DomainGroup,
    error::{FabricLibError, Result},
    imm_count::ImmCountMap,
    mr::MemoryRegion,
    provider::{DomainCompletionEntry, RdmaDomain, RdmaDomainInfo},
    provider_dispatch::DomainInfo,
    verbs::VerbsDomain,
};

#[allow(
    clippy::enum_variant_names,
    clippy::large_enum_variant,
    reason = "variants are queue commands; always boxed when sent"
)]
pub(crate) enum WorkerCommand {
    SubmitTransfer {
        transfer_id: TransferId,
        request: TransferRequest,
        tx_counter: Option<TransferCounter>,
    },
    SubmitSend {
        transfer_id: TransferId,
        mr: MemoryRegionHandle,
        ptr: NonNull<c_void>,
        len: usize,
        addr: DomainAddress,
    },
    SubmitRecv {
        transfer_id: TransferId,
        mr: MemoryRegionHandle,
        ptr: NonNull<c_void>,
        len: usize,
    },
}

unsafe impl Send for WorkerCommand {}
unsafe impl Sync for WorkerCommand {}

pub(crate) enum WorkerCall {
    RegisterMRLocal {
        region: MemoryRegion,
        ret: oneshot::Sender<Result<MemoryRegionHandle>>,
    },
    RegisterMRAllowRemote {
        region: MemoryRegion,
        ret: oneshot::Sender<Result<(MemoryRegionHandle, MemoryRegionDescriptor)>>,
    },
    UnregisterMR {
        ptr: NonNull<c_void>,
        ret: oneshot::Sender<()>,
    },
    AddPeerGroup {
        addrs: Vec<SmallVec<DomainAddress>>,
        ret: oneshot::Sender<Result<PeerGroupHandle>>,
    },
}

unsafe impl Send for WorkerCall {}

pub(crate) struct Worker {
    pub domain_list: Vec<DomainInfo>,
    pub pin_worker_cpu: Option<u16>,
}

unsafe impl Send for Worker {}

pub(crate) struct InitializingWorker {
    worker_handle: std::thread::JoinHandle<()>,
    init_worker_rx: oneshot::Receiver<Result<InitializedWorker>>,
    cq_rx: crossbeam_channel::Receiver<TransferCompletionEntry>,
}

struct InitializedWorker {
    stop_signal: Arc<AtomicBool>,
    aggregated_link_speed: u64,
    address_list: Vec<DomainAddress>,
    call_tx: crossbeam_channel::Sender<WorkerCall>,
    cmd_tx: crossbeam_channel::Sender<Box<WorkerCommand>>,
}

pub(crate) struct WorkerHandle {
    pub aggregated_link_speed: u64,
    pub address_list: Vec<DomainAddress>,
    pub worker_call_tx: crossbeam_channel::Sender<WorkerCall>,
    pub cmd_tx: crossbeam_channel::Sender<Box<WorkerCommand>>,
    pub cq_rx: crossbeam_channel::Receiver<TransferCompletionEntry>,
    worker_stop_signal: Arc<AtomicBool>,
    worker_handle: std::thread::JoinHandle<()>,
}

impl WorkerHandle {
    pub(crate) fn stop(&self) {
        self.worker_stop_signal.store(true, SeqCst);
    }

    /// Wait for the worker thread to exit. Call after [`Self::stop`];
    /// must not panic since it runs from `FabricEngine::drop`.
    pub(crate) fn join(self) {
        if self.worker_handle.join().is_err() {
            log::error!("RDMA worker thread panicked");
        }
    }
}

impl Worker {
    pub(crate) fn spawn(self, imm_count_map: Arc<ImmCountMap>) -> Result<InitializingWorker> {
        let (init_worker_tx, init_worker_rx) = oneshot::channel();

        // Collect Verbs domains. EFA was removed during the port; DomainInfo
        // remains opaque so callers do not depend on provider internals.
        let verbs_domain_list = self
            .domain_list
            .into_iter()
            .map(DomainInfo::into_verbs)
            .collect::<Vec<_>>();

        // Callback queue.
        let (cq_tx, cq_rx) = crossbeam_channel::bounded(128);

        // Spawn thread
        let worker_thread_builder =
            std::thread::Builder::new().name("tx_engine_domain_worker".to_string());
        macro_rules! spawn_for_n {
            ($n:literal) => {
                worker_thread_builder.spawn(move || {
                    rdma_worker_thread::<VerbsDomain, $n>(
                        verbs_domain_list,
                        self.pin_worker_cpu,
                        imm_count_map,
                        init_worker_tx,
                        cq_tx,
                    )
                })
            };
        }
        let worker_handle = match verbs_domain_list.len() {
            1 => spawn_for_n!(1),
            2 => spawn_for_n!(2),
            3 => spawn_for_n!(3),
            4 => spawn_for_n!(4),
            5 => spawn_for_n!(5),
            6 => spawn_for_n!(6),
            7 => spawn_for_n!(7),
            8 => spawn_for_n!(8),
            _ => {
                return Err(FabricLibError::Custom(
                    "Only support 1 to 8 domains per worker for Verbs",
                ));
            }
        };
        let worker_handle =
            worker_handle.map_err(|_| FabricLibError::Custom("Failed to spawn worker thread"))?;

        Ok(InitializingWorker {
            worker_handle,
            init_worker_rx,
            cq_rx,
        })
    }
}

impl InitializingWorker {
    pub(crate) fn wait_init(self) -> Result<WorkerHandle> {
        let init_worker = self
            .init_worker_rx
            .recv()
            .map_err(|_| FabricLibError::Custom("Failed to receive worker init"))?;

        let init_worker_args = match init_worker {
            Ok(init) => init,
            Err(e) => {
                self.worker_handle
                    .join()
                    .expect("Failed to join worker thread");
                return Err(e);
            }
        };

        Ok(WorkerHandle {
            worker_stop_signal: init_worker_args.stop_signal,
            aggregated_link_speed: init_worker_args.aggregated_link_speed,
            address_list: init_worker_args.address_list,
            worker_call_tx: init_worker_args.call_tx,
            cmd_tx: init_worker_args.cmd_tx,
            cq_rx: self.cq_rx,
            worker_handle: self.worker_handle,
        })
    }
}

fn rdma_worker_thread<D: RdmaDomain, const N: usize>(
    domain_list: Vec<D::Info>,
    maybe_pin_cpu: Option<u16>,
    imm_count_map: Arc<ImmCountMap>,
    init_tx: oneshot::Sender<Result<InitializedWorker>>,
    cq_tx: crossbeam_channel::Sender<TransferCompletionEntry>,
) {
    // Pin CPU if specified
    if let Some(cpu) = maybe_pin_cpu {
        let names: Vec<_> = domain_list.iter().map(|info| info.name()).collect();
        debug!("Pin Domain Worker CPU {} for {:?}", cpu, names);
        if let Err(e) = pin_cpu(cpu as usize) {
            // Ignore send error
            let _ = init_tx.send(Err(FabricLibError::Errno(e)));
            return;
        }
    }

    // Create domains
    //
    // NOTE(lequn): We'd like to create the domain after pinning the CPU so that
    // the allocated resources are on the correct NUMA node.
    let mut domains = Vec::with_capacity(N);
    for info in domain_list {
        match D::open(info, imm_count_map.clone()) {
            Ok(domain) => {
                domains.push(domain);
            }
            Err(e) => {
                let _ = init_tx.send(Err(e)); // Ignore send error
                return;
            }
        }
    }
    let address_list = domains.iter().map(|d| d.addr().clone()).collect();

    // Create domain group
    let domains: [D; N] = domains.try_into().unwrap_or_else(|v: Vec<D>| {
        panic!(
            "The number of domains mismatch the const generic N: {} != {}",
            v.len(),
            N
        )
    });
    let mut group = DomainGroup::new(domains);

    // Create channels
    let (call_tx, call_rx) = crossbeam_channel::bounded(128);
    let (cmd_tx, cmd_rx) = crossbeam_channel::bounded(128);

    // Initialization complete
    let stop_signal = Arc::new(AtomicBool::new(false));
    let init = InitializedWorker {
        stop_signal: stop_signal.clone(),
        aggregated_link_speed: group.aggregate_link_speed(),
        address_list,
        call_tx,
        cmd_tx,
    };
    if init_tx.send(Ok(init)).is_err() {
        // Failed to send init message. Caller has discarded the thread.
        // Let's just exit.
        return;
    }

    // Main loop
    while !stop_signal.load(SeqCst) {
        std::hint::spin_loop();
        let ret = worker_step(&mut group, &call_rx, &cmd_rx, &cq_tx, &stop_signal);
        if ret.is_err() {
            debug!("RDMA worker thread exiting after worker channel closed");
            break;
        }
    }
}

/// Send a completion without blocking the worker forever: if the completion
/// queue is full and the engine is stopping (the callback thread may already
/// be gone), give up so the thread can exit and be joined.
fn send_completion(
    cq_tx: &crossbeam_channel::Sender<TransferCompletionEntry>,
    stop_signal: &AtomicBool,
    mut comp: TransferCompletionEntry,
) -> std::result::Result<(), ()> {
    loop {
        match cq_tx.try_send(comp) {
            Ok(()) => return Ok(()),
            Err(crossbeam_channel::TrySendError::Disconnected(_)) => return Err(()),
            Err(crossbeam_channel::TrySendError::Full(c)) => {
                if stop_signal.load(SeqCst) {
                    return Err(());
                }
                comp = c;
                std::hint::spin_loop();
            }
        }
    }
}

fn worker_step<D: RdmaDomain, const N: usize>(
    group: &mut DomainGroup<D, N>,
    call_rx: &crossbeam_channel::Receiver<WorkerCall>,
    cmd_rx: &crossbeam_channel::Receiver<Box<WorkerCommand>>,
    cq_tx: &crossbeam_channel::Sender<TransferCompletionEntry>,
    stop_signal: &AtomicBool,
) -> std::result::Result<(), ()> {
    // Process function call
    match call_rx.try_recv() {
        Ok(call) => match call {
            WorkerCall::RegisterMRLocal { region, ret } => {
                let result = group.register_mr_local(&region);
                ret.send(result).map_err(|_| ())?;
            }
            WorkerCall::RegisterMRAllowRemote { region, ret } => {
                let result = group.register_mr_allow_remote(&region);
                ret.send(result).map_err(|_| ())?;
            }
            WorkerCall::UnregisterMR { ptr, ret } => {
                group.unregister_mr(ptr);
                ret.send(()).map_err(|_| ())?;
            }
            WorkerCall::AddPeerGroup { addrs, ret } => {
                let result = group.add_peer_group(addrs);
                ret.send(result).map_err(|_| ())?;
            }
        },
        Err(TryRecvError::Disconnected) => {
            // Channel disconnected, exit the thread
            return Err(());
        }
        Err(TryRecvError::Empty) => {
            // No function call, continue
        }
    }

    // Process worker command
    match cmd_rx.try_recv() {
        Ok(cmd) => match *cmd {
            WorkerCommand::SubmitTransfer {
                transfer_id,
                request,
                tx_counter,
            } => {
                let result = group.submit_transfer_request(transfer_id, request, tx_counter);
                if let Err(e) = result {
                    let comp = TransferCompletionEntry::Error(transfer_id, e);
                    send_completion(cq_tx, stop_signal, comp)?;
                }
            }
            WorkerCommand::SubmitSend {
                transfer_id,
                mr,
                ptr,
                len,
                addr,
            } => {
                let result = group.submit_send(transfer_id, mr, ptr, len, addr);
                if let Err(e) = result {
                    let comp = TransferCompletionEntry::Error(transfer_id, e);
                    send_completion(cq_tx, stop_signal, comp)?;
                }
            }
            WorkerCommand::SubmitRecv {
                transfer_id,
                mr,
                ptr,
                len,
            } => {
                let result = group.submit_recv(transfer_id, mr, ptr, len);
                if let Err(e) = result {
                    let comp = TransferCompletionEntry::Error(transfer_id, e);
                    send_completion(cq_tx, stop_signal, comp)?;
                }
            }
        },
        Err(TryRecvError::Disconnected) => {
            // Channel disconnected, exit the thread
            return Err(());
        }
        Err(TryRecvError::Empty) => {
            // No command, continue
        }
    }

    // Make progress
    group.poll_progress();

    // Send completions
    while let Some(comp) = group.get_completion() {
        let tx_comp = match comp {
            DomainCompletionEntry::Recv {
                transfer_id,
                data_len,
            } => TransferCompletionEntry::Recv {
                transfer_id,
                data_len,
            },
            DomainCompletionEntry::Send(transfer_id) => TransferCompletionEntry::Send(transfer_id),
            DomainCompletionEntry::Transfer(transfer_id) => {
                TransferCompletionEntry::Transfer(transfer_id)
            }
            DomainCompletionEntry::ImmData(imm) => TransferCompletionEntry::ImmData(imm),
            DomainCompletionEntry::ImmCountReached(imm) => {
                TransferCompletionEntry::ImmCountReached(imm)
            }
            DomainCompletionEntry::Error(transfer_id, err) => {
                TransferCompletionEntry::Error(transfer_id, err)
            }
        };
        send_completion(cq_tx, stop_signal, tx_comp)?;
    }
    Ok(())
}
