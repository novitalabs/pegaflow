use std::collections::HashMap;
use std::future::{Future, IntoFuture};
use std::pin::pin;
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::mpsc as std_mpsc;
use std::sync::{Mutex, MutexGuard};
use std::task::{Context, Poll, Wake};
use std::thread;
use std::time::{Duration, Instant};

use mea::oneshot;
use pegaflow_transfer::{
    ConnectionStatus, ImmCompletion, ImmCompletionReceiver, MemoryRegion, TransferDesc,
    TransferEngine, TransferOp,
};

const TEST_BYTES: usize = 64 * 1024;
const REMOTE_D_ADDR: &str = "it-d";
const REMOTE_P_ADDR: &str = "it-p";
static RDMA_IT_LOCK: Mutex<()> = Mutex::new(());

struct TestBuffer {
    bytes: Box<[u8]>,
}

impl TestBuffer {
    fn new(len: usize, fill: u8) -> Self {
        Self {
            bytes: vec![fill; len].into_boxed_slice(),
        }
    }

    fn ptr(&mut self) -> NonNull<u8> {
        NonNull::new(self.bytes.as_mut_ptr()).expect("boxed slice pointer is non-null")
    }

    fn len(&self) -> usize {
        self.bytes.len()
    }

    fn fill_pattern(&mut self, seed: u8) {
        for (idx, byte) in self.bytes.iter_mut().enumerate() {
            *byte = seed.wrapping_add((idx % 251) as u8);
        }
    }

    fn as_slice(&self) -> &[u8] {
        &self.bytes
    }
}

struct EnginePair {
    p: TransferEngine,
    d: TransferEngine,
}

impl EnginePair {
    fn new(nics: &[String], p_buf: &mut TestBuffer, d_buf: &mut TestBuffer) -> Self {
        let start = Instant::now();
        let p = TransferEngine::new(nics).expect("P engine init");
        p.register_memory(&[MemoryRegion {
            ptr: p_buf.ptr(),
            len: p_buf.len(),
        }])
        .expect("P register_memory");

        let d = TransferEngine::new(nics).expect("D engine init");
        d.register_memory(&[MemoryRegion {
            ptr: d_buf.ptr(),
            len: d_buf.len(),
        }])
        .expect("D register_memory");

        let p_meta = match p.get_or_prepare(REMOTE_D_ADDR).expect("P prepare") {
            ConnectionStatus::Prepared(meta) => meta,
            ConnectionStatus::Existing | ConnectionStatus::Connecting => {
                panic!("unexpected P connection status")
            }
        };
        let d_meta = match d.get_or_prepare(REMOTE_P_ADDR).expect("D prepare") {
            ConnectionStatus::Prepared(meta) => meta,
            ConnectionStatus::Existing | ConnectionStatus::Connecting => {
                panic!("unexpected D connection status")
            }
        };

        p.complete_handshake(REMOTE_D_ADDR, &p_meta, &d_meta)
            .expect("P complete_handshake");
        d.complete_handshake(REMOTE_P_ADDR, &d_meta, &p_meta)
            .expect("D complete_handshake");
        eprintln!(
            "[write_imm_it] setup+handshake nics={} elapsed_us={}",
            nics.join(","),
            start.elapsed().as_micros()
        );

        Self { p, d }
    }
}

struct PdReceiveManager {
    cmd_tx: std_mpsc::Sender<ManagerCommand>,
}

struct LeaseReady {
    imm_data: u32,
    completions: Vec<ImmCompletion>,
}

enum ManagerCommand {
    Prepare {
        imm_data: u32,
        expected_count: usize,
        done_tx: oneshot::Sender<LeaseReady>,
    },
}

struct LeaseState {
    expected_count: usize,
    completions: Vec<ImmCompletion>,
    done_tx: oneshot::Sender<LeaseReady>,
}

impl PdReceiveManager {
    fn spawn(mut imm_rx: ImmCompletionReceiver) -> Self {
        let (cmd_tx, cmd_rx) = std_mpsc::channel::<ManagerCommand>();
        thread::Builder::new()
            .name("pd-receive-manager-it".to_string())
            .spawn(move || {
                let mut leases = HashMap::<u32, LeaseState>::new();
                loop {
                    while let Ok(cmd) = cmd_rx.try_recv() {
                        match cmd {
                            ManagerCommand::Prepare {
                                imm_data,
                                expected_count,
                                done_tx,
                            } => {
                                assert!(expected_count > 0);
                                let previous = leases.insert(
                                    imm_data,
                                    LeaseState {
                                        expected_count,
                                        completions: Vec::with_capacity(expected_count),
                                        done_tx,
                                    },
                                );
                                assert!(previous.is_none(), "duplicate imm_data in test manager");
                            }
                        }
                    }

                    match imm_rx.try_recv() {
                        Ok(completion) => {
                            let Some(lease) = leases.get_mut(&completion.imm_data) else {
                                panic!(
                                    "received unexpected imm_data={} in test manager",
                                    completion.imm_data
                                );
                            };
                            lease.completions.push(completion);
                            if lease.completions.len() == lease.expected_count {
                                let lease =
                                    leases.remove(&completion.imm_data).expect("lease exists");
                                let _ = lease.done_tx.send(LeaseReady {
                                    imm_data: completion.imm_data,
                                    completions: lease.completions,
                                });
                            }
                        }
                        Err(mea::mpsc::TryRecvError::Empty) => {
                            thread::sleep(Duration::from_micros(50));
                        }
                        Err(mea::mpsc::TryRecvError::Disconnected) => return,
                    }
                }
            })
            .expect("spawn P/D receive manager");
        Self { cmd_tx }
    }

    fn prepare(&self, imm_data: u32, expected_count: usize) -> oneshot::Receiver<LeaseReady> {
        let (done_tx, done_rx) = oneshot::channel();
        self.cmd_tx
            .send(ManagerCommand::Prepare {
                imm_data,
                expected_count,
                done_tx,
            })
            .expect("manager command channel");
        done_rx
    }
}

#[test]
#[ignore = "requires a real RDMA NIC; run with PEGAFLOW_IT_NICS=mlx5_0"]
fn pd_caller_demuxes_data_write_with_final_imm() {
    let _guard = rdma_it_guard();
    let Some(nics) = rdma_it_nics() else {
        return;
    };

    let mut p_buf = TestBuffer::new(TEST_BYTES, 0);
    let mut d_buf = TestBuffer::new(TEST_BYTES, 0);
    p_buf.fill_pattern(0x31);

    let pair = EnginePair::new(&nics, &mut p_buf, &mut d_buf);
    let imm_rx = pair
        .d
        .take_imm_receiver()
        .expect("D imm receiver should be available once");
    assert!(
        pair.d.take_imm_receiver().is_none(),
        "imm receiver is single-consumer"
    );
    let manager = PdReceiveManager::spawn(imm_rx);

    let imm_data = 0x00ab_c001;
    let lease_ready = manager.prepare(imm_data, nics.len());

    let write_start = Instant::now();
    let write_rx = pair
        .p
        .batch_transfer_async(
            TransferOp::Write,
            REMOTE_D_ADDR,
            &[TransferDesc {
                local_ptr: p_buf.ptr(),
                remote_ptr: d_buf.ptr(),
                len: TEST_BYTES,
            }],
        )
        .expect("P data write");
    wait_transfer_all(write_rx, "data write");
    let write_elapsed = write_start.elapsed();

    let imm_start = Instant::now();
    let imm_sends = pair
        .p
        .write_imm_async(REMOTE_D_ADDR, imm_data)
        .expect("P write_imm");
    wait_transfer_all(imm_sends, "write_imm");
    let imm_send_elapsed = imm_start.elapsed();

    let ready_start = Instant::now();
    let ready = recv_oneshot(lease_ready, "lease ready");
    let ready_wait_elapsed = ready_start.elapsed();
    assert_eq!(ready.imm_data, imm_data);
    assert_eq!(ready.completions.len(), nics.len());
    assert_eq!(d_buf.as_slice(), p_buf.as_slice());
    eprintln!(
        "[write_imm_it] data+final_imm bytes={} nics={} data_write_us={} imm_send_us={} ready_wait_us={} total_push_us={}",
        TEST_BYTES,
        nics.len(),
        write_elapsed.as_micros(),
        imm_send_elapsed.as_micros(),
        ready_wait_elapsed.as_micros(),
        write_start.elapsed().as_micros()
    );
}

#[test]
#[ignore = "requires a real RDMA NIC; run with PEGAFLOW_IT_NICS=mlx5_0"]
fn pd_caller_can_keep_multiple_pd_requests_in_flight() {
    let _guard = rdma_it_guard();
    let Some(nics) = rdma_it_nics() else {
        return;
    };

    let mut p_buf = TestBuffer::new(TEST_BYTES, 0);
    let mut d_buf = TestBuffer::new(TEST_BYTES, 0);
    let pair = EnginePair::new(&nics, &mut p_buf, &mut d_buf);
    let manager = PdReceiveManager::spawn(
        pair.d
            .take_imm_receiver()
            .expect("D imm receiver should be available"),
    );

    let first_imm = 0x00ab_c101;
    let second_imm = 0x00ab_c102;
    let first_ready = manager.prepare(first_imm, nics.len());
    let second_ready = manager.prepare(second_imm, nics.len());

    let second_start = Instant::now();
    wait_transfer_all(
        pair.p
            .write_imm_async(REMOTE_D_ADDR, second_imm)
            .expect("second write_imm"),
        "second write_imm",
    );
    let second_send_elapsed = second_start.elapsed();
    let first_start = Instant::now();
    wait_transfer_all(
        pair.p
            .write_imm_async(REMOTE_D_ADDR, first_imm)
            .expect("first write_imm"),
        "first write_imm",
    );
    let first_send_elapsed = first_start.elapsed();

    let second_ready_start = Instant::now();
    let second = recv_oneshot(second_ready, "second ready");
    let second_ready_elapsed = second_ready_start.elapsed();
    let first_ready_start = Instant::now();
    let first = recv_oneshot(first_ready, "first ready");
    let first_ready_elapsed = first_ready_start.elapsed();

    assert_eq!(second.imm_data, second_imm);
    assert_eq!(second.completions.len(), nics.len());
    assert_eq!(first.imm_data, first_imm);
    assert_eq!(first.completions.len(), nics.len());
    eprintln!(
        "[write_imm_it] multi_inflight nics={} second_send_us={} first_send_us={} second_ready_wait_us={} first_ready_wait_us={}",
        nics.len(),
        second_send_elapsed.as_micros(),
        first_send_elapsed.as_micros(),
        second_ready_elapsed.as_micros(),
        first_ready_elapsed.as_micros()
    );
}

fn rdma_it_guard() -> MutexGuard<'static, ()> {
    RDMA_IT_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner())
}

fn rdma_it_nics() -> Option<Vec<String>> {
    let value = match std::env::var("PEGAFLOW_IT_NICS") {
        Ok(value) => value,
        Err(_) => {
            eprintln!("skipping RDMA IT: PEGAFLOW_IT_NICS is not set");
            return None;
        }
    };
    let nics: Vec<String> = value
        .split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(ToOwned::to_owned)
        .collect();
    if nics.is_empty() {
        eprintln!("skipping RDMA IT: PEGAFLOW_IT_NICS is empty");
        return None;
    }
    Some(nics)
}

fn wait_transfer_all(
    receivers: Vec<oneshot::Receiver<pegaflow_transfer::Result<usize>>>,
    label: &str,
) {
    assert!(
        !receivers.is_empty(),
        "{label} should submit at least one WR"
    );
    for rx in receivers {
        let result = recv_oneshot(rx, label);
        result.unwrap_or_else(|error| panic!("{label} failed: {error}"));
    }
}

fn recv_oneshot<T: Send + 'static>(rx: oneshot::Receiver<T>, label: &str) -> T {
    block_on_timeout(rx, Duration::from_secs(5))
        .unwrap_or_else(|| panic!("{label} timed out"))
        .unwrap_or_else(|error| panic!("{label} sender dropped: {error}"))
}

fn block_on_timeout<I>(future: I, timeout: Duration) -> Option<I::Output>
where
    I: IntoFuture + Send + 'static,
    I::IntoFuture: Send,
    I::Output: Send + 'static,
{
    let (tx, rx) = std_mpsc::channel();
    thread::spawn(move || {
        let _ = tx.send(block_on(future.into_future()));
    });
    rx.recv_timeout(timeout).ok()
}

fn block_on<F: Future>(future: F) -> F::Output {
    struct ThreadWaker(thread::Thread);

    impl Wake for ThreadWaker {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
    }

    let waker = Arc::new(ThreadWaker(thread::current())).into();
    let mut cx = Context::from_waker(&waker);
    let mut future = pin!(future);
    let deadline = Instant::now() + Duration::from_secs(60);

    loop {
        match future.as_mut().poll(&mut cx) {
            Poll::Ready(result) => return result,
            Poll::Pending => {
                assert!(Instant::now() < deadline, "future did not complete");
                thread::park_timeout(Duration::from_millis(10));
            }
        }
    }
}
