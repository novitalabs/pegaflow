use std::{
    collections::HashMap,
    io::{BufRead, BufReader, Write},
    net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs},
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

use parking_lot::Mutex;
use sideway::{
    ibverbs::{
        AccessFlags,
        completion::{GenericCompletionQueue, PollCompletionQueueError, WorkCompletionStatus},
        device::{DeviceInfo, DeviceList},
        device_context::DeviceContext,
        memory_region::MemoryRegion,
        protection_domain::ProtectionDomain,
        queue_pair::{
            GenericQueuePair, PostSendGuard, QueuePair, QueuePairState, SetScatterGatherEntry,
            WorkRequestFlags,
        },
    },
    rdmacm::communication_manager::{
        ConnectionParameter, EventChannel, EventType, Identifier, PortSpace,
    },
};

use crate::{
    api::WorkerConfig,
    backend::RdmaBackend,
    error::{Result, TransferError},
};

struct SidewayRuntime {
    device_ctx: Arc<DeviceContext>,
    pd: Arc<ProtectionDomain>,
    _event_channel: Arc<EventChannel>,
    _listener_id: Arc<Identifier>,
}

struct RegisteredMemoryEntry {
    base_ptr: u64,
    len: usize,
    mr: Arc<MemoryRegion>,
}

struct RemoteMemoryInfo {
    rkey: u32,
    available_len: usize,
}

struct ActiveSession {
    _event_channel: Arc<EventChannel>,
    _id: Arc<Identifier>,
    qp: Mutex<GenericQueuePair>,
    send_cq: GenericCompletionQueue,
    _recv_cq: GenericCompletionQueue,
}

struct PassiveSession {
    _id: Arc<Identifier>,
    _qp: GenericQueuePair,
    _send_cq: GenericCompletionQueue,
    _recv_cq: GenericCompletionQueue,
}

#[derive(Default)]
struct SidewayState {
    config: Option<WorkerConfig>,
    runtime: Option<Arc<SidewayRuntime>>,
    registered: HashMap<u64, RegisteredMemoryEntry>,
    sessions: HashMap<String, Arc<ActiveSession>>,
}

#[derive(Default)]
pub struct SidewayBackend {
    state: Arc<Mutex<SidewayState>>,
}

impl SidewayBackend {
    pub fn new() -> Self {
        Self::default()
    }

    fn ensure_initialized(state: &SidewayState) -> Result<&WorkerConfig> {
        state.config.as_ref().ok_or(TransferError::NotInitialized)
    }

    fn build_socket_addr(bind_addr: &str, rpc_port: u16) -> Result<SocketAddr> {
        if let Ok(addr) = bind_addr.parse::<SocketAddr>() {
            return Ok(addr);
        }
        let mut resolved = (bind_addr, rpc_port).to_socket_addrs().map_err(|error| {
            TransferError::AddressResolution(format!("addr={bind_addr}:{rpc_port}, error={error}"))
        })?;
        resolved.next().ok_or_else(|| {
            TransferError::AddressResolution(format!(
                "addr={bind_addr}:{rpc_port}, no resolved address"
            ))
        })
    }

    fn parse_u64_token(token: &str) -> Option<u64> {
        let trimmed = token.trim();
        if let Some(hex) = trimmed
            .strip_prefix("0x")
            .or_else(|| trimmed.strip_prefix("0X"))
        {
            return u64::from_str_radix(hex, 16).ok();
        }
        trimmed.parse::<u64>().ok()
    }

    fn resolve_socket_addr(endpoint: &str) -> Result<SocketAddr> {
        if let Ok(addr) = endpoint.parse::<SocketAddr>() {
            return Ok(addr);
        }
        let mut resolved = endpoint.to_socket_addrs().map_err(|error| {
            TransferError::AddressResolution(format!("endpoint={endpoint}, error={error}"))
        })?;
        resolved.next().ok_or_else(|| {
            TransferError::AddressResolution(format!("endpoint={endpoint}, no resolved address"))
        })
    }

    fn metadata_addr_from_session(session_id: &str) -> Result<SocketAddr> {
        let addr = Self::resolve_socket_addr(session_id)?;
        let Some(port) = addr.port().checked_add(1) else {
            return Err(TransferError::InvalidArgument(
                "metadata port overflow from session_id",
            ));
        };
        Ok(SocketAddr::new(addr.ip(), port))
    }

    fn find_registered_entry(
        state: &SidewayState,
        ptr: u64,
        len: usize,
    ) -> Option<(u64, usize, Arc<MemoryRegion>)> {
        let end = ptr.checked_add(len as u64)?;
        state.registered.values().find_map(|entry| {
            let entry_end = entry.base_ptr.checked_add(entry.len as u64)?;
            if ptr >= entry.base_ptr && end <= entry_end {
                Some((entry.base_ptr, entry.len, Arc::clone(&entry.mr)))
            } else {
                None
            }
        })
    }

    fn create_qp(
        runtime: &SidewayRuntime,
    ) -> Result<(
        GenericQueuePair,
        GenericCompletionQueue,
        GenericCompletionQueue,
    )> {
        let mut cq_builder = runtime.device_ctx.create_cq_builder();
        cq_builder.setup_cqe(64);
        let send_cq: GenericCompletionQueue = cq_builder
            .build()
            .map_err(|error| TransferError::Backend(error.to_string()))?
            .into();
        let recv_cq: GenericCompletionQueue = cq_builder
            .build()
            .map_err(|error| TransferError::Backend(error.to_string()))?
            .into();

        let mut qp_builder = runtime.pd.create_qp_builder();
        qp_builder
            .setup_send_cq(send_cq.clone())
            .setup_recv_cq(recv_cq.clone());
        let qp: GenericQueuePair = qp_builder
            .build()
            .map_err(|error| TransferError::Backend(error.to_string()))?
            .into();

        Ok((qp, send_cq, recv_cq))
    }

    fn setup_qp_to_init(id: &Identifier, qp: &mut GenericQueuePair) -> Result<()> {
        let mut init_attr = id
            .get_qp_attr(QueuePairState::Init)
            .map_err(|error| TransferError::Backend(error.to_string()))?;
        init_attr.setup_access_flags(
            AccessFlags::LocalWrite | AccessFlags::RemoteWrite | AccessFlags::RemoteRead,
        );
        qp.modify(&init_attr)
            .map_err(|error| TransferError::Backend(error.to_string()))?;
        Ok(())
    }

    fn setup_qp_to_rtr_rts(id: &Identifier, qp: &mut GenericQueuePair) -> Result<()> {
        let rtr_attr = id
            .get_qp_attr(QueuePairState::ReadyToReceive)
            .map_err(|error| TransferError::Backend(error.to_string()))?;
        qp.modify(&rtr_attr)
            .map_err(|error| TransferError::Backend(error.to_string()))?;
        let rts_attr = id
            .get_qp_attr(QueuePairState::ReadyToSend)
            .map_err(|error| TransferError::Backend(error.to_string()))?;
        qp.modify(&rts_attr)
            .map_err(|error| TransferError::Backend(error.to_string()))?;
        Ok(())
    }

    fn spawn_passive_cm_loop(runtime: Arc<SidewayRuntime>) {
        let event_channel = Arc::clone(&runtime._event_channel);
        let _ = thread::Builder::new()
            .name("pegaflow-sideway-passive".to_string())
            .spawn(move || {
                let mut passive_sessions: Vec<PassiveSession> = Vec::new();
                while let Ok(event) = event_channel.get_cm_event() {
                    if event.event_type() != EventType::ConnectRequest {
                        continue;
                    }
                    let Some(new_id) = event.cm_id() else {
                        continue;
                    };

                    let Ok((mut qp, send_cq, recv_cq)) = Self::create_qp(&runtime) else {
                        continue;
                    };
                    if Self::setup_qp_to_init(&new_id, &mut qp).is_err() {
                        continue;
                    }
                    if Self::setup_qp_to_rtr_rts(&new_id, &mut qp).is_err() {
                        continue;
                    }

                    let mut conn_param = ConnectionParameter::default();
                    conn_param.setup_qp_number(qp.qp_number());
                    if new_id.accept(conn_param).is_err() {
                        continue;
                    }

                    passive_sessions.push(PassiveSession {
                        _id: new_id,
                        _qp: qp,
                        _send_cq: send_cq,
                        _recv_cq: recv_cq,
                    });
                }
            });
    }

    fn handle_metadata_request(line: &str, state: &Arc<Mutex<SidewayState>>) -> String {
        let parts = line.split_whitespace().collect::<Vec<_>>();
        if parts.len() != 3 || parts[0] != "GET" {
            return "ERR bad_request\n".to_string();
        }
        let Some(ptr) = Self::parse_u64_token(parts[1]) else {
            return "ERR bad_ptr\n".to_string();
        };
        let Some(len_u64) = Self::parse_u64_token(parts[2]) else {
            return "ERR bad_len\n".to_string();
        };
        let Ok(len) = usize::try_from(len_u64) else {
            return "ERR bad_len\n".to_string();
        };

        let guard = state.lock();
        let Some((base_ptr, entry_len, mr)) = Self::find_registered_entry(&guard, ptr, len) else {
            return "ERR not_found\n".to_string();
        };
        let Some(entry_end) = base_ptr.checked_add(entry_len as u64) else {
            return "ERR not_found\n".to_string();
        };
        let Some(available_u64) = entry_end.checked_sub(ptr) else {
            return "ERR not_found\n".to_string();
        };
        let Ok(available_len) = usize::try_from(available_u64) else {
            return "ERR not_found\n".to_string();
        };
        format!("OK {} {}\n", mr.rkey(), available_len)
    }

    fn spawn_metadata_listener(
        config: &WorkerConfig,
        state: Arc<Mutex<SidewayState>>,
    ) -> Result<()> {
        let bind_addr = Self::build_socket_addr(&config.bind_addr, config.rpc_port)?;
        let Some(metadata_port) = config.rpc_port.checked_add(1) else {
            return Err(TransferError::InvalidArgument("metadata port overflow"));
        };
        let metadata_addr = SocketAddr::new(bind_addr.ip(), metadata_port);
        let listener = TcpListener::bind(metadata_addr)
            .map_err(|error| TransferError::Backend(format!("metadata bind failed: {error}")))?;

        thread::Builder::new()
            .name("pegaflow-sideway-metadata".to_string())
            .spawn(move || {
                for stream in listener.incoming() {
                    let Ok(mut stream) = stream else {
                        continue;
                    };
                    let mut line = String::new();
                    {
                        let mut reader = BufReader::new(&mut stream);
                        if reader.read_line(&mut line).is_err() {
                            continue;
                        }
                    }
                    let response = Self::handle_metadata_request(&line, &state);
                    if stream.write_all(response.as_bytes()).is_err() {
                        continue;
                    }
                }
            })
            .map_err(|error| TransferError::Backend(error.to_string()))?;

        Ok(())
    }

    fn create_runtime(
        config: &WorkerConfig,
        state: Arc<Mutex<SidewayState>>,
    ) -> Result<Arc<SidewayRuntime>> {
        let device_list =
            DeviceList::new().map_err(|error| TransferError::Backend(error.to_string()))?;
        let device = device_list
            .iter()
            .find(|device| device.name() == config.nic_name)
            .ok_or_else(|| TransferError::DeviceNotFound(config.nic_name.clone()))?;

        let device_ctx = device
            .open()
            .map_err(|error| TransferError::Backend(error.to_string()))?;
        let pd = device_ctx
            .alloc_pd()
            .map_err(|error| TransferError::Backend(error.to_string()))?;

        let event_channel =
            EventChannel::new().map_err(|error| TransferError::Backend(error.to_string()))?;
        let listener_id = event_channel
            .create_id(PortSpace::Tcp)
            .map_err(|error| TransferError::Backend(error.to_string()))?;
        let socket_addr = Self::build_socket_addr(&config.bind_addr, config.rpc_port)?;
        listener_id
            .bind_addr(socket_addr)
            .map_err(|error| TransferError::Backend(error.to_string()))?;
        listener_id
            .listen(128)
            .map_err(|error| TransferError::Backend(error.to_string()))?;

        let runtime = Arc::new(SidewayRuntime {
            device_ctx,
            pd,
            _event_channel: event_channel,
            _listener_id: listener_id,
        });

        Self::spawn_passive_cm_loop(Arc::clone(&runtime));
        Self::spawn_metadata_listener(config, state)?;

        Ok(runtime)
    }

    fn create_active_session(
        runtime: &SidewayRuntime,
        session_id: &str,
    ) -> Result<Arc<ActiveSession>> {
        let remote_addr = Self::resolve_socket_addr(session_id)?;
        let event_channel =
            EventChannel::new().map_err(|error| TransferError::Backend(error.to_string()))?;
        let id = event_channel
            .create_id(PortSpace::Tcp)
            .map_err(|error| TransferError::Backend(error.to_string()))?;

        id.resolve_addr(None, remote_addr, Duration::from_secs(2))
            .map_err(|error| TransferError::Backend(error.to_string()))?;

        let mut qp: Option<GenericQueuePair> = None;
        let mut send_cq: Option<GenericCompletionQueue> = None;
        let mut recv_cq: Option<GenericCompletionQueue> = None;
        let mut connected = false;

        while !connected {
            let event = event_channel
                .get_cm_event()
                .map_err(|error| TransferError::Backend(error.to_string()))?;
            match event.event_type() {
                EventType::AddressResolved => {
                    id.resolve_route(Duration::from_secs(2))
                        .map_err(|error| TransferError::Backend(error.to_string()))?;
                }
                EventType::RouteResolved => {
                    let (mut created_qp, created_send_cq, created_recv_cq) =
                        Self::create_qp(runtime)?;
                    Self::setup_qp_to_init(&id, &mut created_qp)?;

                    let mut conn_param = ConnectionParameter::default();
                    conn_param.setup_qp_number(created_qp.qp_number());
                    id.connect(conn_param)
                        .map_err(|error| TransferError::Backend(error.to_string()))?;

                    qp = Some(created_qp);
                    send_cq = Some(created_send_cq);
                    recv_cq = Some(created_recv_cq);
                }
                EventType::ConnectResponse => {
                    let Some(qp_ref) = qp.as_mut() else {
                        return Err(TransferError::Backend(
                            "connect response before QP creation".to_string(),
                        ));
                    };
                    Self::setup_qp_to_rtr_rts(&id, qp_ref)?;
                    id.establish()
                        .map_err(|error| TransferError::Backend(error.to_string()))?;
                }
                EventType::Established => {
                    connected = true;
                }
                EventType::AddressError
                | EventType::RouteError
                | EventType::ConnectError
                | EventType::Unreachable
                | EventType::Rejected => {
                    return Err(TransferError::Backend(format!(
                        "rdma connect failed: event={:?}, session_id={session_id}",
                        event.event_type()
                    )));
                }
                _ => {}
            }
        }

        let Some(qp) = qp else {
            return Err(TransferError::Backend("session QP missing".to_string()));
        };
        let Some(send_cq) = send_cq else {
            return Err(TransferError::Backend(
                "session send CQ missing".to_string(),
            ));
        };
        let Some(recv_cq) = recv_cq else {
            return Err(TransferError::Backend(
                "session recv CQ missing".to_string(),
            ));
        };

        Ok(Arc::new(ActiveSession {
            _event_channel: event_channel,
            _id: id,
            qp: Mutex::new(qp),
            send_cq,
            _recv_cq: recv_cq,
        }))
    }

    fn query_remote_memory(
        session_id: &str,
        remote_ptr: u64,
        len: usize,
    ) -> Result<RemoteMemoryInfo> {
        let metadata_addr = Self::metadata_addr_from_session(session_id)?;
        let mut stream = TcpStream::connect_timeout(&metadata_addr, Duration::from_millis(500))
            .map_err(|error| {
                TransferError::Backend(format!(
                    "metadata connect failed: addr={metadata_addr}, error={error}"
                ))
            })?;
        stream
            .set_read_timeout(Some(Duration::from_millis(500)))
            .ok();
        stream
            .set_write_timeout(Some(Duration::from_millis(500)))
            .ok();

        let request = format!("GET {remote_ptr:#x} {len}\n");
        stream
            .write_all(request.as_bytes())
            .map_err(|error| TransferError::Backend(format!("metadata write failed: {error}")))?;

        let mut response = String::new();
        let mut reader = BufReader::new(stream);
        reader
            .read_line(&mut response)
            .map_err(|error| TransferError::Backend(format!("metadata read failed: {error}")))?;

        let parts = response.split_whitespace().collect::<Vec<_>>();
        if parts.first() == Some(&"OK") && parts.len() == 3 {
            let Some(rkey_u64) = Self::parse_u64_token(parts[1]) else {
                return Err(TransferError::Backend(
                    "metadata response bad rkey".to_string(),
                ));
            };
            let Some(available_u64) = Self::parse_u64_token(parts[2]) else {
                return Err(TransferError::Backend(
                    "metadata response bad len".to_string(),
                ));
            };
            let Ok(rkey) = u32::try_from(rkey_u64) else {
                return Err(TransferError::Backend("metadata rkey overflow".to_string()));
            };
            let Ok(available_len) = usize::try_from(available_u64) else {
                return Err(TransferError::Backend("metadata len overflow".to_string()));
            };
            return Ok(RemoteMemoryInfo {
                rkey,
                available_len,
            });
        }

        Err(TransferError::Backend(format!(
            "metadata query failed: response={}",
            response.trim()
        )))
    }

    fn wait_send_completion(
        send_cq: &GenericCompletionQueue,
        wr_id: u64,
        timeout: Duration,
    ) -> Result<()> {
        let deadline = Instant::now() + timeout;
        loop {
            match send_cq.start_poll() {
                Ok(mut poller) => {
                    for wc in &mut poller {
                        if wc.wr_id() != wr_id {
                            continue;
                        }
                        if wc.status() != WorkCompletionStatus::Success as u32 {
                            return Err(TransferError::Backend(format!(
                                "send completion failed: status={}, opcode={}, vendor_err={}",
                                wc.status(),
                                wc.opcode(),
                                wc.vendor_err()
                            )));
                        }
                        return Ok(());
                    }
                }
                Err(PollCompletionQueueError::CompletionQueueEmpty) => {}
                Err(error) => {
                    return Err(TransferError::Backend(format!(
                        "poll send CQ failed: {error}"
                    )));
                }
            }

            if Instant::now() >= deadline {
                return Err(TransferError::Backend(
                    "send completion timeout".to_string(),
                ));
            }
            thread::sleep(Duration::from_micros(50));
        }
    }

    fn post_write(
        session: &ActiveSession,
        local_mr: Arc<MemoryRegion>,
        local_ptr: u64,
        remote_ptr: u64,
        len: usize,
        remote_rkey: u32,
    ) -> Result<()> {
        if len > u32::MAX as usize {
            return Err(TransferError::InvalidArgument(
                "len exceeds RDMA SGE length limit",
            ));
        }

        let wr_id = 1_u64;
        {
            let mut qp = session.qp.lock();
            let mut guard = qp.start_post_send();
            let wr = guard
                .construct_wr(wr_id, WorkRequestFlags::Signaled)
                .setup_write(remote_rkey, remote_ptr);
            unsafe {
                wr.setup_sge(local_mr.lkey(), local_ptr, len as u32);
            }
            guard
                .post()
                .map_err(|error| TransferError::Backend(error.to_string()))?;
        }
        Self::wait_send_completion(&session.send_cq, wr_id, Duration::from_secs(2))?;
        Ok(())
    }
}

impl RdmaBackend for SidewayBackend {
    fn initialize(&self, config: WorkerConfig) -> Result<()> {
        if config.bind_addr.trim().is_empty() {
            return Err(TransferError::InvalidArgument("bind_addr is empty"));
        }
        if config.nic_name.trim().is_empty() {
            return Err(TransferError::InvalidArgument("nic_name is empty"));
        }
        if config.rpc_port == 0 {
            return Err(TransferError::InvalidArgument("rpc_port must be non-zero"));
        }
        if config.rpc_port == u16::MAX {
            return Err(TransferError::InvalidArgument("rpc_port must be < 65535"));
        }

        let runtime = Self::create_runtime(&config, Arc::clone(&self.state))?;
        let mut state = self.state.lock();
        state.config = Some(config);
        state.runtime = Some(runtime);
        state.registered.clear();
        state.sessions.clear();
        Ok(())
    }

    fn rpc_port(&self) -> Result<u16> {
        let state = self.state.lock();
        Ok(Self::ensure_initialized(&state)?.rpc_port)
    }

    fn register_memory(&self, ptr: u64, len: usize) -> Result<()> {
        if ptr == 0 {
            return Err(TransferError::InvalidArgument("ptr must be non-zero"));
        }
        if len == 0 {
            return Err(TransferError::InvalidArgument("len must be non-zero"));
        }

        let mut state = self.state.lock();
        Self::ensure_initialized(&state)?;
        let runtime = state
            .runtime
            .as_ref()
            .ok_or(TransferError::NotInitialized)?;
        let mr = unsafe {
            runtime.pd.reg_mr(
                ptr as usize,
                len,
                AccessFlags::LocalWrite | AccessFlags::RemoteWrite | AccessFlags::RemoteRead,
            )
        }
        .map_err(|error| TransferError::Backend(error.to_string()))?;
        state.registered.insert(
            ptr,
            RegisteredMemoryEntry {
                base_ptr: ptr,
                len,
                mr,
            },
        );
        Ok(())
    }

    fn unregister_memory(&self, ptr: u64) -> Result<()> {
        if ptr == 0 {
            return Err(TransferError::InvalidArgument("ptr must be non-zero"));
        }
        let mut state = self.state.lock();
        Self::ensure_initialized(&state)?;
        let removed = state.registered.remove(&ptr);
        if removed.is_none() {
            return Err(TransferError::MemoryNotRegistered { ptr });
        }
        Ok(())
    }

    fn transfer_sync_write(
        &self,
        session_id: &str,
        local_ptr: u64,
        remote_ptr: u64,
        len: usize,
    ) -> Result<usize> {
        if session_id.trim().is_empty() {
            return Err(TransferError::InvalidArgument("session_id is empty"));
        }
        if local_ptr == 0 {
            return Err(TransferError::InvalidArgument("local_ptr must be non-zero"));
        }
        if remote_ptr == 0 {
            return Err(TransferError::InvalidArgument(
                "remote_ptr must be non-zero",
            ));
        }
        if len == 0 {
            return Err(TransferError::InvalidArgument("len must be non-zero"));
        }

        let (runtime, local_mr, existing_session) = {
            let state = self.state.lock();
            Self::ensure_initialized(&state)?;
            let Some(runtime) = state.runtime.as_ref() else {
                return Err(TransferError::NotInitialized);
            };
            let Some((_, _, mr)) = Self::find_registered_entry(&state, local_ptr, len) else {
                return Err(TransferError::MemoryNotRegistered { ptr: local_ptr });
            };
            (
                Arc::clone(runtime),
                mr,
                state.sessions.get(session_id).cloned(),
            )
        };

        let session = if let Some(session) = existing_session {
            session
        } else {
            let created = Self::create_active_session(&runtime, session_id)?;
            let mut state = self.state.lock();
            state
                .sessions
                .entry(session_id.to_string())
                .or_insert_with(|| Arc::clone(&created))
                .clone()
        };

        let remote = Self::query_remote_memory(session_id, remote_ptr, len)?;
        if len > remote.available_len {
            return Err(TransferError::InvalidArgument(
                "len exceeds remote registered memory",
            ));
        }

        Self::post_write(&session, local_mr, local_ptr, remote_ptr, len, remote.rkey)?;
        Ok(len)
    }
}

#[cfg(test)]
mod tests {
    use super::SidewayBackend;
    use crate::{api::WorkerConfig, backend::RdmaBackend, error::TransferError};

    #[test]
    fn build_socket_addr_parses_socket_addr() {
        let addr = SidewayBackend::build_socket_addr("127.0.0.1:50055", 1).expect("must parse");
        assert_eq!(addr.port(), 50055);
    }

    #[test]
    fn initialize_rejects_invalid_input() {
        let backend = SidewayBackend::new();
        let error = backend
            .initialize(WorkerConfig {
                bind_addr: "".to_string(),
                nic_name: "mlx5_0".to_string(),
                rpc_port: 50055,
            })
            .expect_err("must fail");
        assert_eq!(error, TransferError::InvalidArgument("bind_addr is empty"));
    }
}
