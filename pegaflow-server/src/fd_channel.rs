//! Host-local fd side-channel for native VMM clients.
//!
//! A native client exports its fused KV allocation as a POSIX file descriptor
//! (`CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR`). That fd is a kernel object with
//! process-local meaning — it cannot travel inside a gRPC message. The client
//! sends it here over a Unix domain socket via `SCM_RIGHTS`, tagged with its
//! `(instance_id, device_id)`, before it issues `register_context_batch`. The
//! gRPC handler then claims the fd by the same key and imports the VMM
//! allocation from it.
//!
//! This is why native clients can offer GPUDirect RDMA where CUDA IPC cannot: a
//! VMM POSIX-fd allocation can be handed to `ibv_reg_dmabuf_mr`, an imported
//! `cuIpcOpenMemHandle` pointer cannot.

use std::collections::HashMap;
use std::os::fd::{FromRawFd, OwnedFd, RawFd};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::net::UnixListener;
use tokio::sync::Notify;

/// Key identifying which registration an fd belongs to. Matches the tuple the
/// gRPC `register_context_batch` handler reconstructs from its request.
type FdKey = (String, i32);

/// Received fds waiting to be claimed by their registration RPC, plus a waker so
/// a handler that arrives before its fd can await it instead of racing.
#[derive(Default)]
struct Inbox {
    fds: HashMap<FdKey, OwnedFd>,
    arrived: Arc<Notify>,
}

/// Server-side endpoint of the fd side-channel. Holds received fds until the
/// matching registration RPC claims them.
#[derive(Clone)]
pub struct FdChannel {
    path: String,
    inbox: Arc<Mutex<Inbox>>,
}

impl FdChannel {
    /// Bind the Unix socket at `path` and spawn an accept loop. Each connection
    /// carries exactly one fd plus its `instance_id\0device_id` key.
    pub fn bind(path: String) -> std::io::Result<Self> {
        // A stale socket file from a previous run would make bind fail with
        // EADDRINUSE; the server owns this path exclusively, so clear it.
        let _ = std::fs::remove_file(&path);
        let listener = UnixListener::bind(&path)?;
        let channel = Self {
            path,
            inbox: Arc::new(Mutex::new(Inbox::default())),
        };
        let inbox = Arc::clone(&channel.inbox);
        tokio::spawn(async move {
            loop {
                match listener.accept().await {
                    Ok((stream, _)) => {
                        let inbox = Arc::clone(&inbox);
                        // One fd per connection; handle concurrently so a slow
                        // sender can't head-of-line block others.
                        tokio::spawn(async move {
                            if let Err(err) = recv_one(stream, &inbox).await {
                                log::warn!("fd side-channel: dropping connection: {err}");
                            }
                        });
                    }
                    Err(err) => {
                        log::error!("fd side-channel accept failed: {err}");
                        break;
                    }
                }
            }
        });
        Ok(channel)
    }

    /// Claim the fd registered for `key`, waiting up to `timeout` for it to
    /// arrive if the registration RPC beat its fd. Returns `None` on timeout.
    pub async fn take(&self, key: FdKey, timeout: Duration) -> Option<OwnedFd> {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            let notify = {
                let mut inbox = self.inbox.lock().expect("fd inbox poisoned");
                if let Some(fd) = inbox.fds.remove(&key) {
                    return Some(fd);
                }
                Arc::clone(&inbox.arrived)
            };
            // Wait for the next arrival or the deadline, then re-check the map.
            if tokio::time::timeout_at(deadline, notify.notified())
                .await
                .is_err()
            {
                return None;
            }
        }
    }
}

impl Drop for FdChannel {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Read one `(key, fd)` from a connection and deposit it in the inbox.
async fn recv_one(stream: tokio::net::UnixStream, inbox: &Mutex<Inbox>) -> std::io::Result<()> {
    // SCM_RIGHTS requires recvmsg; tokio has no wrapper, so drive the raw fd
    // through a std socket once it is readable. The payload is small and arrives
    // in one datagram-sized write, so a single readable notification suffices.
    stream.readable().await?;
    let std_stream = stream.into_std()?;
    std_stream.set_nonblocking(false)?;
    let raw = std::os::fd::AsRawFd::as_raw_fd(&std_stream);
    let (key, fd) = recv_fd_with_key(raw)?;
    // std_stream owns `raw` and closes it on drop; the received `fd` is separate.
    drop(std_stream);
    let notify = {
        let mut guard = inbox.lock().expect("fd inbox poisoned");
        guard.fds.insert(key, fd);
        Arc::clone(&guard.arrived)
    };
    notify.notify_waiters();
    Ok(())
}

/// Blocking `recvmsg` that pulls one fd (SCM_RIGHTS) plus a
/// `instance_id\0device_id` payload off `sock`.
fn recv_fd_with_key(sock: RawFd) -> std::io::Result<(FdKey, OwnedFd)> {
    // Payload: "<instance_id>\0<device_id>", capped generously.
    let mut buf = [0u8; 512];
    let mut iov = libc::iovec {
        iov_base: buf.as_mut_ptr().cast(),
        iov_len: buf.len(),
    };
    // Control buffer sized for exactly one fd.
    let mut cmsg_space = [0u8; unsafe_cmsg_space()];
    let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
    msg.msg_iov = &mut iov;
    msg.msg_iovlen = 1;
    msg.msg_control = cmsg_space.as_mut_ptr().cast();
    msg.msg_controllen = cmsg_space.len();

    // SAFETY: msg is fully initialized with valid iov/control buffers.
    let n = unsafe { libc::recvmsg(sock, &mut msg, 0) };
    if n < 0 {
        return Err(std::io::Error::last_os_error());
    }
    if (msg.msg_flags & libc::MSG_CTRUNC) != 0 {
        return Err(std::io::Error::other("fd control message truncated"));
    }

    // Extract the single fd from the first SCM_RIGHTS control message.
    // SAFETY: msg has a valid control buffer populated by recvmsg above.
    let cmsg = unsafe { libc::CMSG_FIRSTHDR(&msg) };
    if cmsg.is_null() {
        return Err(std::io::Error::other("fd side-channel: no control message"));
    }
    // SAFETY: cmsg points into our control buffer.
    let (level, ctype) = unsafe { ((*cmsg).cmsg_level, (*cmsg).cmsg_type) };
    if level != libc::SOL_SOCKET || ctype != libc::SCM_RIGHTS {
        return Err(std::io::Error::other(
            "fd side-channel: unexpected control message",
        ));
    }
    // SAFETY: CMSG_DATA points at the fd payload of a SCM_RIGHTS message.
    let raw_fd = unsafe { std::ptr::read_unaligned(libc::CMSG_DATA(cmsg).cast::<RawFd>()) };
    // SAFETY: recvmsg transferred ownership of this fd to us.
    let fd = unsafe { OwnedFd::from_raw_fd(raw_fd) };

    let payload = &buf[..n as usize];
    let key = parse_key(payload)?;
    Ok((key, fd))
}

/// Parse `"<instance_id>\0<device_id>"` into an `(instance_id, device_id)` key.
fn parse_key(payload: &[u8]) -> std::io::Result<FdKey> {
    let mut parts = payload.splitn(2, |&b| b == 0);
    let instance = parts
        .next()
        .ok_or_else(|| std::io::Error::other("fd side-channel: missing instance_id"))?;
    let device = parts
        .next()
        .ok_or_else(|| std::io::Error::other("fd side-channel: missing device_id"))?;
    let instance_id = std::str::from_utf8(instance)
        .map_err(|_| std::io::Error::other("fd side-channel: instance_id not utf-8"))?
        .to_string();
    let device_id: i32 = std::str::from_utf8(device)
        .ok()
        .and_then(|s| s.trim_end_matches('\0').parse().ok())
        .ok_or_else(|| std::io::Error::other("fd side-channel: bad device_id"))?;
    Ok((instance_id, device_id))
}

/// `CMSG_SPACE(size_of::<RawFd>())` as a const for the control buffer. Wrapped
/// in a const fn since `CMSG_SPACE` is not const in libc.
const fn unsafe_cmsg_space() -> usize {
    // CMSG_SPACE(4) rounds the header + 4-byte fd up to the alignment boundary;
    // 64 bytes covers it on all supported platforms with margin.
    64
}
