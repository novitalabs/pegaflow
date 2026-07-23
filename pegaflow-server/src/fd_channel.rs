//! Host-local Unix socket that receives VMM allocation fds (SCM_RIGHTS) for
//! native `register_context_batch`. Wire payload: `instance_id\0device_id` plus
//! one fd per connection.

use std::collections::HashMap;
use std::os::fd::{FromRawFd, OwnedFd, RawFd};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::net::UnixListener;
use tokio::sync::Notify;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct RegistrationKey {
    pub(crate) instance_id: String,
    pub(crate) device_id: i32,
}

/// Bound UDS path plus fds waiting to be claimed by `register_context_batch`.
#[derive(Clone)]
pub struct FdChannel {
    path: String,
    pending: Arc<Mutex<HashMap<RegistrationKey, OwnedFd>>>,
    arrived: Arc<Notify>,
}

impl FdChannel {
    pub fn bind(path: String) -> std::io::Result<Self> {
        remove_stale_socket(&path);
        let listener = UnixListener::bind(&path)?;
        let channel = Self {
            path,
            pending: Arc::new(Mutex::new(HashMap::new())),
            arrived: Arc::new(Notify::new()),
        };
        let pending = Arc::clone(&channel.pending);
        let arrived = Arc::clone(&channel.arrived);
        tokio::spawn(async move {
            loop {
                match listener.accept().await {
                    Ok((stream, _)) => {
                        let pending = Arc::clone(&pending);
                        let arrived = Arc::clone(&arrived);
                        tokio::spawn(async move {
                            if let Err(err) = recv_one(stream, &pending, &arrived).await {
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

    /// Wait up to `timeout` for the fd tagged with `key`.
    pub(crate) async fn take(&self, key: RegistrationKey, timeout: Duration) -> Option<OwnedFd> {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            {
                let mut pending = self.pending.lock().expect("fd pending map poisoned");
                if let Some(fd) = pending.remove(&key) {
                    return Some(fd);
                }
            }
            if tokio::time::timeout_at(deadline, self.arrived.notified())
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
        remove_stale_socket(&self.path);
    }
}

fn remove_stale_socket(path: &str) {
    match std::fs::remove_file(path) {
        Ok(()) => log::info!("fd side-channel: removed existing socket {path}"),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
        Err(err) => log::warn!("fd side-channel: failed to remove socket {path}: {err}"),
    }
}

async fn recv_one(
    stream: tokio::net::UnixStream,
    pending: &Mutex<HashMap<RegistrationKey, OwnedFd>>,
    arrived: &Notify,
) -> std::io::Result<()> {
    // SCM_RIGHTS needs recvmsg; wait until readable then use a blocking std socket.
    stream.readable().await?;
    let std_stream = stream.into_std()?;
    std_stream.set_nonblocking(false)?;
    let raw = std::os::fd::AsRawFd::as_raw_fd(&std_stream);
    let (key, fd) = recv_fd_with_key(raw)?;
    drop(std_stream);

    pending
        .lock()
        .expect("fd pending map poisoned")
        .insert(key, fd);
    arrived.notify_waiters();
    Ok(())
}

fn recv_fd_with_key(sock: RawFd) -> std::io::Result<(RegistrationKey, OwnedFd)> {
    let mut buf = [0u8; 512];
    let mut iov = libc::iovec {
        iov_base: buf.as_mut_ptr().cast(),
        iov_len: buf.len(),
    };
    let mut cmsg_space = [0u8; CMSG_SPACE_FOR_FD];
    let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
    msg.msg_iov = &mut iov;
    msg.msg_iovlen = 1;
    msg.msg_control = cmsg_space.as_mut_ptr().cast();
    msg.msg_controllen = cmsg_space.len();

    // SAFETY: iov and control buffers are valid for the duration of recvmsg.
    let n = unsafe { libc::recvmsg(sock, &mut msg, 0) };
    if n < 0 {
        return Err(std::io::Error::last_os_error());
    }
    if (msg.msg_flags & libc::MSG_CTRUNC) != 0 {
        return Err(std::io::Error::other("fd control message truncated"));
    }

    // SAFETY: control buffer was filled by recvmsg.
    let cmsg = unsafe { libc::CMSG_FIRSTHDR(&msg) };
    if cmsg.is_null() {
        return Err(std::io::Error::other("fd side-channel: no control message"));
    }
    // SAFETY: cmsg points into cmsg_space.
    let (level, ctype) = unsafe { ((*cmsg).cmsg_level, (*cmsg).cmsg_type) };
    if level != libc::SOL_SOCKET || ctype != libc::SCM_RIGHTS {
        return Err(std::io::Error::other(
            "fd side-channel: unexpected control message",
        ));
    }
    // SAFETY: SCM_RIGHTS payload is one RawFd.
    let raw_fd = unsafe { std::ptr::read_unaligned(libc::CMSG_DATA(cmsg).cast::<RawFd>()) };
    // SAFETY: ownership transferred by recvmsg.
    let fd = unsafe { OwnedFd::from_raw_fd(raw_fd) };

    let key = parse_key(&buf[..n as usize])?;
    Ok((key, fd))
}

fn parse_key(payload: &[u8]) -> std::io::Result<RegistrationKey> {
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
    Ok(RegistrationKey {
        instance_id,
        device_id,
    })
}

/// Bytes for one SCM_RIGHTS fd (`CMSG_SPACE(sizeof(int))` with margin).
const CMSG_SPACE_FOR_FD: usize = 64;
