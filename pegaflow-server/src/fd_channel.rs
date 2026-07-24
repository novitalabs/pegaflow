//! Host-local Unix socket that receives VMM allocation fds (SCM_RIGHTS) for
//! native `register_context_batch`. Wire payload: `instance_id\0device_id` plus
//! exactly one fd per connection.

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::os::fd::{FromRawFd, OwnedFd, RawFd};
use std::os::unix::fs::{FileTypeExt, MetadataExt};
use std::os::unix::net::UnixStream as ProbeUnixStream;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::net::UnixListener;
use tokio::sync::{Notify, Semaphore, watch};

const MAX_KEY_PAYLOAD: usize = 256;
const MAX_PENDING_FDS: usize = 1024;
const MAX_CONNECTIONS: usize = 64;
const PENDING_FD_TTL: Duration = Duration::from_secs(60);
#[cfg(not(test))]
const RECEIVE_TIMEOUT: Duration = Duration::from_secs(5);
#[cfg(test)]
const RECEIVE_TIMEOUT: Duration = Duration::from_millis(100);
const MAX_RECEIVED_FDS: usize = 16;
const CMSG_SPACE_FOR_FDS: usize = unsafe {
    // SAFETY: CMSG_SPACE is a pure layout calculation from the payload length.
    libc::CMSG_SPACE((MAX_RECEIVED_FDS * std::mem::size_of::<RawFd>()) as libc::c_uint) as usize
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct RegistrationKey {
    pub(crate) instance_id: String,
    pub(crate) device_id: i32,
}

struct PendingFd {
    fd: OwnedFd,
    inserted_at: Instant,
}

struct Inner {
    path: String,
    socket_identity: (u64, u64),
    pending: Mutex<HashMap<RegistrationKey, PendingFd>>,
    arrived: Notify,
    shutdown: watch::Sender<()>,
}

impl Drop for Inner {
    fn drop(&mut self) {
        let _ = self.shutdown.send(());
        match std::fs::symlink_metadata(&self.path) {
            Ok(metadata)
                if metadata.file_type().is_socket()
                    && (metadata.dev(), metadata.ino()) == self.socket_identity =>
            {
                if let Err(err) = std::fs::remove_file(&self.path) {
                    log::warn!(
                        "fd side-channel: failed to remove socket {}: {err}",
                        self.path
                    );
                }
            }
            Ok(_) => log::warn!(
                "fd side-channel: refusing to remove replaced path {}",
                self.path
            ),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => log::warn!(
                "fd side-channel: failed to inspect socket {}: {err}",
                self.path
            ),
        }
    }
}

/// UDS side-channel; the last handle stops the listener and unlinks the socket.
#[derive(Clone)]
pub struct FdChannel(Arc<Inner>);

impl FdChannel {
    pub fn bind(path: String) -> std::io::Result<Self> {
        let listener = match UnixListener::bind(&path) {
            Ok(listener) => listener,
            Err(err) if err.kind() == std::io::ErrorKind::AddrInUse => {
                remove_stale_socket(&path)?;
                UnixListener::bind(&path)?
            }
            Err(err) => return Err(err),
        };
        let metadata = std::fs::symlink_metadata(&path)?;
        let (shutdown, mut shutdown_rx) = watch::channel(());
        let connection_slots = Arc::new(Semaphore::new(MAX_CONNECTIONS));
        let channel = Self(Arc::new(Inner {
            path,
            socket_identity: (metadata.dev(), metadata.ino()),
            pending: Mutex::new(HashMap::new()),
            arrived: Notify::new(),
            shutdown,
        }));
        let inner = Arc::downgrade(&channel.0);
        tokio::spawn(async move {
            loop {
                let accepted = tokio::select! {
                    accepted = listener.accept() => accepted,
                    _ = shutdown_rx.changed() => break,
                };
                match accepted {
                    Ok((stream, _)) => {
                        let Ok(slot) = Arc::clone(&connection_slots).try_acquire_owned() else {
                            log::warn!("fd side-channel: connection capacity reached");
                            continue;
                        };
                        let inner = inner.clone();
                        let shutdown = shutdown_rx.clone();
                        tokio::spawn(async move {
                            let _slot = slot;
                            if let Err(err) = recv_one(stream, inner, shutdown).await {
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

    pub(crate) async fn take(&self, key: RegistrationKey, timeout: Duration) -> Option<OwnedFd> {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            // Create the notification future before checking the map so an
            // arrival cannot land between the check and subscription.
            let notified = self.0.arrived.notified();
            {
                let mut pending = self.0.pending.lock().expect("fd pending map poisoned");
                pending.retain(|_, entry| entry.inserted_at.elapsed() < PENDING_FD_TTL);
                if let Some(entry) = pending.remove(&key) {
                    return Some(entry.fd);
                }
            }
            if tokio::time::timeout_at(deadline, notified).await.is_err() {
                log::warn!(
                    "fd side-channel: timed out waiting for fd instance_id={} device_id={}",
                    key.instance_id,
                    key.device_id
                );
                return None;
            }
        }
    }
}

fn remove_stale_socket(path: &str) -> std::io::Result<()> {
    let metadata = std::fs::symlink_metadata(path)?;
    if !metadata.file_type().is_socket() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::AddrInUse,
            format!("fd side-channel path is not a socket: {path}"),
        ));
    }
    let identity = (metadata.dev(), metadata.ino());
    match ProbeUnixStream::connect(path) {
        Ok(_) => Err(std::io::Error::new(
            std::io::ErrorKind::AddrInUse,
            format!("fd side-channel already has a live listener: {path}"),
        )),
        Err(err) if err.kind() == std::io::ErrorKind::ConnectionRefused => {
            let current = std::fs::symlink_metadata(path)?;
            if !current.file_type().is_socket() || (current.dev(), current.ino()) != identity {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::AddrInUse,
                    format!("fd side-channel path changed while probing: {path}"),
                ));
            }
            std::fs::remove_file(path)
        }
        Err(err) => Err(err),
    }
}

async fn recv_one(
    stream: tokio::net::UnixStream,
    inner: std::sync::Weak<Inner>,
    mut shutdown: watch::Receiver<()>,
) -> std::io::Result<()> {
    tokio::select! {
        result = tokio::time::timeout(RECEIVE_TIMEOUT, stream.readable()) => {
            result.map_err(|_| std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "fd side-channel receive timed out",
            ))??;
        },
        _ = shutdown.changed() => {
            return Err(std::io::Error::other("fd side-channel closed"));
        }
    }
    let std_stream = stream.into_std()?;
    std_stream.set_nonblocking(false)?;
    let raw = std::os::fd::AsRawFd::as_raw_fd(&std_stream);
    let (key, fd) = recv_fd_with_key(raw)?;
    drop(std_stream);

    let inner = inner
        .upgrade()
        .ok_or_else(|| std::io::Error::other("fd side-channel closed"))?;
    let mut pending = inner.pending.lock().expect("fd pending map poisoned");
    pending.retain(|_, entry| entry.inserted_at.elapsed() < PENDING_FD_TTL);
    if pending.len() >= MAX_PENDING_FDS {
        return Err(std::io::Error::other(
            "fd side-channel: pending fd capacity reached",
        ));
    }
    match pending.entry(key) {
        Entry::Vacant(slot) => {
            slot.insert(PendingFd {
                fd,
                inserted_at: Instant::now(),
            });
            drop(pending);
            inner.arrived.notify_waiters();
            Ok(())
        }
        Entry::Occupied(slot) => {
            let key = slot.key();
            Err(std::io::Error::other(format!(
                "duplicate pending fd for instance_id={} device_id={}",
                key.instance_id, key.device_id
            )))
        }
    }
}

fn recv_fd_with_key(sock: RawFd) -> std::io::Result<(RegistrationKey, OwnedFd)> {
    let mut buf = [0u8; MAX_KEY_PAYLOAD];
    let mut iov = libc::iovec {
        iov_base: buf.as_mut_ptr().cast(),
        iov_len: buf.len(),
    };
    let mut cmsg_space = [0u8; CMSG_SPACE_FOR_FDS];
    let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
    msg.msg_iov = &mut iov;
    msg.msg_iovlen = 1;
    msg.msg_control = cmsg_space.as_mut_ptr().cast();
    msg.msg_controllen = cmsg_space.len();

    // SAFETY: iov and control buffers are valid for the duration of recvmsg.
    let n = unsafe { libc::recvmsg(sock, &mut msg, libc::MSG_CMSG_CLOEXEC) };
    if n < 0 {
        return Err(std::io::Error::last_os_error());
    }
    let mut received_fds = Vec::new();
    let mut unexpected_control = false;
    // SAFETY: CMSG_FIRSTHDR/CMSG_NXTHDR only walk the kernel-populated buffer.
    let mut cmsg = unsafe { libc::CMSG_FIRSTHDR(&msg) };
    while !cmsg.is_null() {
        // SAFETY: cmsg points inside msg_control.
        let header = unsafe { &*cmsg };
        if header.cmsg_level == libc::SOL_SOCKET && header.cmsg_type == libc::SCM_RIGHTS {
            let header_len = unsafe { libc::CMSG_LEN(0) as usize };
            if header.cmsg_len < header_len {
                return Err(std::io::Error::other(
                    "fd side-channel: malformed SCM_RIGHTS message",
                ));
            }
            let payload_len = header.cmsg_len - header_len;
            if payload_len % std::mem::size_of::<RawFd>() != 0 {
                return Err(std::io::Error::other(
                    "fd side-channel: malformed SCM_RIGHTS payload",
                ));
            }
            for index in 0..payload_len / std::mem::size_of::<RawFd>() {
                // SAFETY: the kernel reported a complete RawFd at this index.
                let raw_fd = unsafe {
                    std::ptr::read_unaligned(libc::CMSG_DATA(cmsg).cast::<RawFd>().add(index))
                };
                // SAFETY: ownership of every received descriptor transferred to us.
                received_fds.push(unsafe { OwnedFd::from_raw_fd(raw_fd) });
            }
        } else {
            unexpected_control = true;
        }
        // SAFETY: msg and the control buffer remain valid.
        cmsg = unsafe { libc::CMSG_NXTHDR(&msg, cmsg) };
    }
    if (msg.msg_flags & (libc::MSG_CTRUNC | libc::MSG_TRUNC)) != 0
        || unexpected_control
        || received_fds.len() != 1
    {
        return Err(std::io::Error::other(
            "fd side-channel: expected exactly one complete fd message",
        ));
    }
    let fd = received_fds.pop().expect("one fd checked above");
    Ok((parse_key(&buf[..n as usize])?, fd))
}

fn parse_key(payload: &[u8]) -> std::io::Result<RegistrationKey> {
    let mut parts = payload.splitn(2, |&byte| byte == 0);
    let instance_id = std::str::from_utf8(parts.next().unwrap_or_default())
        .map_err(|_| std::io::Error::other("fd side-channel: instance_id not utf-8"))?;
    if !valid_instance_id(instance_id) {
        return Err(std::io::Error::other(
            "fd side-channel: invalid instance_id",
        ));
    }
    let device_id = parts
        .next()
        .and_then(|bytes| std::str::from_utf8(bytes).ok())
        .and_then(|text| text.trim_end_matches('\0').parse().ok())
        .ok_or_else(|| std::io::Error::other("fd side-channel: bad device_id"))?;
    Ok(RegistrationKey {
        instance_id: instance_id.to_string(),
        device_id,
    })
}

pub(crate) fn valid_instance_id(instance_id: &str) -> bool {
    !instance_id.is_empty()
        && instance_id.len() <= 128
        && instance_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::os::fd::AsRawFd;
    use std::os::unix::net::{UnixListener as StdUnixListener, UnixStream as StdUnixStream};
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_sock_path(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        std::env::temp_dir().join(format!("pegaflow-fd-test-{label}-{nanos}.sock"))
    }

    fn key(instance_id: &str, device_id: i32) -> RegistrationKey {
        RegistrationKey {
            instance_id: instance_id.to_string(),
            device_id,
        }
    }

    fn tagged_pipe(tag: u8) -> (OwnedFd, std::fs::File) {
        let mut fds = [0 as RawFd; 2];
        // SAFETY: pipe(2) writes two owned descriptors into fds.
        assert_eq!(unsafe { libc::pipe(fds.as_mut_ptr()) }, 0);
        // SAFETY: both descriptors are open and uniquely owned here.
        let (read_end, write_end) =
            unsafe { (OwnedFd::from_raw_fd(fds[0]), OwnedFd::from_raw_fd(fds[1])) };
        let mut write_file = std::fs::File::from(write_end);
        write_file.write_all(&[tag]).unwrap();
        (read_end, write_file)
    }

    fn send_fds(path: &str, key: &RegistrationKey, fds: &[RawFd]) {
        let stream = StdUnixStream::connect(path).unwrap();
        let mut payload = format!("{}\0{}", key.instance_id, key.device_id).into_bytes();
        let mut iov = libc::iovec {
            iov_base: payload.as_mut_ptr().cast(),
            iov_len: payload.len(),
        };
        let rights_bytes = std::mem::size_of_val(fds);
        let mut control =
            vec![0u8; unsafe { libc::CMSG_SPACE(rights_bytes as libc::c_uint) as usize }];
        let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
        msg.msg_iov = &mut iov;
        msg.msg_iovlen = 1;
        msg.msg_control = control.as_mut_ptr().cast();
        msg.msg_controllen = control.len();
        // SAFETY: control and iov remain live for sendmsg.
        unsafe {
            let cmsg = libc::CMSG_FIRSTHDR(&msg);
            (*cmsg).cmsg_level = libc::SOL_SOCKET;
            (*cmsg).cmsg_type = libc::SCM_RIGHTS;
            (*cmsg).cmsg_len = libc::CMSG_LEN(rights_bytes as libc::c_uint) as _;
            for (index, fd) in fds.iter().enumerate() {
                std::ptr::write_unaligned(libc::CMSG_DATA(cmsg).cast::<RawFd>().add(index), *fd);
            }
            msg.msg_controllen = (*cmsg).cmsg_len;
            assert!(libc::sendmsg(stream.as_raw_fd(), &msg, 0) >= 0);
        }
    }

    fn read_tag(fd: &OwnedFd) -> u8 {
        let mut tag = [0];
        std::fs::File::from(fd.try_clone().unwrap())
            .read_exact(&mut tag)
            .unwrap();
        tag[0]
    }

    #[test]
    fn key_uses_uuid_instance_and_device() {
        assert_eq!(
            parse_key(b"018f8f75-b82e-7c10-a7d4-01abc2345678\x003").unwrap(),
            key("018f8f75-b82e-7c10-a7d4-01abc2345678", 3)
        );
        assert!(parse_key(b"bad id\x000").is_err());
    }

    #[tokio::test]
    async fn fd_before_take_is_delivered() {
        let path = temp_sock_path("before");
        let channel = FdChannel::bind(path.to_string_lossy().into_owned()).unwrap();
        let registration = key("instance-a", 0);
        let (fd, _writer) = tagged_pipe(0xA1);
        send_fds(path.to_str().unwrap(), &registration, &[fd.as_raw_fd()]);

        let received = channel
            .take(registration, Duration::from_secs(1))
            .await
            .unwrap();
        assert_eq!(read_tag(&received), 0xA1);
    }

    #[tokio::test]
    async fn take_waits_for_later_fd() {
        let path = temp_sock_path("after");
        let path_string = path.to_string_lossy().into_owned();
        let channel = FdChannel::bind(path_string.clone()).unwrap();
        let registration = key("instance-b", 0);
        let waiter = tokio::spawn({
            let channel = channel.clone();
            let registration = registration.clone();
            async move { channel.take(registration, Duration::from_secs(1)).await }
        });
        tokio::task::yield_now().await;
        let (fd, _writer) = tagged_pipe(0xB2);
        send_fds(&path_string, &registration, &[fd.as_raw_fd()]);

        assert_eq!(read_tag(&waiter.await.unwrap().unwrap()), 0xB2);
    }

    #[tokio::test]
    async fn duplicate_and_multi_fd_messages_do_not_replace_first() {
        let path = temp_sock_path("duplicate");
        let path_string = path.to_string_lossy().into_owned();
        let channel = FdChannel::bind(path_string.clone()).unwrap();
        let registration = key("instance-c", 0);
        let (first, mut first_writer) = tagged_pipe(0xC3);
        let (second, mut second_writer) = tagged_pipe(0xD4);
        send_fds(&path_string, &registration, &[first.as_raw_fd()]);
        tokio::time::sleep(Duration::from_millis(20)).await;
        send_fds(
            &path_string,
            &registration,
            &[first.as_raw_fd(), second.as_raw_fd()],
        );

        let received = channel
            .take(registration, Duration::from_secs(1))
            .await
            .unwrap();
        assert_eq!(read_tag(&received), 0xC3);
        drop(first);
        drop(second);
        drop(received);
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(first_writer.write_all(&[1]).is_err());
        assert!(second_writer.write_all(&[1]).is_err());
    }

    #[tokio::test]
    async fn last_handle_stops_listener_and_removes_path() {
        let path = temp_sock_path("drop");
        let channel = FdChannel::bind(path.to_string_lossy().into_owned()).unwrap();
        let mut idle = StdUnixStream::connect(&path).unwrap();
        tokio::time::sleep(RECEIVE_TIMEOUT + Duration::from_millis(20)).await;
        let mut byte = [0];
        assert_eq!(
            idle.read(&mut byte).unwrap(),
            0,
            "idle connection must close"
        );
        drop(channel);
        tokio::time::timeout(Duration::from_secs(1), async {
            while path.exists() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn bind_recovers_only_stale_socket_paths() {
        let stale_path = temp_sock_path("stale");
        drop(StdUnixListener::bind(&stale_path).unwrap());
        drop(FdChannel::bind(stale_path.to_string_lossy().into_owned()).unwrap());

        let live_path = temp_sock_path("live");
        let live = StdUnixListener::bind(&live_path).unwrap();
        assert_eq!(
            FdChannel::bind(live_path.to_string_lossy().into_owned())
                .err()
                .unwrap()
                .kind(),
            std::io::ErrorKind::AddrInUse
        );
        drop(live);
        std::fs::remove_file(live_path).unwrap();

        let file_path = temp_sock_path("file");
        std::fs::write(&file_path, b"keep").unwrap();
        assert!(FdChannel::bind(file_path.to_string_lossy().into_owned()).is_err());
        assert_eq!(std::fs::read(&file_path).unwrap(), b"keep");
        std::fs::remove_file(file_path).unwrap();
    }
}
