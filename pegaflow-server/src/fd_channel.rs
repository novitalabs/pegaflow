//! Host-local Unix socket that receives VMM allocation fds (SCM_RIGHTS) for
//! native `register_context_batch`. Wire payload: `instance_id\0device_id` plus
//! one fd per connection.

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::os::fd::{FromRawFd, OwnedFd, RawFd};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::net::UnixListener;
use tokio::sync::Notify;

/// `instance_id\0device_id` payload; generous so long instance ids do not truncate.
const MAX_KEY_PAYLOAD: usize = 4096;
/// Control buffer for one SCM_RIGHTS fd.
const CMSG_SPACE_FOR_FD: usize = unsafe {
    // SAFETY: CMSG_SPACE is a pure layout calculation from the payload length.
    libc::CMSG_SPACE(std::mem::size_of::<RawFd>() as libc::c_uint) as usize
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct RegistrationKey {
    pub(crate) instance_id: String,
    pub(crate) device_id: i32,
}

struct Inner {
    path: String,
    pending: Mutex<HashMap<RegistrationKey, OwnedFd>>,
    arrived: Notify,
}

impl Drop for Inner {
    fn drop(&mut self) {
        remove_stale_socket(&self.path);
    }
}

/// UDS side-channel; last `Arc` drop unlinks the socket file.
#[derive(Clone)]
pub struct FdChannel(Arc<Inner>);

impl FdChannel {
    pub fn bind(path: String) -> std::io::Result<Self> {
        remove_stale_socket(&path);
        let listener = UnixListener::bind(&path)?;
        let channel = Self(Arc::new(Inner {
            path,
            pending: Mutex::new(HashMap::new()),
            arrived: Notify::new(),
        }));
        let inner = Arc::clone(&channel.0);
        tokio::spawn(async move {
            loop {
                match listener.accept().await {
                    Ok((stream, _)) => {
                        let inner = Arc::clone(&inner);
                        tokio::spawn(async move {
                            if let Err(err) = recv_one(stream, &inner).await {
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
                let mut pending = self.0.pending.lock().expect("fd pending map poisoned");
                if let Some(fd) = pending.remove(&key) {
                    return Some(fd);
                }
            }
            if tokio::time::timeout_at(deadline, self.0.arrived.notified())
                .await
                .is_err()
            {
                return None;
            }
        }
    }
}

fn remove_stale_socket(path: &str) {
    match std::fs::remove_file(path) {
        Ok(()) => log::info!("fd side-channel: removed existing socket {path}"),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
        Err(err) => log::warn!("fd side-channel: failed to remove socket {path}: {err}"),
    }
}

async fn recv_one(stream: tokio::net::UnixStream, inner: &Inner) -> std::io::Result<()> {
    // SCM_RIGHTS needs recvmsg; wait until readable then use a blocking std socket.
    stream.readable().await?;
    let std_stream = stream.into_std()?;
    std_stream.set_nonblocking(false)?;
    let raw = std::os::fd::AsRawFd::as_raw_fd(&std_stream);
    let (key, fd) = recv_fd_with_key(raw)?;
    drop(std_stream);

    let mut pending = inner.pending.lock().expect("fd pending map poisoned");
    match pending.entry(key) {
        Entry::Vacant(slot) => {
            slot.insert(fd);
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::os::fd::AsRawFd;
    use std::os::unix::net::UnixStream as StdUnixStream;
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

    /// Pipe whose read end carries `tag` once written on the write end.
    /// Returns `(read_end_to_send, write_end_keep_alive)`.
    fn tagged_pipe(tag: u8) -> (OwnedFd, std::fs::File) {
        let mut fds = [0 as RawFd; 2];
        // SAFETY: pipe(2) with a two-slot output array.
        assert_eq!(unsafe { libc::pipe(fds.as_mut_ptr()) }, 0);
        // SAFETY: pipe ends are open and owned here.
        let (read_end, write_end) =
            unsafe { (OwnedFd::from_raw_fd(fds[0]), OwnedFd::from_raw_fd(fds[1])) };
        let mut write_file = std::fs::File::from(write_end);
        write_file.write_all(&[tag]).expect("write tag into pipe");
        (read_end, write_file)
    }

    fn send_fd(sock_path: &str, reg: &RegistrationKey, pass: RawFd) {
        let stream = StdUnixStream::connect(sock_path).expect("connect fd side-channel");
        let payload = format!("{}\0{}", reg.instance_id, reg.device_id);
        let mut payload_bytes = payload.into_bytes();
        let mut iov = libc::iovec {
            iov_base: payload_bytes.as_mut_ptr().cast(),
            iov_len: payload_bytes.len(),
        };
        let mut cmsg_buf = vec![0u8; CMSG_SPACE_FOR_FD];
        let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
        msg.msg_iov = &mut iov;
        msg.msg_iovlen = 1;
        msg.msg_control = cmsg_buf.as_mut_ptr().cast();
        msg.msg_controllen = cmsg_buf.len();

        // SAFETY: msg_control points at cmsg_buf; CMSG_* walk that buffer only.
        unsafe {
            let cmsg = libc::CMSG_FIRSTHDR(&msg);
            assert!(!cmsg.is_null());
            (*cmsg).cmsg_level = libc::SOL_SOCKET;
            (*cmsg).cmsg_type = libc::SCM_RIGHTS;
            (*cmsg).cmsg_len = libc::CMSG_LEN(std::mem::size_of::<RawFd>() as libc::c_uint) as _;
            std::ptr::write_unaligned(libc::CMSG_DATA(cmsg).cast::<RawFd>(), pass);
            msg.msg_controllen = (*cmsg).cmsg_len;
            let n = libc::sendmsg(stream.as_raw_fd(), &msg, 0);
            assert!(n >= 0, "sendmsg: {}", std::io::Error::last_os_error());
        }
    }

    fn read_tag(fd: &OwnedFd) -> u8 {
        let mut buf = [0u8; 1];
        let mut file = std::fs::File::from(fd.try_clone().expect("clone fd"));
        file.read_exact(&mut buf).expect("read tag");
        buf[0]
    }

    #[test]
    fn parse_key_splits_instance_and_device() {
        // "engine-abc" + NUL + "3" (\x00 is null; next char is '3')
        let k = parse_key(b"engine-abc\x003").unwrap();
        assert_eq!(k.instance_id, "engine-abc");
        assert_eq!(k.device_id, 3);
    }

    #[test]
    fn parse_key_rejects_missing_nul() {
        assert!(parse_key(b"no-separator").is_err());
    }

    #[tokio::test]
    async fn fd_before_take_delivers_tagged_pipe() {
        let path = temp_sock_path("before");
        let channel = FdChannel::bind(path.to_string_lossy().into_owned()).unwrap();
        let reg = key("inst-a", 0);
        let (pass, _hold_write) = tagged_pipe(0xA1);
        send_fd(path.to_str().unwrap(), &reg, pass.as_raw_fd());

        let got = channel
            .take(reg, Duration::from_secs(2))
            .await
            .expect("fd should be pending");
        assert_eq!(read_tag(&got), 0xA1);
    }

    #[tokio::test]
    async fn take_waits_when_fd_arrives_later() {
        let path = temp_sock_path("wait");
        let channel = FdChannel::bind(path.to_string_lossy().into_owned()).unwrap();
        let reg = key("inst-b", 1);
        let path_str = path.to_string_lossy().into_owned();
        let reg_send = reg.clone();

        let take = tokio::spawn({
            let channel = channel.clone();
            async move { channel.take(reg, Duration::from_secs(2)).await }
        });
        tokio::time::sleep(Duration::from_millis(50)).await;

        let (pass, _hold_write) = tagged_pipe(0xB2);
        tokio::task::spawn_blocking(move || {
            send_fd(&path_str, &reg_send, pass.as_raw_fd());
            // Keep pass alive until sendmsg returns (SCM_RIGHTS dups the fd).
            drop(pass);
        })
        .await
        .unwrap();

        let got = take.await.unwrap().expect("fd after wait");
        assert_eq!(read_tag(&got), 0xB2);
    }

    #[tokio::test]
    async fn take_times_out_without_fd() {
        let path = temp_sock_path("timeout");
        let channel = FdChannel::bind(path.to_string_lossy().into_owned()).unwrap();
        let got = channel
            .take(key("missing", 0), Duration::from_millis(80))
            .await;
        assert!(got.is_none());
    }

    #[tokio::test]
    async fn duplicate_key_keeps_first_fd() {
        let path = temp_sock_path("dup");
        let path_str = path.to_string_lossy().into_owned();
        let channel = FdChannel::bind(path_str.clone()).unwrap();
        let reg = key("inst-dup", 0);

        let (first, _w1) = tagged_pipe(0x11);
        let (second, _w2) = tagged_pipe(0x22);
        send_fd(&path_str, &reg, first.as_raw_fd());
        // Allow the first connection to be accepted and inserted.
        tokio::time::sleep(Duration::from_millis(50)).await;
        send_fd(&path_str, &reg, second.as_raw_fd());
        tokio::time::sleep(Duration::from_millis(50)).await;

        let got = channel
            .take(reg, Duration::from_secs(1))
            .await
            .expect("first fd remains");
        assert_eq!(
            read_tag(&got),
            0x11,
            "duplicate must not replace pending fd"
        );
    }
}
