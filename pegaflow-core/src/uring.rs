// Copyright 2025 foyer Project Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Adapted for PegaFlow: simplified to a single-file engine with blocking read/write APIs.

use io_uring::{IoUring, opcode, types::Fd};
use log::{info, warn};
use std::io;
use std::os::unix::io::RawFd;
use std::sync::mpsc;
use std::thread::JoinHandle;
use tokio::sync::oneshot;

/// Configuration for io_uring engine.
#[derive(Debug, Clone)]
pub struct UringConfig {
    pub threads: usize,
    pub io_depth: usize,
    /// Enable SQ polling; requires kernel support. Off by default.
    pub sqpoll: bool,
    /// Idle time in ms before the kernel SQ poll thread sleeps.
    pub sqpoll_idle: u32,
}

impl Default for UringConfig {
    fn default() -> Self {
        Self {
            threads: 1,
            io_depth: 128,
            sqpoll: false,
            sqpoll_idle: 10,
        }
    }
}

#[derive(Clone, Copy)]
enum IoType {
    Read,
    Write,
    Readv,
    Writev,
}

struct IoCtx {
    io_type: IoType,
    ptr: *mut u8,
    len: usize,
    offset: u64,
    complete: oneshot::Sender<io::Result<usize>>,
    iovecs: Option<Box<[libc::iovec]>>,
}

unsafe impl Send for IoCtx {}
unsafe impl Sync for IoCtx {}

struct UringShard {
    rx: mpsc::Receiver<IoCtx>,
    uring: IoUring,
    io_depth: usize,
}

impl UringShard {
    fn run(mut self, fd: RawFd) {
        let mut inflight = 0usize;
        let mut channel_closed = false;

        loop {
            // Try to prepare as many as possible up to io_depth.
            while inflight < self.io_depth && !channel_closed {
                let next = if inflight == 0 {
                    // If idle, block until we have at least one ctx.
                    match self.rx.recv() {
                        Ok(ctx) => Some(ctx),
                        Err(e) => {
                            warn!("io_uring shard recv closed: {e}");
                            channel_closed = true;
                            None
                        }
                    }
                } else {
                    match self.rx.try_recv() {
                        Ok(ctx) => Some(ctx),
                        Err(mpsc::TryRecvError::Disconnected) => {
                            channel_closed = true;
                            None
                        }
                        Err(mpsc::TryRecvError::Empty) => None,
                    }
                };
                let ctx = match next {
                    Some(ctx) => ctx,
                    None => break,
                };

                let fd = Fd(fd);
                let sqe = match ctx.io_type {
                    IoType::Read => opcode::Read::new(fd, ctx.ptr, ctx.len as _)
                        .offset(ctx.offset)
                        .build(),
                    IoType::Write => opcode::Write::new(fd, ctx.ptr, ctx.len as _)
                        .offset(ctx.offset)
                        .build(),
                    IoType::Readv => {
                        let iovecs_ptr = ctx
                            .iovecs
                            .as_ref()
                            .expect("readv must have iovecs")
                            .as_ptr();
                        opcode::Readv::new(fd, iovecs_ptr, ctx.len as _)
                            .offset(ctx.offset)
                            .build()
                    }
                    IoType::Writev => {
                        // Safety: iovecs must remain valid until completion
                        let iovecs_ptr = ctx
                            .iovecs
                            .as_ref()
                            .expect("writev must have iovecs")
                            .as_ptr();
                        opcode::Writev::new(fd, iovecs_ptr, ctx.len as _)
                            .offset(ctx.offset)
                            .build()
                    }
                };

                let data = Box::into_raw(Box::new(ctx)) as u64;
                let sqe = sqe.user_data(data);
                // Safety: we keep ctx boxed and rely on user_data to free it in completion.
                unsafe {
                    self.uring
                        .submission()
                        .push(&sqe)
                        .expect("submission queue full")
                };
                inflight += 1;
            }

            // If channel is closed and no inflight requests, exit gracefully.
            if channel_closed && inflight == 0 {
                info!("io_uring shard shutting down gracefully");
                return;
            }

            if inflight == 0 {
                continue;
            }

            // Submit and wait for at least one completion.
            if let Err(e) = self.uring.submit_and_wait(1) {
                // Fatal error; drop all inflight requests.
                warn!("io_uring submit_and_wait failed: {}, shutting down", e);
                while let Some(cqe) = self.uring.completion().next() {
                    let data = cqe.user_data();
                    if data != 0 {
                        // Safety: data was produced from Box::into_raw.
                        let ctx = unsafe { Box::from_raw(data as *mut IoCtx) };
                        let _ = ctx.complete.send(Err(io::Error::other(format!(
                            "io_uring submit failed: {e}"
                        ))));
                    }
                }
                info!("io_uring shard shut down due to submit error");
                return;
            }

            let mut completed = 0usize;
            for cqe in self.uring.completion() {
                completed += 1;
                let data = cqe.user_data();
                if data == 0 {
                    warn!("io_uring completion with user_data=0, res={}", cqe.result());
                    continue;
                }
                let ctx = unsafe { Box::from_raw(data as *mut IoCtx) };
                let res = cqe.result();
                let send_res = if res < 0 {
                    Err(io::Error::from_raw_os_error(-res))
                } else {
                    Ok(res as usize)
                };
                let _ = ctx.complete.send(send_res);
            }
            inflight = inflight.saturating_sub(completed);
        }
    }
}

/// io_uring based engine for single-file read/write.
pub struct UringIoEngine {
    txs: Vec<mpsc::SyncSender<IoCtx>>,
    #[allow(dead_code)]
    handles: Vec<JoinHandle<()>>,
}

impl UringIoEngine {
    pub fn new(fd: RawFd, cfg: UringConfig) -> io::Result<Self> {
        if cfg.threads == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "threads must be > 0",
            ));
        }

        let mut txs = Vec::with_capacity(cfg.threads);
        let mut handles = Vec::with_capacity(cfg.threads);

        for _ in 0..cfg.threads {
            let (tx, rx) = mpsc::sync_channel(cfg.io_depth * 2);
            let mut builder = IoUring::builder();
            if cfg.sqpoll {
                builder.setup_sqpoll(cfg.sqpoll_idle);
            }
            let uring = builder.build(cfg.io_depth as u32)?;
            let shard = UringShard {
                rx,
                uring,
                io_depth: cfg.io_depth,
            };
            let handle = std::thread::Builder::new()
                .name("pegaflow-uring".to_string())
                .spawn(move || shard.run(fd))?;
            txs.push(tx);
            handles.push(handle);
        }

        Ok(Self { txs, handles })
    }

    fn pick_tx(&self, offset: u64) -> &mpsc::SyncSender<IoCtx> {
        let idx = if self.txs.len() == 1 {
            0
        } else {
            (offset as usize / 4096) % self.txs.len()
        };
        &self.txs[idx]
    }

    pub fn read_at_async(
        &self,
        ptr: *mut u8,
        len: usize,
        offset: u64,
    ) -> io::Result<oneshot::Receiver<io::Result<usize>>> {
        let (tx, rx) = oneshot::channel();
        let ctx = IoCtx {
            io_type: IoType::Read,
            ptr,
            len,
            offset,
            complete: tx,
            iovecs: None,
        };
        self.pick_tx(offset).send(ctx).map_err(|e| {
            io::Error::new(
                io::ErrorKind::BrokenPipe,
                format!("io_uring read send failed: {e}"),
            )
        })?;
        Ok(rx)
    }

    pub fn write_at_async(
        &self,
        ptr: *const u8,
        len: usize,
        offset: u64,
    ) -> io::Result<oneshot::Receiver<io::Result<usize>>> {
        let (tx, rx) = oneshot::channel();
        let ctx = IoCtx {
            io_type: IoType::Write,
            ptr: ptr as *mut u8,
            len,
            offset,
            complete: tx,
            iovecs: None,
        };
        self.pick_tx(offset).send(ctx).map_err(|e| {
            io::Error::new(
                io::ErrorKind::BrokenPipe,
                format!("io_uring write send failed: {e}"),
            )
        })?;
        Ok(rx)
    }

    /// Vectorized read (readv) - reads into multiple buffers in a single syscall.
    ///
    /// # Arguments
    /// * `iovecs` - Array of (ptr, len) pairs to read into sequentially
    /// * `offset` - File offset to start reading
    ///
    /// # Safety
    /// Caller must ensure all buffer pointers remain valid until the returned receiver completes.
    pub fn readv_at_async(
        &self,
        iovecs: Vec<(*mut u8, usize)>,
        offset: u64,
    ) -> io::Result<oneshot::Receiver<io::Result<usize>>> {
        if iovecs.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "readv requires at least one iovec",
            ));
        }

        let iovecs_libc: Box<[libc::iovec]> = iovecs
            .iter()
            .map(|(ptr, len)| libc::iovec {
                iov_base: *ptr as *mut libc::c_void,
                iov_len: *len,
            })
            .collect();

        let iovec_count = iovecs_libc.len();
        let (tx, rx) = oneshot::channel();
        let ctx = IoCtx {
            io_type: IoType::Readv,
            ptr: std::ptr::null_mut(),
            len: iovec_count,
            offset,
            complete: tx,
            iovecs: Some(iovecs_libc),
        };

        self.pick_tx(offset).send(ctx).map_err(|e| {
            io::Error::new(
                io::ErrorKind::BrokenPipe,
                format!("io_uring readv send failed: {e}"),
            )
        })?;
        Ok(rx)
    }

    /// Vectorized write (writev) - writes multiple buffers in a single syscall.
    ///
    /// # Arguments
    /// * `iovecs` - Array of (ptr, len) pairs to write sequentially
    /// * `offset` - File offset to start writing
    ///
    /// # Safety
    /// Caller must ensure all buffer pointers remain valid until the returned receiver completes.
    pub fn writev_at_async(
        &self,
        iovecs: Vec<(*const u8, usize)>,
        offset: u64,
    ) -> io::Result<oneshot::Receiver<io::Result<usize>>> {
        if iovecs.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "writev requires at least one iovec",
            ));
        }

        // Convert to libc::iovec
        let iovecs_libc: Box<[libc::iovec]> = iovecs
            .iter()
            .map(|(ptr, len)| libc::iovec {
                iov_base: *ptr as *mut libc::c_void,
                iov_len: *len,
            })
            .collect();

        let iovec_count = iovecs_libc.len();
        let (tx, rx) = oneshot::channel();
        let ctx = IoCtx {
            io_type: IoType::Writev,
            ptr: std::ptr::null_mut(), // not used for writev
            len: iovec_count,          // number of iovecs
            offset,
            complete: tx,
            iovecs: Some(iovecs_libc),
        };

        self.pick_tx(offset).send(ctx).map_err(|e| {
            io::Error::new(
                io::ErrorKind::BrokenPipe,
                format!("io_uring writev send failed: {e}"),
            )
        })?;
        Ok(rx)
    }
}

impl Drop for UringIoEngine {
    fn drop(&mut self) {
        // Drop senders to unblock shards, then join.
        self.txs.clear();
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}
