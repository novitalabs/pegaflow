//! Shared-memory synchronization state for async KV cache loading.
//!
//! Only `LoadState` and `PrepareLoadState` are supported. Other platforms are not
//! supported; compilation will fail outside x86_64 Linux.

#[cfg(not(all(target_arch = "x86_64", target_os = "linux")))]
compile_error!("shared sync state is only supported on x86_64 Linux");

use shared_memory::{Shmem, ShmemConf, ShmemError};
use std::{
    alloc::Layout,
    fmt,
    ptr::NonNull,
    sync::atomic::{AtomicI64, AtomicU32, AtomicU64, Ordering},
};
use uuid::Uuid;

/// State values for LoadState.
pub const LOAD_STATE_PENDING: i64 = 0;
pub const LOAD_STATE_SUCCESS: i64 = 1;
pub const LOAD_STATE_ERROR: i64 = -1;

/// State values for PrepareLoadState.
///
/// ```text
///                    +------+
///                    |ERROR |
///                    +------+
///                       ^
///                       |
/// +---------+       +-------+       +---------+
/// | request | ----> |PENDING| ----> |NO_PLAN  |
/// +---------+       +-------+       +---------+
///                       |
///                       v
///                    +------+
///                    | PLAN |
///                    +------+
/// ```
///
/// `NO_PLAN` means no external KV hit, so no load plan is returned.
pub const PREPARE_LOAD_STATE_PENDING: i64 = 0;
pub const PREPARE_LOAD_STATE_NO_PLAN: i64 = 1;
pub const PREPARE_LOAD_STATE_PLAN: i64 = 2;
const PREPARE_LOAD_STATE_ERROR: i64 = -1;

/// Magic and version for LoadState header validation.
const LOAD_STATE_MAGIC: u32 = 0x5046_4c44; // 'PFLD'
pub const LOAD_STATE_VERSION: u32 = 1;
const PREPARE_LOAD_STATE_MAGIC: u32 = 0x5046_4c50; // 'PFLP'
pub const PREPARE_LOAD_STATE_VERSION: u32 = 1;

/// Bytes reserved at the start of the mapping to store the aligned offset.
const LOAD_STATE_META_SIZE: usize = std::mem::size_of::<usize>();
// Only support 64-bit systems for now.
const _: () = assert!(std::mem::size_of::<usize>() == 8);

#[repr(C)]
struct LoadStateHeader {
    magic: AtomicU32,
    version: AtomicU32,
}

/// In-memory layout for LoadState.
#[repr(C, align(8))]
struct LoadStateMem {
    header: LoadStateHeader,
    state: AtomicI64,
}

#[repr(C)]
struct PrepareLoadStateHeader {
    magic: AtomicU32,
    version: AtomicU32,
}

/// In-memory layout for PrepareLoadState.
#[repr(C, align(8))]
struct PrepareLoadStateMem {
    header: PrepareLoadStateHeader,
    state: AtomicI64,
    num_tokens: AtomicU64,
    plan_id: AtomicU64,
}

/// Detailed error type for LoadState creation/attachment.
#[derive(Debug)]
pub enum LoadStateError {
    CreateShmem(ShmemError),
    OpenShmem(ShmemError),
    MappingTooSmall { actual: usize, required: usize },
    Overflow,
    InvalidOffset { offset: usize, len: usize },
    Misaligned { ptr: usize, align: usize },
    NullPointer,
    InvalidHeader { magic: u32, version: u32 },
}

#[derive(Debug)]
pub enum PrepareLoadStateError {
    CreateShmem(ShmemError),
    OpenShmem(ShmemError),
    MappingTooSmall { actual: usize, required: usize },
    Overflow,
    InvalidOffset { offset: usize, len: usize },
    Misaligned { ptr: usize, align: usize },
    NullPointer,
    InvalidHeader { magic: u32, version: u32 },
}

impl fmt::Display for LoadStateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadStateError::CreateShmem(err) => write!(f, "failed to create shared memory: {err}"),
            LoadStateError::OpenShmem(err) => write!(f, "failed to open shared memory: {err}"),
            LoadStateError::MappingTooSmall { actual, required } => write!(
                f,
                "shared memory too small (actual={actual} bytes, required={required} bytes)"
            ),
            LoadStateError::Overflow => {
                f.write_str("address calculation overflowed when aligning LoadState mapping")
            }
            LoadStateError::InvalidOffset { offset, len } => write!(
                f,
                "invalid LoadState offset (offset={offset}, len={len}) recorded in shared memory"
            ),
            LoadStateError::Misaligned { ptr, align } => {
                write!(f, "aligned pointer {ptr:#x} is not {align}-byte aligned")
            }
            LoadStateError::NullPointer => {
                f.write_str("aligned LoadState pointer resolved to null")
            }
            LoadStateError::InvalidHeader { magic, version } => write!(
                f,
                "invalid LoadState header (magic={magic:#x}, version={version}, expected magic={LOAD_STATE_MAGIC:#x}, version={LOAD_STATE_VERSION})"
            ),
        }
    }
}

impl std::error::Error for LoadStateError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            LoadStateError::CreateShmem(err) | LoadStateError::OpenShmem(err) => Some(err),
            _ => None,
        }
    }
}

impl fmt::Display for PrepareLoadStateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PrepareLoadStateError::CreateShmem(err) => {
                write!(f, "failed to create prepare shared memory: {err}")
            }
            PrepareLoadStateError::OpenShmem(err) => {
                write!(f, "failed to open prepare shared memory: {err}")
            }
            PrepareLoadStateError::MappingTooSmall { actual, required } => write!(
                f,
                "prepare shared memory too small (actual={actual} bytes, required={required} bytes)"
            ),
            PrepareLoadStateError::Overflow => {
                f.write_str("address calculation overflowed when aligning PrepareLoadState mapping")
            }
            PrepareLoadStateError::InvalidOffset { offset, len } => write!(
                f,
                "invalid PrepareLoadState offset (offset={offset}, len={len}) recorded in shared memory"
            ),
            PrepareLoadStateError::Misaligned { ptr, align } => {
                write!(
                    f,
                    "aligned PrepareLoadState pointer {ptr:#x} is not {align}-byte aligned"
                )
            }
            PrepareLoadStateError::NullPointer => {
                f.write_str("aligned PrepareLoadState pointer resolved to null")
            }
            PrepareLoadStateError::InvalidHeader { magic, version } => write!(
                f,
                "invalid PrepareLoadState header (magic={magic:#x}, version={version}, expected magic={PREPARE_LOAD_STATE_MAGIC:#x}, version={PREPARE_LOAD_STATE_VERSION})"
            ),
        }
    }
}

impl std::error::Error for PrepareLoadStateError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            PrepareLoadStateError::CreateShmem(err) | PrepareLoadStateError::OpenShmem(err) => {
                Some(err)
            }
            _ => None,
        }
    }
}

/// Batch-level synchronization state for async KV cache loading.
///
/// The connector creates this, passes the `shm_name` to the server, and
/// periodically polls `get()` to see whether the async load completed.
pub struct LoadState {
    shmem: Shmem,
    ptr: NonNull<LoadStateMem>,
}

pub struct PrepareLoadSnapshot {
    pub state: i64,
    pub num_tokens: u64,
    pub plan_id: u64,
}

/// Request-level synchronization state for async prepare-load.
pub struct PrepareLoadState {
    shmem: Shmem,
    ptr: NonNull<PrepareLoadStateMem>,
}

impl fmt::Debug for LoadState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LoadState")
            .field("shmem", &self.shmem.get_os_id())
            .field("ptr", &self.ptr.as_ptr())
            .finish()
    }
}

// SAFETY: LoadState accesses shared memory exclusively through atomic fields.
// The pointer is validated for alignment and size on creation/attach.
unsafe impl Send for LoadState {}
unsafe impl Sync for LoadState {}

// SAFETY: PrepareLoadState accesses shared memory exclusively through atomic
// fields. The pointer is validated for alignment and size on creation/attach.
unsafe impl Send for PrepareLoadState {}
unsafe impl Sync for PrepareLoadState {}

impl LoadState {
    /// Create a new LoadState (creates shared memory).
    ///
    /// The state is initialized to PENDING (0) and the header is stamped with
    /// a magic value and version for validation when attaching.
    pub fn new() -> Result<Self, LoadStateError> {
        let shm_name = format!("pega_load_{}", Uuid::new_v4().as_simple());
        let shmem = ShmemConf::new()
            .os_id(&shm_name)
            .size(load_state_allocation_size())
            .create()
            .map_err(LoadStateError::CreateShmem)?;

        let ptr = init_new_mapping(&shmem)?;
        let mem = unsafe { ptr.as_ref() };
        mem.header.magic.store(LOAD_STATE_MAGIC, Ordering::Release);
        mem.header
            .version
            .store(LOAD_STATE_VERSION, Ordering::Release);
        mem.state.store(LOAD_STATE_PENDING, Ordering::Release);

        Ok(Self { shmem, ptr })
    }

    /// Attach to an existing LoadState by shared memory name.
    pub(crate) fn attach(shm_name: &str) -> Result<Self, LoadStateError> {
        let shmem = ShmemConf::new()
            .os_id(shm_name)
            .open()
            .map_err(LoadStateError::OpenShmem)?;
        let ptr = attach_mapping(&shmem)?;
        let mem = unsafe { ptr.as_ref() };
        validate_header(mem)?;

        Ok(Self { shmem, ptr })
    }

    /// Get the shared memory identifier.
    pub fn shm_name(&self) -> &str {
        self.shmem.get_os_id()
    }

    /// Get current state value (non-blocking).
    pub fn get(&self) -> i64 {
        self.mem().state.load(Ordering::Acquire)
    }

    /// Set state to SUCCESS (1). Called by server when all transfers complete.
    pub(crate) fn set_completed(&self) {
        self.mem()
            .state
            .store(LOAD_STATE_SUCCESS, Ordering::Release);
    }

    /// Set state to ERROR (-1). Called by server on transfer failure.
    pub(crate) fn set_error(&self) {
        self.mem().state.store(LOAD_STATE_ERROR, Ordering::Release);
    }

    fn mem(&self) -> &LoadStateMem {
        // SAFETY: `ptr` is validated for size/alignment on creation/attach,
        // and the underlying memory lives for the lifetime of `shmem`.
        unsafe { self.ptr.as_ref() }
    }
}

impl PrepareLoadState {
    /// Create a new PrepareLoadState (creates shared memory).
    pub fn new() -> Result<Self, PrepareLoadStateError> {
        let shm_name = format!("pega_prepare_{}", Uuid::new_v4().as_simple());
        let shmem = ShmemConf::new()
            .os_id(&shm_name)
            .size(prepare_load_state_allocation_size())
            .create()
            .map_err(PrepareLoadStateError::CreateShmem)?;

        let ptr = init_new_prepare_mapping(&shmem)?;
        let mem = unsafe { ptr.as_ref() };
        mem.header
            .magic
            .store(PREPARE_LOAD_STATE_MAGIC, Ordering::Release);
        mem.header
            .version
            .store(PREPARE_LOAD_STATE_VERSION, Ordering::Release);
        mem.state
            .store(PREPARE_LOAD_STATE_PENDING, Ordering::Release);
        mem.num_tokens.store(0, Ordering::Release);
        mem.plan_id.store(0, Ordering::Release);

        Ok(Self { shmem, ptr })
    }

    /// Attach to an existing PrepareLoadState by shared memory name.
    pub fn attach(shm_name: &str) -> Result<Self, PrepareLoadStateError> {
        let shmem = ShmemConf::new()
            .os_id(shm_name)
            .open()
            .map_err(PrepareLoadStateError::OpenShmem)?;
        let ptr = attach_prepare_mapping(&shmem)?;
        let mem = unsafe { ptr.as_ref() };
        validate_prepare_header(mem)?;

        Ok(Self { shmem, ptr })
    }

    pub fn shm_name(&self) -> &str {
        self.shmem.get_os_id()
    }

    pub fn snapshot(&self) -> PrepareLoadSnapshot {
        let mem = self.mem();
        let state = mem.state.load(Ordering::Acquire);
        PrepareLoadSnapshot {
            state,
            num_tokens: mem.num_tokens.load(Ordering::Acquire),
            plan_id: mem.plan_id.load(Ordering::Acquire),
        }
    }

    pub fn set_ready_no_plan(&self) {
        self.publish(PREPARE_LOAD_STATE_NO_PLAN, 0, 0);
    }

    pub fn set_ready_plan(&self, num_tokens: u64, plan_id: u64) {
        self.publish(PREPARE_LOAD_STATE_PLAN, num_tokens, plan_id);
    }

    pub fn set_error(&self) {
        self.publish(PREPARE_LOAD_STATE_ERROR, 0, 0);
    }

    fn publish(&self, state: i64, num_tokens: u64, plan_id: u64) {
        let mem = self.mem();
        mem.num_tokens.store(num_tokens, Ordering::Release);
        mem.plan_id.store(plan_id, Ordering::Release);
        mem.state.store(state, Ordering::Release);
    }

    fn mem(&self) -> &PrepareLoadStateMem {
        // SAFETY: `ptr` is validated for size/alignment on creation/attach,
        // and the underlying memory lives for the lifetime of `shmem`.
        unsafe { self.ptr.as_ref() }
    }
}

fn load_state_layout() -> Layout {
    Layout::new::<LoadStateMem>()
}

fn prepare_load_state_layout() -> Layout {
    Layout::new::<PrepareLoadStateMem>()
}

fn load_state_allocation_size() -> usize {
    let layout = load_state_layout();
    LOAD_STATE_META_SIZE + layout.size() + layout.align()
}

fn prepare_load_state_allocation_size() -> usize {
    let layout = prepare_load_state_layout();
    LOAD_STATE_META_SIZE + layout.size() + layout.align()
}

fn align_up(value: usize, align: usize) -> Option<usize> {
    debug_assert!(align.is_power_of_two());
    let mask = align - 1;
    value.checked_add(mask).map(|v| v & !mask)
}

fn init_new_mapping(shmem: &Shmem) -> Result<NonNull<LoadStateMem>, LoadStateError> {
    let layout = load_state_layout();
    if shmem.len() < load_state_allocation_size() {
        return Err(LoadStateError::MappingTooSmall {
            actual: shmem.len(),
            required: load_state_allocation_size(),
        });
    }

    let base = shmem.as_ptr() as usize;
    let aligned = align_up(
        base.checked_add(LOAD_STATE_META_SIZE)
            .ok_or(LoadStateError::Overflow)?,
        layout.align(),
    )
    .ok_or(LoadStateError::Overflow)?;
    let offset = aligned.checked_sub(base).ok_or(LoadStateError::Overflow)?;

    if offset
        .checked_add(layout.size())
        .is_none_or(|end| end > shmem.len())
    {
        return Err(LoadStateError::MappingTooSmall {
            actual: shmem.len(),
            required: offset.saturating_add(layout.size()),
        });
    }

    // Record the offset so attach() can find the aligned region.
    // SAFETY: The mapping has at least LOAD_STATE_META_SIZE (8) bytes.
    // write_unaligned is used because the base pointer has no usize alignment
    // guarantee. No other references to this memory exist (freshly created).
    unsafe {
        (shmem.as_ptr() as *mut usize).write_unaligned(offset);
    }

    if aligned % layout.align() != 0 {
        return Err(LoadStateError::Misaligned {
            ptr: aligned,
            align: layout.align(),
        });
    }

    NonNull::new(aligned as *mut LoadStateMem).ok_or(LoadStateError::NullPointer)
}

fn init_new_prepare_mapping(
    shmem: &Shmem,
) -> Result<NonNull<PrepareLoadStateMem>, PrepareLoadStateError> {
    let layout = prepare_load_state_layout();
    if shmem.len() < prepare_load_state_allocation_size() {
        return Err(PrepareLoadStateError::MappingTooSmall {
            actual: shmem.len(),
            required: prepare_load_state_allocation_size(),
        });
    }

    let base = shmem.as_ptr() as usize;
    let aligned = align_up(
        base.checked_add(LOAD_STATE_META_SIZE)
            .ok_or(PrepareLoadStateError::Overflow)?,
        layout.align(),
    )
    .ok_or(PrepareLoadStateError::Overflow)?;
    let offset = aligned
        .checked_sub(base)
        .ok_or(PrepareLoadStateError::Overflow)?;

    if offset
        .checked_add(layout.size())
        .is_none_or(|end| end > shmem.len())
    {
        return Err(PrepareLoadStateError::MappingTooSmall {
            actual: shmem.len(),
            required: offset + layout.size(),
        });
    }

    if aligned % layout.align() != 0 {
        return Err(PrepareLoadStateError::Misaligned {
            ptr: aligned,
            align: layout.align(),
        });
    }

    unsafe {
        (shmem.as_ptr() as *mut usize).write_unaligned(offset);
    }

    NonNull::new(aligned as *mut PrepareLoadStateMem).ok_or(PrepareLoadStateError::NullPointer)
}

fn attach_mapping(shmem: &Shmem) -> Result<NonNull<LoadStateMem>, LoadStateError> {
    let layout = load_state_layout();
    if shmem.len() < load_state_allocation_size() {
        return Err(LoadStateError::MappingTooSmall {
            actual: shmem.len(),
            required: load_state_allocation_size(),
        });
    }

    let base = shmem.as_ptr() as usize;
    // SAFETY: The mapping has at least LOAD_STATE_META_SIZE bytes (checked above).
    // read_unaligned is used because base pointer alignment is not guaranteed.
    // The returned offset is validated below before use.
    let offset = unsafe { (shmem.as_ptr() as *const usize).read_unaligned() };

    if offset < LOAD_STATE_META_SIZE {
        return Err(LoadStateError::InvalidOffset {
            offset,
            len: shmem.len(),
        });
    }

    let ptr_addr = base.checked_add(offset).ok_or(LoadStateError::Overflow)?;

    if offset
        .checked_add(layout.size())
        .is_none_or(|end| end > shmem.len())
    {
        return Err(LoadStateError::MappingTooSmall {
            actual: shmem.len(),
            required: offset.saturating_add(layout.size()),
        });
    }

    if ptr_addr % layout.align() != 0 {
        return Err(LoadStateError::Misaligned {
            ptr: ptr_addr,
            align: layout.align(),
        });
    }

    NonNull::new(ptr_addr as *mut LoadStateMem).ok_or(LoadStateError::NullPointer)
}

fn attach_prepare_mapping(
    shmem: &Shmem,
) -> Result<NonNull<PrepareLoadStateMem>, PrepareLoadStateError> {
    let layout = prepare_load_state_layout();
    if shmem.len() < prepare_load_state_allocation_size() {
        return Err(PrepareLoadStateError::MappingTooSmall {
            actual: shmem.len(),
            required: prepare_load_state_allocation_size(),
        });
    }

    let base = shmem.as_ptr() as usize;
    let offset = unsafe { (shmem.as_ptr() as *const usize).read_unaligned() };
    if offset < LOAD_STATE_META_SIZE || offset >= shmem.len() {
        return Err(PrepareLoadStateError::InvalidOffset {
            offset,
            len: shmem.len(),
        });
    }

    let aligned = base
        .checked_add(offset)
        .ok_or(PrepareLoadStateError::Overflow)?;
    if offset
        .checked_add(layout.size())
        .is_none_or(|end| end > shmem.len())
    {
        return Err(PrepareLoadStateError::MappingTooSmall {
            actual: shmem.len(),
            required: offset + layout.size(),
        });
    }
    if aligned % layout.align() != 0 {
        return Err(PrepareLoadStateError::Misaligned {
            ptr: aligned,
            align: layout.align(),
        });
    }

    NonNull::new(aligned as *mut PrepareLoadStateMem).ok_or(PrepareLoadStateError::NullPointer)
}

fn validate_header(mem: &LoadStateMem) -> Result<(), LoadStateError> {
    let magic = mem.header.magic.load(Ordering::Acquire);
    let version = mem.header.version.load(Ordering::Acquire);
    if magic != LOAD_STATE_MAGIC || version != LOAD_STATE_VERSION {
        return Err(LoadStateError::InvalidHeader { magic, version });
    }
    Ok(())
}

fn validate_prepare_header(mem: &PrepareLoadStateMem) -> Result<(), PrepareLoadStateError> {
    let magic = mem.header.magic.load(Ordering::Acquire);
    let version = mem.header.version.load(Ordering::Acquire);
    if magic != PREPARE_LOAD_STATE_MAGIC || version != PREPARE_LOAD_STATE_VERSION {
        return Err(PrepareLoadStateError::InvalidHeader { magic, version });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    #[test]
    fn create_and_attach_round_trip() {
        let load_state = LoadState::new().expect("create LoadState");
        assert_eq!(load_state.get(), LOAD_STATE_PENDING);

        let shm_name = load_state.shm_name().to_string();
        load_state.set_completed();

        let attached = LoadState::attach(&shm_name).expect("attach LoadState");
        assert_eq!(attached.get(), LOAD_STATE_SUCCESS);
    }

    #[test]
    fn attach_rejects_too_small_mapping() {
        let shm_name = format!("pega_test_small_{}", Uuid::new_v4().as_simple());
        let _mapping = ShmemConf::new().os_id(&shm_name).size(1).create().unwrap();

        let err =
            LoadState::attach(&shm_name).expect_err("should fail to attach too small mapping");
        assert!(matches!(
            err,
            LoadStateError::MappingTooSmall {
                actual: _,
                required: _
            }
        ));
    }

    #[test]
    fn attach_rejects_invalid_header() {
        let load_state = LoadState::new().expect("create LoadState");
        let shm_name = load_state.shm_name().to_string();

        // Corrupt the header magic to force a validation failure.
        unsafe {
            let mem = load_state.ptr.as_ref();
            mem.header.magic.store(0xDEADBEEF, Ordering::Release);
        }

        let err = LoadState::attach(&shm_name).unwrap_err();
        assert!(matches!(err, LoadStateError::InvalidHeader { .. }));
    }
}
