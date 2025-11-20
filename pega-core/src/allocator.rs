use offset_allocator::{Allocation as RawAllocation, Allocator as RawAllocator};
use std::fmt::{Display, Formatter, Result as FmtResult};

/// Error variants returned by the scaled allocator wrapper.
#[derive(Debug, PartialEq, Eq)]
pub enum AllocatorError {
    /// The configured unit size must be greater than zero.
    InvalidUnitSize,
    /// The requested capacity cannot be represented with the underlying u32 allocator.
    CapacityTooLarge { total_bytes: u64, unit_size: u64 },
    /// The requested allocation is too large for the underlying allocator.
    RequestTooLarge {
        requested_bytes: u64,
        unit_size: u64,
    },
}

impl Display for AllocatorError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            AllocatorError::InvalidUnitSize => write!(f, "unit size must be greater than zero"),
            AllocatorError::CapacityTooLarge {
                total_bytes,
                unit_size,
            } => write!(
                f,
                "capacity {} bytes with unit {} overflows u32 backing allocator",
                total_bytes, unit_size
            ),
            AllocatorError::RequestTooLarge {
                requested_bytes,
                unit_size,
            } => write!(
                f,
                "request {} bytes with unit {} overflows u32 backing allocator",
                requested_bytes, unit_size
            ),
        }
    }
}

/// Represents an allocated region in bytes.
#[derive(Clone, Copy)]
pub struct Allocation {
    /// Offset from the start of the managed region, in bytes.
    pub offset_bytes: u64,
    /// Actual size of the allocation in bytes (rounded up to `unit_size`).
    pub size_bytes: u64,
    raw: RawAllocation,
}

impl std::fmt::Debug for Allocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Allocation")
            .field("offset_bytes", &self.offset_bytes)
            .field("size_bytes", &self.size_bytes)
            .finish()
    }
}

/// A thin scaling wrapper around `offset-allocator` that allows byte-level APIs
/// while the underlying allocator works with u32-sized units.
#[derive(Debug)]
pub struct ScaledOffsetAllocator {
    unit_size: u64,
    total_units: u32,
    inner: RawAllocator,
}

impl ScaledOffsetAllocator {
    /// Create a new allocator that automatically chooses the smallest unit size that
    /// keeps the backing allocator within its u32 capacity.
    ///
    /// For example, 64 GiB will scale to 16-byte units to fit under the ~4 GiB u32 ceiling.
    pub fn new(total_bytes: u64) -> Result<Self, AllocatorError> {
        let max_units = u32::MAX as u64;
        let unit_size = if total_bytes > max_units {
            ceil_div(total_bytes, max_units)
        } else {
            1
        };
        Self::new_with_unit_size(total_bytes, unit_size)
    }

    /// Create a new allocator that manages `total_bytes`, rounding up to multiples of `unit_size`.
    pub fn new_with_unit_size(total_bytes: u64, unit_size: u64) -> Result<Self, AllocatorError> {
        if unit_size == 0 {
            return Err(AllocatorError::InvalidUnitSize);
        }

        let total_units = ceil_div(total_bytes, unit_size);
        let total_units_u32 =
            u32::try_from(total_units).map_err(|_| AllocatorError::CapacityTooLarge {
                total_bytes,
                unit_size,
            })?;

        Ok(Self {
            unit_size,
            total_units: total_units_u32,
            inner: RawAllocator::new(total_units_u32),
        })
    }

    /// Allocate `size_bytes`, rounded up to the allocator's unit size.
    pub fn allocate(&mut self, size_bytes: u64) -> Result<Option<Allocation>, AllocatorError> {
        if size_bytes == 0 {
            return Ok(None);
        }

        let units = ceil_div(size_bytes, self.unit_size);
        let units_u32 = u32::try_from(units).map_err(|_| AllocatorError::RequestTooLarge {
            requested_bytes: size_bytes,
            unit_size: self.unit_size,
        })?;

        let Some(raw) = self.inner.allocate(units_u32) else {
            return Ok(None);
        };

        let allocated_units = self.inner.allocation_size(raw) as u64;
        Ok(Some(Allocation {
            offset_bytes: raw.offset as u64 * self.unit_size,
            size_bytes: allocated_units * self.unit_size,
            raw,
        }))
    }

    /// Free a previously allocated region.
    pub fn free(&mut self, allocation: Allocation) {
        self.inner.free(allocation.raw);
    }

    /// Return the total managed capacity in bytes (rounded up to `unit_size`).
    pub fn total_bytes(&self) -> u64 {
        self.unit_size * self.total_units as u64
    }

    /// Return a storage report converted to bytes.
    pub fn storage_report(&self) -> StorageReportBytes {
        let report = self.inner.storage_report();
        StorageReportBytes {
            total_free_bytes: report.total_free_space as u64 * self.unit_size,
            largest_free_allocation_bytes: report.largest_free_region as u64 * self.unit_size,
        }
    }
}

/// Storage report expressed in bytes.
#[derive(Debug, PartialEq, Eq)]
pub struct StorageReportBytes {
    pub total_free_bytes: u64,
    pub largest_free_allocation_bytes: u64,
}

fn ceil_div(lhs: u64, rhs: u64) -> u64 {
    let quotient = lhs / rhs;
    if lhs % rhs == 0 {
        quotient
    } else {
        quotient + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_allocator_with_scaled_capacity() {
        let allocator =
            ScaledOffsetAllocator::new_with_unit_size(10 * 1024 * 1024 * 1024, 64).unwrap();
        assert_eq!(allocator.unit_size, 64);
        assert_eq!(
            allocator.total_units,
            (10 * 1024 * 1024 * 1024u64 / 64) as u32
        );
    }

    #[test]
    fn allocate_rounds_up_to_unit_size() {
        let mut allocator = ScaledOffsetAllocator::new_with_unit_size(1024, 64).unwrap();
        let allocation = allocator.allocate(1).unwrap().unwrap();
        assert_eq!(allocation.offset_bytes, 0);
        assert_eq!(allocation.size_bytes, 64);

        let storage = allocator.storage_report();
        assert_eq!(storage.total_free_bytes, 960);
        assert_eq!(storage.largest_free_allocation_bytes, 960);
    }

    #[test]
    fn reuse_after_free_merges_neighboring_regions() {
        let mut allocator = ScaledOffsetAllocator::new_with_unit_size(256, 64).unwrap();
        let a = allocator.allocate(64).unwrap().unwrap();
        let b = allocator.allocate(64).unwrap().unwrap();

        allocator.free(a);
        allocator.free(b);

        let merged = allocator.allocate(128).unwrap().unwrap();
        assert_eq!(merged.offset_bytes, 0);
        assert_eq!(merged.size_bytes, 128);
    }

    #[test]
    fn rejects_unreasonably_large_requests() {
        let mut allocator = ScaledOffsetAllocator::new_with_unit_size(1024 * 1024, 1).unwrap();
        let too_large = u64::from(u32::MAX) * 2;
        let err = allocator.allocate(too_large).unwrap_err();
        assert_eq!(
            err,
            AllocatorError::RequestTooLarge {
                requested_bytes: too_large,
                unit_size: 1
            }
        );
    }

    #[test]
    fn rejects_invalid_unit_size() {
        let err = ScaledOffsetAllocator::new_with_unit_size(1024, 0).unwrap_err();
        assert_eq!(err, AllocatorError::InvalidUnitSize);
    }
}
