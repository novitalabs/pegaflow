//! Engine-neutral rules for reusing offloaded KV state.
//!
//! Everything here is a pure function: no I/O, no RPC, no knowledge of vLLM
//! or any other engine. Connectors translate their cache layout into
//! [`Group`]s and [`Region`]s, then ask this crate two questions:
//!
//! - [`resolve`]: given which blocks the store holds, what is the longest
//!   prefix every group can resume from?
//! - [`saveable`]: when compute advances, which blocks of a group became
//!   complete and may be stored?
//!
//! The store side asks one more: [`seal_plan`] turns every rank's sharding
//! declaration into the exact list of fragments a block needs before it is
//! complete.

mod groups;
mod keys;
mod sharding;

pub use groups::{Group, InvalidGroup, Kind, needed, resolve, saveable};
pub use keys::key;
pub use sharding::{Fragment, GroupPlan, RankDecl, Region, SealError, SealPlan, Shard, seal_plan};
