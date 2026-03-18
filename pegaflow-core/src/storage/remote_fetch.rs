// Type aliases for cross-node remote fetch, used by PrefetchScheduler and
// remote_fetch_worker.

use std::sync::Arc;

use tokio::sync::oneshot;

use crate::block::{BlockKey, SealedBlock};

/// Blocks fetched from a remote node, ready to insert into ReadCache.
pub(crate) type RemoteFetchResult = Vec<(BlockKey, Arc<SealedBlock>)>;

/// Closure that dispatches a remote fetch to a background tokio task.
/// Injected by the server layer (captures MetaServerQueryClient, PegaflowClientPool,
/// MooncakeTransferEngine, PinnedAllocator).
pub(crate) type RemoteFetchFn =
    Arc<dyn Fn(Vec<BlockKey>, oneshot::Sender<RemoteFetchResult>) + Send + Sync>;
