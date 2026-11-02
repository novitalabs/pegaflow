//! gRPC service for the content-addressed v2 protocol.

use std::sync::Arc;

use log::debug;
use pegaflow_core::kv_v2::{
    KeyList, KvV2Error, KvV2Store, ObjectKey, ObjectRead, ObjectWrite, Run,
};
use tonic::{Request, Response, Status, async_trait};

use crate::proto::engine_v2 as pb;
use crate::proto::engine_v2::engine_v2_server::EngineV2;

#[derive(Clone)]
pub struct GrpcEngineV2Service {
    store: Arc<KvV2Store>,
}

impl GrpcEngineV2Service {
    pub fn new(store: Arc<KvV2Store>) -> Self {
        Self { store }
    }

    /// Run a blocking store call off the async workers.
    async fn blocking<T, F>(&self, f: F) -> Result<T, Status>
    where
        T: Send + 'static,
        F: FnOnce(&KvV2Store) -> Result<T, KvV2Error> + Send + 'static,
    {
        let store = Arc::clone(&self.store);
        tokio::task::spawn_blocking(move || f(&store))
            .await
            .map_err(|e| Status::internal(format!("store task failed: {e}")))?
            .map_err(map_error)
    }
}

fn map_error(err: KvV2Error) -> Status {
    match err {
        KvV2Error::Invalid(_) => Status::invalid_argument(err.to_string()),
        KvV2Error::NotFound(_) => Status::not_found(err.to_string()),
        KvV2Error::Exhausted(_) => Status::resource_exhausted(err.to_string()),
        KvV2Error::Cuda(_) => Status::internal(err.to_string()),
    }
}

fn digest(bytes: &[u8], field: &str) -> Result<[u8; 32], Status> {
    bytes
        .try_into()
        .map_err(|_| Status::invalid_argument(format!("{field} has {} bytes, not 32", bytes.len())))
}

fn key(k: Option<pb::ObjectKey>) -> Result<ObjectKey, Status> {
    let k = k.ok_or_else(|| Status::invalid_argument("missing object key"))?;
    Ok(ObjectKey {
        fingerprint: digest(&k.fingerprint, "fingerprint")?,
        prefix: digest(&k.prefix, "prefix")?,
    })
}

fn run(r: pb::Run) -> Run {
    Run {
        region: r.region,
        region_offset: r.region_offset,
        object_offset: r.object_offset,
        width: r.width,
        rows: r.rows,
        region_stride: r.region_stride,
        object_stride: r.object_stride,
    }
}

#[async_trait]
impl EngineV2 for GrpcEngineV2Service {
    async fn register(
        &self,
        request: Request<pb::RegisterRequest>,
    ) -> Result<Response<pb::RegisterResponse>, Status> {
        let req = request.into_inner();
        let server_version = env!("CARGO_PKG_VERSION");
        if req.version != server_version {
            return Err(Status::failed_precondition(format!(
                "version mismatch: client={} server={server_version}",
                req.version
            )));
        }
        let regions: Vec<(Vec<u8>, u64)> = req
            .regions
            .into_iter()
            .map(|r| (r.ipc_handle, r.size))
            .collect();
        self.blocking(move |s| s.register(&req.instance_id, req.rank, req.device_id, &regions))
            .await?;
        Ok(Response::new(pb::RegisterResponse {}))
    }

    async fn unregister(
        &self,
        request: Request<pb::UnregisterRequest>,
    ) -> Result<Response<pb::UnregisterResponse>, Status> {
        let req = request.into_inner();
        let removed = self
            .blocking(move |s| Ok(s.unregister(&req.instance_id)))
            .await?;
        Ok(Response::new(pb::UnregisterResponse {
            removed_devices: removed as u32,
        }))
    }

    async fn query(
        &self,
        request: Request<pb::QueryRequest>,
    ) -> Result<Response<pb::QueryResponse>, Status> {
        let lists = request
            .into_inner()
            .lists
            .into_iter()
            .map(|l| {
                Ok(KeyList {
                    keys: l
                        .keys
                        .into_iter()
                        .map(|k| key(Some(k)))
                        .collect::<Result<_, Status>>()?,
                    stop_at_miss: l.stop_at_miss,
                })
            })
            .collect::<Result<Vec<_>, Status>>()?;
        let (lease, hits) = self.store.query(&lists);
        debug!("v2 query: lease={lease} lists={}", hits.len());
        Ok(Response::new(pb::QueryResponse {
            lease,
            lists: hits.into_iter().map(|hits| pb::HitList { hits }).collect(),
        }))
    }

    async fn save(
        &self,
        request: Request<pb::SaveRequest>,
    ) -> Result<Response<pb::SaveResponse>, Status> {
        let req = request.into_inner();
        let writes = req
            .writes
            .into_iter()
            .map(|w| {
                Ok(ObjectWrite {
                    key: key(w.key)?,
                    size: w.size,
                    runs: w.runs.into_iter().map(run).collect(),
                })
            })
            .collect::<Result<Vec<_>, Status>>()?;
        let sealed = self
            .blocking(move |s| s.save(&req.instance_id, req.rank, &writes))
            .await?;
        Ok(Response::new(pb::SaveResponse {
            sealed: sealed as u32,
        }))
    }

    async fn load(
        &self,
        request: Request<pb::LoadRequest>,
    ) -> Result<Response<pb::LoadResponse>, Status> {
        let req = request.into_inner();
        let reads = req
            .reads
            .into_iter()
            .map(|r| {
                Ok(ObjectRead {
                    key: key(r.key)?,
                    runs: r.runs.into_iter().map(run).collect(),
                })
            })
            .collect::<Result<Vec<_>, Status>>()?;
        self.blocking(move |s| s.load(&req.instance_id, req.rank, req.lease, &reads))
            .await?;
        Ok(Response::new(pb::LoadResponse {}))
    }

    async fn release(
        &self,
        request: Request<pb::ReleaseRequest>,
    ) -> Result<Response<pb::ReleaseResponse>, Status> {
        let lease = request.into_inner().lease;
        if !self.store.release(lease) {
            return Err(Status::not_found(format!("lease {lease}")));
        }
        Ok(Response::new(pb::ReleaseResponse {}))
    }
}
