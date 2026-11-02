pub mod proto {
    #[allow(
        clippy::allow_attributes_without_reason,
        reason = "prost/tonic generated modules emit allow attributes"
    )]
    pub mod engine {
        tonic::include_proto!("pegaflow");
    }

    #[allow(
        clippy::allow_attributes_without_reason,
        reason = "prost/tonic generated modules emit allow attributes"
    )]
    pub mod engine_v2 {
        tonic::include_proto!("pegaflow.v2");
    }
}
