/// Exact wire capability version. Registration is fenced on this string, so a
/// client built before the native-arena contract fails before any memory
/// changes hands.
pub const VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), "+native-arena-v1");

pub mod proto {
    #[allow(
        clippy::allow_attributes_without_reason,
        reason = "prost/tonic generated modules emit allow attributes"
    )]
    pub mod engine {
        tonic::include_proto!("pegaflow");
    }
}
