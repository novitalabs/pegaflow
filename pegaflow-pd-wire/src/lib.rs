//! Sealed wire schema for the PegaFlow P/D KV-push handshake.
//!
//! This crate is the single source of truth for the payload a decode (D) node
//! sends to a prefill (P) node to describe where prefilled KV blocks must be
//! RDMA-written and how completion is signalled. Anything that crosses the
//! D -> P boundary must serialize to exactly this schema; both the PegaFlow
//! vLLM connector and external engines consume these types.
//!
//! # Wire format
//!
//! The handshake is JSON. Per-layer `block_ids` may be hoisted to the
//! handshake level when every layer shares the same list (the "compact"
//! form); a layer-level `block_ids` always wins over the handshake-level one.
//!
//! # Completion contract
//!
//! The producer finishes a transfer with an RDMA WRITE_WITH_IMM per source
//! rank. The consumer waits for `expected_imm_count` arrivals of `imm_id`.
//! Failure and abort are signalled by XOR-ing dedicated flag bits into
//! `imm_id`, so a valid `imm_id` must keep both flag bits clear.

use std::collections::HashSet;
use std::fmt;
use std::num::NonZeroU32;

use serde::{Deserialize, Deserializer, Serialize};

/// XOR-ed into `imm_id` to signal a failed transfer.
pub const FAIL_IMM_FLAG: u32 = 0x8000_0000;
/// XOR-ed into `imm_id` to signal an acknowledged consumer abort.
pub const ABORT_IMM_FLAG: u32 = 0x4000_0000;
/// Bits an `imm_id` must keep clear so the flag encodings stay collision-free.
pub const IMM_FLAG_BITS: u32 = FAIL_IMM_FLAG | ABORT_IMM_FLAG;
/// Largest valid `imm_id`.
pub const MAX_IMM_ID: u32 = !IMM_FLAG_BITS;

/// The failure immediate for a given `imm_id`.
pub fn fail_imm(imm_id: u32) -> u32 {
    imm_id ^ FAIL_IMM_FLAG
}

/// The abort-ack immediate for a given `imm_id`.
pub fn abort_imm(imm_id: u32) -> u32 {
    imm_id ^ ABORT_IMM_FLAG
}

/// Schema violation found while parsing or validating a handshake.
#[derive(Debug)]
pub struct WireError(String);

impl WireError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for WireError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "pd-wire: {}", self.0)
    }
}

impl std::error::Error for WireError {}

/// RDMA memory-region descriptor: remote virtual address plus one
/// `(domain address, rkey)` pair per NIC the region is registered on.
///
/// The domain address string format is owned by the transfer engine; this
/// crate treats it as opaque.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MrDesc {
    pub ptr: u64,
    pub addr_rkey_list: Vec<(String, u64)>,
}

/// One contiguous registered address range inside a layer.
///
/// Block `b` of this region lives at `base_addr + b * block_stride` and is
/// `block_len` bytes long. A missing `block_stride` means tightly packed
/// (`block_stride == block_len`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Region {
    pub region_idx: u32,
    pub base_addr: u64,
    pub block_len: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub block_stride: Option<u64>,
}

impl Region {
    pub fn block_stride(&self) -> u64 {
        self.block_stride.unwrap_or(self.block_len)
    }
}

/// Destination layout of one KV-cache layer on the consumer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Layer {
    pub layer_name: String,
    pub layer_idx: u64,
    /// Consumer block ids this layer receives, strictly increasing.
    /// `None` in the compact form: fall back to [`Handshake::block_ids`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub block_ids: Option<Vec<u64>>,
    pub regions: Vec<Region>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mr_desc: Option<MrDesc>,
}

/// The D -> P handshake: where to RDMA-write each layer's KV blocks and how
/// to signal completion. One handshake per consumer rank.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Handshake {
    pub request_id: String,
    pub engine_id: String,
    pub tp_rank: u32,
    pub tp_size: u32,
    pub block_size: u64,
    /// Compact form: block ids shared by every layer without its own list.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub block_ids: Option<Vec<u64>>,
    pub layers: Vec<Layer>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub imm_id: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fail_imm_id: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub abort_imm_id: Option<u32>,
    /// WRITE_WITH_IMM arrivals of `imm_id` that mean "transfer complete".
    #[serde(
        default = "default_expected_imm_count",
        deserialize_with = "de_expected_imm_count"
    )]
    pub expected_imm_count: NonZeroU32,
}

fn default_expected_imm_count() -> NonZeroU32 {
    NonZeroU32::MIN
}

fn de_expected_imm_count<'de, D>(deserializer: D) -> Result<NonZeroU32, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Option::<u32>::deserialize(deserializer)?;
    NonZeroU32::new(value.unwrap_or(1))
        .ok_or_else(|| serde::de::Error::custom("expected_imm_count must be positive"))
}

impl Handshake {
    /// Parse and validate a handshake from its wire JSON.
    ///
    /// This is the only way to obtain a `Handshake` from untrusted bytes;
    /// a successful return guarantees every invariant of the schema.
    pub fn from_json(json: &str) -> Result<Self, WireError> {
        let handshake: Self = serde_json::from_str(json)
            .map_err(|err| WireError::new(format!("invalid handshake JSON: {err}")))?;
        handshake.validate()?;
        Ok(handshake)
    }

    /// Validate and serialize to wire JSON.
    pub fn to_json(&self) -> Result<String, WireError> {
        self.validate()?;
        serde_json::to_string(self)
            .map_err(|err| WireError::new(format!("handshake serialization failed: {err}")))
    }

    /// Block ids for `layer`, resolving the compact form.
    pub fn layer_block_ids<'a>(&'a self, layer: &'a Layer) -> Option<&'a [u64]> {
        layer.block_ids.as_deref().or(self.block_ids.as_deref())
    }

    /// Failure immediate, derived from `imm_id` unless overridden.
    pub fn fail_imm_id(&self) -> Option<u32> {
        self.fail_imm_id.or(self.imm_id.map(fail_imm))
    }

    /// Abort-ack immediate, derived from `imm_id` unless overridden.
    pub fn abort_imm_id(&self) -> Option<u32> {
        self.abort_imm_id.or(self.imm_id.map(abort_imm))
    }

    pub fn validate(&self) -> Result<(), WireError> {
        if self.request_id.is_empty() {
            return Err(WireError::new("request_id must not be empty"));
        }
        if self.tp_size == 0 || self.tp_rank >= self.tp_size {
            return Err(WireError::new(format!(
                "invalid tp_rank/tp_size: {}/{}",
                self.tp_rank, self.tp_size
            )));
        }
        if self.block_size == 0 {
            return Err(WireError::new("block_size must be positive"));
        }
        self.validate_imm_ids()?;
        let mut seen_layer_idx = HashSet::new();
        for layer in &self.layers {
            self.validate_layer(layer)?;
            if !seen_layer_idx.insert(layer.layer_idx) {
                return Err(WireError::new(format!(
                    "duplicate layer_idx {}",
                    layer.layer_idx
                )));
            }
        }
        Ok(())
    }

    fn validate_imm_ids(&self) -> Result<(), WireError> {
        let Some(imm_id) = self.imm_id else {
            return Ok(());
        };
        if imm_id & IMM_FLAG_BITS != 0 {
            return Err(WireError::new(format!(
                "imm_id {imm_id:#x} uses reserved flag bits (max {MAX_IMM_ID:#x})"
            )));
        }
        let fail = self.fail_imm_id().expect("imm_id is set");
        let abort = self.abort_imm_id().expect("imm_id is set");
        if fail == imm_id || abort == imm_id || abort == fail {
            return Err(WireError::new(format!(
                "imm_id/fail_imm_id/abort_imm_id must be distinct: {imm_id:#x}/{fail:#x}/{abort:#x}"
            )));
        }
        Ok(())
    }

    fn validate_layer(&self, layer: &Layer) -> Result<(), WireError> {
        if layer.layer_name.is_empty() {
            return Err(WireError::new(format!(
                "layer {} has an empty layer_name",
                layer.layer_idx
            )));
        }
        let block_ids = self.layer_block_ids(layer).ok_or_else(|| {
            WireError::new(format!(
                "layer {} has no block_ids and the handshake has no shared block_ids",
                layer.layer_idx
            ))
        })?;
        if block_ids.is_empty() {
            return Err(WireError::new(format!(
                "layer {} has empty block_ids",
                layer.layer_idx
            )));
        }
        if !block_ids.windows(2).all(|pair| pair[0] < pair[1]) {
            return Err(WireError::new(format!(
                "layer {} block_ids must be strictly increasing",
                layer.layer_idx
            )));
        }
        if layer.regions.is_empty() {
            return Err(WireError::new(format!(
                "layer {} has no regions",
                layer.layer_idx
            )));
        }
        for (position, region) in layer.regions.iter().enumerate() {
            if region.region_idx as usize != position {
                return Err(WireError::new(format!(
                    "layer {} regions must be ordered by region_idx",
                    layer.layer_idx
                )));
            }
            if region.base_addr == 0 || region.block_len == 0 {
                return Err(WireError::new(format!(
                    "layer {} region {} has an invalid address range",
                    layer.layer_idx, region.region_idx
                )));
            }
            if region.block_stride() < region.block_len {
                return Err(WireError::new(format!(
                    "layer {} region {} block_stride {} is smaller than block_len {}",
                    layer.layer_idx,
                    region.region_idx,
                    region.block_stride(),
                    region.block_len
                )));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compact-form payload exactly as the Python connector emits it
    /// (`handshake_to_compact_dict` + `json.dumps`), including explicit nulls.
    const COMPACT_FIXTURE: &str = r#"{
        "request_id": "cmpl-42-0",
        "engine_id": "d-node0",
        "tp_rank": 0,
        "tp_size": 2,
        "block_size": 64,
        "block_ids": [3, 7, 9],
        "layers": [
            {
                "layer_name": "model.layers.0.self_attn.attn",
                "layer_idx": 0,
                "regions": [
                    {"region_idx": 0, "base_addr": 4096, "block_len": 1024},
                    {"region_idx": 1, "base_addr": 1048576, "block_len": 1024, "block_stride": 2048}
                ],
                "mr_desc": {"ptr": 4096, "addr_rkey_list": [["verbs:mlx5_0:1", 17]]}
            },
            {
                "layer_name": "model.layers.1.self_attn.attn",
                "layer_idx": 1,
                "block_ids": [3, 7],
                "regions": [{"region_idx": 0, "base_addr": 8192, "block_len": 1024}],
                "mr_desc": {"ptr": 8192, "addr_rkey_list": [["verbs:mlx5_0:1", 18]]}
            }
        ],
        "imm_id": 5,
        "fail_imm_id": null,
        "abort_imm_id": null,
        "expected_imm_count": 2
    }"#;

    fn fixture() -> Handshake {
        Handshake::from_json(COMPACT_FIXTURE).expect("fixture must parse")
    }

    #[test]
    fn compact_fixture_parses_and_resolves_block_ids() {
        let handshake = fixture();
        assert_eq!(
            handshake.layer_block_ids(&handshake.layers[0]),
            Some(&[3, 7, 9][..]),
            "layer without block_ids falls back to the shared list"
        );
        assert_eq!(
            handshake.layer_block_ids(&handshake.layers[1]),
            Some(&[3, 7][..]),
            "layer-level block_ids override the shared list"
        );
        assert_eq!(handshake.expected_imm_count.get(), 2);
        assert_eq!(handshake.layers[0].regions[0].block_stride(), 1024);
        assert_eq!(handshake.layers[0].regions[1].block_stride(), 2048);
        assert_eq!(
            handshake.layers[0].mr_desc.as_ref().unwrap().addr_rkey_list,
            vec![("verbs:mlx5_0:1".to_string(), 17)]
        );
    }

    #[test]
    fn imm_helpers_derive_fail_and_abort() {
        let handshake = fixture();
        assert_eq!(handshake.fail_imm_id(), Some(5 ^ FAIL_IMM_FLAG));
        assert_eq!(handshake.abort_imm_id(), Some(5 ^ ABORT_IMM_FLAG));
        assert_eq!(fail_imm(abort_imm(5)), 5 ^ IMM_FLAG_BITS);
    }

    #[test]
    fn json_round_trip_preserves_the_handshake() {
        let handshake = fixture();
        let round_tripped = Handshake::from_json(&handshake.to_json().unwrap()).unwrap();
        assert_eq!(round_tripped, handshake);
    }

    #[test]
    fn missing_expected_imm_count_defaults_to_one() {
        let json = COMPACT_FIXTURE.replace(r#""expected_imm_count": 2"#, r#""dummy": 0"#);
        assert_eq!(
            Handshake::from_json(&json)
                .unwrap()
                .expected_imm_count
                .get(),
            1
        );
        let json = COMPACT_FIXTURE.replace(
            r#""expected_imm_count": 2"#,
            r#""expected_imm_count": null"#,
        );
        assert_eq!(
            Handshake::from_json(&json)
                .unwrap()
                .expected_imm_count
                .get(),
            1
        );
    }

    fn rejects(mutate: impl FnOnce(&mut Handshake), expected_message: &str) {
        let mut handshake = fixture();
        mutate(&mut handshake);
        let err = handshake
            .validate()
            .expect_err(expected_message)
            .to_string();
        assert!(
            err.contains(expected_message),
            "expected {expected_message:?} in {err:?}"
        );
    }

    #[test]
    fn validation_rejects_schema_violations() {
        rejects(|h| h.imm_id = Some(FAIL_IMM_FLAG | 1), "reserved flag bits");
        rejects(|h| h.fail_imm_id = Some(5), "must be distinct");
        rejects(|h| h.layers[1].layer_idx = 0, "duplicate layer_idx");
        rejects(
            |h| h.layers[1].block_ids = Some(vec![7, 3]),
            "strictly increasing",
        );
        rejects(|h| h.layers[1].block_ids = Some(vec![]), "empty block_ids");
        rejects(
            |h| {
                h.block_ids = None;
                h.layers[0].block_ids = None;
            },
            "no block_ids",
        );
        rejects(
            |h| h.layers[0].regions[1].region_idx = 0,
            "ordered by region_idx",
        );
        rejects(|h| h.layers[0].regions.clear(), "no regions");
        rejects(
            |h| h.layers[0].regions[0].base_addr = 0,
            "invalid address range",
        );
        rejects(
            |h| h.layers[0].regions[1].block_stride = Some(8),
            "smaller than block_len",
        );
        rejects(|h| h.tp_rank = 2, "invalid tp_rank/tp_size");
        rejects(|h| h.request_id.clear(), "request_id");
    }

    #[test]
    fn zero_expected_imm_count_is_rejected_at_parse_time() {
        let json =
            COMPACT_FIXTURE.replace(r#""expected_imm_count": 2"#, r#""expected_imm_count": 0"#);
        let err = Handshake::from_json(&json).unwrap_err().to_string();
        assert!(err.contains("expected_imm_count"), "{err}");
    }
}
