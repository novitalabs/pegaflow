use crate::domain_address::DomainAddress;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RcEndpoint {
    pub(crate) gid: [u8; 16],
    pub(crate) lid: u16,
    pub(crate) qp_num: u32,
    pub(crate) psn: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct RegisteredMemoryRegion {
    pub(crate) base_ptr: u64,
    pub(crate) len: u64,
    pub(crate) rkey: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum ConnectRespError {
    TooManyRegisteredMemoryRegions,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ControlMessage {
    ConnectReq {
        request_id: u64,
        src_ud: DomainAddress,
        rc: RcEndpoint,
    },
    ConnectResp {
        request_id: u64,
        src_ud: DomainAddress,
        rc: RcEndpoint,
        remote_memory_regions: Vec<RegisteredMemoryRegion>,
    },
    ConnectRespErr {
        request_id: u64,
        src_ud: DomainAddress,
        error: ConnectRespError,
    },
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
struct WireRcEndpoint {
    gid: [u8; 16],
    lid: u16,
    qp_num: u32,
    psn: u32,
}

impl From<RcEndpoint> for WireRcEndpoint {
    fn from(value: RcEndpoint) -> Self {
        Self {
            gid: value.gid,
            lid: value.lid,
            qp_num: value.qp_num,
            psn: value.psn,
        }
    }
}

impl From<WireRcEndpoint> for RcEndpoint {
    fn from(value: WireRcEndpoint) -> Self {
        Self {
            gid: value.gid,
            lid: value.lid,
            qp_num: value.qp_num,
            psn: value.psn,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
enum WireControlMessage {
    ConnectReq {
        request_id: u64,
        src_ud: [u8; DomainAddress::BYTES],
        rc: WireRcEndpoint,
    },
    ConnectResp {
        request_id: u64,
        src_ud: [u8; DomainAddress::BYTES],
        rc: WireRcEndpoint,
        remote_memory_regions: Vec<RegisteredMemoryRegion>,
    },
    ConnectRespErr {
        request_id: u64,
        src_ud: [u8; DomainAddress::BYTES],
        error: ConnectRespError,
    },
}

impl WireControlMessage {
    fn from_message(message: &ControlMessage) -> Self {
        match message {
            ControlMessage::ConnectReq {
                request_id,
                src_ud,
                rc,
            } => Self::ConnectReq {
                request_id: *request_id,
                src_ud: domain_to_wire(src_ud),
                rc: (*rc).into(),
            },
            ControlMessage::ConnectResp {
                request_id,
                src_ud,
                rc,
                remote_memory_regions,
            } => Self::ConnectResp {
                request_id: *request_id,
                src_ud: domain_to_wire(src_ud),
                rc: (*rc).into(),
                remote_memory_regions: remote_memory_regions.clone(),
            },
            ControlMessage::ConnectRespErr {
                request_id,
                src_ud,
                error,
            } => Self::ConnectRespErr {
                request_id: *request_id,
                src_ud: domain_to_wire(src_ud),
                error: *error,
            },
        }
    }

    fn into_message(self) -> Option<ControlMessage> {
        match self {
            Self::ConnectReq {
                request_id,
                src_ud,
                rc,
            } => Some(ControlMessage::ConnectReq {
                request_id,
                src_ud: wire_to_domain(src_ud)?,
                rc: rc.into(),
            }),
            Self::ConnectResp {
                request_id,
                src_ud,
                rc,
                remote_memory_regions,
            } => Some(ControlMessage::ConnectResp {
                request_id,
                src_ud: wire_to_domain(src_ud)?,
                rc: rc.into(),
                remote_memory_regions,
            }),
            Self::ConnectRespErr {
                request_id,
                src_ud,
                error,
            } => Some(ControlMessage::ConnectRespErr {
                request_id,
                src_ud: wire_to_domain(src_ud)?,
                error,
            }),
        }
    }
}

fn domain_to_wire(src_ud: &DomainAddress) -> [u8; DomainAddress::BYTES] {
    let mut bytes = [0_u8; DomainAddress::BYTES];
    let src_bytes = src_ud.to_bytes();
    bytes.copy_from_slice(src_bytes.as_ref());
    bytes
}

fn wire_to_domain(bytes: [u8; DomainAddress::BYTES]) -> Option<DomainAddress> {
    DomainAddress::from_bytes(&bytes)
}

impl ControlMessage {
    pub(crate) fn request_id(&self) -> u64 {
        match self {
            ControlMessage::ConnectReq { request_id, .. }
            | ControlMessage::ConnectResp { request_id, .. }
            | ControlMessage::ConnectRespErr { request_id, .. } => *request_id,
        }
    }

    pub(crate) fn kind(&self) -> &'static str {
        match self {
            ControlMessage::ConnectReq { .. } => "connect_req",
            ControlMessage::ConnectResp { .. } => "connect_resp",
            ControlMessage::ConnectRespErr { .. } => "connect_resp_err",
        }
    }
}

pub(crate) fn encode_message(message: &ControlMessage) -> Vec<u8> {
    let wire = WireControlMessage::from_message(message);
    let serialized = bincode::serde::encode_to_vec(&wire, bincode::config::standard())
        .expect("control message serialization should not fail");
    lz4_flex::compress_prepend_size(&serialized)
}

pub(crate) fn decode_message(bytes: &[u8]) -> Option<ControlMessage> {
    let decompressed = lz4_flex::decompress_size_prepended(bytes).ok()?;
    let (wire, consumed) = bincode::serde::decode_from_slice::<WireControlMessage, _>(
        &decompressed,
        bincode::config::standard(),
    )
    .ok()?;
    if consumed != decompressed.len() {
        return None;
    }
    wire.into_message()
}

#[cfg(test)]
mod tests {
    use super::{
        ConnectRespError, ControlMessage, RcEndpoint, RegisteredMemoryRegion, decode_message,
        encode_message,
    };
    use crate::domain_address::DomainAddress;

    fn sample_addr(seed: u8) -> DomainAddress {
        DomainAddress::from_parts(
            [seed; 16],
            100 + seed as u16,
            200 + seed as u32,
            0x1111_1111,
        )
    }

    fn sample_rc(seed: u8) -> RcEndpoint {
        RcEndpoint {
            gid: [seed; 16],
            lid: 300 + seed as u16,
            qp_num: 400 + seed as u32,
            psn: 500 + seed as u32,
        }
    }

    fn sample_region(seed: u8) -> RegisteredMemoryRegion {
        RegisteredMemoryRegion {
            base_ptr: 0x1000 * (seed as u64 + 1),
            len: 0x2000,
            rkey: 10_000 + seed as u32,
        }
    }

    fn assert_roundtrip(message: ControlMessage) {
        let encoded = encode_message(&message);
        let decoded = decode_message(&encoded).expect("decode");
        assert_eq!(decoded, message);
    }

    #[test]
    fn roundtrip_connect_req() {
        assert_roundtrip(ControlMessage::ConnectReq {
            request_id: 1,
            src_ud: sample_addr(1),
            rc: sample_rc(2),
        });
    }

    #[test]
    fn roundtrip_connect_resp() {
        assert_roundtrip(ControlMessage::ConnectResp {
            request_id: 2,
            src_ud: sample_addr(2),
            rc: sample_rc(3),
            remote_memory_regions: vec![sample_region(1), sample_region(2)],
        });
    }

    #[test]
    fn roundtrip_connect_resp_err() {
        assert_roundtrip(ControlMessage::ConnectRespErr {
            request_id: 3,
            src_ud: sample_addr(4),
            error: ConnectRespError::TooManyRegisteredMemoryRegions,
        });
    }

    #[test]
    fn decode_rejects_invalid_compressed_payload() {
        assert!(decode_message(&[1_u8, 2, 3, 4]).is_none());
    }

    #[test]
    fn decode_rejects_garbage_after_decompress() {
        let garbage = lz4_flex::compress_prepend_size(&[1_u8, 2, 3, 4]);
        assert!(decode_message(&garbage).is_none());
    }

    #[test]
    fn decode_rejects_trailing_payload() {
        let message = ControlMessage::ConnectRespErr {
            request_id: 5,
            src_ud: sample_addr(5),
            error: ConnectRespError::TooManyRegisteredMemoryRegions,
        };
        let encoded = encode_message(&message);
        let mut decoded = lz4_flex::decompress_size_prepended(&encoded).expect("decompress");
        decoded.push(1_u8);
        let malformed = lz4_flex::compress_prepend_size(&decoded);
        assert!(decode_message(&malformed).is_none());
    }
}
