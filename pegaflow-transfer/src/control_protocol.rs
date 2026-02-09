use crate::domain_address::DomainAddress;

#[derive(Clone, Copy, Debug)]
pub(crate) struct RcEndpoint {
    pub(crate) gid: [u8; 16],
    pub(crate) lid: u16,
    pub(crate) qp_num: u32,
    pub(crate) psn: u32,
}

impl RcEndpoint {
    const BYTES: usize = 26;

    fn to_bytes(self) -> [u8; Self::BYTES] {
        let mut bytes = [0_u8; Self::BYTES];
        bytes[..16].copy_from_slice(&self.gid);
        bytes[16..18].copy_from_slice(&self.lid.to_le_bytes());
        bytes[18..22].copy_from_slice(&self.qp_num.to_le_bytes());
        bytes[22..26].copy_from_slice(&self.psn.to_le_bytes());
        bytes
    }

    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != Self::BYTES {
            return None;
        }
        let mut gid = [0_u8; 16];
        gid.copy_from_slice(&bytes[..16]);
        Some(Self {
            gid,
            lid: u16::from_le_bytes(bytes[16..18].try_into().ok()?),
            qp_num: u32::from_le_bytes(bytes[18..22].try_into().ok()?),
            psn: u32::from_le_bytes(bytes[22..26].try_into().ok()?),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum MessageType {
    ConnectReq = 1,
    ConnectResp = 2,
    MrQueryReq = 3,
    MrQueryResp = 4,
}

impl MessageType {
    fn from_u8(raw: u8) -> Option<Self> {
        match raw {
            1 => Some(Self::ConnectReq),
            2 => Some(Self::ConnectResp),
            3 => Some(Self::MrQueryReq),
            4 => Some(Self::MrQueryResp),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
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
    },
    MrQueryReq {
        request_id: u64,
        src_ud: DomainAddress,
        ptr: u64,
        len: u64,
    },
    MrQueryResp {
        request_id: u64,
        src_ud: DomainAddress,
        found: bool,
        rkey: u32,
        available_len: u64,
    },
}

impl ControlMessage {
    pub(crate) fn request_id(&self) -> u64 {
        match self {
            ControlMessage::ConnectReq { request_id, .. }
            | ControlMessage::ConnectResp { request_id, .. }
            | ControlMessage::MrQueryReq { request_id, .. }
            | ControlMessage::MrQueryResp { request_id, .. } => *request_id,
        }
    }

    pub(crate) fn kind(&self) -> &'static str {
        match self {
            ControlMessage::ConnectReq { .. } => "connect_req",
            ControlMessage::ConnectResp { .. } => "connect_resp",
            ControlMessage::MrQueryReq { .. } => "mr_query_req",
            ControlMessage::MrQueryResp { .. } => "mr_query_resp",
        }
    }
}

pub(crate) fn encode_message(message: &ControlMessage) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(96);
    match message {
        ControlMessage::ConnectReq {
            request_id,
            src_ud,
            rc,
        } => {
            bytes.push(MessageType::ConnectReq as u8);
            bytes.extend_from_slice(&request_id.to_le_bytes());
            let src_bytes = src_ud.to_bytes();
            bytes.extend_from_slice(&src_bytes);
            bytes.extend_from_slice(&rc.to_bytes());
        }
        ControlMessage::ConnectResp {
            request_id,
            src_ud,
            rc,
        } => {
            bytes.push(MessageType::ConnectResp as u8);
            bytes.extend_from_slice(&request_id.to_le_bytes());
            let src_bytes = src_ud.to_bytes();
            bytes.extend_from_slice(&src_bytes);
            bytes.extend_from_slice(&rc.to_bytes());
        }
        ControlMessage::MrQueryReq {
            request_id,
            src_ud,
            ptr,
            len,
        } => {
            bytes.push(MessageType::MrQueryReq as u8);
            bytes.extend_from_slice(&request_id.to_le_bytes());
            let src_bytes = src_ud.to_bytes();
            bytes.extend_from_slice(&src_bytes);
            bytes.extend_from_slice(&ptr.to_le_bytes());
            bytes.extend_from_slice(&len.to_le_bytes());
        }
        ControlMessage::MrQueryResp {
            request_id,
            src_ud,
            found,
            rkey,
            available_len,
        } => {
            bytes.push(MessageType::MrQueryResp as u8);
            bytes.extend_from_slice(&request_id.to_le_bytes());
            let src_bytes = src_ud.to_bytes();
            bytes.extend_from_slice(&src_bytes);
            bytes.push(u8::from(*found));
            bytes.extend_from_slice(&rkey.to_le_bytes());
            bytes.extend_from_slice(&available_len.to_le_bytes());
        }
    }
    bytes
}

pub(crate) fn decode_message(bytes: &[u8]) -> Option<ControlMessage> {
    if bytes.len() < 1 + 8 + DomainAddress::BYTES {
        return None;
    }
    let kind = MessageType::from_u8(bytes[0])?;
    let request_id = u64::from_le_bytes(bytes[1..9].try_into().ok()?);
    let src_ud = DomainAddress::from_bytes(&bytes[9..(9 + DomainAddress::BYTES)])?;
    let payload = &bytes[(9 + DomainAddress::BYTES)..];

    match kind {
        MessageType::ConnectReq => Some(ControlMessage::ConnectReq {
            request_id,
            src_ud,
            rc: RcEndpoint::from_bytes(payload)?,
        }),
        MessageType::ConnectResp => Some(ControlMessage::ConnectResp {
            request_id,
            src_ud,
            rc: RcEndpoint::from_bytes(payload)?,
        }),
        MessageType::MrQueryReq => {
            if payload.len() != 16 {
                return None;
            }
            Some(ControlMessage::MrQueryReq {
                request_id,
                src_ud,
                ptr: u64::from_le_bytes(payload[..8].try_into().ok()?),
                len: u64::from_le_bytes(payload[8..16].try_into().ok()?),
            })
        }
        MessageType::MrQueryResp => {
            if payload.len() != 13 {
                return None;
            }
            Some(ControlMessage::MrQueryResp {
                request_id,
                src_ud,
                found: payload[0] != 0,
                rkey: u32::from_le_bytes(payload[1..5].try_into().ok()?),
                available_len: u64::from_le_bytes(payload[5..13].try_into().ok()?),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ControlMessage, RcEndpoint, decode_message, encode_message};
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

    fn assert_roundtrip(message: ControlMessage) {
        let encoded = encode_message(&message);
        let decoded = decode_message(&encoded).expect("decode");
        match (message, decoded) {
            (
                ControlMessage::ConnectReq {
                    request_id: a_id,
                    src_ud: a_ud,
                    rc: a_rc,
                },
                ControlMessage::ConnectReq {
                    request_id: b_id,
                    src_ud: b_ud,
                    rc: b_rc,
                },
            ) => {
                assert_eq!(a_id, b_id);
                assert_eq!(a_ud.to_bytes(), b_ud.to_bytes());
                assert_eq!(a_rc.gid, b_rc.gid);
                assert_eq!(a_rc.lid, b_rc.lid);
                assert_eq!(a_rc.qp_num, b_rc.qp_num);
                assert_eq!(a_rc.psn, b_rc.psn);
            }
            (
                ControlMessage::ConnectResp {
                    request_id: a_id,
                    src_ud: a_ud,
                    rc: a_rc,
                },
                ControlMessage::ConnectResp {
                    request_id: b_id,
                    src_ud: b_ud,
                    rc: b_rc,
                },
            ) => {
                assert_eq!(a_id, b_id);
                assert_eq!(a_ud.to_bytes(), b_ud.to_bytes());
                assert_eq!(a_rc.gid, b_rc.gid);
                assert_eq!(a_rc.lid, b_rc.lid);
                assert_eq!(a_rc.qp_num, b_rc.qp_num);
                assert_eq!(a_rc.psn, b_rc.psn);
            }
            (
                ControlMessage::MrQueryReq {
                    request_id: a_id,
                    src_ud: a_ud,
                    ptr: a_ptr,
                    len: a_len,
                },
                ControlMessage::MrQueryReq {
                    request_id: b_id,
                    src_ud: b_ud,
                    ptr: b_ptr,
                    len: b_len,
                },
            ) => {
                assert_eq!(a_id, b_id);
                assert_eq!(a_ud.to_bytes(), b_ud.to_bytes());
                assert_eq!(a_ptr, b_ptr);
                assert_eq!(a_len, b_len);
            }
            (
                ControlMessage::MrQueryResp {
                    request_id: a_id,
                    src_ud: a_ud,
                    found: a_found,
                    rkey: a_rkey,
                    available_len: a_len,
                },
                ControlMessage::MrQueryResp {
                    request_id: b_id,
                    src_ud: b_ud,
                    found: b_found,
                    rkey: b_rkey,
                    available_len: b_len,
                },
            ) => {
                assert_eq!(a_id, b_id);
                assert_eq!(a_ud.to_bytes(), b_ud.to_bytes());
                assert_eq!(a_found, b_found);
                assert_eq!(a_rkey, b_rkey);
                assert_eq!(a_len, b_len);
            }
            _ => panic!("decoded variant mismatch"),
        }
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
        });
    }

    #[test]
    fn roundtrip_mr_query_req() {
        assert_roundtrip(ControlMessage::MrQueryReq {
            request_id: 3,
            src_ud: sample_addr(3),
            ptr: 0x1234,
            len: 0x5678,
        });
    }

    #[test]
    fn roundtrip_mr_query_resp() {
        assert_roundtrip(ControlMessage::MrQueryResp {
            request_id: 4,
            src_ud: sample_addr(4),
            found: true,
            rkey: 999,
            available_len: 1024,
        });
    }

    #[test]
    fn decode_rejects_unknown_message_type() {
        let mut bytes = vec![9_u8; 1 + 8 + DomainAddress::BYTES];
        bytes[0] = 99;
        assert!(decode_message(&bytes).is_none());
    }

    #[test]
    fn decode_rejects_truncated_header() {
        assert!(decode_message(&[0_u8; 1 + 8 + DomainAddress::BYTES - 1]).is_none());
    }

    #[test]
    fn decode_rejects_bad_payload_len() {
        let message = ControlMessage::MrQueryReq {
            request_id: 5,
            src_ud: sample_addr(5),
            ptr: 10,
            len: 20,
        };
        let mut bytes = encode_message(&message);
        bytes.pop();
        assert!(decode_message(&bytes).is_none());
    }
}
