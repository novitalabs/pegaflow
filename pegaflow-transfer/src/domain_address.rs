use std::fmt::{Display, Formatter};

use bytes::Bytes;

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct DomainAddress(pub Bytes);

impl DomainAddress {
    pub const BYTES: usize = 26;

    pub fn to_bytes(&self) -> Bytes {
        self.0.clone()
    }

    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != Self::BYTES {
            return None;
        }
        Some(Self(Bytes::copy_from_slice(bytes)))
    }

    pub(crate) fn from_parts(gid: [u8; 16], lid: u16, qp_num: u32, qkey: u32) -> Self {
        let mut bytes = [0_u8; Self::BYTES];
        bytes[..16].copy_from_slice(&gid);
        bytes[16..18].copy_from_slice(&lid.to_le_bytes());
        bytes[18..22].copy_from_slice(&qp_num.to_le_bytes());
        bytes[22..26].copy_from_slice(&qkey.to_le_bytes());
        Self(Bytes::copy_from_slice(&bytes))
    }

    pub(crate) fn gid(&self) -> [u8; 16] {
        let mut gid = [0_u8; 16];
        gid.copy_from_slice(&self.0[..16]);
        gid
    }

    pub(crate) fn lid(&self) -> u16 {
        u16::from_le_bytes(self.0[16..18].try_into().expect("validated length"))
    }

    pub(crate) fn qp_num(&self) -> u32 {
        u32::from_le_bytes(self.0[18..22].try_into().expect("validated length"))
    }

    pub(crate) fn qkey(&self) -> u32 {
        u32::from_le_bytes(self.0[22..26].try_into().expect("validated length"))
    }
}

impl std::fmt::Debug for DomainAddress {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "DomainAddress {{ gid: {:?}, lid: {}, qp_num: {}, qkey: {} }}",
            self.gid(),
            self.lid(),
            self.qp_num(),
            self.qkey()
        )
    }
}

impl Display for DomainAddress {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "DomainAddress(lid={}, qp_num={}, qkey={})",
            self.lid(),
            self.qp_num(),
            self.qkey()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::DomainAddress;

    #[test]
    fn roundtrip_bytes() {
        let addr = DomainAddress::from_parts([7_u8; 16], 123, 456, 789);
        let bytes = addr.to_bytes();
        let decoded = DomainAddress::from_bytes(&bytes).expect("decode");
        assert_eq!(decoded, addr);
    }

    #[test]
    fn from_bytes_rejects_invalid_len() {
        assert!(DomainAddress::from_bytes(&[]).is_none());
        assert!(DomainAddress::from_bytes(&[1_u8; DomainAddress::BYTES - 1]).is_none());
        assert!(DomainAddress::from_bytes(&[1_u8; DomainAddress::BYTES + 1]).is_none());
    }

    #[test]
    fn accessors_decode_fields() {
        let addr = DomainAddress::from_parts([1_u8; 16], 2, 3, 4);
        assert_eq!(addr.gid(), [1_u8; 16]);
        assert_eq!(addr.lid(), 2);
        assert_eq!(addr.qp_num(), 3);
        assert_eq!(addr.qkey(), 4);
    }
}
