use bitflags::bitflags;
use ironrdp_core::{
    Decode, DecodeResult, Encode, EncodeResult, ReadCursor, WriteCursor, cast_length, ensure_fixed_part_size,
    ensure_size, invalid_field_err, not_enough_bytes_err, other_err, read_padding, write_padding,
};
use num_derive::FromPrimitive;
use num_traits::FromPrimitive as _;

use crate::codecs::rfx::FrameAcknowledgePdu;
use crate::input::InputEventPdu;
use crate::rdp::autodetect::{AutoDetectRequest, AutoDetectResponse};
use crate::rdp::capability_sets::{ClientConfirmActive, ServerDemandActive};
use crate::rdp::client_info;
use crate::rdp::finalization_messages::{ControlPdu, FontPdu, MonitorLayoutPdu, SynchronizePdu};
use crate::rdp::refresh_rectangle::RefreshRectanglePdu;
use crate::rdp::server_error_info::ServerSetErrorInfoPdu;
use crate::rdp::session_info::SaveSessionInfoPdu;
use crate::rdp::suppress_output::SuppressOutputPdu;

pub const BASIC_SECURITY_HEADER_SIZE: usize = 4;
pub const SHARE_DATA_HEADER_COMPRESSION_MASK: u8 = 0xF;
const SHARE_CONTROL_HEADER_MASK: u16 = 0xF;
const SHARE_CONTROL_HEADER_SIZE: usize = 2 * 3 + 4;

const PROTOCOL_VERSION: u16 = 0x10;

// ShareDataHeader
const PADDING_FIELD_SIZE: usize = 1;
const STREAM_ID_FIELD_SIZE: usize = 1;
const UNCOMPRESSED_LENGTH_FIELD_SIZE: usize = 2;
const PDU_TYPE_FIELD_SIZE: usize = 1;
const COMPRESSION_TYPE_FIELD_SIZE: usize = 1;
const COMPRESSED_LENGTH_FIELD_SIZE: usize = 2;

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
pub struct BasicSecurityHeader {
    pub flags: BasicSecurityHeaderFlags,
}

impl BasicSecurityHeader {
    const NAME: &'static str = "BasicSecurityHeader";

    pub const FIXED_PART_SIZE: usize = BASIC_SECURITY_HEADER_SIZE;
}

impl Encode for BasicSecurityHeader {
    fn encode(&self, dst: &mut WriteCursor<'_>) -> EncodeResult<()> {
        ensure_fixed_part_size!(in: dst);

        dst.write_u16(self.flags.bits());
        dst.write_u16(0); // flags_hi
        Ok(())
    }

    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn size(&self) -> usize {
        Self::FIXED_PART_SIZE
    }
}

impl<'de> Decode<'de> for BasicSecurityHeader {
    fn decode(src: &mut ReadCursor<'de>) -> DecodeResult<Self> {
        ensure_fixed_part_size!(in: src);

        let flags = BasicSecurityHeaderFlags::from_bits(src.read_u16())
            .ok_or_else(|| invalid_field_err!("securityHeader", "invalid basic security header"))?;
        let _flags_hi = src.read_u16(); // unused

        Ok(Self { flags })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
pub struct ShareControlHeader {
    pub share_control_pdu: ShareControlPdu,
    pub pdu_source: u16,
    pub share_id: u32,
}

impl ShareControlHeader {
    const NAME: &'static str = "ShareControlHeader";

    const FIXED_PART_SIZE: usize = SHARE_CONTROL_HEADER_SIZE;
}

impl Encode for ShareControlHeader {
    fn encode(&self, dst: &mut WriteCursor<'_>) -> EncodeResult<()> {
        ensure_size!(in: dst, size: self.size());

        let pdu_type_with_version = PROTOCOL_VERSION | self.share_control_pdu.share_header_type().as_u16();

        dst.write_u16(cast_length!(
            "len",
            self.share_control_pdu.size() + SHARE_CONTROL_HEADER_SIZE
        )?);
        dst.write_u16(pdu_type_with_version);
        dst.write_u16(self.pdu_source);
        dst.write_u32(self.share_id);

        self.share_control_pdu.encode(dst)
    }

    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn size(&self) -> usize {
        Self::FIXED_PART_SIZE + self.share_control_pdu.size()
    }
}

impl<'de> Decode<'de> for ShareControlHeader {
    fn decode(src: &mut ReadCursor<'de>) -> DecodeResult<Self> {
        ensure_fixed_part_size!(in: src);

        let total_length = usize::from(src.read_u16());
        let pdu_type_with_version = src.read_u16();
        let pdu_source = src.read_u16();

        let pdu_type = ShareControlPduType::from_u16(pdu_type_with_version & SHARE_CONTROL_HEADER_MASK)
            .ok_or_else(|| invalid_field_err!("pdu_type", "invalid pdu type"))?;
        let pdu_version = pdu_type_with_version & !SHARE_CONTROL_HEADER_MASK;
        if pdu_version != PROTOCOL_VERSION {
            return Err(invalid_field_err!("pdu_version", "invalid PDU version"));
        }

        // The Enhanced Security Server Redirection PDU (MS-RDPBCGR 2.2.13.3)
        // has NO shareId — the share control header is followed by pad2Octets
        // and then the redirection packet itself (which begins with its own
        // 0x0400 marker). Every other share control PDU carries a shareId here.
        let share_id = if pdu_type == ShareControlPduType::ServerRedirect {
            ensure_size!(in: src, size: 2);
            read_padding!(src, 2);
            0
        } else {
            ensure_size!(in: src, size: 4);
            src.read_u32()
        };

        let share_pdu = ShareControlPdu::from_type(src, pdu_type)?;
        let header = Self {
            share_control_pdu: share_pdu,
            pdu_source,
            share_id,
        };

        if pdu_type == ShareControlPduType::DataPdu {
            // Some windows version have an issue where
            // there is some padding not part of the inner unit.
            // Consume that data
            let header_length = header.size();

            if header_length != total_length {
                if total_length < header_length {
                    return Err(not_enough_bytes_err!(total_length, header_length));
                }

                let padding = total_length - header_length;
                ensure_size!(in: src, size: padding);
                read_padding!(src, padding);
            }
        }

        Ok(header)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
pub enum ShareControlPdu {
    ServerDemandActive(ServerDemandActive),
    ClientConfirmActive(ClientConfirmActive),
    Data(ShareDataHeader),
    ServerDeactivateAll(ServerDeactivateAll),
    ServerRedirect(RdpServerRedirectionPacket),
}

impl ShareControlPdu {
    const NAME: &'static str = "ShareControlPdu";

    pub fn as_short_name(&self) -> &str {
        match self {
            ShareControlPdu::ServerDemandActive(_) => "Server Demand Active PDU",
            ShareControlPdu::ClientConfirmActive(_) => "Client Confirm Active PDU",
            ShareControlPdu::Data(_) => "Data PDU",
            ShareControlPdu::ServerDeactivateAll(_) => "Server Deactivate All PDU",
            ShareControlPdu::ServerRedirect(_) => "Server Redirection PDU",
        }
    }

    pub fn share_header_type(&self) -> ShareControlPduType {
        match self {
            ShareControlPdu::ServerDemandActive(_) => ShareControlPduType::DemandActivePdu,
            ShareControlPdu::ClientConfirmActive(_) => ShareControlPduType::ConfirmActivePdu,
            ShareControlPdu::Data(_) => ShareControlPduType::DataPdu,
            ShareControlPdu::ServerDeactivateAll(_) => ShareControlPduType::DeactivateAllPdu,
            ShareControlPdu::ServerRedirect(_) => ShareControlPduType::ServerRedirect,
        }
    }

    pub fn from_type(src: &mut ReadCursor<'_>, share_type: ShareControlPduType) -> DecodeResult<Self> {
        match share_type {
            ShareControlPduType::DemandActivePdu => {
                Ok(ShareControlPdu::ServerDemandActive(ServerDemandActive::decode(src)?))
            }
            ShareControlPduType::ConfirmActivePdu => {
                Ok(ShareControlPdu::ClientConfirmActive(ClientConfirmActive::decode(src)?))
            }
            ShareControlPduType::DataPdu => Ok(ShareControlPdu::Data(ShareDataHeader::decode(src)?)),
            ShareControlPduType::DeactivateAllPdu => {
                Ok(ShareControlPdu::ServerDeactivateAll(ServerDeactivateAll::decode(src)?))
            }
            ShareControlPduType::ServerRedirect => {
                Ok(ShareControlPdu::ServerRedirect(RdpServerRedirectionPacket::decode(src)?))
            }
        }
    }
}

impl Encode for ShareControlPdu {
    fn encode(&self, dst: &mut WriteCursor<'_>) -> EncodeResult<()> {
        match self {
            ShareControlPdu::ServerDemandActive(pdu) => pdu.encode(dst),
            ShareControlPdu::ClientConfirmActive(pdu) => pdu.encode(dst),
            ShareControlPdu::Data(share_data_header) => share_data_header.encode(dst),
            ShareControlPdu::ServerDeactivateAll(deactivate_all) => deactivate_all.encode(dst),
            ShareControlPdu::ServerRedirect(packet) => packet.encode(dst),
        }
    }

    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn size(&self) -> usize {
        match self {
            ShareControlPdu::ServerDemandActive(pdu) => pdu.size(),
            ShareControlPdu::ClientConfirmActive(pdu) => pdu.size(),
            ShareControlPdu::Data(share_data_header) => share_data_header.size(),
            ShareControlPdu::ServerDeactivateAll(deactivate_all) => deactivate_all.size(),
            ShareControlPdu::ServerRedirect(packet) => packet.size(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
pub struct ShareDataHeader {
    pub share_data_pdu: ShareDataPdu,
    pub stream_priority: StreamPriority,
    pub compression_flags: CompressionFlags,
    pub compression_type: client_info::CompressionType,
}

impl ShareDataHeader {
    const NAME: &'static str = "ShareDataHeader";

    const FIXED_PART_SIZE: usize = PADDING_FIELD_SIZE
        + STREAM_ID_FIELD_SIZE
        + UNCOMPRESSED_LENGTH_FIELD_SIZE
        + PDU_TYPE_FIELD_SIZE
        + COMPRESSION_TYPE_FIELD_SIZE
        + COMPRESSED_LENGTH_FIELD_SIZE;
}

impl Encode for ShareDataHeader {
    fn encode(&self, dst: &mut WriteCursor<'_>) -> EncodeResult<()> {
        ensure_size!(in: dst, size: self.size());

        if self.compression_flags.is_empty() {
            let compression_flags_with_type = self.compression_flags.bits() | self.compression_type.as_u8();

            write_padding!(dst, 1);
            dst.write_u8(self.stream_priority.as_u8());
            dst.write_u16(cast_length!("uncompressedLength", self.share_data_pdu.size())?);
            dst.write_u8(self.share_data_pdu.share_header_type().as_u8());
            dst.write_u8(compression_flags_with_type);
            dst.write_u16(0); // compressed length

            self.share_data_pdu.encode(dst)
        } else {
            Err(other_err!("Compression is not implemented"))
        }
    }

    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn size(&self) -> usize {
        Self::FIXED_PART_SIZE + self.share_data_pdu.size()
    }
}

impl<'de> Decode<'de> for ShareDataHeader {
    fn decode(src: &mut ReadCursor<'de>) -> DecodeResult<Self> {
        ensure_fixed_part_size!(in: src);

        read_padding!(src, 1);
        let stream_priority = StreamPriority::from_u8(src.read_u8())
            .ok_or_else(|| invalid_field_err!("streamPriority", "Invalid stream priority"))?;
        let _uncompressed_length = src.read_u16();
        let pdu_type = ShareDataPduType::from_u8(src.read_u8())
            .ok_or_else(|| invalid_field_err!("pduType", "Invalid pdu type"))?;
        let compression_flags_with_type = src.read_u8();

        let compression_flags =
            CompressionFlags::from_bits_retain(compression_flags_with_type & !SHARE_DATA_HEADER_COMPRESSION_MASK);
        let compression_type =
            client_info::CompressionType::from_u8(compression_flags_with_type & SHARE_DATA_HEADER_COMPRESSION_MASK)
                .ok_or_else(|| invalid_field_err!("compressionType", "Invalid compression type"))?;
        let _compressed_length = src.read_u16();

        let share_data_pdu = ShareDataPdu::from_type(src, pdu_type)?;

        Ok(Self {
            share_data_pdu,
            stream_priority,
            compression_flags,
            compression_type,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
pub enum ShareDataPdu {
    Synchronize(SynchronizePdu),
    Control(ControlPdu),
    FontList(FontPdu),
    FontMap(FontPdu),
    MonitorLayout(MonitorLayoutPdu),
    SaveSessionInfo(SaveSessionInfoPdu),
    FrameAcknowledge(FrameAcknowledgePdu),
    ServerSetErrorInfo(ServerSetErrorInfoPdu),
    Input(InputEventPdu),
    ShutdownRequest,
    ShutdownDenied,
    SuppressOutput(SuppressOutputPdu),
    RefreshRectangle(RefreshRectanglePdu),
    Update(Vec<u8>),
    Pointer(Vec<u8>),
    PlaySound(Vec<u8>),
    SetKeyboardIndicators(Vec<u8>),
    BitmapCachePersistentList(Vec<u8>),
    BitmapCacheErrorPdu(Vec<u8>),
    SetKeyboardImeStatus(Vec<u8>),
    OffscreenCacheErrorPdu(Vec<u8>),
    DrawNineGridErrorPdu(Vec<u8>),
    DrawGdiPusErrorPdu(Vec<u8>),
    ArcStatusPdu(Vec<u8>),
    StatusInfoPdu(Vec<u8>),
    /// Auto-Detect Request (server to client)
    AutoDetectReq(AutoDetectRequest),
    /// Auto-Detect Response (client to server)
    AutoDetectRsp(AutoDetectResponse),
}

impl ShareDataPdu {
    const NAME: &'static str = "ShareDataPdu";

    pub fn as_short_name(&self) -> &str {
        match self {
            ShareDataPdu::Synchronize(_) => "Synchronize PDU",
            ShareDataPdu::Control(_) => "Control PDU",
            ShareDataPdu::FontList(_) => "FontList PDU",
            ShareDataPdu::FontMap(_) => "Font Map PDU",
            ShareDataPdu::MonitorLayout(_) => "Monitor Layout PDU",
            ShareDataPdu::SaveSessionInfo(_) => "Save session info PDU",
            ShareDataPdu::FrameAcknowledge(_) => "Frame Acknowledge PDU",
            ShareDataPdu::ServerSetErrorInfo(_) => "Server Set Error Info PDU",
            ShareDataPdu::Input(_) => "Server Input PDU",
            ShareDataPdu::ShutdownRequest => "Shutdown Request PDU",
            ShareDataPdu::ShutdownDenied => "Shutdown Denied PDU",
            ShareDataPdu::SuppressOutput(_) => "Suppress Output PDU",
            ShareDataPdu::RefreshRectangle(_) => "Refresh Rectangle PDU",
            ShareDataPdu::Update(_) => "Update PDU",
            ShareDataPdu::Pointer(_) => "Pointer PDU",
            ShareDataPdu::PlaySound(_) => "Play Sound PDU",
            ShareDataPdu::SetKeyboardIndicators(_) => "Set Keyboard Indicators PDU",
            ShareDataPdu::BitmapCachePersistentList(_) => "Bitmap Cache Persistent List PDU",
            ShareDataPdu::BitmapCacheErrorPdu(_) => "Bitmap Cache Error PDU",
            ShareDataPdu::SetKeyboardImeStatus(_) => "Set Keyboard IME Status PDU",
            ShareDataPdu::OffscreenCacheErrorPdu(_) => "Offscreen Cache Error PDU",
            ShareDataPdu::DrawNineGridErrorPdu(_) => "Draw Nine Grid Error PDU",
            ShareDataPdu::DrawGdiPusErrorPdu(_) => "Draw GDI PUS Error PDU",
            ShareDataPdu::ArcStatusPdu(_) => "Arc Status PDU",
            ShareDataPdu::StatusInfoPdu(_) => "Status Info PDU",
            ShareDataPdu::AutoDetectReq(_) => "Auto-Detect Request PDU",
            ShareDataPdu::AutoDetectRsp(_) => "Auto-Detect Response PDU",
        }
    }

    pub fn share_header_type(&self) -> ShareDataPduType {
        match self {
            ShareDataPdu::Synchronize(_) => ShareDataPduType::Synchronize,
            ShareDataPdu::Control(_) => ShareDataPduType::Control,
            ShareDataPdu::FontList(_) => ShareDataPduType::FontList,
            ShareDataPdu::FontMap(_) => ShareDataPduType::FontMap,
            ShareDataPdu::MonitorLayout(_) => ShareDataPduType::MonitorLayoutPdu,
            ShareDataPdu::SaveSessionInfo(_) => ShareDataPduType::SaveSessionInfo,
            ShareDataPdu::FrameAcknowledge(_) => ShareDataPduType::FrameAcknowledgePdu,
            ShareDataPdu::ServerSetErrorInfo(_) => ShareDataPduType::SetErrorInfoPdu,
            ShareDataPdu::Input(_) => ShareDataPduType::Input,
            ShareDataPdu::ShutdownRequest => ShareDataPduType::ShutdownRequest,
            ShareDataPdu::ShutdownDenied => ShareDataPduType::ShutdownDenied,
            ShareDataPdu::SuppressOutput(_) => ShareDataPduType::SuppressOutput,
            ShareDataPdu::RefreshRectangle(_) => ShareDataPduType::RefreshRectangle,
            ShareDataPdu::Update(_) => ShareDataPduType::Update,
            ShareDataPdu::Pointer(_) => ShareDataPduType::Pointer,
            ShareDataPdu::PlaySound(_) => ShareDataPduType::PlaySound,
            ShareDataPdu::SetKeyboardIndicators(_) => ShareDataPduType::SetKeyboardIndicators,
            ShareDataPdu::BitmapCachePersistentList(_) => ShareDataPduType::BitmapCachePersistentList,
            ShareDataPdu::BitmapCacheErrorPdu(_) => ShareDataPduType::BitmapCacheErrorPdu,
            ShareDataPdu::SetKeyboardImeStatus(_) => ShareDataPduType::SetKeyboardImeStatus,
            ShareDataPdu::OffscreenCacheErrorPdu(_) => ShareDataPduType::OffscreenCacheErrorPdu,
            ShareDataPdu::DrawNineGridErrorPdu(_) => ShareDataPduType::DrawNineGridErrorPdu,
            ShareDataPdu::DrawGdiPusErrorPdu(_) => ShareDataPduType::DrawGdiPusErrorPdu,
            ShareDataPdu::ArcStatusPdu(_) => ShareDataPduType::ArcStatusPdu,
            ShareDataPdu::StatusInfoPdu(_) => ShareDataPduType::StatusInfoPdu,
            ShareDataPdu::AutoDetectReq(_) | ShareDataPdu::AutoDetectRsp(_) => ShareDataPduType::AutoDetect,
        }
    }

    fn from_type(src: &mut ReadCursor<'_>, share_type: ShareDataPduType) -> DecodeResult<Self> {
        match share_type {
            ShareDataPduType::Synchronize => Ok(ShareDataPdu::Synchronize(SynchronizePdu::decode(src)?)),
            ShareDataPduType::Control => Ok(ShareDataPdu::Control(ControlPdu::decode(src)?)),
            ShareDataPduType::FontList => Ok(ShareDataPdu::FontList(FontPdu::decode(src)?)),
            ShareDataPduType::FontMap => Ok(ShareDataPdu::FontMap(FontPdu::decode(src)?)),
            ShareDataPduType::MonitorLayoutPdu => Ok(ShareDataPdu::MonitorLayout(MonitorLayoutPdu::decode(src)?)),
            ShareDataPduType::SaveSessionInfo => Ok(ShareDataPdu::SaveSessionInfo(SaveSessionInfoPdu::decode(src)?)),
            ShareDataPduType::FrameAcknowledgePdu => {
                Ok(ShareDataPdu::FrameAcknowledge(FrameAcknowledgePdu::decode(src)?))
            }
            ShareDataPduType::SetErrorInfoPdu => {
                Ok(ShareDataPdu::ServerSetErrorInfo(ServerSetErrorInfoPdu::decode(src)?))
            }
            ShareDataPduType::Input => Ok(ShareDataPdu::Input(InputEventPdu::decode(src)?)),
            ShareDataPduType::ShutdownRequest => Ok(ShareDataPdu::ShutdownRequest),
            ShareDataPduType::ShutdownDenied => Ok(ShareDataPdu::ShutdownDenied),
            ShareDataPduType::SuppressOutput => Ok(ShareDataPdu::SuppressOutput(SuppressOutputPdu::decode(src)?)),
            ShareDataPduType::RefreshRectangle => Ok(ShareDataPdu::RefreshRectangle(RefreshRectanglePdu::decode(src)?)),
            ShareDataPduType::Update => Ok(ShareDataPdu::Update(src.remaining().to_vec())),
            ShareDataPduType::Pointer => Ok(ShareDataPdu::Pointer(src.remaining().to_vec())),
            ShareDataPduType::PlaySound => Ok(ShareDataPdu::PlaySound(src.remaining().to_vec())),
            ShareDataPduType::SetKeyboardIndicators => {
                Ok(ShareDataPdu::SetKeyboardIndicators(src.remaining().to_vec()))
            }
            ShareDataPduType::BitmapCachePersistentList => {
                Ok(ShareDataPdu::BitmapCachePersistentList(src.remaining().to_vec()))
            }
            ShareDataPduType::BitmapCacheErrorPdu => Ok(ShareDataPdu::BitmapCacheErrorPdu(src.remaining().to_vec())),
            ShareDataPduType::SetKeyboardImeStatus => Ok(ShareDataPdu::SetKeyboardImeStatus(src.remaining().to_vec())),
            ShareDataPduType::OffscreenCacheErrorPdu => {
                Ok(ShareDataPdu::OffscreenCacheErrorPdu(src.remaining().to_vec()))
            }
            ShareDataPduType::DrawNineGridErrorPdu => Ok(ShareDataPdu::DrawNineGridErrorPdu(src.remaining().to_vec())),
            ShareDataPduType::DrawGdiPusErrorPdu => Ok(ShareDataPdu::DrawGdiPusErrorPdu(src.remaining().to_vec())),
            ShareDataPduType::ArcStatusPdu => Ok(ShareDataPdu::ArcStatusPdu(src.remaining().to_vec())),
            ShareDataPduType::StatusInfoPdu => Ok(ShareDataPdu::StatusInfoPdu(src.remaining().to_vec())),
            ShareDataPduType::AutoDetect => {
                ensure_size!(in: src, size: 2);
                let type_id = src.remaining()[1];
                if type_id == crate::rdp::autodetect::TYPE_ID_AUTODETECT_REQUEST {
                    Ok(ShareDataPdu::AutoDetectReq(AutoDetectRequest::decode(src)?))
                } else {
                    Ok(ShareDataPdu::AutoDetectRsp(AutoDetectResponse::decode(src)?))
                }
            }
        }
    }
}

impl Encode for ShareDataPdu {
    fn encode(&self, dst: &mut WriteCursor<'_>) -> EncodeResult<()> {
        match self {
            ShareDataPdu::Synchronize(pdu) => pdu.encode(dst),
            ShareDataPdu::Control(pdu) => pdu.encode(dst),
            ShareDataPdu::FontList(pdu) | ShareDataPdu::FontMap(pdu) => pdu.encode(dst),
            ShareDataPdu::MonitorLayout(pdu) => pdu.encode(dst),
            ShareDataPdu::SaveSessionInfo(pdu) => pdu.encode(dst),
            ShareDataPdu::FrameAcknowledge(pdu) => pdu.encode(dst),
            ShareDataPdu::ServerSetErrorInfo(pdu) => pdu.encode(dst),
            ShareDataPdu::Input(pdu) => pdu.encode(dst),
            ShareDataPdu::ShutdownRequest | ShareDataPdu::ShutdownDenied => Ok(()),
            ShareDataPdu::SuppressOutput(pdu) => pdu.encode(dst),
            ShareDataPdu::RefreshRectangle(pdu) => pdu.encode(dst),
            ShareDataPdu::AutoDetectReq(pdu) => pdu.encode(dst),
            ShareDataPdu::AutoDetectRsp(pdu) => pdu.encode(dst),
            _ => Err(other_err!("Encoding not implemented")),
        }
    }

    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn size(&self) -> usize {
        match self {
            ShareDataPdu::Synchronize(pdu) => pdu.size(),
            ShareDataPdu::Control(pdu) => pdu.size(),
            ShareDataPdu::FontList(pdu) | ShareDataPdu::FontMap(pdu) => pdu.size(),
            ShareDataPdu::MonitorLayout(pdu) => pdu.size(),
            ShareDataPdu::SaveSessionInfo(pdu) => pdu.size(),
            ShareDataPdu::FrameAcknowledge(pdu) => pdu.size(),
            ShareDataPdu::ServerSetErrorInfo(pdu) => pdu.size(),
            ShareDataPdu::Input(pdu) => pdu.size(),
            ShareDataPdu::ShutdownRequest | ShareDataPdu::ShutdownDenied => 0,
            ShareDataPdu::SuppressOutput(pdu) => pdu.size(),
            ShareDataPdu::RefreshRectangle(pdu) => pdu.size(),
            ShareDataPdu::Update(buffer)
            | ShareDataPdu::Pointer(buffer)
            | ShareDataPdu::PlaySound(buffer)
            | ShareDataPdu::SetKeyboardIndicators(buffer)
            | ShareDataPdu::BitmapCachePersistentList(buffer)
            | ShareDataPdu::BitmapCacheErrorPdu(buffer)
            | ShareDataPdu::SetKeyboardImeStatus(buffer)
            | ShareDataPdu::OffscreenCacheErrorPdu(buffer)
            | ShareDataPdu::DrawNineGridErrorPdu(buffer)
            | ShareDataPdu::DrawGdiPusErrorPdu(buffer)
            | ShareDataPdu::ArcStatusPdu(buffer)
            | ShareDataPdu::StatusInfoPdu(buffer) => buffer.len(),
            ShareDataPdu::AutoDetectReq(pdu) => pdu.size(),
            ShareDataPdu::AutoDetectRsp(pdu) => pdu.size(),
        }
    }
}

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
    pub struct BasicSecurityHeaderFlags: u16 {
        const EXCHANGE_PKT = 0x0001;
        const TRANSPORT_REQ = 0x0002;
        const TRANSPORT_RSP = 0x0004;
        const ENCRYPT = 0x0008;
        const RESET_SEQNO = 0x0010;
        const IGNORE_SEQNO = 0x0020;
        const INFO_PKT = 0x0040;
        const LICENSE_PKT = 0x0080;
        const LICENSE_ENCRYPT_CS = 0x0100;
        const LICENSE_ENCRYPT_SC = 0x0200;
        const REDIRECTION_PKT = 0x0400;
        const SECURE_CHECKSUM = 0x0800;
        const AUTODETECT_REQ = 0x1000;
        const AUTODETECT_RSP = 0x2000;
        const HEARTBEAT = 0x4000;
        const FLAGSHI_VALID = 0x8000;
    }
}

#[repr(u8)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, FromPrimitive)]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
pub enum StreamPriority {
    Undefined = 0,
    Low = 1,
    Medium = 2,
    High = 4,
}

impl StreamPriority {
    #[expect(
        clippy::as_conversions,
        reason = "guarantees discriminant layout, and as is the only way to cast enum -> primitive"
    )]
    fn as_u8(self) -> u8 {
        self as u8
    }
}

#[repr(u16)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, FromPrimitive)]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
pub enum ShareControlPduType {
    DemandActivePdu = 0x1,
    ConfirmActivePdu = 0x3,
    DeactivateAllPdu = 0x6,
    DataPdu = 0x7,
    ServerRedirect = 0xa,
}

impl ShareControlPduType {
    #[expect(
        clippy::as_conversions,
        reason = "guarantees discriminant layout, and as is the only way to cast enum -> primitive"
    )]
    fn as_u16(self) -> u16 {
        self as u16
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, FromPrimitive)]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[repr(u8)]
pub enum ShareDataPduType {
    Update = 0x02,
    Control = 0x14,
    Pointer = 0x1b,
    Input = 0x1c,
    Synchronize = 0x1f,
    RefreshRectangle = 0x21,
    PlaySound = 0x22,
    SuppressOutput = 0x23,
    ShutdownRequest = 0x24,
    ShutdownDenied = 0x25,
    SaveSessionInfo = 0x26,
    FontList = 0x27,
    FontMap = 0x28,
    SetKeyboardIndicators = 0x29,
    BitmapCachePersistentList = 0x2b,
    BitmapCacheErrorPdu = 0x2c,
    SetKeyboardImeStatus = 0x2d,
    OffscreenCacheErrorPdu = 0x2e,
    SetErrorInfoPdu = 0x2f,
    DrawNineGridErrorPdu = 0x30,
    DrawGdiPusErrorPdu = 0x31,
    ArcStatusPdu = 0x32,
    StatusInfoPdu = 0x36,
    MonitorLayoutPdu = 0x37,
    FrameAcknowledgePdu = 0x38,
    /// Auto-Detect Request or Response ([MS-RDPBCGR 2.2.14]).
    ///
    /// The headerTypeId field within the PDU body discriminates direction:
    /// 0x00 for server-to-client requests, 0x01 for client-to-server responses.
    ///
    /// [MS-RDPBCGR 2.2.14]: https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-rdpbcgr/dc672839-4f4e-40b1-a71c-cd6a959baa38
    AutoDetect = 0x3b,
}

impl ShareDataPduType {
    #[expect(
        clippy::as_conversions,
        reason = "guarantees discriminant layout, and as is the only way to cast enum -> primitive"
    )]
    fn as_u8(self) -> u8 {
        self as u8
    }
}

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
    pub struct CompressionFlags: u8 {
        const COMPRESSED = 0x20;
        const AT_FRONT = 0x40;
        const FLUSHED = 0x80;

        const _ = !0;
    }
}

/// 2.2.3.1 Server Deactivate All PDU
///
/// [2.2.3.1]: https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-rdpbcgr/8a29971a-df3c-48da-add2-8ed9a05edc89
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
pub struct ServerDeactivateAll;

impl ServerDeactivateAll {
    const FIXED_PART_SIZE: usize = 2 /* length_source_descriptor */ + 1 /* source_descriptor */;
}

impl Decode<'_> for ServerDeactivateAll {
    fn decode(src: &mut ReadCursor<'_>) -> DecodeResult<Self> {
        // Some servers (notably XRDP and older Windows versions) send a short
        // Deactivate All PDU without the sourceDescriptor field. FreeRDP
        // handles this by treating any remaining data as optional.
        if src.len() >= Self::FIXED_PART_SIZE {
            let length_source_descriptor = src.read_u16();
            ensure_size!(in: src, size: length_source_descriptor.into());
            let _ = src.read_slice(length_source_descriptor.into());
        }
        Ok(Self)
    }
}

impl Encode for ServerDeactivateAll {
    fn encode(&self, dst: &mut WriteCursor<'_>) -> EncodeResult<()> {
        ensure_fixed_part_size!(in: dst);
        // A 16-bit, unsigned integer. The size in bytes of the sourceDescriptor field.
        dst.write_u16(1);
        // Variable number of bytes. The source descriptor. This field SHOULD be set to 0x00.
        dst.write_u8(0);
        Ok(())
    }

    fn name(&self) -> &'static str {
        "Server Deactivate All"
    }

    fn size(&self) -> usize {
        Self::FIXED_PART_SIZE
    }
}

bitflags! {
    /// RedirFlags bit field of the [`RdpServerRedirectionPacket`].
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
    pub struct RedirectionFlags: u32 {
        const LB_TARGET_NET_ADDRESS = 0x0000_0001;
        const LB_LOAD_BALANCE_INFO = 0x0000_0002;
        const LB_USERNAME = 0x0000_0004;
        const LB_DOMAIN = 0x0000_0008;
        const LB_PASSWORD = 0x0000_0010;
        const LB_DONTSTOREUSERNAME = 0x0000_0020;
        const LB_SMARTCARD_LOGON = 0x0000_0040;
        const LB_NOREDIRECT = 0x0000_0080;
        const LB_TARGET_FQDN = 0x0000_0100;
        const LB_TARGET_NETBIOS_NAME = 0x0000_0200;
        const LB_TARGET_NET_ADDRESSES = 0x0000_0800;
        const LB_CLIENT_TSV_URL = 0x0000_1000;
        const LB_SERVER_TSV_CAPABLE = 0x0000_2000;
        const LB_PASSWORD_IS_PK_ENCRYPTED = 0x0000_4000;
        const LB_REDIRECTION_GUID = 0x0000_8000;
        const LB_TARGET_CERTIFICATE = 0x0001_0000;

        const _ = !0;
    }
}

const SEC_REDIRECTION_PKT_MARKER: u16 = 0x0400;

/// 2.2.13.2.1 Server Redirection Packet (RDP_SERVER_REDIRECTION_PACKET)
///
/// Sent (wrapped in a Share Control Header, `pduType` = `ServerRedirect`) either
/// during the connection sequence or, as here, deferred until after the client
/// is already active — e.g. GNOME Remote Desktop's headless "Remote Login" mode
/// hands off from its greeter-level daemon to a per-user session this way. Every
/// optional field below is opaque to the client: it is not parsed further, only
/// forwarded verbatim (routing token on the X.224 reconnect, and username/domain/
/// password/redirection GUID into the RDSTLS Authentication Request PDU).
///
/// [2.2.13.2.1]: https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-rdpbcgr/df3d59e6-30a8-4a36-bd2d-9d11bcd96c3e
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
pub struct RdpServerRedirectionPacket {
    pub session_id: u32,
    pub flags: RedirectionFlags,
    pub target_net_address: Option<Vec<u8>>,
    /// Opaque routing token. When `LB_TARGET_NET_ADDRESS` is not set, this MUST
    /// be sent as the X.224 routing token (see `nego::NegoRequestData::RoutingToken`)
    /// on reconnect.
    pub load_balance_info: Option<Vec<u8>>,
    /// UTF-16LE, null-terminated, exactly as it appeared on the wire.
    pub user_name: Option<Vec<u8>>,
    pub domain: Option<Vec<u8>>,
    /// If `flags` contains `LB_PASSWORD_IS_PK_ENCRYPTED`, this is an opaque blob
    /// pre-encrypted by the server for RDSTLS and MUST NOT be decrypted or
    /// modified by the client.
    pub password: Option<Vec<u8>>,
    pub target_fqdn: Option<Vec<u8>>,
    pub target_netbios_name: Option<Vec<u8>>,
    pub tsv_url: Option<Vec<u8>>,
    /// Base64-encoded GUID (UTF-16), forwarded verbatim into RDSTLS.
    pub redirection_guid: Option<Vec<u8>>,
    /// Base64-encoded Target Certificate Container (UTF-16). The TLS certificate
    /// presented on reconnect SHOULD match this.
    pub target_certificate: Option<Vec<u8>>,
    pub target_net_addresses: Option<Vec<u8>>,
}

impl RdpServerRedirectionPacket {
    const FIXED_PART_SIZE: usize = 2 /* Flags */ + 2 /* Length */ + 4 /* SessionID */ + 4 /* RedirFlags */;

    fn optional_fields_size(&self) -> usize {
        [
            &self.target_net_address,
            &self.load_balance_info,
            &self.user_name,
            &self.domain,
            &self.password,
            &self.target_fqdn,
            &self.target_netbios_name,
            &self.tsv_url,
            &self.redirection_guid,
            &self.target_certificate,
            &self.target_net_addresses,
        ]
        .into_iter()
        .filter_map(|field| field.as_ref())
        .map(|field| 4 /* length prefix */ + field.len())
        .sum()
    }
}

fn read_length_prefixed_field(src: &mut ReadCursor<'_>) -> DecodeResult<Vec<u8>> {
    ensure_size!(in: src, size: 4);
    let len = cast_length!("fieldLength", src.read_u32())?;
    ensure_size!(in: src, size: len);
    Ok(src.read_slice(len).to_vec())
}

fn write_length_prefixed_field(dst: &mut WriteCursor<'_>, field: &[u8]) -> EncodeResult<()> {
    dst.write_u32(cast_length!("fieldLength", field.len())?);
    dst.write_slice(field);
    Ok(())
}

impl Decode<'_> for RdpServerRedirectionPacket {
    fn decode(src: &mut ReadCursor<'_>) -> DecodeResult<Self> {
        ensure_size!(in: src, size: Self::FIXED_PART_SIZE);

        let marker = src.read_u16();
        if marker != SEC_REDIRECTION_PKT_MARKER {
            return Err(invalid_field_err!("flags", "expected SEC_REDIRECTION_PKT (0x0400)"));
        }
        let _length = src.read_u16();
        let session_id = src.read_u32();
        let flags = RedirectionFlags::from_bits_retain(src.read_u32());

        let target_net_address = flags
            .contains(RedirectionFlags::LB_TARGET_NET_ADDRESS)
            .then(|| read_length_prefixed_field(src))
            .transpose()?;
        let load_balance_info = flags
            .contains(RedirectionFlags::LB_LOAD_BALANCE_INFO)
            .then(|| read_length_prefixed_field(src))
            .transpose()?;
        let user_name = flags
            .contains(RedirectionFlags::LB_USERNAME)
            .then(|| read_length_prefixed_field(src))
            .transpose()?;
        let domain = flags
            .contains(RedirectionFlags::LB_DOMAIN)
            .then(|| read_length_prefixed_field(src))
            .transpose()?;
        let password = flags
            .contains(RedirectionFlags::LB_PASSWORD)
            .then(|| read_length_prefixed_field(src))
            .transpose()?;
        let target_fqdn = flags
            .contains(RedirectionFlags::LB_TARGET_FQDN)
            .then(|| read_length_prefixed_field(src))
            .transpose()?;
        let target_netbios_name = flags
            .contains(RedirectionFlags::LB_TARGET_NETBIOS_NAME)
            .then(|| read_length_prefixed_field(src))
            .transpose()?;
        let tsv_url = flags
            .contains(RedirectionFlags::LB_CLIENT_TSV_URL)
            .then(|| read_length_prefixed_field(src))
            .transpose()?;
        let redirection_guid = flags
            .contains(RedirectionFlags::LB_REDIRECTION_GUID)
            .then(|| read_length_prefixed_field(src))
            .transpose()?;
        let target_certificate = flags
            .contains(RedirectionFlags::LB_TARGET_CERTIFICATE)
            .then(|| read_length_prefixed_field(src))
            .transpose()?;
        let target_net_addresses = flags
            .contains(RedirectionFlags::LB_TARGET_NET_ADDRESSES)
            .then(|| read_length_prefixed_field(src))
            .transpose()?;
        // Any remaining bytes are the optional 8-byte Pad field ("MUST be
        // ignored" per spec) — nothing left to read.

        Ok(Self {
            session_id,
            flags,
            target_net_address,
            load_balance_info,
            user_name,
            domain,
            password,
            target_fqdn,
            target_netbios_name,
            tsv_url,
            redirection_guid,
            target_certificate,
            target_net_addresses,
        })
    }
}

impl Encode for RdpServerRedirectionPacket {
    fn encode(&self, dst: &mut WriteCursor<'_>) -> EncodeResult<()> {
        ensure_size!(in: dst, size: self.size());

        dst.write_u16(SEC_REDIRECTION_PKT_MARKER);
        dst.write_u16(cast_length!("length", self.size())?);
        dst.write_u32(self.session_id);
        dst.write_u32(self.flags.bits());

        for field in [
            &self.target_net_address,
            &self.load_balance_info,
            &self.user_name,
            &self.domain,
            &self.password,
            &self.target_fqdn,
            &self.target_netbios_name,
            &self.tsv_url,
            &self.redirection_guid,
            &self.target_certificate,
            &self.target_net_addresses,
        ] {
            if let Some(field) = field {
                write_length_prefixed_field(dst, field)?;
            }
        }

        Ok(())
    }

    fn name(&self) -> &'static str {
        "Server Redirection Packet"
    }

    fn size(&self) -> usize {
        Self::FIXED_PART_SIZE + self.optional_fields_size()
    }
}

#[cfg(test)]
mod server_redirection_tests {
    use super::*;

    /// Decodes a gnome-remote-desktop-style deferred Server Redirection PDU:
    /// share control header (6 bytes, no shareId!) + pad2Octets + packet.
    /// Guards the offset bug where shareId was read unconditionally, eating
    /// pad2Octets + the packet's 0x0400 marker.
    #[test]
    fn decodes_share_control_wrapped_server_redirection() {
        let cookie = b"Cookie: msts=1611166392\r\n";
        let user: Vec<u8> = "u\0".encode_utf16().flat_map(|c| c.to_le_bytes()).collect();

        let mut body = Vec::new();
        body.extend(0x0400u16.to_le_bytes()); // Flags: SEC_REDIRECTION_PKT
        let flags = RedirectionFlags::LB_LOAD_BALANCE_INFO | RedirectionFlags::LB_USERNAME;
        let length = 12 + 4 + cookie.len() + 4 + user.len();
        body.extend((u16::try_from(length).unwrap()).to_le_bytes());
        body.extend(0u32.to_le_bytes()); // SessionID
        body.extend(flags.bits().to_le_bytes());
        body.extend((u32::try_from(cookie.len()).unwrap()).to_le_bytes());
        body.extend_from_slice(cookie);
        body.extend((u32::try_from(user.len()).unwrap()).to_le_bytes());
        body.extend_from_slice(&user);

        let mut wire = Vec::new();
        wire.extend((u16::try_from(6 + 2 + body.len()).unwrap()).to_le_bytes()); // totalLength
        wire.extend((PROTOCOL_VERSION | ShareControlPduType::ServerRedirect.as_u16()).to_le_bytes());
        wire.extend(0u16.to_le_bytes()); // pduSource
        wire.extend(0u16.to_le_bytes()); // pad2Octets — NOT a shareId
        wire.extend_from_slice(&body);

        let mut cursor = ReadCursor::new(&wire);
        let header = ShareControlHeader::decode(&mut cursor).expect("decode");
        let ShareControlPdu::ServerRedirect(packet) = header.share_control_pdu else {
            panic!("expected ServerRedirect, got {}", header.share_control_pdu.as_short_name());
        };
        assert_eq!(packet.flags, flags);
        assert_eq!(packet.load_balance_info.as_deref(), Some(cookie.as_slice()));
        assert_eq!(packet.user_name.as_deref(), Some(user.as_slice()));
        assert_eq!(packet.password, None);
    }
}
