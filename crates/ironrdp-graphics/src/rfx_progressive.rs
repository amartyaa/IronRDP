//! RFX Progressive codec decoder (MS-RDPRFX progressive mode, carried by
//! MS-RDPEGFX `RDPGFX_WIRE_TO_SURFACE_PDU_2`).
//!
//! This is a faithful Rust port of the decode path in FreeRDP's
//! `libfreerdp/codec/progressive.c` (Apache-2.0), layered on top of IronRDP's
//! existing RFX DSP primitives (`rlgr`, `dwt`, `dwt_extrapolate`,
//! `subband_reconstruction`) and the progressive PDU parser in
//! `ironrdp_pdu::codecs::rfx::progressive`.
//!
//! Progressive sends a coarse first pass for a whole region, then refinement
//! ("upgrade") passes that sharpen it — so the client sees a complete frame
//! quickly instead of the raster tile-march of legacy RFX. Tiles carry
//! persistent state across passes (the DWT coefficient buffer `current` and the
//! tri-state `sign` buffer), held per-surface in a tile grid keyed by
//! `(x_idx, y_idx)`.

use core::fmt;

use ironrdp_pdu::codecs::rfx::progressive::{
    decode_progressive_stream, ComponentCodecQuant, ProgressiveBlock, ProgressiveCodecQuant, ProgressiveRegion,
    ProgressiveTile,
};
use ironrdp_pdu::codecs::rfx::EntropyAlgorithm;
use ironrdp_pdu::geometry::InclusiveRectangle;

use crate::color_conversion::{ycbcr_to_rgba, YCbCrBuffer};
use crate::{dwt, dwt_extrapolate, rlgr, subband_reconstruction};

const TILE_SIZE: usize = 64;
const TILE_PIXELS: usize = TILE_SIZE * TILE_SIZE; // 4096
const TILE_RGBA: usize = TILE_PIXELS * 4;

/// Tile flag bit: tile coefficients are a delta from the previous frame.
const RFX_TILE_DIFFERENCE: u8 = 0x01;

// Sub-band offsets/lengths within the 4096-coefficient tile buffer.
// Band order: HL1, LH1, HH1, HL2, LH2, HH2, HL3, LH3, HH3, LL3.
const OFF_NORMAL: [usize; 10] = [0, 1024, 2048, 3072, 3328, 3584, 3840, 3904, 3968, 4032];
const LEN_NORMAL: [usize; 10] = [1024, 1024, 1024, 256, 256, 256, 64, 64, 64, 64];
const OFF_EXTRAP: [usize; 10] = [0, 1023, 2046, 3007, 3279, 3551, 3807, 3879, 3951, 4015];
const LEN_EXTRAP: [usize; 10] = [1023, 1023, 961, 272, 272, 256, 72, 72, 64, 81];

/// Decode error for the progressive codec.
#[derive(Debug)]
pub struct ProgressiveError(pub String);

impl fmt::Display for ProgressiveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "rfx-progressive: {}", self.0)
    }
}

impl std::error::Error for ProgressiveError {}

type Result<T> = core::result::Result<T, ProgressiveError>;

/// Summary of one `decode()` call — surfaced for verbose logging at the call site.
#[derive(Debug, Default, Clone)]
pub struct ProgressiveUpdate {
    /// Dirty rectangles relative to the surface origin (0, 0).
    pub dirty: Vec<InclusiveRectangle>,
    pub tiles_simple: u32,
    pub tiles_first: u32,
    pub tiles_upgrade: u32,
    /// Number of regions seen and the rect count of the first region (diagnostics).
    pub regions: u32,
    pub region0_rects: u32,
    /// Whether the region used the reduce-extrapolate DWT variant.
    pub extrapolate: bool,
    /// Count of tiles whose flags requested inter-frame coeff diff.
    pub coeff_diff_tiles: u32,
    /// Tiles skipped due to a decode error (a bad tile no longer aborts the frame).
    pub errors: u32,
    /// First per-tile error message, for diagnosis.
    pub first_error: Option<String>,
    /// Tiles that decoded to (near-)black pixels — distinguishes a decode/data bug
    /// from an unpainted region.
    pub black_tiles: u32,
    /// Grid position (x_idx, y_idx) of the first black tile.
    pub first_black: Option<(u16, u16)>,
}

/// Convert a `ComponentCodecQuant` into a band-ordered array
/// `[HL1, LH1, HH1, HL2, LH2, HH2, HL3, LH3, HH3, LL3]`.
fn bands(q: &ComponentCodecQuant) -> [i32; 10] {
    [
        i32::from(q.hl1),
        i32::from(q.lh1),
        i32::from(q.hh1),
        i32::from(q.hl2),
        i32::from(q.lh2),
        i32::from(q.hh2),
        i32::from(q.hl3),
        i32::from(q.lh3),
        i32::from(q.hh3),
        i32::from(q.ll3),
    ]
}

/// `quantProgValFull`: full-quality progressive quant is all-zero (FreeRDP).
const PROG_QUANT_FULL: [i32; 10] = [0; 10];

/// Left-shift a coefficient band in place by `shift` bits (the progressive
/// dequantization step — FreeRDP `progressive_rfx_decode_block` /
/// `lShiftC_16s_inplace`). `shift == 0` is a no-op.
fn lshift_band(buffer: &mut [i16], off: usize, len: usize, shift: i32) {
    if shift <= 0 {
        return;
    }
    let s = shift as u32;
    for v in &mut buffer[off..off + len] {
        // Mimic FreeRDP's 16-bit left shift (wraps on overflow).
        *v = (*v as u16).wrapping_shl(s) as i16;
    }
}

/// Persistent per-tile state, held in the surface tile grid.
struct TileState {
    /// DWT coefficient buffer per component (Y, Cb, Cr) — accumulated across passes.
    current: [Box<[i16; TILE_PIXELS]>; 3],
    /// Tri-state sign buffer per component (set on first pass, used by upgrades).
    sign: [Box<[i16; TILE_PIXELS]>; 3],
    /// Persisted per-band bit positions per component (for upgrade numBits).
    bit_pos: [[i32; 10]; 3],
    pass: u16,
}

impl TileState {
    fn new() -> Self {
        Self {
            current: [
                Box::new([0; TILE_PIXELS]),
                Box::new([0; TILE_PIXELS]),
                Box::new([0; TILE_PIXELS]),
            ],
            sign: [
                Box::new([0; TILE_PIXELS]),
                Box::new([0; TILE_PIXELS]),
                Box::new([0; TILE_PIXELS]),
            ],
            bit_pos: [[0; 10]; 3],
            pass: 0,
        }
    }
}

/// MSB-first bit reader over a byte slice. Reads past the end yield zero bits,
/// matching FreeRDP's zero-padded `wBitStream` accumulator behavior.
struct BitStream<'a> {
    data: &'a [u8],
    pos: usize, // in bits
}

impl<'a> BitStream<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    #[inline]
    fn read_bit(&mut self) -> u32 {
        let byte = self.data.get(self.pos >> 3).copied().unwrap_or(0);
        let bit = (byte >> (7 - (self.pos & 7))) & 1;
        self.pos += 1;
        u32::from(bit)
    }

    #[inline]
    fn read_bits(&mut self, n: u32) -> u32 {
        let mut v = 0u32;
        for _ in 0..n {
            v = (v << 1) | self.read_bit();
        }
        v
    }
}

/// Upgrade-pass bitstream state (FreeRDP `RFX_PROGRESSIVE_UPGRADE_STATE`).
/// SRL (for previously-zero coefficients) and RAW (for non-zero) are consumed
/// interleaved per coefficient, decided by the tri-state sign.
struct UpgradeState<'a> {
    srl: BitStream<'a>,
    raw: BitStream<'a>,
    kp: i32,
    nz: i32,
    mode: bool, // false = zero-encoding, true = unary
}

impl<'a> UpgradeState<'a> {
    fn new(srl_data: &'a [u8], raw_data: &'a [u8]) -> Self {
        Self {
            srl: BitStream::new(srl_data),
            raw: BitStream::new(raw_data),
            kp: 8,
            nz: 0,
            mode: false,
        }
    }

    /// FreeRDP `rawShift`: read `num_bits` from the RAW stream as a magnitude.
    #[inline]
    fn raw_shift(&mut self, num_bits: u32) -> i32 {
        self.raw.read_bits(num_bits) as i32
    }

    /// FreeRDP `progressive_rfx_srl_read`: one value from the SRL stream.
    fn srl_read(&mut self, num_bits: u32) -> i16 {
        if self.nz > 0 {
            self.nz -= 1;
            return 0;
        }

        let k = (self.kp / 8) as u32;

        if !self.mode {
            // Zero-encoding mode.
            let bit = self.srl.read_bit();
            if bit == 0 {
                // '0' bit: a run of (1 << k) zeros.
                self.nz = 1i32 << k;
                self.kp += 4;
                if self.kp > 80 {
                    self.kp = 80;
                }
                self.nz -= 1;
                return 0;
            }
            // '1' bit: unary encoding is next; nz = next k bits.
            self.nz = 0;
            self.mode = true;
            if k > 0 {
                self.nz = self.srl.read_bits(k) as i32;
            }
            if self.nz > 0 {
                self.nz -= 1;
                return 0;
            }
        }

        // Unary encoding.
        self.mode = false;
        let sign = self.srl.read_bit();

        if self.kp < 6 {
            self.kp = 0;
        } else {
            self.kp -= 6;
        }

        if num_bits == 1 {
            return if sign != 0 { -1 } else { 1 };
        }

        let mut mag: u32 = 1;
        let max: u32 = (1u32 << num_bits) - 1;
        while mag < max {
            if self.srl.read_bit() != 0 {
                break;
            }
            mag += 1;
        }

        let mag = mag.min(i16::MAX as u32) as i16;
        if sign != 0 {
            -mag
        } else {
            mag
        }
    }
}

/// A surface's progressive decoder: a persistent tile grid plus scratch buffers.
pub struct ProgressiveDecoder {
    width: u16,
    height: u16,
    grid_width: usize,
    grid_height: usize,
    tiles: Vec<Option<Box<TileState>>>,
    // Scratch reused across tiles to avoid per-tile allocation on the hot path.
    buffer: [Box<[i16; TILE_PIXELS]>; 3],
    dwt_temp: Box<[i16; TILE_PIXELS]>,
    rgba_tile: Box<[u8; TILE_RGBA]>,
}

impl ProgressiveDecoder {
    /// Create a decoder for a surface of the given pixel dimensions.
    pub fn new(width: u16, height: u16) -> Self {
        let grid_width = (usize::from(width) + TILE_SIZE - 1) / TILE_SIZE;
        let grid_height = (usize::from(height) + TILE_SIZE - 1) / TILE_SIZE;
        let mut tiles = Vec::new();
        tiles.resize_with(grid_width * grid_height, || None);
        Self {
            width,
            height,
            grid_width,
            grid_height,
            tiles,
            buffer: [
                Box::new([0; TILE_PIXELS]),
                Box::new([0; TILE_PIXELS]),
                Box::new([0; TILE_PIXELS]),
            ],
            dwt_temp: Box::new([0; TILE_PIXELS]),
            rgba_tile: Box::new([0; TILE_RGBA]),
        }
    }

    /// Drop all persistent tile state (e.g. on `ResetGraphics`).
    pub fn reset(&mut self) {
        for t in &mut self.tiles {
            *t = None;
        }
    }

    pub fn width(&self) -> u16 {
        self.width
    }

    pub fn height(&self) -> u16 {
        self.height
    }

    /// Decode a WireToSurface2 progressive bitmap stream into `dst` (an RGBA
    /// framebuffer for this surface, `dst_stride` bytes per row). Returns the
    /// updated regions (relative to the surface origin) plus per-pass tile counts.
    pub fn decode(&mut self, data: &[u8], dst: &mut [u8], dst_stride: usize) -> Result<ProgressiveUpdate> {
        let blocks =
            decode_progressive_stream(data).map_err(|e| ProgressiveError(format!("parse stream: {e}")))?;

        let mut update = ProgressiveUpdate::default();

        for block in blocks {
            match block {
                ProgressiveBlock::Region(region) => {
                    self.process_region(&region, dst, dst_stride, &mut update)?;
                }
                // Sync / Context / FrameBegin / FrameEnd carry no pixels. Context
                // selects subband-diffing (unused by the decode math) and tile
                // size (always 64). Extrapolate is a per-region flag.
                ProgressiveBlock::Sync(_)
                | ProgressiveBlock::FrameBegin(_)
                | ProgressiveBlock::FrameEnd(_)
                | ProgressiveBlock::Context(_) => {}
            }
        }

        Ok(update)
    }

    fn process_region(
        &mut self,
        region: &ProgressiveRegion<'_>,
        dst: &mut [u8],
        dst_stride: usize,
        update: &mut ProgressiveUpdate,
    ) -> Result<()> {
        let extrapolate = region.uses_reduce_extrapolate();
        update.extrapolate = extrapolate;
        if update.regions == 0 {
            update.region0_rects = region.rects.len() as u32;
        }
        update.regions += 1;

        for tile in &region.tiles {
            // A single bad tile must not abandon the rest of the frame (that left
            // large black regions). Decode each tile independently; on error,
            // record it and move on so the rest of the screen still paints.
            let res = self.decode_one_tile(region, tile, extrapolate, update);
            match res {
                Ok(()) => {
                    let (x_idx, y_idx) = (usize::from(tile.x_idx()), usize::from(tile.y_idx()));
                    // Diagnostic: did this tile decode to (near-)black pixels?
                    if self.tile_is_black() {
                        update.black_tiles += 1;
                        if update.first_black.is_none() {
                            update.first_black = Some((tile.x_idx(), tile.y_idx()));
                        }
                    }
                    let tx = x_idx * TILE_SIZE;
                    let ty = y_idx * TILE_SIZE;
                    // Composite into fb AND record exactly what painted as dirty.
                    // We deliberately do NOT mark whole region rectangles dirty:
                    // a region can contain tiles we skipped (decode error), and
                    // pushing the raw rect would blit the still-black framebuffer
                    // over good canvas content (the black-square artifacts).
                    self.blit_tile(dst, dst_stride, tx, ty, region, &mut update.dirty);
                }
                Err(e) => {
                    update.errors += 1;
                    if update.first_error.is_none() {
                        update.first_error =
                            Some(format!("tile ({},{}): {}", tile.x_idx(), tile.y_idx(), e.0));
                    }
                }
            }
        }

        Ok(())
    }

    /// Decode a single tile into `self.rgba_tile` and the persistent tile state.
    fn decode_one_tile(
        &mut self,
        region: &ProgressiveRegion<'_>,
        tile: &ProgressiveTile<'_>,
        extrapolate: bool,
        update: &mut ProgressiveUpdate,
    ) -> Result<()> {
        let x_idx = usize::from(tile.x_idx());
        let y_idx = usize::from(tile.y_idx());
        if x_idx >= self.grid_width || y_idx >= self.grid_height {
            return Err(ProgressiveError(format!(
                "index ({x_idx},{y_idx}) outside grid {}x{}",
                self.grid_width, self.grid_height
            )));
        }
        let zidx = y_idx * self.grid_width + x_idx;

        match tile {
            ProgressiveTile::Simple(t) => {
                if t.flags & RFX_TILE_DIFFERENCE != 0 {
                    update.coeff_diff_tiles += 1;
                }
                let quant = [
                    quant_at(region, t.quant_idx_y)?,
                    quant_at(region, t.quant_idx_cb)?,
                    quant_at(region, t.quant_idx_cr)?,
                ];
                let prog = [PROG_QUANT_FULL; 3];
                let data = [t.y_data, t.cb_data, t.cr_data];
                self.decode_tile_first(zidx, &quant, &prog, &data, t.flags, extrapolate)?;
                update.tiles_simple += 1;
            }
            ProgressiveTile::First(t) => {
                if t.flags & RFX_TILE_DIFFERENCE != 0 {
                    update.coeff_diff_tiles += 1;
                }
                let quant = [
                    quant_at(region, t.quant_idx_y)?,
                    quant_at(region, t.quant_idx_cb)?,
                    quant_at(region, t.quant_idx_cr)?,
                ];
                let prog = prog_quant_set(region, t.quality)?;
                let data = [t.y_data, t.cb_data, t.cr_data];
                self.decode_tile_first(zidx, &quant, &prog, &data, t.flags, extrapolate)?;
                update.tiles_first += 1;
            }
            ProgressiveTile::Upgrade(t) => {
                let quant = [
                    quant_at(region, t.quant_idx_y)?,
                    quant_at(region, t.quant_idx_cb)?,
                    quant_at(region, t.quant_idx_cr)?,
                ];
                let prog = prog_quant_set(region, t.quality)?;
                let srl = [t.y_srl_data, t.cb_srl_data, t.cr_srl_data];
                let raw = [t.y_raw_data, t.cb_raw_data, t.cr_raw_data];
                self.decode_tile_upgrade(zidx, &quant, &prog, &srl, &raw, extrapolate)?;
                update.tiles_upgrade += 1;
            }
        }
        Ok(())
    }

    /// First-pass (or simple) tile decode — FreeRDP `progressive_decompress_tile_first`.
    fn decode_tile_first(
        &mut self,
        zidx: usize,
        quant: &[[i32; 10]; 3],
        prog: &[[i32; 10]; 3],
        data: &[&[u8]; 3],
        flags: u8,
        extrapolate: bool,
    ) -> Result<()> {
        let coeff_diff = flags & RFX_TILE_DIFFERENCE != 0;

        let state = self.tiles[zidx].get_or_insert_with(|| Box::new(TileState::new()));
        state.pass = 1;

        for comp in 0..3 {
            let q = quant[comp];
            let qp = prog[comp];
            let mut bit_pos = [0i32; 10];
            let mut shift = [0i32; 10];
            for b in 0..10 {
                bit_pos[b] = q[b] + qp[b];
                // shift = quant + progQuant - 1 (FreeRDP "-6 + 5 = -1").
                shift[b] = (q[b] + qp[b] - 1).max(0);
            }
            state.bit_pos[comp] = bit_pos;

            decode_component_first(
                data[comp],
                &shift,
                self.buffer[comp].as_mut(),
                state.current[comp].as_mut(),
                state.sign[comp].as_mut(),
                coeff_diff,
                extrapolate,
                self.dwt_temp.as_mut(),
            )?;
        }

        self.to_rgba();
        Ok(())
    }

    /// Upgrade-pass tile decode — FreeRDP `progressive_decompress_tile_upgrade`.
    fn decode_tile_upgrade(
        &mut self,
        zidx: usize,
        quant: &[[i32; 10]; 3],
        prog: &[[i32; 10]; 3],
        srl: &[&[u8]; 3],
        raw: &[&[u8]; 3],
        extrapolate: bool,
    ) -> Result<()> {
        let state = match self.tiles[zidx].as_mut() {
            Some(s) => s,
            // An upgrade with no prior first-pass is malformed; skip it rather
            // than panic so a single bad tile doesn't kill the session.
            None => return Err(ProgressiveError("upgrade before first pass".to_owned())),
        };
        state.pass = state.pass.saturating_add(1);

        for comp in 0..3 {
            let q = quant[comp];
            let qp = prog[comp];
            let mut new_bit_pos = [0i32; 10];
            let mut num_bits = [0i32; 10];
            let mut shift = [0i32; 10];
            for b in 0..10 {
                new_bit_pos[b] = q[b] + qp[b];
                num_bits[b] = (state.bit_pos[comp][b] - new_bit_pos[b]).max(0);
                shift[b] = (q[b] + qp[b] - 1).max(0);
            }
            state.bit_pos[comp] = new_bit_pos;

            upgrade_component(
                &shift,
                &num_bits,
                self.buffer[comp].as_mut(),
                state.current[comp].as_mut(),
                state.sign[comp].as_mut(),
                srl[comp],
                raw[comp],
                extrapolate,
                self.dwt_temp.as_mut(),
            );
        }

        self.to_rgba();
        Ok(())
    }

    /// Cheap check: does the just-decoded `rgba_tile` look (near-)black? Samples
    /// a 9-point grid rather than scanning all 4096 pixels.
    fn tile_is_black(&self) -> bool {
        const THRESH: u8 = 10;
        for sy in [4usize, 32, 60] {
            for sx in [4usize, 32, 60] {
                let o = (sy * TILE_SIZE + sx) * 4;
                if self.rgba_tile[o] > THRESH || self.rgba_tile[o + 1] > THRESH || self.rgba_tile[o + 2] > THRESH {
                    return false;
                }
            }
        }
        true
    }

    /// Convert the three post-DWT component buffers into `self.rgba_tile`.
    fn to_rgba(&mut self) {
        let ycbcr = YCbCrBuffer {
            y: self.buffer[0].as_slice(),
            cb: self.buffer[1].as_slice(),
            cr: self.buffer[2].as_slice(),
        };
        // ycbcr_to_rgba treats the buffers as a flat pixel run; output length
        // (TILE_RGBA) fixes the pixel count at 4096 = one 64x64 tile.
        let _ = ycbcr_to_rgba(ycbcr, self.rgba_tile.as_mut_slice());
    }

    /// Composite the decoded 64x64 RGBA tile at (tx, ty) into `dst`, clipped to
    /// the surface bounds and the region's rectangles, recording each painted
    /// sub-rectangle (surface-local, inclusive) into `dirty`.
    fn blit_tile(
        &self,
        dst: &mut [u8],
        dst_stride: usize,
        tx: usize,
        ty: usize,
        region: &ProgressiveRegion<'_>,
        dirty: &mut Vec<InclusiveRectangle>,
    ) {
        let surf_w = usize::from(self.width);
        let surf_h = usize::from(self.height);

        for r in &region.rects {
            let rx = usize::from(r.x);
            let ry = usize::from(r.y);
            let rright = rx + usize::from(r.width);
            let rbottom = ry + usize::from(r.height);

            // Intersect the tile box [tx, ty, +64] with this region rect and the
            // surface bounds.
            let x0 = tx.max(rx);
            let y0 = ty.max(ry);
            let x1 = (tx + TILE_SIZE).min(rright).min(surf_w);
            let y1 = (ty + TILE_SIZE).min(rbottom).min(surf_h);
            if x1 <= x0 || y1 <= y0 {
                continue;
            }

            for y in y0..y1 {
                let src_row = (y - ty) * TILE_SIZE;
                let dst_row = y * dst_stride;
                for x in x0..x1 {
                    let src = (src_row + (x - tx)) * 4;
                    let dst_off = dst_row + x * 4;
                    if dst_off + 4 <= dst.len() {
                        dst[dst_off..dst_off + 4].copy_from_slice(&self.rgba_tile[src..src + 4]);
                    }
                }
            }

            dirty.push(InclusiveRectangle {
                left: x0 as u16,
                top: y0 as u16,
                right: (x1 - 1) as u16,
                bottom: (y1 - 1) as u16,
            });
        }
    }
}

fn quant_at(region: &ProgressiveRegion<'_>, idx: u8) -> Result<[i32; 10]> {
    region
        .quant_vals
        .get(usize::from(idx))
        .map(bands)
        .ok_or_else(|| ProgressiveError(format!("quant index {idx} out of range")))
}

/// Resolve the progressive quant set for a tile `quality`. `0xFF` selects the
/// all-zero full-quality set; otherwise it indexes the region's prog-quant table.
fn prog_quant_set(region: &ProgressiveRegion<'_>, quality: u8) -> Result<[[i32; 10]; 3]> {
    if quality == 0xFF {
        return Ok([PROG_QUANT_FULL; 3]);
    }
    let pq: &ProgressiveCodecQuant = region
        .quant_prog_vals
        .get(usize::from(quality))
        .ok_or_else(|| ProgressiveError(format!("prog-quant quality {quality} out of range")))?;
    Ok([bands(&pq.y_quant), bands(&pq.cb_quant), bands(&pq.cr_quant)])
}

/// First-pass component decode — FreeRDP `progressive_rfx_decode_component`.
#[expect(clippy::too_many_arguments)]
fn decode_component_first(
    data: &[u8],
    shift: &[i32; 10],
    buffer: &mut [i16; TILE_PIXELS],
    current: &mut [i16; TILE_PIXELS],
    sign: &mut [i16; TILE_PIXELS],
    coeff_diff: bool,
    extrapolate: bool,
    temp: &mut [i16; TILE_PIXELS],
) -> Result<()> {
    rlgr::decode(EntropyAlgorithm::Rlgr1, data, buffer.as_mut_slice())
        .map_err(|e| ProgressiveError(format!("rlgr decode: {e}")))?;

    // Sign snapshot is the raw RLGR output (before reconstruction/dequant).
    sign.copy_from_slice(buffer.as_slice());

    if !extrapolate {
        // LL3 differential first, then dequant all 10 bands.
        subband_reconstruction::decode(&mut buffer[4032..4096]);
        for b in 0..10 {
            lshift_band(buffer.as_mut_slice(), OFF_NORMAL[b], LEN_NORMAL[b], shift[b]);
        }
    } else {
        // Dequant HL1..HH3, LL3 differential, then dequant LL3.
        for b in 0..9 {
            lshift_band(buffer.as_mut_slice(), OFF_EXTRAP[b], LEN_EXTRAP[b], shift[b]);
        }
        subband_reconstruction::decode(&mut buffer[4015..4096]);
        lshift_band(buffer.as_mut_slice(), OFF_EXTRAP[9], LEN_EXTRAP[9], shift[9]);
    }

    dwt_2d_decode(buffer, current, coeff_diff, extrapolate, false, temp);
    Ok(())
}

/// Upgrade component decode — FreeRDP `progressive_rfx_upgrade_component`.
/// Always uses the extrapolate band layout for the `current` coefficient buffer
/// (progressive upgrade only occurs with reduce-extrapolate regions).
#[expect(clippy::too_many_arguments)]
fn upgrade_component(
    shift: &[i32; 10],
    num_bits: &[i32; 10],
    buffer: &mut [i16; TILE_PIXELS],
    current: &mut [i16; TILE_PIXELS],
    sign: &mut [i16; TILE_PIXELS],
    srl_data: &[u8],
    raw_data: &[u8],
    extrapolate: bool,
    temp: &mut [i16; TILE_PIXELS],
) {
    let mut st = UpgradeState::new(srl_data, raw_data);

    // HL1..HH3 use the tri-state sign (SRL for zeros, RAW for non-zeros).
    for b in 0..9 {
        let off = OFF_EXTRAP[b];
        let len = LEN_EXTRAP[b];
        upgrade_block(
            &mut st,
            &mut current[off..off + len],
            &mut sign[off..off + len],
            shift[b],
            num_bits[b],
            true,
        );
    }
    // LL3 reads all refinement bits from RAW.
    let off = OFF_EXTRAP[9];
    let len = LEN_EXTRAP[9];
    upgrade_block(
        &mut st,
        &mut current[off..off + len],
        &mut sign[off..off + len],
        shift[9],
        num_bits[9],
        false,
    );

    dwt_2d_decode(buffer, current, false, extrapolate, true, temp);
}

/// Refine one sub-band — FreeRDP `progressive_rfx_upgrade_block`.
fn upgrade_block(st: &mut UpgradeState<'_>, cur: &mut [i16], sign: &mut [i16], shift: i32, num_bits: i32, use_sign: bool) {
    if num_bits < 1 {
        return;
    }
    let nb = num_bits as u32;
    let len = cur.len();

    if !use_sign {
        // LL3: every coefficient refined from the RAW stream.
        for i in 0..len {
            let input = st.raw_shift(nb);
            cur[i] = (i32::from(cur[i]) + (input << shift)) as i16;
        }
        return;
    }

    for i in 0..len {
        let input: i32 = if sign[i] > 0 {
            st.raw_shift(nb)
        } else if sign[i] < 0 {
            -st.raw_shift(nb)
        } else {
            let v = st.srl_read(nb);
            sign[i] = v;
            i32::from(v)
        };
        cur[i] = (i32::from(cur[i]) + (input << shift)) as i16;
    }
}

/// FreeRDP `progressive_rfx_dwt_2d_decode`: move coefficients between the
/// scratch `buffer` and the persistent `current`, then run the inverse DWT.
///
/// * `reverse` (upgrade): load `current` into `buffer`, then iDWT `buffer`.
/// * first pass, no diff: copy `buffer` into `current`, then iDWT `buffer`.
/// * first pass, coeff diff: `buffer = current = buffer + current`, then iDWT.
fn dwt_2d_decode(
    buffer: &mut [i16; TILE_PIXELS],
    current: &mut [i16; TILE_PIXELS],
    coeff_diff: bool,
    extrapolate: bool,
    reverse: bool,
    temp: &mut [i16; TILE_PIXELS],
) {
    if reverse {
        buffer.copy_from_slice(current.as_slice());
    } else if !coeff_diff {
        current.copy_from_slice(buffer.as_slice());
    } else {
        for i in 0..TILE_PIXELS {
            let v = buffer[i].wrapping_add(current[i]);
            buffer[i] = v;
            current[i] = v;
        }
    }

    if extrapolate {
        dwt_extrapolate::decode(buffer.as_mut_slice(), temp.as_mut_slice());
    } else {
        dwt::decode(buffer.as_mut_slice(), temp.as_mut_slice());
    }
}
