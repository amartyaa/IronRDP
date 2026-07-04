//! ClearCodec bitmap decoder (MS-RDPEGFX 2.2.4.1 `CLEARCODEC_BITMAP_STREAM`,
//! carried by `RDPGFX_WIRE_TO_SURFACE_PDU_1` with `codecId = 0x8` ClearCodec).
//!
//! A faithful Rust port of the decode path in FreeRDP's
//! `libfreerdp/codec/clear.c` (Apache-2.0). ClearCodec is mandatory for any
//! client advertising RDPGFX CAPVERSION_8+ — Windows uses it heavily for
//! text and UI regions, mixing it per-region with RFX-Progressive and
//! uncompressed blits. A composition is up to three layers painted in order:
//! residual (coarse RLE background), bands (columns of "vBars" with two
//! LRU-ish caches), and subcodecs (per-rect raw BGR24 or RLEX palette runs).
//! On top sits a 4000-entry glyph cache keyed by explicit server indices,
//! caching the *composed* output of small rects.
//!
//! The decoder is stateful across the whole session (caches survive
//! `ResetGraphics`) and decodes each PDU into a caller-provided RGBA buffer
//! of exactly the destination-rect size.
//!
//! Not implemented: the NSCodec subcodec (`subcodecId = 1`). Windows in a
//! CAPVERSION_8 EGFX session has not been observed to use it (it prefers raw
//! and RLEX inside ClearCodec); a rect using it fails with a clear error and
//! is skipped by the caller rather than aborting the session.

use core::fmt;

const FLAG_GLYPH_INDEX: u8 = 0x01;
const FLAG_GLYPH_HIT: u8 = 0x02;
const FLAG_CACHE_RESET: u8 = 0x04;

const GLYPH_CACHE_SIZE: usize = 4000;
const VBAR_CACHE_SIZE: usize = 32768;
const SHORT_VBAR_CACHE_SIZE: usize = 16384;

const BYTES_PER_PIXEL: usize = 4; // RGBA8888 throughout

#[derive(Debug)]
pub struct ClearDecodeError(pub &'static str);

impl fmt::Display for ClearDecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ClearCodec: {}", self.0)
    }
}

impl core::error::Error for ClearDecodeError {}

type ClearResult<T> = Result<T, ClearDecodeError>;

/// floor(log2(n)), with `log2_floor(0) = 0` — matching the reference
/// decoder's `CLEAR_LOG2_FLOOR` table (index 0 holds 0).
fn log2_floor(n: u8) -> u32 {
    if n == 0 { 0 } else { 7 - n.leading_zeros() }
}

/// Little sequential reader over the stream; every read is bounds-checked.
struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }

    fn u8(&mut self) -> ClearResult<u8> {
        let v = *self.data.get(self.pos).ok_or(ClearDecodeError("stream underrun"))?;
        self.pos += 1;
        Ok(v)
    }

    fn u16(&mut self) -> ClearResult<u16> {
        let b = self.bytes(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn u32(&mut self) -> ClearResult<u32> {
        let b = self.bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn bytes(&mut self, n: usize) -> ClearResult<&'a [u8]> {
        let end = self.pos.checked_add(n).ok_or(ClearDecodeError("stream underrun"))?;
        let s = self.data.get(self.pos..end).ok_or(ClearDecodeError("stream underrun"))?;
        self.pos = end;
        Ok(s)
    }
}

/// Reads the shared extended run-length encoding: u8, escaped to u16 at 0xFF,
/// escaped to u32 at 0xFFFF. Returns (value, bytes consumed).
fn read_run_length(r: &mut Reader<'_>) -> ClearResult<(u32, usize)> {
    let first = u32::from(r.u8()?);
    if first < 0xFF {
        return Ok((first, 1));
    }
    let second = u32::from(r.u16()?);
    if second < 0xFFFF {
        return Ok((second, 3));
    }
    Ok((r.u32()?, 7))
}

#[derive(Default)]
struct GlyphEntry {
    /// Composed RGBA pixels of a previously decoded rect.
    pixels: Vec<u8>,
}

#[derive(Default, Clone)]
struct VBarEntry {
    /// RGBA pixel column.
    pixels: Vec<u8>,
}

pub struct ClearDecoder {
    seq_number: u8,
    seq_started: bool,
    glyph_cache: Vec<GlyphEntry>,
    vbar: Vec<VBarEntry>,
    vbar_cursor: usize,
    short_vbar: Vec<VBarEntry>,
    short_vbar_cursor: usize,
}

impl Default for ClearDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl ClearDecoder {
    pub fn new() -> Self {
        Self {
            seq_number: 0,
            seq_started: false,
            glyph_cache: (0..GLYPH_CACHE_SIZE).map(|_| GlyphEntry::default()).collect(),
            vbar: vec![VBarEntry::default(); VBAR_CACHE_SIZE],
            vbar_cursor: 0,
            short_vbar: vec![VBarEntry::default(); SHORT_VBAR_CACHE_SIZE],
            short_vbar_cursor: 0,
        }
    }

    /// Decodes one `CLEARCODEC_BITMAP_STREAM` into `out`, an RGBA8888 buffer of
    /// exactly `width * height * 4` bytes (the destination rectangle size).
    /// Prior content of `out` is preserved where the stream paints nothing
    /// (layers are sparse by design).
    pub fn decode(&mut self, src: &[u8], width: u16, height: u16, out: &mut [u8]) -> ClearResult<()> {
        let w = usize::from(width);
        let h = usize::from(height);
        if w == 0 || h == 0 {
            return Err(ClearDecodeError("empty destination rectangle"));
        }
        if w * h > 1024 * 1024 * 16 {
            return Err(ClearDecodeError("destination rectangle too large"));
        }
        if out.len() != w * h * BYTES_PER_PIXEL {
            return Err(ClearDecodeError("output buffer size mismatch"));
        }

        let mut r = Reader::new(src);
        let glyph_flags = r.u8()?;
        let seq_number = r.u8()?;

        // FreeRDP hard-fails on a sequence mismatch, which permanently
        // desyncs (every later PDU also mismatches, blacking out regions
        // forever). Adopt the server's sequence with a resync instead — the
        // caches are index-addressed, so decoding remains well-defined.
        if !self.seq_started {
            self.seq_started = true;
        } else if seq_number != self.seq_number {
            // ponytail: silent resync; add a tracing dep here if this needs visibility
        }
        self.seq_number = seq_number.wrapping_add(1);

        if glyph_flags & FLAG_CACHE_RESET != 0 {
            self.vbar_cursor = 0;
            self.short_vbar_cursor = 0;
        }

        // Glyph handling. A HIT copies the cached composed rect and finishes.
        // An INDEX (without HIT) decodes normally, then stores the composed
        // rect back into the cache slot.
        let mut store_glyph: Option<usize> = None;
        if glyph_flags & FLAG_GLYPH_HIT != 0 && glyph_flags & FLAG_GLYPH_INDEX == 0 {
            return Err(ClearDecodeError("GLYPH_HIT without GLYPH_INDEX"));
        }
        if glyph_flags & FLAG_GLYPH_INDEX != 0 {
            if w * h > 1024 * 1024 {
                return Err(ClearDecodeError("glyph rect larger than 1024x1024"));
            }
            let glyph_index = usize::from(r.u16()?);
            if glyph_index >= GLYPH_CACHE_SIZE {
                return Err(ClearDecodeError("glyph index out of range"));
            }
            if glyph_flags & FLAG_GLYPH_HIT != 0 {
                let entry = &self.glyph_cache[glyph_index];
                if entry.pixels.len() < out.len() {
                    return Err(ClearDecodeError("glyph cache entry smaller than rect"));
                }
                out.copy_from_slice(&entry.pixels[..out.len()]);
                return Ok(());
            }
            store_glyph = Some(glyph_index);
        }

        // Composition payload header. It may legitimately be absent only for
        // glyph hits (handled above), so absence here is an error — except
        // FreeRDP tolerates it when both glyph flags were set.
        if r.remaining() < 12 {
            return Err(ClearDecodeError("missing composition payload header"));
        }
        let residual_byte_count = r.u32()? as usize;
        let bands_byte_count = r.u32()? as usize;
        let subcodec_byte_count = r.u32()? as usize;

        if residual_byte_count > 0 {
            let layer = r.bytes(residual_byte_count)?;
            decode_residual(layer, w, h, out)?;
        }
        if bands_byte_count > 0 {
            let layer = r.bytes(bands_byte_count)?;
            self.decode_bands(layer, w, h, out)?;
        }
        if subcodec_byte_count > 0 {
            let layer = r.bytes(subcodec_byte_count)?;
            decode_subcodecs(layer, w, h, out)?;
        }

        if let Some(index) = store_glyph {
            let entry = &mut self.glyph_cache[index];
            entry.pixels.clear();
            entry.pixels.extend_from_slice(out);
        }

        Ok(())
    }

    /// The bands layer: vertical strips ("vBars") of full band height,
    /// composed from a background color plus a cached or inline "short vBar"
    /// segment, then blitted column by column.
    fn decode_bands(&mut self, layer: &[u8], w: usize, h: usize, out: &mut [u8]) -> ClearResult<()> {
        let mut r = Reader::new(layer);

        while r.remaining() > 0 {
            let x_start = usize::from(r.u16()?);
            let x_end = usize::from(r.u16()?);
            let y_start = usize::from(r.u16()?);
            let y_end = usize::from(r.u16()?);
            let blue = r.u8()?;
            let green = r.u8()?;
            let red = r.u8()?;
            let bg = [red, green, blue, 0xFF];

            if x_end < x_start {
                return Err(ClearDecodeError("band xEnd < xStart"));
            }
            if y_end < y_start {
                return Err(ClearDecodeError("band yEnd < yStart"));
            }
            let vbar_count = x_end - x_start + 1;
            let vbar_height = y_end - y_start + 1;
            if vbar_height > 52 {
                return Err(ClearDecodeError("band vBar height > 52"));
            }

            for i in 0..vbar_count {
                let header = r.u16()?;

                // Which pixels make up this bar, and does the composed bar
                // need to be (re)built into the long-vBar cache?
                let mut short_pixels: &[u8] = &[];
                let mut vbar_y_on = 0usize;
                let mut rebuild = false;
                let mut cached_index: Option<usize> = None;

                if header & 0xC000 == 0x4000 {
                    // SHORT_VBAR_CACHE_HIT
                    let idx = usize::from(header & 0x3FFF);
                    vbar_y_on = usize::from(r.u8()?);
                    let entry = &self.short_vbar[idx];
                    short_pixels = &entry.pixels;
                    rebuild = true;
                } else if header & 0xC000 == 0x0000 {
                    // SHORT_VBAR_CACHE_MISS: inline pixels, cached for later
                    vbar_y_on = usize::from(header & 0xFF);
                    let vbar_y_off = usize::from((header >> 8) & 0x3F);
                    if vbar_y_off < vbar_y_on {
                        return Err(ClearDecodeError("short vBar yOff < yOn"));
                    }
                    let count = vbar_y_off - vbar_y_on;
                    if count > 52 {
                        return Err(ClearDecodeError("short vBar pixel count > 52"));
                    }
                    let raw = r.bytes(count * 3)?;
                    let entry = &mut self.short_vbar[self.short_vbar_cursor];
                    entry.pixels.clear();
                    for px in raw.chunks_exact(3) {
                        entry.pixels.extend_from_slice(&[px[2], px[1], px[0], 0xFF]);
                    }
                    self.short_vbar_cursor = (self.short_vbar_cursor + 1) % SHORT_VBAR_CACHE_SIZE;
                    // Re-borrow immutably for composition below.
                    let idx = (self.short_vbar_cursor + SHORT_VBAR_CACHE_SIZE - 1) % SHORT_VBAR_CACHE_SIZE;
                    short_pixels = &self.short_vbar[idx].pixels;
                    rebuild = true;
                } else if header & 0x8000 == 0x8000 {
                    // VBAR_CACHE_HIT: fully composed bar cached
                    let idx = usize::from(header & 0x7FFF);
                    cached_index = Some(idx);
                } else {
                    return Err(ClearDecodeError("invalid vBar header"));
                }

                if rebuild {
                    // Compose: bg above, short pixels in the middle, bg below.
                    let mut bar = Vec::with_capacity(vbar_height * BYTES_PER_PIXEL);
                    let short_count = short_pixels.len() / BYTES_PER_PIXEL;
                    for y in 0..vbar_height {
                        if y >= vbar_y_on && y < vbar_y_on + short_count {
                            let o = (y - vbar_y_on) * BYTES_PER_PIXEL;
                            bar.extend_from_slice(&short_pixels[o..o + BYTES_PER_PIXEL]);
                        } else {
                            bar.extend_from_slice(&bg);
                        }
                    }
                    self.vbar[self.vbar_cursor].pixels = bar;
                    cached_index = Some(self.vbar_cursor);
                    self.vbar_cursor = (self.vbar_cursor + 1) % VBAR_CACHE_SIZE;
                }

                let bar = &self.vbar[cached_index.expect("set on all accepted paths")].pixels;
                // A long-cache hit after CACHE_RESET may reference an empty
                // slot; FreeRDP paints dummy (zero) data. We skip the blit.
                let bar_h = (bar.len() / BYTES_PER_PIXEL).min(vbar_height).min(h);

                let x = x_start + i;
                if x >= w {
                    return Err(ClearDecodeError("band column outside rect"));
                }
                for y in 0..bar_h {
                    let dy = y_start + y;
                    if dy >= h {
                        return Err(ClearDecodeError("band row outside rect"));
                    }
                    let dst = (dy * w + x) * BYTES_PER_PIXEL;
                    let srcp = y * BYTES_PER_PIXEL;
                    out[dst..dst + BYTES_PER_PIXEL].copy_from_slice(&bar[srcp..srcp + BYTES_PER_PIXEL]);
                }
            }
        }

        Ok(())
    }
}

/// The residual layer: a plain RLE of (B, G, R, runLength) covering the whole
/// rect in row-major order. Must cover exactly `w * h` pixels.
fn decode_residual(layer: &[u8], w: usize, h: usize, out: &mut [u8]) -> ClearResult<()> {
    let mut r = Reader::new(layer);
    let pixel_count = w * h;
    let mut pixel_index = 0usize;

    while r.remaining() > 0 {
        let b = r.u8()?;
        let g = r.u8()?;
        let red = r.u8()?;
        let (run, _) = read_run_length(&mut r)?;
        let run = run as usize;

        if pixel_index + run > pixel_count {
            return Err(ClearDecodeError("residual run overflows rect"));
        }
        let rgba = [red, g, b, 0xFF];
        let start = pixel_index * BYTES_PER_PIXEL;
        for px in out[start..start + run * BYTES_PER_PIXEL].chunks_exact_mut(BYTES_PER_PIXEL) {
            px.copy_from_slice(&rgba);
        }
        pixel_index += run;
    }

    if pixel_index != pixel_count {
        return Err(ClearDecodeError("residual does not cover rect"));
    }
    Ok(())
}

/// The subcodec layer: a list of sub-rects, each raw BGR24 (`subcodecId = 0`)
/// or RLEX palette runs (`subcodecId = 2`). NSCodec (`1`) is not supported.
fn decode_subcodecs(layer: &[u8], w: usize, h: usize, out: &mut [u8]) -> ClearResult<()> {
    let mut r = Reader::new(layer);

    while r.remaining() > 0 {
        let x_start = usize::from(r.u16()?);
        let y_start = usize::from(r.u16()?);
        let sw = usize::from(r.u16()?);
        let sh = usize::from(r.u16()?);
        let byte_count = r.u32()? as usize;
        let subcodec_id = r.u8()?;
        let data = r.bytes(byte_count)?;

        if x_start + sw > w || y_start + sh > h {
            return Err(ClearDecodeError("subcodec rect outside destination"));
        }

        match subcodec_id {
            0 => {
                // Raw BGR24, rows tightly packed.
                if byte_count != sw * sh * 3 {
                    return Err(ClearDecodeError("raw subcodec size mismatch"));
                }
                for y in 0..sh {
                    let src_row = &data[y * sw * 3..(y + 1) * sw * 3];
                    let dst = ((y_start + y) * w + x_start) * BYTES_PER_PIXEL;
                    let dst_row = &mut out[dst..dst + sw * BYTES_PER_PIXEL];
                    for (s, d) in src_row.chunks_exact(3).zip(dst_row.chunks_exact_mut(BYTES_PER_PIXEL)) {
                        d.copy_from_slice(&[s[2], s[1], s[0], 0xFF]);
                    }
                }
            }
            2 => decode_rlex(data, sw, sh, x_start, y_start, w, h, out)?,
            1 => return Err(ClearDecodeError("NSCodec subcodec not supported")),
            _ => return Err(ClearDecodeError("unknown subcodec id")),
        }
    }

    Ok(())
}

/// RLEX: a small palette (BGR triplets), then runs of `(background run,
/// ascending palette suite)` packed as bitfields sized by the palette count.
#[expect(clippy::too_many_arguments, reason = "mirrors the reference decoder's geometry plumbing")]
fn decode_rlex(
    data: &[u8],
    sw: usize,
    sh: usize,
    x_start: usize,
    y_start: usize,
    w: usize,
    h: usize,
    out: &mut [u8],
) -> ClearResult<()> {
    let mut r = Reader::new(data);

    let palette_count = r.u8()?;
    if palette_count == 0 || palette_count > 127 {
        return Err(ClearDecodeError("RLEX palette count out of range"));
    }
    let mut palette = [[0u8; 4]; 128];
    for entry in palette.iter_mut().take(usize::from(palette_count)) {
        let b = r.u8()?;
        let g = r.u8()?;
        let red = r.u8()?;
        *entry = [red, g, b, 0xFF];
    }

    let pixel_count = sw * sh;
    let mut pixel_index = 0usize;
    let num_bits = log2_floor(palette_count - 1).wrapping_add(1) as u16;

    let mut x = 0usize;
    let mut y = 0usize;
    let mut put = |x: &mut usize, y: &mut usize, color: [u8; 4]| {
        let dx = x_start + *x;
        let dy = y_start + *y;
        if dx < w && dy < h {
            let dst = (dy * w + dx) * BYTES_PER_PIXEL;
            out[dst..dst + BYTES_PER_PIXEL].copy_from_slice(&color);
        }
        *x += 1;
        if *x >= sw {
            *y += 1;
            *x = 0;
        }
    };

    while r.remaining() > 0 {
        let tmp = u16::from(r.u8()?);
        let (run, _) = read_run_length(&mut r)?;
        let run = run as usize;

        let suite_depth = ((tmp >> num_bits) & ((1u16 << (8 - num_bits)) - 1)) as u8;
        let stop_index = (tmp & ((1u16 << num_bits) - 1)) as u8;
        let start_index = stop_index.wrapping_sub(suite_depth);

        if start_index >= palette_count || stop_index >= palette_count {
            return Err(ClearDecodeError("RLEX palette index out of range"));
        }

        if pixel_index + run > pixel_count {
            return Err(ClearDecodeError("RLEX background run overflows rect"));
        }
        let bg = palette[usize::from(start_index)];
        for _ in 0..run {
            put(&mut x, &mut y, bg);
        }
        pixel_index += run;

        let suite_len = usize::from(suite_depth) + 1;
        if pixel_index + suite_len > pixel_count {
            return Err(ClearDecodeError("RLEX suite overflows rect"));
        }
        for s in 0..suite_len {
            put(&mut x, &mut y, palette[usize::from(start_index) + s]);
        }
        pixel_index += suite_len;
    }

    if pixel_index != pixel_count {
        return Err(ClearDecodeError("RLEX does not cover rect"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test vectors from [MS-RDPEGFX] 4.1.1.1 (same ones FreeRDP's
    // TestFreeRDPCodecClear uses). Example 1 is a glyph hit that needs a
    // pre-filled cache, so like FreeRDP we skip it.

    /// Example 2: bands + subcodec layers, 78x17.
    const EXAMPLE_2: &[u8] = &[
        0x00, 0x0d, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x82, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x4e, 0x00, 0x11, 0x00, 0x75, 0x00, 0x00, 0x00, 0x02, 0x0e, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0xdb, 0xff,
        0xff, 0x00, 0x3a, 0x90, 0xff, 0xb6, 0x66, 0x66, 0xb6, 0xff, 0xb6, 0x66, 0x00, 0x90, 0xdb, 0xff, 0x00, 0x00,
        0x3a, 0xdb, 0x90, 0x3a, 0x3a, 0x90, 0xdb, 0x66, 0x00, 0x00, 0xff, 0xff, 0xb6, 0x64, 0x64, 0x64, 0x11, 0x04,
        0x11, 0x4c, 0x11, 0x4c, 0x11, 0x4c, 0x11, 0x4c, 0x11, 0x4c, 0x00, 0x47, 0x13, 0x00, 0x01, 0x01, 0x04, 0x00,
        0x01, 0x00, 0x00, 0x47, 0x16, 0x00, 0x11, 0x02, 0x00, 0x47, 0x29, 0x00, 0x11, 0x01, 0x00, 0x49, 0x0a, 0x00,
        0x01, 0x00, 0x04, 0x00, 0x01, 0x00, 0x00, 0x4a, 0x0a, 0x00, 0x09, 0x00, 0x01, 0x00, 0x00, 0x47, 0x05, 0x00,
        0x01, 0x01, 0x1c, 0x00, 0x01, 0x00, 0x11, 0x4c, 0x11, 0x4c, 0x11, 0x4c, 0x00, 0x47, 0x0d, 0x4d, 0x00, 0x4d,
    ];

    /// Example 3: bands layer with short vBar cache misses/hits, 64x24.
    const EXAMPLE_3: &[u8] = &[
        0x00, 0xdf, 0x0e, 0x00, 0x00, 0x00, 0x8b, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xfe, 0xfe, 0xfe, 0xff,
        0x80, 0x05, 0xff, 0xff, 0xff, 0x40, 0xfe, 0xfe, 0xfe, 0x40, 0x00, 0x00, 0x3f, 0x00, 0x03, 0x00, 0x0b, 0x00,
        0xfe, 0xfe, 0xfe, 0xc5, 0xd0, 0xc6, 0xd0, 0xc7, 0xd0, 0x68, 0xd4, 0x69, 0xd4, 0x6a, 0xd4, 0x6b, 0xd4, 0x6c,
        0xd4, 0x6d, 0xd4, 0x1a, 0xd4, 0x1a, 0xd4, 0xa6, 0xd0, 0x6e, 0xd4, 0x6f, 0xd4, 0x70, 0xd4, 0x71, 0xd4, 0x72,
        0xd4, 0x73, 0xd4, 0x74, 0xd4, 0x21, 0xd4, 0x22, 0xd4, 0x23, 0xd4, 0x24, 0xd4, 0x25, 0xd4, 0xd9, 0xd0, 0xda,
        0xd0, 0xdb, 0xd0, 0xc5, 0xd0, 0xc5, 0xd0, 0xdc, 0xd0, 0xc2, 0xd0, 0x21, 0xd4, 0x22, 0xd4, 0x23, 0xd4, 0x24,
        0xd4, 0x25, 0xd4, 0xc9, 0xd0, 0xca, 0xd0, 0x5a, 0xd4, 0x2b, 0xd1, 0x28, 0xd1, 0x2c, 0xd1, 0x75, 0xd4, 0x27,
        0xd4, 0x28, 0xd4, 0x29, 0xd4, 0x2a, 0xd4, 0x1a, 0xd4, 0x1a, 0xd4, 0x1a, 0xd4, 0xb7, 0xd0, 0xb8, 0xd0, 0xb9,
        0xd0, 0xba, 0xd0, 0xbb, 0xd0, 0xbc, 0xd0, 0xbd, 0xd0, 0xbe, 0xd0, 0xbf, 0xd0, 0xc0, 0xd0, 0xc1, 0xd0, 0xc2,
        0xd0, 0xc3, 0xd0, 0xc4, 0xd0,
    ];

    /// Example 4: glyph index store with residual + subcodec layers, 7x15.
    const EXAMPLE_4: &[u8] = &[
        0x01, 0x0b, 0x78, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x06, 0x00, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xb6, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xb6, 0x66, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xb6, 0x66, 0xdb, 0x90, 0x3a, 0xff, 0xff, 0xb6, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0x46, 0x91, 0x47, 0x91, 0x48, 0x91, 0x49, 0x91, 0x4a, 0x91, 0x1b, 0x91,
    ];

    fn decode(width: u16, height: u16, src: &[u8]) -> Vec<u8> {
        let mut decoder = ClearDecoder::new();
        let mut out = vec![0u8; usize::from(width) * usize::from(height) * 4];
        decoder
            .decode(src, width, height, &mut out)
            .unwrap_or_else(|e| panic!("decode failed: {e}"));
        out
    }

    #[test]
    fn spec_example_2_decodes() {
        decode(78, 17, EXAMPLE_2);
    }

    #[test]
    fn spec_example_3_decodes() {
        let out = decode(64, 24, EXAMPLE_3);
        // The band background color is 0xFEFEFE — expect it somewhere.
        assert!(out.chunks_exact(4).any(|px| px == [0xFE, 0xFE, 0xFE, 0xFF]));
    }

    #[test]
    fn spec_example_4_stores_glyph_and_hits() {
        let mut decoder = ClearDecoder::new();
        let mut out = vec![0u8; 7 * 15 * 4];
        decoder
            .decode(EXAMPLE_4, 7, 15, &mut out)
            .unwrap_or_else(|e| panic!("decode failed: {e}"));

        // Example 4 stores glyph index 0x0078; a follow-up GLYPH_HIT for the
        // same index must reproduce the identical composed rect.
        let hit = [0x03u8, 0x0c, 0x78, 0x00]; // flags=INDEX|HIT, seq, index 0x0078
        let mut out2 = vec![0u8; 7 * 15 * 4];
        decoder
            .decode(&hit, 7, 15, &mut out2)
            .unwrap_or_else(|e| panic!("glyph hit decode failed: {e}"));
        assert_eq!(out, out2);
    }

    #[test]
    fn residual_only_fill() {
        // 2x2 solid red via residual layer: header + residual RLE (B,G,R,run=4).
        let src = [
            0x00, 0x00, // glyphFlags, seqNumber
            0x04, 0x00, 0x00, 0x00, // residualByteCount
            0x00, 0x00, 0x00, 0x00, // bandsByteCount
            0x00, 0x00, 0x00, 0x00, // subcodecByteCount
            0x00, 0x00, 0xFF, 0x04, // B=0 G=0 R=255, run 4
        ];
        let out = decode(2, 2, &src);
        assert_eq!(out, [0xFF, 0x00, 0x00, 0xFF].repeat(4));
    }
}
