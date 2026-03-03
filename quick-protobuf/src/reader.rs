//! A module to manage protobuf deserialization
//!
//! There are actually two main *readers*
//! - a `BytesReader` which parses data from a `&[u8]`
//! - a `Reader` which is a wrapper on `BytesReader` which has its own buffer. It provides
//!   convenient functions to the user such as `from_file`
//!
//! It is advised, for convenience to directly work with a `Reader`.

use core::iter::FusedIterator;
#[cfg(feature = "std")]
use std::borrow::ToOwned;
#[cfg(feature = "std")]
use std::fs::File;
#[cfg(feature = "std")]
use std::io::Read;
#[cfg(feature = "std")]
use std::path::Path;

use core::convert::TryFrom;
use core::convert::TryInto;

#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(not(feature = "std"))]
use alloc::borrow::Cow;
#[cfg(not(feature = "std"))]
use alloc::borrow::ToOwned;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::errors::{Error, Result};
use crate::message::MessageRead;


const WIRE_TYPE_VARINT: u8 = 0;
const WIRE_TYPE_FIXED64: u8 = 1;
const WIRE_TYPE_LENGTH_DELIMITED: u8 = 2;
const WIRE_TYPE_START_GROUP: u8 = 3;
const WIRE_TYPE_END_GROUP: u8 = 4;
const WIRE_TYPE_FIXED32: u8 = 5;

/// Branchless varint32 decode from a u64 containing raw bytes (little-endian).
/// Returns `Some((value, byte_length))` if the varint terminates within 5 bytes,
/// or `None` for 10-byte negative-i32 encodings where all 5 low bytes have
/// continuation bits set.
#[cfg(any(target_arch = "aarch64", test))]
#[cfg_attr(feature = "std", inline(always))]
fn decode_varint32_branchless(raw: u64) -> Option<(u32, usize)> {
    const MSB_MASK: u64 = 0x8080_8080_8080_8080;
    const FIRST_5_MSB: u64 = 0x0000_0080_8080_8080;

    // Identify terminator bytes (MSB = 0 means end of varint)
    let term = !raw & MSB_MASK;

    // If no terminator in first 5 bytes, this is a >5 byte varint (negative i32)
    if term & FIRST_5_MSB == 0 {
        return None;
    }

    // Isolate first terminator bit position
    let first_term = term & term.wrapping_neg();
    // Byte mask: all bits from 0 up to and including the terminator byte
    let byte_mask = first_term | first_term.wrapping_sub(1);
    let masked = raw & byte_mask;

    // Extract 7-bit payload groups and combine into u32
    let value = ((masked & 0x0000_007F)
        | ((masked >> 1) & 0x0000_3F80)
        | ((masked >> 2) & 0x001F_C000)
        | ((masked >> 3) & 0x0FE0_0000)
        | ((masked >> 4) & 0xF000_0000)) as u32;

    let len = (first_term.trailing_zeros() / 8 + 1) as usize;
    Some((value, len))
}

/// Branchless varint64 decode from a u64 containing raw bytes (little-endian).
/// Returns `Some((value, byte_length))` if the varint terminates within 8 bytes,
/// or `None` if all 8 bytes have continuation bits set (9-10 byte varint).
#[cfg(any(target_arch = "aarch64", test))]
#[cfg_attr(feature = "std", inline(always))]
fn decode_varint64_branchless(raw: u64) -> Option<(u64, usize)> {
    const MSB_MASK: u64 = 0x8080_8080_8080_8080;

    // Identify terminator bytes (MSB = 0 means end of varint)
    let term = !raw & MSB_MASK;

    // If no terminator in all 8 bytes, this is a 9-10 byte varint
    if term == 0 {
        return None;
    }

    // Isolate first terminator bit position
    let first_term = term & term.wrapping_neg();
    // Byte mask: all bits from 0 up to and including the terminator byte
    let byte_mask = first_term | first_term.wrapping_sub(1);
    let masked = raw & byte_mask;

    // Extract 7-bit payload groups and combine into u64
    let value = (masked & 0x0000_0000_0000_007F)
        | ((masked >> 1) & 0x0000_0000_0000_3F80)
        | ((masked >> 2) & 0x0000_0000_001F_C000)
        | ((masked >> 3) & 0x0000_0000_0FE0_0000)
        | ((masked >> 4) & 0x0000_0007_F000_0000)
        | ((masked >> 5) & 0x0000_03F8_0000_0000)
        | ((masked >> 6) & 0x0001_FC00_0000_0000)
        | ((masked >> 7) & 0x00FE_0000_0000_0000);

    let len = (first_term.trailing_zeros() / 8 + 1) as usize;
    Some((value, len))
}

/// NEON-accelerated batch varint32 decode for packed fields.
/// Decodes varints from `buf[start..end]` into `out`, using NEON to detect
/// varint boundaries in 16-byte chunks and branchless scalar decode for values.
/// Returns the position after the last successfully decoded varint.
/// Varints longer than 5 bytes (negative i32 encoding) are not handled;
/// the caller must use scalar decode for the remainder.
///
/// # Safety
/// Caller must ensure `start <= end <= buf.len()`.
#[cfg(target_arch = "aarch64")]
#[cfg_attr(feature = "std", inline)]
unsafe fn batch_decode_varint32_neon(
    buf: &[u8],
    start: usize,
    end: usize,
    out: &mut Vec<i32>,
) -> usize {
    use core::arch::aarch64::*;

    let mut pos = start;

    // Process 16-byte chunks. Need pos + 16 <= end (message boundary)
    // AND pos + 16 <= buf.len() (buffer boundary) for the NEON load.
    // Branchless decode loads 8 bytes, so varints starting at local offset <= 8
    // are safe (pos + 8 + 8 <= pos + 16 <= buf.len()).
    while pos + 16 <= end && pos + 16 <= buf.len() {
        // Load 16 bytes and extract continuation-bit mask via NEON
        let data = vld1q_u8(buf.as_ptr().add(pos));
        let msbs = vshrq_n_u8(data, 7);

        // Convert 16-byte vector to 16-bit scalar mask using
        // power-of-2 weighted reduction (NEON movemask equivalent)
        static POWERS: [u8; 16] = [
            1, 2, 4, 8, 16, 32, 64, 128,
            1, 2, 4, 8, 16, 32, 64, 128,
        ];
        let powers = vld1q_u8(POWERS.as_ptr());
        let weighted = vmulq_u8(msbs, powers);
        let sum16 = vpaddlq_u8(weighted);
        let sum32 = vpaddlq_u16(sum16);
        let sum64 = vpaddlq_u32(sum32);
        let lo = vgetq_lane_u64(sum64, 0) as u8;
        let hi = vgetq_lane_u64(sum64, 1) as u8;
        let cont_mask = ((hi as u16) << 8) | (lo as u16);

        // Terminator mask: bit i set where byte i is the last byte of a varint
        let term_mask = !cont_mask;

        if term_mask == 0 {
            // No complete varint in 16 bytes - fall back to scalar
            break;
        }

        let mut local_pos: usize = 0;
        let mut remaining_term = term_mask;

        while remaining_term != 0 {
            // Safe decode zone: local_pos <= 8 guarantees 8 bytes available
            // for branchless u64 load (pos + 8 + 8 <= pos + 16 <= buf.len())
            if local_pos > 8 {
                break;
            }

            let term_bit = remaining_term.trailing_zeros() as usize;
            let varint_len = term_bit - local_pos + 1;

            // >5 byte varint (negative i32 encoding) - defer to scalar
            if varint_len > 5 {
                break;
            }

            let raw = u64::from_le_bytes(
                buf[pos + local_pos..pos + local_pos + 8]
                    .try_into()
                    .unwrap(),
            );

            match decode_varint32_branchless(raw) {
                Some((value, len)) => {
                    out.push(value as i32);
                    local_pos += len;
                }
                None => break,
            }

            // Clear all bits up to and including this terminator
            if term_bit < 15 {
                remaining_term &= !((2u16 << term_bit) - 1);
            } else {
                remaining_term = 0;
            }
        }

        if local_pos == 0 {
            break;
        }

        pos += local_pos;
    }

    pos
}

/// A struct to read protocol binary files
///
/// # Examples
///
/// ```rust
/// # mod foo_bar {
/// #     use quick_protobuf::{MessageRead, BytesReader, Result};
/// #     pub struct Foo {}
/// #     pub struct Bar {}
/// #     pub struct FooBar { pub foos: Vec<Foo>, pub bars: Vec<Bar>, }
/// #     impl<'a> MessageRead<'a> for FooBar {
/// #         fn from_reader(_: &mut BytesReader, _: &[u8]) -> Result<Self> {
/// #              Ok(FooBar { foos: vec![], bars: vec![] })
/// #         }
/// #     }
/// # }
///
/// // FooBar is a message generated from a proto file
/// // in parcicular it contains a `from_reader` function
/// use foo_bar::FooBar;
/// use quick_protobuf::{BytesReader, MessageRead};
///
/// fn main() {
///     // bytes is a buffer on the data we want to deserialize
///     // typically bytes is read from a `Read`:
///     // r.read_to_end(&mut bytes).expect("cannot read bytes");
///     let mut bytes: Vec<u8>;
///     # bytes = vec![];
///
///     // we can build a bytes reader directly out of the bytes
///     let mut reader = BytesReader::from_bytes(&bytes);
///
///     // now using the generated module decoding is as easy as:
///     let foobar = FooBar::from_reader(&mut reader, &bytes).expect("Cannot read FooBar");
///
///     // if instead the buffer contains a length delimited stream of message we could use:
///     // while !r.is_eof() {
///     //     let foobar: FooBar = r.read_message(&bytes).expect(...);
///     //     ...
///     // }
///     println!(
///         "Found {} foos and {} bars",
///         foobar.foos.len(),
///         foobar.bars.len()
///     );
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BytesReader {
    start: usize,
    end: usize,
}

impl BytesReader {
    /// Creates a new reader from chunks of data
    pub fn from_bytes(bytes: &[u8]) -> BytesReader {
        BytesReader {
            start: 0,
            end: bytes.len(),
        }
    }

    /// Returns the current read position.
    #[inline(always)]
    pub fn start(&self) -> usize {
        self.start
    }

    /// Returns the current end-of-message boundary.
    #[inline(always)]
    pub fn end(&self) -> usize {
        self.end
    }

    /// Sets the current read position.
    /// SAFETY: caller must ensure `pos <= self.end`.
    #[inline(always)]
    pub unsafe fn set_start(&mut self, pos: usize) {
        self.start = pos;
    }

    /// Sets the end-of-message boundary.
    /// SAFETY: caller must ensure `end <= bytes.len()` for the active buffer.
    #[inline(always)]
    pub unsafe fn set_end(&mut self, end: usize) {
        self.end = end;
    }

    /// Reads next tag, `None` if all bytes have been read
    #[cfg_attr(feature = "std", inline(always))]
    pub fn next_tag(&mut self, bytes: &[u8]) -> Result<u32> {
        self.read_varint32(bytes)
    }

    /// Reads the next byte
    #[cfg_attr(feature = "std", inline(always))]
    pub fn read_u8(&mut self, bytes: &[u8]) -> Result<u8> {
        if self.start >= self.end {
            return Err(Error::UnexpectedEndOfBuffer);
        }
        // SAFETY: start < end (checked above) and end <= bytes.len() (BytesReader invariant),
        // so start < bytes.len().
        let b = unsafe { *bytes.get_unchecked(self.start) };
        self.start += 1;
        Ok(b)
    }

    /// Reads the next varint encoded u32
    #[cfg_attr(feature = "std", inline(always))]
    pub fn read_varint32(&mut self, bytes: &[u8]) -> Result<u32> {
        // ARM64 branchless fast path: load 8 bytes as u64, decode without branches
        #[cfg(target_arch = "aarch64")]
        {
            if self.start + 8 <= self.end && self.end <= bytes.len() {
                // SAFETY: start + 8 <= end <= bytes.len(), so reading 8 bytes at start is valid.
                // [u8; 8] has alignment 1 so the pointer cast is always aligned.
                let raw = u64::from_le_bytes(unsafe {
                    *(bytes.as_ptr().add(self.start) as *const [u8; 8])
                });

                if let Some((value, len)) = decode_varint32_branchless(raw) {
                    self.start += len;
                    return Ok(value);
                }

                // Negative i32 encoded as 10-byte varint: extract value from first
                // 5 bytes, then skip remaining continuation bytes.
                let result = ((raw & 0x0000_007F)
                    | ((raw >> 1) & 0x0000_3F80)
                    | ((raw >> 2) & 0x001F_C000)
                    | ((raw >> 3) & 0x0FE0_0000)
                    | ((raw >> 4) & 0xF000_0000)) as u32;

                self.start += 5;
                for _ in 0..5 {
                    if self.read_u8(bytes)? & 0x80 == 0 {
                        return Ok(result);
                    }
                }
                return Err(Error::Varint);
            }
        }

        // Scalar fast path: at least 5 bytes available, skip per-byte bounds checks
        #[cfg(not(target_arch = "aarch64"))]
        {
            if self.start + 5 <= self.end && self.end <= bytes.len() {
                // SAFETY: start + 5 <= end <= bytes.len(), so bytes[start..] has >= 5 elements
                let buf = unsafe { bytes.get_unchecked(self.start..) };

                let b = buf[0];
                if b & 0x80 == 0 {
                    self.start += 1;
                    return Ok(b as u32);
                }
                let mut r = (b & 0x7f) as u32;

                let b = buf[1];
                r |= ((b & 0x7f) as u32) << 7;
                if b & 0x80 == 0 {
                    self.start += 2;
                    return Ok(r);
                }

                let b = buf[2];
                r |= ((b & 0x7f) as u32) << 14;
                if b & 0x80 == 0 {
                    self.start += 3;
                    return Ok(r);
                }

                let b = buf[3];
                r |= ((b & 0x7f) as u32) << 21;
                if b & 0x80 == 0 {
                    self.start += 4;
                    return Ok(r);
                }

                let b = buf[4];
                r |= ((b & 0xf) as u32) << 28;
                if b & 0x80 == 0 {
                    self.start += 5;
                    return Ok(r);
                }

                // Negative i32 encoded as 10-byte varint: discard remaining bytes
                self.start += 5;
                for _ in 0..5 {
                    if self.read_u8(bytes)? & 0x80 == 0 {
                        return Ok(r);
                    }
                }
                return Err(Error::Varint);
            }
        }

        // Slow path: near end of buffer, per-byte bounds checks
        self.read_varint32_slow(bytes)
    }

    /// Slow path for read_varint32 when fewer than 5 bytes remain.
    /// Uses per-byte bounds checks via read_u8.
    #[cold]
    fn read_varint32_slow(&mut self, bytes: &[u8]) -> Result<u32> {
        let mut b = self.read_u8(bytes)?;
        if b & 0x80 == 0 {
            return Ok(b as u32);
        }
        let mut r = (b & 0x7f) as u32;

        b = self.read_u8(bytes)?;
        r |= ((b & 0x7f) as u32) << 7;
        if b & 0x80 == 0 {
            return Ok(r);
        }

        b = self.read_u8(bytes)?;
        r |= ((b & 0x7f) as u32) << 14;
        if b & 0x80 == 0 {
            return Ok(r);
        }

        b = self.read_u8(bytes)?;
        r |= ((b & 0x7f) as u32) << 21;
        if b & 0x80 == 0 {
            return Ok(r);
        }

        b = self.read_u8(bytes)?;
        r |= ((b & 0xf) as u32) << 28;
        if b & 0x80 == 0 {
            return Ok(r);
        }

        for _ in 0..5 {
            if self.read_u8(bytes)? & 0x80 == 0 {
                return Ok(r);
            }
        }

        Err(Error::Varint)
    }

    /// Reads the next varint encoded u64
    #[cfg_attr(feature = "std", inline(always))]
    pub fn read_varint64(&mut self, bytes: &[u8]) -> Result<u64> {
        // ARM64 branchless fast path: load 8 bytes as u64, decode without branches
        #[cfg(target_arch = "aarch64")]
        {
            if self.start + 10 <= self.end && self.end <= bytes.len() {
                // SAFETY: start + 10 <= end <= bytes.len(), so reading 8 bytes at start is valid.
                let raw = u64::from_le_bytes(unsafe {
                    *(bytes.as_ptr().add(self.start) as *const [u8; 8])
                });

                if let Some((value, len)) = decode_varint64_branchless(raw) {
                    self.start += len;
                    return Ok(value);
                }

                // 9-10 byte varint: all first 8 bytes have continuation bits set.
                // Extract 56 payload bits from bytes 0-7 branchlessly.
                let base = (raw & 0x0000_0000_0000_007F)
                    | ((raw >> 1) & 0x0000_0000_0000_3F80)
                    | ((raw >> 2) & 0x0000_0000_001F_C000)
                    | ((raw >> 3) & 0x0000_0000_0FE0_0000)
                    | ((raw >> 4) & 0x0000_0007_F000_0000)
                    | ((raw >> 5) & 0x0000_03F8_0000_0000)
                    | ((raw >> 6) & 0x0001_FC00_0000_0000)
                    | ((raw >> 7) & 0x00FE_0000_0000_0000);

                // SAFETY: start + 10 <= end <= bytes.len(), so bytes[start..start+10] is valid
                let buf = unsafe { bytes.get_unchecked(self.start..) };

                // Byte 8 (part2, first byte)
                let b8 = buf[8];
                if b8 & 0x80 == 0 {
                    self.start += 9;
                    return Ok(base | ((b8 as u64 & 0x7F) << 56));
                }

                // Byte 9 (part2, second byte) - silent truncation of high bits,
                // consistent with slow path and Google's protobuf implementation.
                let b9 = buf[9];
                if b9 & 0x80 == 0 {
                    self.start += 10;
                    let r2 = (b8 as u64 & 0x7F) | ((b9 as u64) << 7);
                    return Ok(base | (r2 << 56));
                }

                // cannot read more than 10 bytes
                self.start += 10;
                return Err(Error::Varint);
            }
        }

        // Scalar fast path: at least 10 bytes available, skip per-byte bounds checks
        #[cfg(not(target_arch = "aarch64"))]
        {
            if self.start + 10 <= self.end && self.end <= bytes.len() {
                // SAFETY: start + 10 <= end <= bytes.len(), so bytes[start..] has >= 10 elements
                let buf = unsafe { bytes.get_unchecked(self.start..) };

                // part0
                let b = buf[0];
                if b & 0x80 == 0 {
                    self.start += 1;
                    return Ok(b as u64);
                }
                let mut r0 = (b & 0x7f) as u32;

                let b = buf[1];
                r0 |= ((b & 0x7f) as u32) << 7;
                if b & 0x80 == 0 {
                    self.start += 2;
                    return Ok(r0 as u64);
                }

                let b = buf[2];
                r0 |= ((b & 0x7f) as u32) << 14;
                if b & 0x80 == 0 {
                    self.start += 3;
                    return Ok(r0 as u64);
                }

                let b = buf[3];
                r0 |= ((b & 0x7f) as u32) << 21;
                if b & 0x80 == 0 {
                    self.start += 4;
                    return Ok(r0 as u64);
                }

                // part1
                let b = buf[4];
                let mut r1 = (b & 0x7f) as u32;
                if b & 0x80 == 0 {
                    self.start += 5;
                    return Ok(r0 as u64 | (r1 as u64) << 28);
                }

                let b = buf[5];
                r1 |= ((b & 0x7f) as u32) << 7;
                if b & 0x80 == 0 {
                    self.start += 6;
                    return Ok(r0 as u64 | (r1 as u64) << 28);
                }

                let b = buf[6];
                r1 |= ((b & 0x7f) as u32) << 14;
                if b & 0x80 == 0 {
                    self.start += 7;
                    return Ok(r0 as u64 | (r1 as u64) << 28);
                }

                let b = buf[7];
                r1 |= ((b & 0x7f) as u32) << 21;
                if b & 0x80 == 0 {
                    self.start += 8;
                    return Ok(r0 as u64 | (r1 as u64) << 28);
                }

                // part2
                let b = buf[8];
                let mut r2 = (b & 0x7f) as u32;
                if b & 0x80 == 0 {
                    self.start += 9;
                    return Ok((r0 as u64 | (r1 as u64) << 28) | (r2 as u64) << 56);
                }

                // Silent truncation of high bits, consistent with slow path and
                // Google's protobuf implementation (see comment in read_varint64_slow).
                let b = buf[9];
                r2 |= (b as u32) << 7;
                if b & 0x80 == 0 {
                    self.start += 10;
                    return Ok((r0 as u64 | (r1 as u64) << 28) | (r2 as u64) << 56);
                }

                // cannot read more than 10 bytes
                self.start += 10;
                return Err(Error::Varint);
            }
        }

        // Slow path: near end of buffer, per-byte bounds checks
        self.read_varint64_slow(bytes)
    }

    /// Slow path for read_varint64 when fewer than 10 bytes remain.
    /// Uses per-byte bounds checks via read_u8.
    #[cold]
    fn read_varint64_slow(&mut self, bytes: &[u8]) -> Result<u64> {
        // part0
        let mut b = self.read_u8(bytes)?;
        if b & 0x80 == 0 {
            return Ok(b as u64);
        }
        let mut r0 = (b & 0x7f) as u32;

        b = self.read_u8(bytes)?;
        r0 |= ((b & 0x7f) as u32) << 7;
        if b & 0x80 == 0 {
            return Ok(r0 as u64);
        }

        b = self.read_u8(bytes)?;
        r0 |= ((b & 0x7f) as u32) << 14;
        if b & 0x80 == 0 {
            return Ok(r0 as u64);
        }

        b = self.read_u8(bytes)?;
        r0 |= ((b & 0x7f) as u32) << 21;
        if b & 0x80 == 0 {
            return Ok(r0 as u64);
        }

        // part1
        b = self.read_u8(bytes)?;
        let mut r1 = (b & 0x7f) as u32;
        if b & 0x80 == 0 {
            return Ok(r0 as u64 | (r1 as u64) << 28);
        }

        b = self.read_u8(bytes)?;
        r1 |= ((b & 0x7f) as u32) << 7;
        if b & 0x80 == 0 {
            return Ok(r0 as u64 | (r1 as u64) << 28);
        }

        b = self.read_u8(bytes)?;
        r1 |= ((b & 0x7f) as u32) << 14;
        if b & 0x80 == 0 {
            return Ok(r0 as u64 | (r1 as u64) << 28);
        }

        b = self.read_u8(bytes)?;
        r1 |= ((b & 0x7f) as u32) << 21;
        if b & 0x80 == 0 {
            return Ok(r0 as u64 | (r1 as u64) << 28);
        }

        // part2
        b = self.read_u8(bytes)?;
        let mut r2 = (b & 0x7f) as u32;
        if b & 0x80 == 0 {
            return Ok((r0 as u64 | (r1 as u64) << 28) | (r2 as u64) << 56);
        }

        // WARNING ABOUT TRUNCATION:
        //
        // For the number to be within valid 64 bit range, some conditions about
        // this last byte must be met:
        // 1. This must be the last byte (MSB not set)
        // 2. No 64-bit overflow (middle 6 bits are beyond 64 bits for the
        //    entire varint, so they cannot be set either)
        //
        // However, for the sake of consistency with Google's own protobuf
        // implementation, and also to allow for any efficient use of those
        // extra bits by users if they wish (this crate is meant for speed
        // optimization anyway) we shall not check for this here.
        //
        // Therefore, THIS FUNCTION SIMPLY IGNORES THE EXTRA BITS, WHICH IS
        // ESSENTIALLY A SILENT TRUNCATION!
        b = self.read_u8(bytes)?;
        r2 |= (b as u32) << 7;
        if b & 0x80 == 0 {
            return Ok((r0 as u64 | (r1 as u64) << 28) | (r2 as u64) << 56);
        }

        // cannot read more than 10 bytes
        Err(Error::Varint)
    }

    /// Reads int32 (varint)
    #[cfg_attr(feature = "std", inline)]
    pub fn read_int32(&mut self, bytes: &[u8]) -> Result<i32> {
        self.read_varint32(bytes).map(|i| i as i32)
    }

    /// Reads int64 (varint)
    #[cfg_attr(feature = "std", inline)]
    pub fn read_int64(&mut self, bytes: &[u8]) -> Result<i64> {
        self.read_varint64(bytes).map(|i| i as i64)
    }

    /// Reads uint32 (varint)
    #[cfg_attr(feature = "std", inline)]
    pub fn read_uint32(&mut self, bytes: &[u8]) -> Result<u32> {
        self.read_varint32(bytes)
    }

    /// Reads uint64 (varint)
    #[cfg_attr(feature = "std", inline)]
    pub fn read_uint64(&mut self, bytes: &[u8]) -> Result<u64> {
        self.read_varint64(bytes)
    }

    /// Reads sint32 (varint)
    #[cfg_attr(feature = "std", inline)]
    pub fn read_sint32(&mut self, bytes: &[u8]) -> Result<i32> {
        // zigzag
        let n = self.read_varint32(bytes)?;
        Ok(((n >> 1) as i32) ^ (-((n & 1) as i32)))
    }

    /// Reads sint64 (varint)
    #[cfg_attr(feature = "std", inline)]
    pub fn read_sint64(&mut self, bytes: &[u8]) -> Result<i64> {
        // zigzag
        let n = self.read_varint64(bytes)?;
        Ok(((n >> 1) as i64) ^ (-((n & 1) as i64)))
    }

    /// Reads fixed64 (little endian u64)
    #[cfg_attr(feature = "std", inline)]
    pub fn read_fixed64(&mut self, bytes: &[u8]) -> Result<u64> {
        let end = self.start + 8;
        if end > self.end {
            return Err(Error::UnexpectedEndOfBuffer);
        }
        // SAFETY: end <= self.end <= bytes.len() (BytesReader invariant), and
        // [u8; 8] has alignment 1 so the pointer cast is always valid.
        let v = u64::from_le_bytes(unsafe {
            *(bytes.as_ptr().add(self.start) as *const [u8; 8])
        });
        self.start = end;
        Ok(v)
    }

    /// Reads fixed32 (little endian u32)
    #[cfg_attr(feature = "std", inline)]
    pub fn read_fixed32(&mut self, bytes: &[u8]) -> Result<u32> {
        let end = self.start + 4;
        if end > self.end {
            return Err(Error::UnexpectedEndOfBuffer);
        }
        // SAFETY: end <= self.end <= bytes.len() (BytesReader invariant), and
        // [u8; 4] has alignment 1 so the pointer cast is always valid.
        let v = u32::from_le_bytes(unsafe {
            *(bytes.as_ptr().add(self.start) as *const [u8; 4])
        });
        self.start = end;
        Ok(v)
    }

    /// Reads sfixed64 (little endian i64)
    #[cfg_attr(feature = "std", inline)]
    pub fn read_sfixed64(&mut self, bytes: &[u8]) -> Result<i64> {
        let end = self.start + 8;
        if end > self.end {
            return Err(Error::UnexpectedEndOfBuffer);
        }
        // SAFETY: end <= self.end <= bytes.len(), [u8; 8] has alignment 1
        let v = i64::from_le_bytes(unsafe {
            *(bytes.as_ptr().add(self.start) as *const [u8; 8])
        });
        self.start = end;
        Ok(v)
    }

    /// Reads sfixed32 (little endian i32)
    #[cfg_attr(feature = "std", inline)]
    pub fn read_sfixed32(&mut self, bytes: &[u8]) -> Result<i32> {
        let end = self.start + 4;
        if end > self.end {
            return Err(Error::UnexpectedEndOfBuffer);
        }
        // SAFETY: end <= self.end <= bytes.len(), [u8; 4] has alignment 1
        let v = i32::from_le_bytes(unsafe {
            *(bytes.as_ptr().add(self.start) as *const [u8; 4])
        });
        self.start = end;
        Ok(v)
    }

    /// Reads float (little endian f32)
    #[cfg_attr(feature = "std", inline)]
    pub fn read_float(&mut self, bytes: &[u8]) -> Result<f32> {
        let end = self.start + 4;
        if end > self.end {
            return Err(Error::UnexpectedEndOfBuffer);
        }
        // SAFETY: end <= self.end <= bytes.len(), [u8; 4] has alignment 1
        let v = f32::from_le_bytes(unsafe {
            *(bytes.as_ptr().add(self.start) as *const [u8; 4])
        });
        self.start = end;
        Ok(v)
    }

    /// Reads double (little endian f64)
    #[cfg_attr(feature = "std", inline)]
    pub fn read_double(&mut self, bytes: &[u8]) -> Result<f64> {
        let end = self.start + 8;
        if end > self.end {
            return Err(Error::UnexpectedEndOfBuffer);
        }
        // SAFETY: end <= self.end <= bytes.len(), [u8; 8] has alignment 1
        let v = f64::from_le_bytes(unsafe {
            *(bytes.as_ptr().add(self.start) as *const [u8; 8])
        });
        self.start = end;
        Ok(v)
    }

    /// Reads bool (varint, check if == 0)
    #[cfg_attr(feature = "std", inline)]
    pub fn read_bool(&mut self, bytes: &[u8]) -> Result<bool> {
        self.read_varint32(bytes).map(|i| i != 0)
    }

    /// Reads enum, encoded as i32
    #[cfg_attr(feature = "std", inline)]
    pub fn read_enum<E: From<i32>>(&mut self, bytes: &[u8]) -> Result<E> {
        self.read_int32(bytes).map(|e| e.into())
    }

    /// First reads a varint and use it as size to read a generic object
    #[cfg_attr(feature = "std", inline(always))]
    pub fn read_len_varint<'a, M, F>(&mut self, bytes: &'a [u8], read: F) -> Result<M>
    where
        F: FnMut(&mut BytesReader, &'a [u8]) -> Result<M>,
    {
        let len = self.read_varint32(bytes)? as usize;
        self.read_len(bytes, read, len)
    }

    /// Reads a certain number of bytes specified by len
    #[cfg_attr(feature = "std", inline(always))]
    fn read_len<'a, M, F>(&mut self, bytes: &'a [u8], mut read: F, len: usize) -> Result<M>
    where
        F: FnMut(&mut BytesReader, &'a [u8]) -> Result<M>,
    {
        let cur_end = self.end;
        if len > cur_end - self.start {
            return Err(Error::UnexpectedEndOfBuffer);
        }
        self.end = self.start + len;
        let v = read(self, bytes);
        self.start = self.end;
        self.end = cur_end;
        v
    }

    /// Reads bytes (Vec<u8>)
    #[cfg_attr(feature = "std", inline)]
    pub fn read_bytes<'a>(&mut self, bytes: &'a [u8]) -> Result<&'a [u8]> {
        self.read_len_varint(bytes, |r, b| {
            // SAFETY: read_len validates len <= (cur_end - start) and sets end = start + len,
            // where cur_end <= b.len() is the BytesReader invariant. So start <= end <= b.len().
            Ok(unsafe { b.get_unchecked(r.start..r.end) })
        })
    }

    /// Reads string (String)
    #[cfg_attr(feature = "std", inline)]
    pub fn read_string<'a>(&mut self, bytes: &'a [u8]) -> Result<&'a str> {
        self.read_len_varint(bytes, |r, b| {
            // SAFETY: same invariant as read_bytes — read_len guarantees start <= end <= b.len()
            let slice = unsafe { b.get_unchecked(r.start..r.end) };
            ::core::str::from_utf8(slice).map_err(|e| e.into())
        })
    }

    /// Reads packed repeated field (Vec<M>)
    ///
    /// Note: packed field are stored as a variable length chunk of data, while regular repeated
    /// fields behaves like an iterator, yielding their tag everytime
    #[cfg_attr(feature = "std", inline)]
    pub fn read_packed<'a, M, F>(&mut self, bytes: &'a [u8], mut read: F) -> Result<Vec<M>>
    where
        F: FnMut(&mut BytesReader, &'a [u8]) -> Result<M>,
    {
        self.read_len_varint(bytes, |r, b| {
            let elem_size = core::mem::size_of::<M>().max(1);
            let mut v = Vec::with_capacity((r.len() / elem_size).min(1024));
            while !r.is_eof() {
                v.push(read(r, b)?);
            }
            Ok(v)
        })
    }

    /// Reads packed repeated int32 field with NEON-accelerated batch decode on ARM64.
    ///
    /// On ARM64, uses NEON intrinsics to detect varint boundaries in parallel across
    /// 16-byte chunks, then decodes individual varints using branchless scalar decode.
    /// On other architectures, falls back to standard per-element varint decode.
    #[cfg_attr(feature = "std", inline)]
    pub fn read_packed_int32(&mut self, bytes: &[u8]) -> Result<Vec<i32>> {
        self.read_len_varint(bytes, |r, b| {
            let capacity = r.len().min(1024);
            let mut v: Vec<i32> = Vec::with_capacity(capacity);

            #[cfg(target_arch = "aarch64")]
            {
                // Safety: r.start <= r.end is maintained by BytesReader invariants,
                // and r.end <= b.len() is enforced by read_len_varint.
                let new_pos = unsafe { batch_decode_varint32_neon(b, r.start, r.end, &mut v) };
                r.start = new_pos;
            }

            // Scalar fallback for remaining bytes (or all bytes on non-aarch64)
            while !r.is_eof() {
                v.push(r.read_varint32(b)? as i32);
            }

            Ok(v)
        })
    }

    /// Reads packed repeated field where M can directly be transmutted from raw bytes
    ///
    /// Note: packed field are stored as a variable length chunk of data, while regular repeated
    /// fields behaves like an iterator, yielding their tag everytime
    #[cfg_attr(feature = "std", inline)]
    pub fn read_packed_fixed<'a, M: Copy + PartialEq>(
        &mut self,
        bytes: &'a [u8],
    ) -> Result<PackedFixed<'a, M>>
    where
        [M]: ToOwned,
    {
        let len = self.read_varint32(bytes)? as usize;
        if self.len() < len {
            return Err(Error::UnexpectedEndOfBuffer);
        }

        // Note the floor divide; we rely on this to guarantee
        // correctness in the rest of this function
        let n = len / ::core::mem::size_of::<M>();
        let target = &bytes[self.start..self.start + (n * ::core::mem::size_of::<M>())];

        self.start += len;
        Ok(PackedFixed::from(target))
    }

    /// Reads a nested message
    ///
    /// First reads a varint and interprets it as the length of the message
    #[cfg_attr(feature = "std", inline)]
    pub fn read_message<'a, M>(&mut self, bytes: &'a [u8]) -> Result<M>
    where
        M: MessageRead<'a>,
    {
        self.read_len_varint(bytes, M::from_reader)
    }

    /// Reads a nested message
    ///
    /// Reads just the message and does not try to read it's size first.
    ///  * 'len' - The length of the message to be read.
    #[cfg_attr(feature = "std", inline)]
    pub fn read_message_by_len<'a, M>(&mut self, bytes: &'a [u8], len: usize) -> Result<M>
    where
        M: MessageRead<'a>,
    {
        self.read_len(bytes, M::from_reader, len)
    }

    /// Reads a map item: (key, value)
    #[cfg_attr(feature = "std", inline)]
    pub fn read_map<'a, K, V, F, G>(
        &mut self,
        bytes: &'a [u8],
        mut read_key: F,
        mut read_val: G,
    ) -> Result<(K, V)>
    where
        F: FnMut(&mut BytesReader, &'a [u8]) -> Result<K>,
        G: FnMut(&mut BytesReader, &'a [u8]) -> Result<V>,
        K: ::core::fmt::Debug + Default,
        V: ::core::fmt::Debug + Default,
    {
        self.read_len_varint(bytes, |r, bytes| {
            let mut k = K::default();
            let mut v = V::default();
            while !r.is_eof() {
                let t = r.read_u8(bytes)?;
                match t >> 3 {
                    1 => k = read_key(r, bytes)?,
                    2 => v = read_val(r, bytes)?,
                    t => return Err(Error::Map(t)),
                }
            }
            Ok((k, v))
        })
    }

    /// Reads unknown data, based on its tag value (which itself gives us the wire_type value)
    #[cfg_attr(feature = "std", inline)]
    pub fn read_unknown(&mut self, bytes: &[u8], tag_value: u32) -> Result<()> {
        // Since `read.varint64()` calls `read_u8()`, which increments
        // `self.start`, we don't need to manually increment `self.start` in
        // control flows that either call `read_varint64()` or error out.
        let offset = match (tag_value & 0x7) as u8 {
            WIRE_TYPE_VARINT => {
                self.read_varint64(bytes)?;
                return Ok(());
            }
            WIRE_TYPE_FIXED64 => 8,
            WIRE_TYPE_FIXED32 => 4,
            WIRE_TYPE_LENGTH_DELIMITED => {
                usize::try_from(self.read_varint64(bytes)?).map_err(|_| Error::Varint)?
            }
            WIRE_TYPE_START_GROUP | WIRE_TYPE_END_GROUP => {
                return Err(Error::Deprecated("group"));
            }
            t => {
                return Err(Error::UnknownWireType(t));
            }
        };

        // Meant to prevent overflowing. Comparison used is *strictly* lesser
        // since `self.end` is given by `len()`; i.e. `self.end` is 1 more than
        // highest index
        if self.end - self.start < offset {
            Err(Error::Varint)
        } else {
            self.start += offset;
            Ok(())
        }
    }

    /// Gets the remaining length of bytes not read yet
    #[cfg_attr(feature = "std", inline(always))]
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    /// Checks if `self.len == 0`
    #[cfg_attr(feature = "std", inline(always))]
    pub fn is_eof(&self) -> bool {
        self.start == self.end
    }

    /// Advance inner cursor to the end
    pub fn read_to_end(&mut self) {
        self.start = self.end;
    }
}

/// A struct to read protobuf data
///
/// Contrary to `BytesReader`, this struct will own a buffer
///
/// # Examples
///
/// ```rust,should_panic
/// # mod foo_bar {
/// #     use quick_protobuf::{MessageRead, BytesReader, Result};
/// #     pub struct Foo {}
/// #     pub struct Bar {}
/// #     pub struct FooBar { pub foos: Vec<Foo>, pub bars: Vec<Bar>, }
/// #     impl<'a> MessageRead<'a> for FooBar {
/// #         fn from_reader(_: &mut BytesReader, _: &[u8]) -> Result<Self> {
/// #              Ok(FooBar { foos: vec![], bars: vec![] })
/// #         }
/// #     }
/// # }
///
/// // FooBar is a message generated from a proto file
/// // In particular it implements the `MessageRead` trait, containing a `from_reader` function.
/// use foo_bar::FooBar;
/// use quick_protobuf::Reader;
///
/// fn main() {
///     // create a reader, which will parse the protobuf binary file and pop events
///     // this reader will read the entire file into an internal buffer
///     let mut reader =
///         Reader::from_file("/path/to/binary/protobuf.bin").expect("Cannot read input file");
///
///     // Use the generated module fns with the reader to convert your data into rust structs.
///     //
///     // Depending on your input file, the message can or not be prefixed with the encoded length
///     // for instance, a *stream* which contains several messages generally split them using this
///     // technique (see https://developers.google.com/protocol-buffers/docs/techniques#streaming)
///     //
///     // To read a message without a length prefix you can directly call `FooBar::from_reader`:
///     // let foobar = reader.read(FooBar::from_reader).expect("Cannot read FooBar message");
///     //
///     // Else to read a length then a message, you can use:
///     let foobar: FooBar = reader
///         .read(|r, b| r.read_message(b))
///         .expect("Cannot read FooBar message");
///     // Reader::read_message uses `FooBar::from_reader` internally through the `MessageRead`
///     // trait.
///
///     println!(
///         "Found {} foos and {} bars!",
///         foobar.foos.len(),
///         foobar.bars.len()
///     );
/// }
/// ```
pub struct Reader {
    buffer: Vec<u8>,
    inner: BytesReader,
}

impl Reader {
    /// Creates a new `Reader`
    #[cfg(feature = "std")]
    #[allow(clippy::uninit_vec)]
    pub fn from_reader<R: Read>(mut r: R, capacity: usize) -> Result<Reader> {
        let mut buf = Vec::with_capacity(capacity);
        unsafe {
            buf.set_len(capacity);
        }
        buf.shrink_to_fit();
        r.read_exact(&mut buf)?;
        Ok(Reader::from_bytes(buf))
    }

    /// Creates a new `Reader` out of a file path
    #[cfg(feature = "std")]
    pub fn from_file<P: AsRef<Path>>(src: P) -> Result<Reader> {
        let len = src.as_ref().metadata().unwrap().len() as usize;
        let f = File::open(src)?;
        Reader::from_reader(f, len)
    }

    /// Creates a new reader consuming the bytes
    pub fn from_bytes(bytes: Vec<u8>) -> Reader {
        let reader = BytesReader {
            start: 0,
            end: bytes.len(),
        };
        Reader {
            buffer: bytes,
            inner: reader,
        }
    }

    /// Run a `BytesReader` dependent function
    #[cfg_attr(feature = "std", inline)]
    pub fn read<'a, M, F>(&'a mut self, mut read: F) -> Result<M>
    where
        F: FnMut(&mut BytesReader, &'a [u8]) -> Result<M>,
    {
        read(&mut self.inner, &self.buffer)
    }

    /// Gets the inner `BytesReader`
    pub fn inner(&mut self) -> &mut BytesReader {
        &mut self.inner
    }

    /// Gets the buffer used internally
    pub fn buffer(&self) -> &[u8] {
        &self.buffer
    }
}

/// Deserialize a `MessageRead from a `&[u8]`
pub fn deserialize_from_slice<'a, M: MessageRead<'a>>(bytes: &'a [u8]) -> Result<M> {
    let mut reader = BytesReader::from_bytes(bytes);
    reader.read_message::<M>(bytes)
}

/// Wrapper enum over packed fixed data, similar to `Cow`.
///
/// When we read packed fixed data, the raw bytes are often misaligned to the
/// data type they represent. We don't want to have to align all the data before
/// reading (especially if we're only accessing a few elements), as this
/// involves lengthy allocations. This enum provides the `Borrowed` variant as a
/// wrapper for such data, holding a reference to the (possibly) misaligned raw
/// bytes and providing methods to read and iterate that are alignment-safe.
///
/// However, it is also convenient for the user to be able to use a
/// `PackedFixed` variant that owns its own data (perhaps when setting the data
/// themselves). It is mainly for this reason that the `Owned` variant is
/// provided, which owns a `Vec<T>`.
///
/// One implementation detail is that the `Owned` variant is always aligned, so
/// no use of `read_unaligned` is necessary. Methods are provided to convert
/// from `Borrowed` to `Owned`, if it is found that it helps compiler
/// optimization (not fully benchmarked at time of writing, seems
/// temperamental).
#[derive(Debug, Clone, Default)]
pub enum PackedFixed<'a, T: Copy + PartialEq> {
    /// Default when no data has been received yet; e.g. when just initialized.
    #[default]
    NoDataYet,
    /// Variant that carries a reference to raw bytes that may or may not be
    /// aligned, representing a packed set of fixed numbers.
    ///
    /// `PackedFixed` methods called on `Borrowed` will use `read_unaligned()`
    /// to interact with the data without copying all bytes to an aligned buffer
    /// in order to avoid delay from that memory allocation. So far, I can't
    /// think of any way to take advantage of the situations when it is
    /// coincidentally aligned.
    Borrowed(&'a [u8]),
    /// Variant that contains an owned vector of numbers.
    Owned(Vec<T>),
}

impl<'a, T: Copy + PartialEq> PackedFixed<'a, T> {
    /// Return the length of the DATA (not the bytes).
    pub fn len(&self) -> usize {
        match self {
            PackedFixed::Borrowed(bytes) => bytes.len() / ::core::mem::size_of::<T>(),
            PackedFixed::Owned(v) => v.len(),
            PackedFixed::NoDataYet => 0,
        }
    }

    /// Mutate in place to `Owned` variant. In the case of `Borrowed`, this
    /// performs a bitwise copy of the entire slice.
    pub fn own(&mut self) {
        match self {
            PackedFixed::NoDataYet => *self = PackedFixed::Owned(Vec::new()),
            PackedFixed::Borrowed(_) => {
                *self = self.make_owned_variant_from_unaligned_buf();
            }
            PackedFixed::Owned(_) => {} // no-op for PackedFixed::Owned, just like Cow
        }
    }

    /// Get a `Vec<T>` of the internal data, moving `self` in the process. The
    /// reason we move `self` is so that calling this on an `Owned` variant
    /// will not require copying data. `Borrowed` variants will trigger a
    /// bitwise copy.
    ///
    /// It would be really nice if this could instead return `&[T]` without
    /// moving `self`, but we can't do this for the `Borrowed` variant, so
    /// we have no such method on `PackedFixed` as a whole. And anyway, this is
    /// what `at()` on `Borrowed` is for.
    pub fn into_vec(self) -> Vec<T> {
        match self {
            PackedFixed::NoDataYet => Vec::new(),
            PackedFixed::Borrowed(_) => self.make_vec_from_unaligned_buf(),
            PackedFixed::Owned(v) => v,
        }
    }

    /// Get the element at index `index`.
    ///
    /// Note that `index` refers to the index of the type `T`, and NOT the byte
    /// index. In the case of `Borrowed`, this index is calculated during
    /// runtime, as if the underlying data was already in form `Vec<T>`.
    pub fn at(&self, index: usize) -> T {
        match self {
            PackedFixed::Borrowed(bytes) => {
                let byte_offset = index * core::mem::size_of::<T>();
                if byte_offset >= bytes.len() {
                    panic!("PackedFixed::at(): Index out of range!");
                }

                let mut ptr = bytes.as_ptr();
                unsafe {
                    ptr = ptr.add(byte_offset);
                    (ptr as *const T).read_unaligned()
                }
            }
            PackedFixed::Owned(v) => v[index],
            PackedFixed::NoDataYet => panic!("Cannot call at() on PackedFixed::NoDataYet!"),
        }
    }

    /// Mutate `self` to `Owned` variant before returning immutable slice
    pub fn to_slice(&mut self) -> &[T] {
        self.own();
        if let PackedFixed::Owned(ref contents) = *self {
            contents
        } else {
            unreachable!();
        }
    }

    /// Mutate `self` to `Owned` variant before returning mutable slice
    pub fn to_mut_slice(&mut self) -> &mut [T] {
        self.own();
        if let PackedFixed::Owned(ref mut contents) = *self {
            contents
        } else {
            unreachable!();
        }
    }

    /// Returns `true` if no data is contained in the enum.
    pub fn is_empty(&self) -> bool {
        match self {
            PackedFixed::NoDataYet => true,
            PackedFixed::Borrowed(bytes) => bytes.is_empty(),
            PackedFixed::Owned(contents) => contents.is_empty(),
        }
    }

    // This method is private and mainly to avoid repetition in code.
    fn make_vec_from_unaligned_buf(&self) -> Vec<T> {
        match &self {
            PackedFixed::Borrowed(bytes) => unsafe {
                let src = bytes.as_ptr();
                let mut buf = Vec::<T>::with_capacity(self.len());
                let dst = buf.as_mut_ptr() as *mut u8;
                ::core::ptr::copy(src, dst, bytes.len()); // careful to use length in bytes here
                buf.set_len(self.len());
                buf
            },
            _ => unreachable!(),
        }
    }

    // This method is private and mainly to avoid repetition in code.
    fn make_owned_variant_from_unaligned_buf(&self) -> Self {
        match &self {
            PackedFixed::Borrowed(_) => PackedFixed::Owned(self.make_vec_from_unaligned_buf()),
            _ => unreachable!(),
        }
    }
}

/// Iterator over `PackedFixed`.
pub struct PackedFixedIntoIter<'a, T: Copy + PartialEq> {
    packed_fixed: PackedFixed<'a, T>,
    index: usize,
}

impl<'a, T: Copy + PartialEq> FusedIterator for PackedFixedIntoIter<'a, T> {}

impl<'a, T: Copy + PartialEq> Iterator for PackedFixedIntoIter<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.packed_fixed.len() {
            None
        } else {
            let res = Some(self.packed_fixed.at(self.index));
            self.index += 1;
            res
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.packed_fixed.len() - self.index;
        (remaining, Some(remaining))
    }
}

impl<'a, T: Copy + PartialEq> ExactSizeIterator for PackedFixedIntoIter<'a, T> {}

impl<'a, T: Copy + PartialEq> IntoIterator for PackedFixed<'a, T> {
    type Item = T;

    type IntoIter = PackedFixedIntoIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter {
            packed_fixed: self,
            index: 0,
        }
    }
}

/// Iterator over `&'a PackedFixed`. Note: This does NOT return references to
/// the iterated elements (which is why we aren't following the convention of
/// calling it `PackedFixedIter`), because:
/// - Due to limitations of `read_unaligned()` (must always copy), we cannot get
///   references to elements of the `Borrowed` variant
/// - The only data types expected to be handled by `PackedFixed` are primitive
///   numeral types anyway.
///
/// This is purely for convenience, so we can iterate over `&PackedFixed`
/// without moving it.
pub struct PackedFixedRefIter<'a, T: Copy + PartialEq> {
    packed_fixed: &'a PackedFixed<'a, T>,
    index: usize,
}

impl<'a, T: Copy + PartialEq> FusedIterator for PackedFixedRefIter<'a, T> {}

impl<'a, T: Copy + PartialEq> Iterator for PackedFixedRefIter<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.packed_fixed.len() {
            None
        } else {
            let res = Some(self.packed_fixed.at(self.index));
            self.index += 1;
            res
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.packed_fixed.len() - self.index;
        (remaining, Some(remaining))
    }
}

impl<'a, T: Copy + PartialEq> ExactSizeIterator for PackedFixedRefIter<'a, T> {}

impl<'a, T: Copy + PartialEq> IntoIterator for &'a PackedFixed<'a, T> {
    type Item = T;

    type IntoIter = PackedFixedRefIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter {
            packed_fixed: self,
            index: 0,
        }
    }
}

impl<'a, T: Copy + PartialEq, const N: usize> From<&'a [u8; N]> for PackedFixed<'a, T> {
    fn from(value: &'a [u8; N]) -> Self {
        Self::Borrowed(value)
    }
}

impl<'a, T: Copy + PartialEq> From<&'a [u8]> for PackedFixed<'a, T> {
    fn from(value: &'a [u8]) -> Self {
        Self::Borrowed(value)
    }
}

impl<'a, T: Copy + PartialEq> From<&'a Vec<u8>> for PackedFixed<'a, T> {
    fn from(value: &'a Vec<u8>) -> Self {
        Self::Borrowed(value)
    }
}

impl<'a, T: Copy + PartialEq> From<Vec<T>> for PackedFixed<'a, T> {
    fn from(value: Vec<T>) -> Self {
        Self::Owned(value)
    }
}

impl<'a, T: Copy + PartialEq> PartialEq for PackedFixed<'a, T> {
    fn eq(&self, other: &Self) -> bool {
        self.into_iter().eq(other)
    }
}

#[test]
fn test_varint() {
    let data = [0x96, 0x01];
    let mut r = BytesReader::from_bytes(&data[..]);
    assert_eq!(150, r.read_varint32(&data[..]).unwrap());
    assert!(r.is_eof());
}

#[test]
fn test_varint32_1byte() {
    // Value 1: single byte varint (no continuation bit)
    let data = [0x01];
    let mut r = BytesReader::from_bytes(&data);
    assert_eq!(1, r.read_varint32(&data).unwrap());
    assert!(r.is_eof());

    // Value 127: largest 1-byte varint
    let data = [0x7f];
    let mut r = BytesReader::from_bytes(&data);
    assert_eq!(127, r.read_varint32(&data).unwrap());
    assert!(r.is_eof());

    // Value 0
    let data = [0x00];
    let mut r = BytesReader::from_bytes(&data);
    assert_eq!(0, r.read_varint32(&data).unwrap());
    assert!(r.is_eof());
}

#[test]
fn test_varint32_2byte() {
    // Value 128: smallest 2-byte varint
    let data = [0x80, 0x01];
    let mut r = BytesReader::from_bytes(&data);
    assert_eq!(128, r.read_varint32(&data).unwrap());
    assert!(r.is_eof());

    // Value 300
    let data = [0xac, 0x02];
    let mut r = BytesReader::from_bytes(&data);
    assert_eq!(300, r.read_varint32(&data).unwrap());
    assert!(r.is_eof());
}

#[test]
fn test_varint32_5byte() {
    // u32::MAX = 4294967295 encoded as 5-byte varint
    let data = [0xff, 0xff, 0xff, 0xff, 0x0f];
    let mut r = BytesReader::from_bytes(&data);
    assert_eq!(u32::MAX, r.read_varint32(&data).unwrap());
    assert!(r.is_eof());
}

#[test]
fn test_varint32_boundary_slow_path() {
    // 5-byte varint with only 4 bytes in buffer should fail (UnexpectedEndOfBuffer).
    // This forces the slow path since < 5 bytes remain.
    let data = [0xff, 0xff, 0xff, 0xff]; // 4 bytes, all with continuation bit
    let mut r = BytesReader::from_bytes(&data);
    assert!(matches!(r.read_varint32(&data).unwrap_err(), Error::UnexpectedEndOfBuffer));
}

#[test]
fn test_varint32_boundary_exact_fit() {
    // 4-byte varint fitting exactly in 4 bytes (slow path since < 5 bytes)
    // Value: 0x0FFFFFFF = 268435455
    let data = [0xff, 0xff, 0xff, 0x7f];
    let mut r = BytesReader::from_bytes(&data);
    assert_eq!(0x0FFFFFFF, r.read_varint32(&data).unwrap());
    assert!(r.is_eof());
}

#[test]
fn test_varint32_negative_i32() {
    // -1 as i32 encoded as 10-byte varint (all continuation bits set except last)
    let data = [0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01];
    let mut r = BytesReader::from_bytes(&data);
    let val = r.read_varint32(&data).unwrap();
    // -1 as i32 bit pattern = 0xFFFFFFFF, but read_varint32 masks byte4 with 0xF
    // so we get 0xFFFFFFFF
    assert_eq!(val as i32, -1);
    assert!(r.is_eof());
}

#[test]
fn test_varint64_1byte() {
    // Value 0
    let data = [0x00];
    let mut r = BytesReader::from_bytes(&data);
    assert_eq!(0u64, r.read_varint64(&data).unwrap());
    assert!(r.is_eof());

    // Value 1
    let data = [0x01];
    let mut r = BytesReader::from_bytes(&data);
    assert_eq!(1u64, r.read_varint64(&data).unwrap());
    assert!(r.is_eof());

    // Value 127: largest 1-byte varint
    let data = [0x7f];
    let mut r = BytesReader::from_bytes(&data);
    assert_eq!(127u64, r.read_varint64(&data).unwrap());
    assert!(r.is_eof());
}

#[test]
fn test_varint64_5byte() {
    // u32::MAX = 4294967295 encoded as 5-byte varint (tests part0 + part1 boundary)
    let data = [0xff, 0xff, 0xff, 0xff, 0x0f];
    let mut r = BytesReader::from_bytes(&data);
    assert_eq!(u32::MAX as u64, r.read_varint64(&data).unwrap());
    assert!(r.is_eof());
}

#[test]
fn test_varint64_10byte() {
    // u64::MAX encoded as 10-byte varint
    // Each of the first 9 bytes has continuation bit set (0xFF),
    // last byte is 0x01 (carries bit 63)
    let data = [0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01];
    let mut r = BytesReader::from_bytes(&data);
    assert_eq!(u64::MAX, r.read_varint64(&data).unwrap());
    assert!(r.is_eof());
}

#[test]
fn test_varint64_boundary_exact_fit() {
    // 5-byte varint64 in exactly 5-byte buffer (slow path since < 10 bytes)
    let data = [0xff, 0xff, 0xff, 0xff, 0x0f];
    let mut r = BytesReader::from_bytes(&data);
    assert_eq!(u32::MAX as u64, r.read_varint64(&data).unwrap());
    assert!(r.is_eof());
}

#[test]
fn test_varint64_boundary_slow_path() {
    // 10-byte varint with only 9 bytes in buffer should fail
    let data = [0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff];
    let mut r = BytesReader::from_bytes(&data);
    assert!(matches!(r.read_varint64(&data).unwrap_err(), Error::UnexpectedEndOfBuffer));
}

#[test]
fn read_size_overflowing_unknown() {
    let bytes = &[
        200, 250, 35, // varint tag with WIRE_TYPE_VARINT -- 589128
        //
        //
        47, // varint itself
        //
        //
        250, 36, // varint tag with WIRE_TYPE_LENGTH_DELIMITED -- 4730
        //
        //
        255, 255, 255, 255, 255, 255, 255, 255, 255, 3, // huge 10-byte length
        //
        //
        255, 255, 227, // unused extra bytes
    ];

    let mut r = BytesReader::from_bytes(bytes);

    assert!(!r.is_eof());
    assert_eq!(r.next_tag(bytes).unwrap(), 589128);
    r.read_unknown(bytes, 589128).unwrap();

    assert!(!r.is_eof());
    assert_eq!(r.next_tag(bytes).unwrap(), 4730);
    let e = r.read_unknown(bytes, 4730).unwrap_err();

    assert!(matches!(e, Error::Varint), "{:?}", e);
}

#[test]
fn test_packed_fixed_iter() {
    let pf: PackedFixed<i32> = vec![1, 2, 3, 4, 5].into();

    let mut total = 0;

    for _ in 0..10 {
        for i in &pf {
            total += i;
        }
    }

    for i in pf {
        total += i;
    }

    assert_eq!(total, (1 + 2 + 3 + 4 + 5) * (10 + 1));
}

#[test]
fn test_packed_fixed_size_hint() {
    let pf: PackedFixed<i32> = vec![1, 2, 3, 4, 5].into();

    // Test ref iterator size_hint
    let mut iter = (&pf).into_iter();
    assert_eq!(iter.size_hint(), (5, Some(5)));
    assert_eq!(iter.len(), 5);
    iter.next();
    assert_eq!(iter.size_hint(), (4, Some(4)));
    assert_eq!(iter.len(), 4);
    iter.next();
    iter.next();
    assert_eq!(iter.size_hint(), (2, Some(2)));
    assert_eq!(iter.len(), 2);
    iter.next();
    iter.next();
    assert_eq!(iter.size_hint(), (0, Some(0)));
    assert_eq!(iter.len(), 0);

    // Test owned iterator size_hint
    let pf2: PackedFixed<i32> = vec![10, 20, 30].into();
    let mut iter = pf2.into_iter();
    assert_eq!(iter.size_hint(), (3, Some(3)));
    assert_eq!(iter.len(), 3);
    iter.next();
    assert_eq!(iter.size_hint(), (2, Some(2)));
    iter.next();
    iter.next();
    assert_eq!(iter.size_hint(), (0, Some(0)));
    assert_eq!(iter.len(), 0);

    // Test Borrowed variant size_hint (len = bytes / size_of::<T>())
    let bytes: [u8; 12] = [
        0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
    ];
    let pf3: PackedFixed<i32> = PackedFixed::Borrowed(&bytes);
    let mut iter = (&pf3).into_iter();
    assert_eq!(iter.size_hint(), (3, Some(3)));
    assert_eq!(iter.len(), 3);
    iter.next();
    assert_eq!(iter.size_hint(), (2, Some(2)));
    assert_eq!(iter.len(), 2);
    iter.next();
    iter.next();
    assert_eq!(iter.size_hint(), (0, Some(0)));
    assert_eq!(iter.len(), 0);
}

#[test]
fn test_packed_fixed_collect() {
    // Test collect on owned iterator
    let pf: PackedFixed<i32> = vec![1, 2, 3, 4, 5].into();
    let collected: Vec<i32> = pf.into_iter().collect();
    assert_eq!(collected, vec![1, 2, 3, 4, 5]);

    // Test collect on ref iterator
    let pf: PackedFixed<i32> = vec![10, 20, 30].into();
    let collected: Vec<i32> = (&pf).into_iter().collect();
    assert_eq!(collected, vec![10, 20, 30]);

    // Test collect on borrowed variant
    let bytes: [u8; 12] = [
        0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
    ];
    let pf: PackedFixed<i32> = PackedFixed::Borrowed(&bytes);
    let collected: Vec<i32> = pf.into_iter().collect();
    assert_eq!(collected, vec![1, 2, 3]);
}

#[test]
fn test_packed_fixed_eq() {
    let v = vec![
        0x01u8, 0x00u8, 0x00u8, 0x00u8, 0x02u8, 0x00u8, 0x00u8, 0x00u8, 0x03u8, 0x00u8, 0x00u8,
        0x00u8,
    ];
    let borrowed: PackedFixed<i32> = PackedFixed::Borrowed(&v);
    let mut owned: PackedFixed<i32> = borrowed.clone();
    owned.own();

    let owned_reversed: PackedFixed<i32> = vec![3, 2, 1].into();

    let v_reversed = vec![
        0x03u8, 0x00u8, 0x00u8, 0x00u8, 0x02u8, 0x00u8, 0x00u8, 0x00u8, 0x01u8, 0x00u8, 0x00u8,
        0x00u8,
    ];
    let borrowed_reversed: PackedFixed<i32> = PackedFixed::Borrowed(&v_reversed);

    let ndy: PackedFixed<i32> = PackedFixed::NoDataYet;
    let ndy2: PackedFixed<i32> = PackedFixed::NoDataYet;
    let def: PackedFixed<i32> = PackedFixed::default();

    assert_eq!(borrowed, owned);
    assert_eq!(borrowed_reversed, owned_reversed);
    assert_eq!(ndy, ndy2);
    assert_eq!(ndy, def);

    assert_ne!(borrowed, owned_reversed);
    assert_ne!(owned, owned_reversed);
    assert_ne!(owned, borrowed_reversed);
    assert_ne!(borrowed, borrowed_reversed);
}

#[test]
fn test_read_len_exceeding_buffer() {
    // Length-delimited field with length 10, but only 3 bytes of data follow
    // Varint 10 = 0x0A, then only 3 bytes of actual data
    let data: &[u8] = &[0x0A, 0x01, 0x02, 0x03];
    let mut reader = BytesReader::from_bytes(data);
    // read_len_varint will read the varint (0x0A = 10), then call read_len with len=10
    // but only 3 bytes remain after the varint, so it should fail
    let result = reader.read_bytes(data);
    assert!(matches!(result.unwrap_err(), Error::UnexpectedEndOfBuffer));
}

#[test]
fn test_varint32_fast_path_1byte() {
    // 1-byte varint in a buffer large enough to trigger fast path (>= 5 bytes)
    let data = [0x01, 0x00, 0x00, 0x00, 0x00];
    let mut r = BytesReader::from_bytes(&data);
    assert_eq!(1, r.read_varint32(&data).unwrap());
    assert_eq!(r.start, 1);
}

#[test]
fn test_varint32_fast_path_2byte() {
    // 2-byte varint (300) in a buffer large enough for fast path
    let data = [0xac, 0x02, 0x00, 0x00, 0x00];
    let mut r = BytesReader::from_bytes(&data);
    assert_eq!(300, r.read_varint32(&data).unwrap());
    assert_eq!(r.start, 2);
}

#[test]
fn test_varint32_fast_path_3byte() {
    // 3-byte varint: 16384 = 0x4000 -> [0x80, 0x80, 0x01]
    let data = [0x80, 0x80, 0x01, 0x00, 0x00];
    let mut r = BytesReader::from_bytes(&data);
    assert_eq!(16384, r.read_varint32(&data).unwrap());
    assert_eq!(r.start, 3);
}

#[test]
fn test_varint32_fast_path_4byte() {
    // 4-byte varint: 2097152 = 0x200000 -> [0x80, 0x80, 0x80, 0x01]
    let data = [0x80, 0x80, 0x80, 0x01, 0x00];
    let mut r = BytesReader::from_bytes(&data);
    assert_eq!(2097152, r.read_varint32(&data).unwrap());
    assert_eq!(r.start, 4);
}

#[test]
fn test_varint32_fast_path_overflow_error() {
    // All 10+ bytes with continuation bit set -> Error::Varint (fast path)
    let data = [0x80; 11];
    let mut r = BytesReader::from_bytes(&data);
    assert!(matches!(r.read_varint32(&data).unwrap_err(), Error::Varint));
}

#[test]
fn test_varint64_fast_path_1byte() {
    // 1-byte varint in buffer >= 10 bytes (fast path)
    let data = [0x01, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let mut r = BytesReader::from_bytes(&data);
    assert_eq!(1u64, r.read_varint64(&data).unwrap());
    assert_eq!(r.start, 1);
}

#[test]
fn test_varint64_fast_path_2byte() {
    // 2-byte varint (300) in buffer >= 10 bytes (fast path)
    let data = [0xac, 0x02, 0, 0, 0, 0, 0, 0, 0, 0];
    let mut r = BytesReader::from_bytes(&data);
    assert_eq!(300u64, r.read_varint64(&data).unwrap());
    assert_eq!(r.start, 2);
}

#[test]
fn test_varint64_fast_path_3byte() {
    // 3-byte varint: 16384 = 1 << 14 -> [0x80, 0x80, 0x01]
    let data = [0x80, 0x80, 0x01, 0, 0, 0, 0, 0, 0, 0];
    let mut r = BytesReader::from_bytes(&data);
    assert_eq!(16384u64, r.read_varint64(&data).unwrap());
    assert_eq!(r.start, 3);
}

#[test]
fn test_varint64_fast_path_4byte() {
    // 4-byte varint: 2097152 = 1 << 21 -> [0x80, 0x80, 0x80, 0x01]
    let data = [0x80, 0x80, 0x80, 0x01, 0, 0, 0, 0, 0, 0];
    let mut r = BytesReader::from_bytes(&data);
    assert_eq!(2097152u64, r.read_varint64(&data).unwrap());
    assert_eq!(r.start, 4);
}

#[test]
fn test_varint64_fast_path_5byte() {
    // 5-byte varint (u32::MAX) in buffer >= 10 bytes (fast path)
    let data = [0xff, 0xff, 0xff, 0xff, 0x0f, 0, 0, 0, 0, 0];
    let mut r = BytesReader::from_bytes(&data);
    assert_eq!(u32::MAX as u64, r.read_varint64(&data).unwrap());
    assert_eq!(r.start, 5);
}

#[test]
fn test_varint64_fast_path_6byte() {
    // 6-byte varint: value = 128 << 28 = 34359738368
    // part1 second byte (buf[5]) exits with lower 7 bits = 1, shifted << 35 from base
    // buf[0..3]: all 0x80 (part0 = 0), buf[4]: 0x80 (r1 = 0), buf[5]: 0x01 (r1 |= 1<<7 = 128)
    let data = [0x80, 0x80, 0x80, 0x80, 0x80, 0x01, 0, 0, 0, 0];
    let mut r = BytesReader::from_bytes(&data);
    assert_eq!(128u64 << 28, r.read_varint64(&data).unwrap());
    assert_eq!(r.start, 6);
}

#[test]
fn test_varint64_fast_path_7byte() {
    // 7-byte varint: value = 16384 << 28 = 4398046511104
    // buf[0..3]: all 0x80 (part0 = 0), buf[4..5]: 0x80 (r1 bits 0-13 = 0),
    // buf[6]: 0x01 (r1 |= 1<<14 = 16384)
    let data = [0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x01, 0, 0, 0];
    let mut r = BytesReader::from_bytes(&data);
    assert_eq!(16384u64 << 28, r.read_varint64(&data).unwrap());
    assert_eq!(r.start, 7);
}

#[test]
fn test_varint64_fast_path_8byte() {
    // 8-byte varint in buffer >= 10 bytes (fast path)
    // Encodes value that uses part0 (28 bits) + part1 (28 bits) = 56 bits
    // 0x00FFFFFFFFFFFFFF = 72057594037927935
    let data = [0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x7f, 0, 0];
    let mut r = BytesReader::from_bytes(&data);
    assert_eq!(0x00FFFFFFFFFFFFFF_u64, r.read_varint64(&data).unwrap());
    assert_eq!(r.start, 8);
}

#[test]
fn test_varint64_fast_path_9byte() {
    // 9-byte varint in buffer >= 10 bytes (fast path)
    // Encodes a value using part0 + part1 + 1 byte of part2
    let data = [0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x7f, 0];
    let mut r = BytesReader::from_bytes(&data);
    let expected = (0x0F_FF_FF_FF_u64) | ((0x0F_FF_FF_FF_u64) << 28) | (0x7F_u64 << 56);
    assert_eq!(expected, r.read_varint64(&data).unwrap());
    assert_eq!(r.start, 9);
}

#[test]
fn test_varint64_fast_path_overflow_error() {
    // All 10 bytes with continuation bit set -> Error::Varint (fast path)
    let data = [0x80; 10];
    let mut r = BytesReader::from_bytes(&data);
    assert!(matches!(r.read_varint64(&data).unwrap_err(), Error::Varint));
}

#[test]
fn test_branchless_varint32_1byte() {
    // Value 0
    let raw = u64::from_le_bytes([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]);
    let (value, len) = decode_varint32_branchless(raw).unwrap();
    assert_eq!(value, 0);
    assert_eq!(len, 1);

    // Value 1 (with garbage in trailing bytes)
    let raw = u64::from_le_bytes([0x01, 0xAB, 0xCD, 0xEF, 0x12, 0x34, 0x56, 0x78]);
    let (value, len) = decode_varint32_branchless(raw).unwrap();
    assert_eq!(value, 1);
    assert_eq!(len, 1);

    // Value 127 (largest 1-byte varint)
    let raw = u64::from_le_bytes([0x7F, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]);
    let (value, len) = decode_varint32_branchless(raw).unwrap();
    assert_eq!(value, 127);
    assert_eq!(len, 1);
}

#[test]
fn test_branchless_varint32_2byte() {
    // Value 128: [0x80, 0x01]
    let raw = u64::from_le_bytes([0x80, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]);
    let (value, len) = decode_varint32_branchless(raw).unwrap();
    assert_eq!(value, 128);
    assert_eq!(len, 2);

    // Value 300: [0xAC, 0x02]
    let raw = u64::from_le_bytes([0xAC, 0x02, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]);
    let (value, len) = decode_varint32_branchless(raw).unwrap();
    assert_eq!(value, 300);
    assert_eq!(len, 2);

    // Value 16383 (max 2-byte): [0xFF, 0x7F]
    let raw = u64::from_le_bytes([0xFF, 0x7F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]);
    let (value, len) = decode_varint32_branchless(raw).unwrap();
    assert_eq!(value, 16383);
    assert_eq!(len, 2);
}

#[test]
fn test_branchless_varint32_3byte() {
    // Value 16384: [0x80, 0x80, 0x01]
    let raw = u64::from_le_bytes([0x80, 0x80, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00]);
    let (value, len) = decode_varint32_branchless(raw).unwrap();
    assert_eq!(value, 16384);
    assert_eq!(len, 3);

    // Value 2097151 (max 3-byte): [0xFF, 0xFF, 0x7F]
    let raw = u64::from_le_bytes([0xFF, 0xFF, 0x7F, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE]);
    let (value, len) = decode_varint32_branchless(raw).unwrap();
    assert_eq!(value, 2097151);
    assert_eq!(len, 3);
}

#[test]
fn test_branchless_varint32_4byte() {
    // Value 2097152: [0x80, 0x80, 0x80, 0x01]
    let raw = u64::from_le_bytes([0x80, 0x80, 0x80, 0x01, 0x00, 0x00, 0x00, 0x00]);
    let (value, len) = decode_varint32_branchless(raw).unwrap();
    assert_eq!(value, 2097152);
    assert_eq!(len, 4);

    // Value 268435455 (max 4-byte): [0xFF, 0xFF, 0xFF, 0x7F]
    let raw = u64::from_le_bytes([0xFF, 0xFF, 0xFF, 0x7F, 0xAA, 0xBB, 0xCC, 0xDD]);
    let (value, len) = decode_varint32_branchless(raw).unwrap();
    assert_eq!(value, 268435455);
    assert_eq!(len, 4);
}

#[test]
fn test_branchless_varint32_5byte() {
    // u32::MAX: [0xFF, 0xFF, 0xFF, 0xFF, 0x0F]
    let raw = u64::from_le_bytes([0xFF, 0xFF, 0xFF, 0xFF, 0x0F, 0x00, 0x00, 0x00]);
    let (value, len) = decode_varint32_branchless(raw).unwrap();
    assert_eq!(value, u32::MAX);
    assert_eq!(len, 5);

    // Value 268435456 (min 5-byte): [0x80, 0x80, 0x80, 0x80, 0x01]
    let raw = u64::from_le_bytes([0x80, 0x80, 0x80, 0x80, 0x01, 0xAA, 0xBB, 0xCC]);
    let (value, len) = decode_varint32_branchless(raw).unwrap();
    assert_eq!(value, 268435456);
    assert_eq!(len, 5);
}

#[test]
fn test_branchless_varint32_negative_i32_returns_none() {
    // -1 as i32 encoded as 10-byte varint: all first 5 bytes have continuation bits
    let raw = u64::from_le_bytes([0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]);
    assert!(decode_varint32_branchless(raw).is_none());

    // All bytes 0x80 (continuation only, no value bits)
    let raw = u64::from_le_bytes([0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80]);
    assert!(decode_varint32_branchless(raw).is_none());
}

#[test]
fn test_branchless_varint32_matches_scalar_for_all_sizes() {
    // Encode a u32 as a varint and verify branchless decode matches
    fn encode_varint32(mut value: u32) -> Vec<u8> {
        let mut result = Vec::new();
        loop {
            if value < 0x80 {
                result.push(value as u8);
                break;
            }
            result.push((value as u8) | 0x80);
            value >>= 7;
        }
        result
    }

    let test_values: &[u32] = &[
        0, 1, 2, 63, 64, 127,               // 1-byte
        128, 255, 256, 300, 16383,           // 2-byte
        16384, 32768, 2097151,               // 3-byte
        2097152, 134217728, 268435455,       // 4-byte
        268435456, u32::MAX / 2, u32::MAX,   // 5-byte
    ];

    for &val in test_values {
        let encoded = encode_varint32(val);
        let mut buf = [0u8; 8];
        buf[..encoded.len()].copy_from_slice(&encoded);
        let raw = u64::from_le_bytes(buf);

        let (decoded, len) = decode_varint32_branchless(raw)
            .unwrap_or_else(|| panic!("branchless decode returned None for value {}", val));
        assert_eq!(decoded, val, "value mismatch for {}", val);
        assert_eq!(len, encoded.len(), "length mismatch for {}", val);

        // Also verify against BytesReader scalar decode
        let mut reader = BytesReader::from_bytes(&buf);
        let scalar_val = reader.read_varint32(&buf).unwrap();
        assert_eq!(decoded, scalar_val, "branchless vs scalar mismatch for {}", val);
    }
}

#[test]
fn test_branchless_varint64_1byte() {
    // Value 0
    let raw = u64::from_le_bytes([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]);
    let (value, len) = decode_varint64_branchless(raw).unwrap();
    assert_eq!(value, 0);
    assert_eq!(len, 1);

    // Value 1 (with garbage in trailing bytes)
    let raw = u64::from_le_bytes([0x01, 0xAB, 0xCD, 0xEF, 0x12, 0x34, 0x56, 0x78]);
    let (value, len) = decode_varint64_branchless(raw).unwrap();
    assert_eq!(value, 1);
    assert_eq!(len, 1);

    // Value 127 (largest 1-byte varint)
    let raw = u64::from_le_bytes([0x7F, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]);
    let (value, len) = decode_varint64_branchless(raw).unwrap();
    assert_eq!(value, 127);
    assert_eq!(len, 1);
}

#[test]
fn test_branchless_varint64_2byte() {
    // Value 128: [0x80, 0x01]
    let raw = u64::from_le_bytes([0x80, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]);
    let (value, len) = decode_varint64_branchless(raw).unwrap();
    assert_eq!(value, 128);
    assert_eq!(len, 2);

    // Value 300: [0xAC, 0x02]
    let raw = u64::from_le_bytes([0xAC, 0x02, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]);
    let (value, len) = decode_varint64_branchless(raw).unwrap();
    assert_eq!(value, 300);
    assert_eq!(len, 2);

    // Value 16383 (max 2-byte): [0xFF, 0x7F]
    let raw = u64::from_le_bytes([0xFF, 0x7F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]);
    let (value, len) = decode_varint64_branchless(raw).unwrap();
    assert_eq!(value, 16383);
    assert_eq!(len, 2);
}

#[test]
fn test_branchless_varint64_5byte() {
    // u32::MAX: [0xFF, 0xFF, 0xFF, 0xFF, 0x0F]
    let raw = u64::from_le_bytes([0xFF, 0xFF, 0xFF, 0xFF, 0x0F, 0x00, 0x00, 0x00]);
    let (value, len) = decode_varint64_branchless(raw).unwrap();
    assert_eq!(value, u32::MAX as u64);
    assert_eq!(len, 5);
}

#[test]
fn test_branchless_varint64_6byte() {
    // Value = 128 << 28 = 34359738368 (uses 6 bytes)
    let raw = u64::from_le_bytes([0x80, 0x80, 0x80, 0x80, 0x80, 0x01, 0x00, 0x00]);
    let (value, len) = decode_varint64_branchless(raw).unwrap();
    assert_eq!(value, 128u64 << 28);
    assert_eq!(len, 6);
}

#[test]
fn test_branchless_varint64_7byte() {
    // Value = 16384 << 28 = 4398046511104 (uses 7 bytes)
    let raw = u64::from_le_bytes([0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x01, 0x00]);
    let (value, len) = decode_varint64_branchless(raw).unwrap();
    assert_eq!(value, 16384u64 << 28);
    assert_eq!(len, 7);
}

#[test]
fn test_branchless_varint64_8byte() {
    // 8-byte varint: 0x00FFFFFFFFFFFFFF (56 bits all set)
    let raw = u64::from_le_bytes([0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x7F]);
    let (value, len) = decode_varint64_branchless(raw).unwrap();
    assert_eq!(value, 0x00FFFFFFFFFFFFFF_u64);
    assert_eq!(len, 8);

    // Minimum 8-byte varint: 2097152 << 28 = 562949953421312
    let raw = u64::from_le_bytes([0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x01]);
    let (value, len) = decode_varint64_branchless(raw).unwrap();
    assert_eq!(value, 2097152u64 << 28);
    assert_eq!(len, 8);
}

#[test]
fn test_branchless_varint64_9_10_byte_returns_none() {
    // All 8 bytes with continuation bits set -> 9-10 byte varint
    let raw = u64::from_le_bytes([0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]);
    assert!(decode_varint64_branchless(raw).is_none());

    let raw = u64::from_le_bytes([0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80]);
    assert!(decode_varint64_branchless(raw).is_none());
}

#[test]
fn test_branchless_varint64_matches_scalar_for_all_sizes() {
    fn encode_varint64(mut value: u64) -> Vec<u8> {
        let mut result = Vec::new();
        loop {
            if value < 0x80 {
                result.push(value as u8);
                break;
            }
            result.push((value as u8) | 0x80);
            value >>= 7;
        }
        result
    }

    let test_values: &[u64] = &[
        0, 1, 2, 63, 64, 127,                           // 1-byte
        128, 255, 256, 300, 16383,                       // 2-byte
        16384, 32768, 2097151,                           // 3-byte
        2097152, 134217728, 268435455,                   // 4-byte
        268435456, u32::MAX as u64,                      // 5-byte
        (u32::MAX as u64) + 1, 1u64 << 35,              // 6-byte
        1u64 << 42, (1u64 << 42) - 1,                   // 7-byte
        1u64 << 49, (1u64 << 56) - 1,                   // 8-byte
        1u64 << 56, 1u64 << 63, u64::MAX,               // 9-10 byte (handled by read_varint64)
    ];

    for &val in test_values {
        let encoded = encode_varint64(val);

        // For 1-8 byte varints, test the branchless decode function directly
        if encoded.len() <= 8 {
            let mut buf = [0u8; 8];
            buf[..encoded.len()].copy_from_slice(&encoded);
            let raw = u64::from_le_bytes(buf);

            let (decoded, len) = decode_varint64_branchless(raw)
                .unwrap_or_else(|| panic!("branchless decode returned None for value {}", val));
            assert_eq!(decoded, val, "value mismatch for {}", val);
            assert_eq!(len, encoded.len(), "length mismatch for {}", val);
        }

        // Verify against BytesReader for all sizes (needs 10-byte buffer for fast path)
        let mut padded = vec![0u8; 10];
        padded[..encoded.len()].copy_from_slice(&encoded);
        let mut reader = BytesReader::from_bytes(&padded);
        let reader_val = reader.read_varint64(&padded).unwrap();
        assert_eq!(reader_val, val, "BytesReader mismatch for {}", val);
        assert_eq!(reader.start, encoded.len(), "BytesReader consumed wrong number of bytes for {}", val);
    }
}

// --- Tests for read_packed_int32 (NEON batch varint32 decode) ---

/// Helper: encode a u32 as a protobuf varint into buf
fn encode_varint_u32(mut value: u32, buf: &mut Vec<u8>) {
    while value >= 0x80 {
        buf.push((value as u8) | 0x80);
        value >>= 7;
    }
    buf.push(value as u8);
}

/// Helper: encode an i32 as a protobuf varint (negative values use 10-byte encoding)
fn encode_varint_i32(value: i32, buf: &mut Vec<u8>) {
    if value >= 0 {
        encode_varint_u32(value as u32, buf);
    } else {
        // Negative i32: encoded as 10-byte varint (sign-extended to u64)
        let mut v = value as u64;
        for _ in 0..9 {
            buf.push((v as u8) | 0x80);
            v >>= 7;
        }
        buf.push(v as u8);
    }
}

/// Helper: build a packed int32 field (length prefix + varint-encoded values)
fn build_packed_int32(values: &[i32]) -> Vec<u8> {
    let mut payload = Vec::new();
    for &v in values {
        encode_varint_i32(v, &mut payload);
    }
    let mut result = Vec::new();
    encode_varint_u32(payload.len() as u32, &mut result);
    result.extend_from_slice(&payload);
    result
}

#[test]
fn test_read_packed_int32_matches_scalar_small_values() {
    // All 1-byte varints (0-127)
    let values: Vec<i32> = (0..20).collect();
    let data = build_packed_int32(&values);
    let mut reader = BytesReader::from_bytes(&data);
    let result = reader.read_packed_int32(&data).unwrap();
    assert_eq!(result, values);
    assert!(reader.is_eof());
}

#[test]
fn test_read_packed_int32_matches_scalar_mixed_sizes() {
    // Mix of 1-byte, 2-byte, 3-byte, 4-byte, and 5-byte varints
    let values = vec![
        0, 1, 127,          // 1-byte
        128, 255, 16383,     // 2-byte
        16384, 2097151,      // 3-byte
        2097152, 268435455,  // 4-byte
        268435456, i32::MAX, // 5-byte
    ];
    let data = build_packed_int32(&values);
    let mut reader = BytesReader::from_bytes(&data);
    let result = reader.read_packed_int32(&data).unwrap();
    assert_eq!(result, values);
}

#[test]
fn test_read_packed_int32_matches_scalar_vs_read_packed() {
    // Compare read_packed_int32 against read_packed with read_int32
    let values: Vec<i32> = (0..50).map(|i| i * 1000).collect();
    let data = build_packed_int32(&values);

    let mut reader1 = BytesReader::from_bytes(&data);
    let result1 = reader1.read_packed_int32(&data).unwrap();

    let mut reader2 = BytesReader::from_bytes(&data);
    let result2 = reader2
        .read_packed(&data, BytesReader::read_int32)
        .unwrap();

    assert_eq!(result1, result2);
}

#[test]
fn test_read_packed_int32_negative_values() {
    // Negative i32 values use 10-byte varint encoding
    let values = vec![-1, -128, -32768, i32::MIN, 0, 1, -1];
    let data = build_packed_int32(&values);
    let mut reader = BytesReader::from_bytes(&data);
    let result = reader.read_packed_int32(&data).unwrap();
    assert_eq!(result, values);
}

#[test]
fn test_read_packed_int32_empty() {
    let data = build_packed_int32(&[]);
    let mut reader = BytesReader::from_bytes(&data);
    let result = reader.read_packed_int32(&data).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_read_packed_int32_single_element() {
    let values = vec![42];
    let data = build_packed_int32(&values);
    let mut reader = BytesReader::from_bytes(&data);
    let result = reader.read_packed_int32(&data).unwrap();
    assert_eq!(result, values);
}

#[test]
fn test_read_packed_int32_exactly_16_payload_bytes() {
    // Build a payload that is exactly 16 bytes
    // 16 single-byte varints (values 0-15)
    let values: Vec<i32> = (0..16).collect();
    let data = build_packed_int32(&values);
    // Verify payload is 16 bytes (length prefix is 1 byte for value 16)
    assert_eq!(data[0], 16); // length prefix
    assert_eq!(data.len(), 17); // 1 prefix + 16 payload
    let mut reader = BytesReader::from_bytes(&data);
    let result = reader.read_packed_int32(&data).unwrap();
    assert_eq!(result, values);
}

#[test]
fn test_read_packed_int32_17_payload_bytes() {
    // Build a payload that is 17 bytes: 17 single-byte varints
    let values: Vec<i32> = (0..17).collect();
    let data = build_packed_int32(&values);
    assert_eq!(data[0], 17);
    assert_eq!(data.len(), 18);
    let mut reader = BytesReader::from_bytes(&data);
    let result = reader.read_packed_int32(&data).unwrap();
    assert_eq!(result, values);
}

#[test]
fn test_read_packed_int32_many_elements() {
    // Stress test with many elements of varying sizes
    let values: Vec<i32> = (0..200)
        .map(|i| match i % 5 {
            0 => i,             // small (1-byte)
            1 => i * 200,       // medium (2-byte)
            2 => i * 40000,     // large (3-byte)
            3 => i * 5000000,   // very large (4-byte)
            _ => i * 100,       // mixed
        })
        .collect();
    let data = build_packed_int32(&values);
    let mut reader = BytesReader::from_bytes(&data);
    let result = reader.read_packed_int32(&data).unwrap();
    assert_eq!(result, values);
}

#[test]
fn test_read_packed_int32_all_max_positive() {
    // All i32::MAX values (5-byte varints)
    let values = vec![i32::MAX; 10];
    let data = build_packed_int32(&values);
    let mut reader = BytesReader::from_bytes(&data);
    let result = reader.read_packed_int32(&data).unwrap();
    assert_eq!(result, values);
}

// ---- Fixed-width read method tests ----

#[test]
fn test_read_fixed32_success() {
    let data: &[u8] = &[0x2A, 0x00, 0x00, 0x00];
    let mut r = BytesReader::from_bytes(data);
    assert_eq!(42u32, r.read_fixed32(data).unwrap());
    assert!(r.is_eof());
}

#[test]
fn test_read_fixed32_insufficient_buffer() {
    let data: &[u8] = &[0x01, 0x02, 0x03]; // 3 bytes, need 4
    let mut r = BytesReader::from_bytes(data);
    assert!(matches!(
        r.read_fixed32(data).unwrap_err(),
        Error::UnexpectedEndOfBuffer
    ));
}

#[test]
fn test_read_fixed64_success() {
    let data: &[u8] = &[0x2A, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
    let mut r = BytesReader::from_bytes(data);
    assert_eq!(42u64, r.read_fixed64(data).unwrap());
    assert!(r.is_eof());
}

#[test]
fn test_read_fixed64_insufficient_buffer() {
    let data: &[u8] = &[0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07]; // 7 bytes, need 8
    let mut r = BytesReader::from_bytes(data);
    assert!(matches!(
        r.read_fixed64(data).unwrap_err(),
        Error::UnexpectedEndOfBuffer
    ));
}

#[test]
fn test_read_sfixed32_success() {
    // -1 in little-endian i32 = 0xFF 0xFF 0xFF 0xFF
    let data: &[u8] = &[0xFF, 0xFF, 0xFF, 0xFF];
    let mut r = BytesReader::from_bytes(data);
    assert_eq!(-1i32, r.read_sfixed32(data).unwrap());
    assert!(r.is_eof());
}

#[test]
fn test_read_sfixed32_insufficient_buffer() {
    let data: &[u8] = &[0x01, 0x02];
    let mut r = BytesReader::from_bytes(data);
    assert!(matches!(
        r.read_sfixed32(data).unwrap_err(),
        Error::UnexpectedEndOfBuffer
    ));
}

#[test]
fn test_read_sfixed64_success() {
    // -1 in little-endian i64
    let data: &[u8] = &[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF];
    let mut r = BytesReader::from_bytes(data);
    assert_eq!(-1i64, r.read_sfixed64(data).unwrap());
    assert!(r.is_eof());
}

#[test]
fn test_read_sfixed64_insufficient_buffer() {
    let data: &[u8] = &[0x01, 0x02, 0x03, 0x04];
    let mut r = BytesReader::from_bytes(data);
    assert!(matches!(
        r.read_sfixed64(data).unwrap_err(),
        Error::UnexpectedEndOfBuffer
    ));
}

#[test]
fn test_read_float_success() {
    // 1.0f32 in little-endian = 0x00 0x00 0x80 0x3F
    let data: &[u8] = &[0x00, 0x00, 0x80, 0x3F];
    let mut r = BytesReader::from_bytes(data);
    assert_eq!(1.0f32, r.read_float(data).unwrap());
    assert!(r.is_eof());
}

#[test]
fn test_read_float_insufficient_buffer() {
    let data: &[u8] = &[0x00, 0x00, 0x80];
    let mut r = BytesReader::from_bytes(data);
    assert!(matches!(
        r.read_float(data).unwrap_err(),
        Error::UnexpectedEndOfBuffer
    ));
}

#[test]
fn test_read_double_success() {
    // 1.0f64 in little-endian = 0x00 0x00 0x00 0x00 0x00 0x00 0xF0 0x3F
    let data: &[u8] = &[0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0x3F];
    let mut r = BytesReader::from_bytes(data);
    assert_eq!(1.0f64, r.read_double(data).unwrap());
    assert!(r.is_eof());
}

#[test]
fn test_read_double_insufficient_buffer() {
    let data: &[u8] = &[0x00, 0x00, 0x00, 0x00, 0x00];
    let mut r = BytesReader::from_bytes(data);
    assert!(matches!(
        r.read_double(data).unwrap_err(),
        Error::UnexpectedEndOfBuffer
    ));
}

#[test]
fn test_read_fixed32_respects_sub_message_end() {
    // Buffer has enough physical bytes, but self.end is set tighter via read_len.
    // Varint length prefix (2) + 2 bytes of payload, then 4 more physical bytes.
    // read_fixed32 inside the length-delimited scope should fail (needs 4, has 2).
    let data: &[u8] = &[
        0x02, // varint length = 2
        0x01, 0x02, // 2 bytes of sub-message
        0x03, 0x04, 0x05, 0x06, // extra bytes outside sub-message
    ];
    let mut r = BytesReader::from_bytes(data);
    let result = r.read_len_varint(data, |r, b| r.read_fixed32(b));
    assert!(matches!(
        result.unwrap_err(),
        Error::UnexpectedEndOfBuffer
    ));
}

// ---- read_u8 sub-message boundary test ----

#[test]
fn test_read_u8_respects_sub_message_end() {
    // Buffer has physical bytes, but a length-delimited field constrains the end.
    // Use read_len_varint with a zero-length prefix to set self.end = self.start,
    // then try to read_u8 inside -- it should fail even though bytes exist.
    let data: &[u8] = &[
        0x00, // varint length = 0
        0xFF, // byte that exists physically but is outside sub-message
    ];
    let mut r = BytesReader::from_bytes(data);
    let result = r.read_len_varint(data, |r, b| r.read_u8(b));
    assert!(matches!(
        result.unwrap_err(),
        Error::UnexpectedEndOfBuffer
    ));
}
