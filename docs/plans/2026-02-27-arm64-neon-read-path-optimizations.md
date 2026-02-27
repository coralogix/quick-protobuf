# ARM64 NEON Read Path Optimizations

## Overview

Add ARM64 NEON-accelerated read path optimizations to BytesReader, targeting AWS Graviton
processors. Three optimization areas identified through read path review: branchless single
varint decode, NEON batch varint decode for packed fields, and NEON-accelerated PackedFixed
bulk conversion. All optimizations are gated on `cfg(target_arch = "aarch64")` with automatic
scalar fallback on other architectures.

## Read Path Review Findings

### High value targets

1. **Branchless single varint decode** (read_varint32, read_varint64)
   - Current code: branch-per-byte, up to 5 branches for varint32, 10 for varint64
   - Optimization: load 8 bytes at once as u64, use bit manipulation + ARM64 CLZ to find
     varint length, extract value branchlessly
   - Covers: next_tag, read_int32, read_uint32, read_sint32, read_bool, read_enum, and all
     length-prefix reads (read_string, read_bytes, read_message, read_packed)
   - Expected impact: 20-40% improvement on multi-byte varints, reduces branch mispredictions

2. **NEON batch varint decode for packed repeated fields** (read_packed with varint types)
   - Current code: tight scalar loop calling read_varint32 per element
   - Optimization: masked-VByte approach - load 16 bytes via NEON, determine all varint
     boundaries from continuation bits in parallel, extract 4+ values simultaneously
   - Expected impact: 2-4x for packed int32/uint32/sint32 fields

### Medium value targets

3. **PackedFixed bulk conversion** (make_vec_from_unaligned_buf)
   - Current code: ptr::copy (already good), but iteration uses per-element read_unaligned
   - ARM64 handles unaligned loads efficiently, so improvement is marginal
   - Skip: not worth the complexity for ARM64 where unaligned loads are fast

### Low value targets (skip)

4. **UTF-8 validation in read_string** - Rust stdlib already uses SIMD internally
5. **Field skipping in read_unknown** - low volume, not worth optimizing

## Context

- Files involved:
  - `quick-protobuf/src/reader.rs` (main target)
  - `quick-protobuf/benches/benches.rs` (benchmark additions)
  - `quick-protobuf/Cargo.toml` (no changes expected - no new deps)
- Related patterns: fast-path/slow-path split, `#[cfg_attr(feature = "std", inline)]` convention
- Dependencies: none - uses `core::arch::aarch64` intrinsics (stable since Rust 1.59, MSRV is 1.62.1)
- Target gating: `cfg(target_arch = "aarch64")` - NEON is mandatory on ARMv8-A, no feature flag needed

## Development Approach

- **Testing approach**: Regular (code first, then tests)
- Complete each task fully before moving to the next
- ARM64-specific code is gated with `cfg(target_arch)`, scalar fallback is the existing code
- All existing tests must continue to pass on all architectures
- **CRITICAL: every task MUST include new/updated tests**
- **CRITICAL: all tests must pass before starting next task**

## Implementation Steps

### Task 1: Branchless varint32 decode

**Files:**
- Modify: `quick-protobuf/src/reader.rs`

Replace the branch-per-byte varint32 fast path with a branchless approach on aarch64:

- [x] Add `#[cfg(target_arch = "aarch64")]` branchless varint32 fast path in `read_varint32`
  - Load 8 bytes as u64 (little-endian) from `bytes[self.start..]`
  - Extract continuation-bit mask: `let cont = val & 0x8080808080808080`
  - Find first zero MSB to determine length: use `(!cont).trailing_zeros() / 8 + 1` (maps to CLZ on ARM64)
  - Apply per-byte masks to strip continuation bits and combine 7-bit groups
  - Clamp to 5 bytes max for varint32
- [x] Keep existing scalar fast path under `#[cfg(not(target_arch = "aarch64"))]`
- [x] Write tests: varint32 decode produces identical results for all varint sizes (1-5 bytes)
- [x] Run project test suite - must pass before task 2

### Task 2: Branchless varint64 decode

**Files:**
- Modify: `quick-protobuf/src/reader.rs`

Apply the same branchless approach to varint64:

- [x] Add `#[cfg(target_arch = "aarch64")]` branchless varint64 fast path in `read_varint64`
  - Same u64-load approach for the first 8 bytes
  - For varints > 8 bytes (9-10), load a second u64 or fall back to scalar for the tail
  - Combine parts using shifts matching the existing r0/r1/r2 decomposition
- [x] Keep existing scalar fast path under `#[cfg(not(target_arch = "aarch64"))]`
- [x] Write tests: varint64 decode produces identical results for all varint sizes (1-10 bytes)
- [x] Run project test suite - must pass before task 3

### Task 3: NEON batch varint32 decode for packed fields

**Files:**
- Modify: `quick-protobuf/src/reader.rs`

Add a NEON-accelerated path for `read_packed` when decoding varint-encoded types:

- [x] Add `#[cfg(target_arch = "aarch64")]` module with NEON batch varint32 decode function
  - Uses `core::arch::aarch64` intrinsics: `vld1q_u8`, `vcltq_u8`, `vandq_u8`, `vgetq_lane_u8`
  - Load 16 bytes via `vld1q_u8`
  - Extract continuation bits via vector comparison against 0x80
  - Convert continuation-bit bitmask to varint boundary positions
  - Decode up to 4+ varints per 16-byte load using lookup tables or shift chains
  - Write decoded u32 values directly into output Vec
  - Handle tail (remaining bytes < 16) with scalar fallback
- [x] Add `read_packed_varint32_neon` method to BytesReader, called from `read_packed` when
    the read closure matches varint32 patterns, or add a dedicated `read_packed_int32` method
- [x] Write tests: batch decode matches scalar decode for varied varint sizes and counts
- [x] Write tests: edge cases - empty packed field, single element, exactly 16 bytes, 17 bytes
- [x] Run project test suite - must pass before task 4

### Task 4: ARM64-targeted benchmarks

**Files:**
- Modify: `quick-protobuf/benches/benches.rs`

Add benchmarks that highlight the SIMD improvements:

- [x] Add `read_packed_int32` benchmark (packed repeated int32 with mixed varint sizes)
- [x] Add `read_packed_int32_small_values` benchmark (1-byte varints, best case for batch decode)
- [x] Add `read_packed_int32_large_values` benchmark (4-5 byte varints)
- [x] Add `read_varint32_multibyte` benchmark (varints that are 2-4 bytes, to measure branchless benefit)
- [x] Verify benchmarks compile and run
- [x] Run project test suite - must pass before task 5

### Task 5: Verify acceptance criteria

- [ ] Run full test suite: `cargo test --manifest-path quick-protobuf/Cargo.toml`
- [ ] Run linter: `cargo clippy --manifest-path quick-protobuf/Cargo.toml`
- [ ] Verify all aarch64-gated code compiles (cross-check with `--target aarch64-unknown-linux-gnu` if available)
- [ ] Verify non-aarch64 builds are unaffected (existing scalar paths unchanged)

### Task 6: Update documentation

- [ ] Update CLAUDE.md with ARM64/NEON optimization patterns and conventions
- [ ] Move this plan to `docs/plans/completed/`
