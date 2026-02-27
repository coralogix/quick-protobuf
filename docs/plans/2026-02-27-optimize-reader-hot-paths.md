# Optimize reader.rs hot paths

## Overview
Fix broken inline attributes and optimize varint decoding - the two highest-impact reader optimizations. The inline attribute bug means zero cross-crate inlining hints are currently applied. The varint fast-path reduces per-byte bounds checks in the hottest code path.

## Context
- Files involved: `quick-protobuf/src/reader.rs`, `quick-protobuf/benches/benches.rs`
- Related patterns: existing unrolled varint loops, `#[cfg_attr(...)]` convention
- Dependencies: none new; byteorder stays

## Development Approach
- **Testing approach**: Regular (code first, then tests)
- Complete each task fully before moving to the next
- **CRITICAL: every task MUST include new/updated tests**
- **CRITICAL: all tests must pass before starting next task**

## Implementation Steps

### Task 1: Fix cfg_attr inline hints

**Files:**
- Modify: `quick-protobuf/src/reader.rs`

All 32 instances of `#[cfg_attr(std, inline)]` and `#[cfg_attr(std, inline(always))]` in reader.rs use incorrect cfg syntax. `cfg(std)` is never true because `std` is not a predefined rustc cfg flag - the correct form is `cfg(feature = "std")`. This means no inline hints are being applied to any reader function, which hurts cross-crate inlining (the primary use case for a library).

- [x] Replace all `#[cfg_attr(std, inline)]` with `#[cfg_attr(feature = "std", inline)]` in reader.rs
- [x] Replace all `#[cfg_attr(std, inline(always))]` with `#[cfg_attr(feature = "std", inline(always))]` in reader.rs
- [x] Verify existing tests still pass (`cargo test --manifest-path quick-protobuf/Cargo.toml --lib`)
- [x] Verify the 60 compiler warnings about unknown cfg `std` are gone

### Task 2: Optimize varint32 with fast-path batch bounds check

**Files:**
- Modify: `quick-protobuf/src/reader.rs`

Currently `read_varint32` calls `read_u8` up to 5 times, each performing an independent bounds check via `bytes.get(self.start)`. For the common case (most protobuf data has plenty of bytes remaining), we can check once that at least 5 bytes are available, then read directly from the slice without per-byte bounds checks. Fall back to the current per-byte approach only when near the end of the buffer.

- [x] Add a private `read_varint32_slow` method containing the current per-byte implementation
- [x] Rewrite `read_varint32` with a fast path: check `self.start + 5 <= self.end` and `self.end <= bytes.len()`, then index directly into the slice
- [x] Fast path updates `self.start` once at the end instead of incrementing per byte
- [x] Fall back to `read_varint32_slow` when fewer than 5 bytes remain
- [x] Add test for varint32 at exact buffer boundary (4 bytes remaining, 5-byte varint)
- [x] Add test for 1-byte, 2-byte, and 5-byte varints
- [x] Run lib tests

### Task 3: Optimize varint64 with fast-path batch bounds check

**Files:**
- Modify: `quick-protobuf/src/reader.rs`

Same approach as Task 2 but for `read_varint64` which reads up to 10 bytes.

- [x] Add a private `read_varint64_slow` method containing the current per-byte implementation
- [x] Rewrite `read_varint64` with a fast path: check `self.start + 10 <= self.end` and `self.end <= bytes.len()`, then index directly
- [x] Fall back to `read_varint64_slow` when fewer than 10 bytes remain
- [x] Add test for varint64 at exact buffer boundary
- [x] Add test for 1-byte, 5-byte, and 10-byte varint64s
- [x] Run lib tests

### Task 4: Add bounds check in read_len and capacity hint in read_packed

**Files:**
- Modify: `quick-protobuf/src/reader.rs`

Two minor improvements: (1) `read_len` currently sets `self.end = self.start + len` without verifying this doesn't exceed the actual buffer boundary - add a bounds check. (2) `read_packed` allocates `Vec::new()` with no capacity hint - estimate from the remaining byte count.

- [x] In `read_len`, add check that `self.start + len <= cur_end`, return `Error::UnexpectedEndOfBuffer` if not
- [x] In `read_packed`, replace `Vec::new()` with `Vec::with_capacity(...)` using `r.len() / estimated_element_size` (conservative: assume 1 byte per element for varints)
- [x] Add test: read_len with length exceeding remaining buffer should return error
- [x] Run lib tests

### Task 5: Expand benchmarks and validate optimizations

**Files:**
- Modify: `quick-protobuf/benches/benches.rs`

Current benchmarks only cover varint reading. Add benchmarks for other reader operations to measure and validate the optimizations.

- [ ] Add benchmark for `read_fixed32` (packed fixed data reading)
- [ ] Add benchmark for `read_fixed64`
- [ ] Add benchmark for `read_string` (length-delimited + UTF-8 validation)
- [ ] Add benchmark for `read_packed` with varint elements
- [ ] Run benchmarks and record results

### Task 6: Verify acceptance criteria

- [ ] Run full lib test suite (`cargo test --manifest-path quick-protobuf/Cargo.toml --lib`)
- [ ] Run clippy (`cargo clippy --manifest-path quick-protobuf/Cargo.toml`)
- [ ] Verify no compiler warnings about unknown cfg `std`
- [ ] Run benchmarks to confirm no regressions
