# Optimize reader fixed-width reads and code layout

## Overview

Second round of reader optimizations following the completed varint fast-path work. Focuses on three areas: (1) eliminating the generic `read_fixed` closure pattern in favor of direct `from_le_bytes` calls, (2) improving code layout with `#[cold]` annotations on slow paths, and (3) adding `size_hint` to PackedFixed iterators for better collection performance.

## Context

- Files involved: `quick-protobuf/src/reader.rs`, `quick-protobuf/benches/benches.rs`
- Related patterns: varint fast-path pattern from previous optimization round
- Dependencies: byteorder stays (still used by writer.rs); reader stops importing it
- Previous work: `docs/plans/completed/2026-02-27-optimize-reader-hot-paths.md` (inline fixes, varint fast paths, read_len bounds check, read_packed capacity hint)

## Findings from inspection

### 1. Generic `read_fixed<M, F>` closure pattern (lines 438-446)

The private `read_fixed` method takes a closure parameter `F: Fn(&[u8]) -> M` which dispatches to byteorder's `LE::read_u64`, `LE::read_u32`, etc. While the compiler monomorphizes and inlines this, the indirection chain is: caller -> read_fixed (generic) -> bytes.get().ok_or() -> closure -> LE::read_u64 -> u64::from_le_bytes. Replacing with direct `from_le_bytes` calls in each method eliminates the generic + closure layer, uses the tighter `self.end` bound instead of `bytes.len()` used by `get()`, and removes the byteorder import from reader.rs.

### 2. Missing `#[cold]` on slow paths (lines 179, 313)

`read_varint32_slow` and `read_varint64_slow` are fallback paths hit only near buffer boundaries. Without `#[cold]`, the compiler treats them as equally likely as the fast path, potentially placing slow-path code in hot cache lines and degrading instruction cache utilization.

### 3. PackedFixed iterators missing `size_hint` (lines 970, 1014)

Both `PackedFixedIntoIter` and `PackedFixedRefIter` implement `Iterator` but don't override `size_hint()`, defaulting to `(0, None)`. This means `collect::<Vec<_>>()` and similar operations cannot pre-allocate, causing repeated reallocations.

## Development Approach

- **Testing approach**: Regular (code first, then tests)
- Complete each task fully before moving to the next
- **CRITICAL: every task MUST include new/updated tests**
- **CRITICAL: all tests must pass before starting next task**

## Implementation Steps

### Task 1: Replace generic read_fixed with direct from_le_bytes implementations

**Files:**
- Modify: `quick-protobuf/src/reader.rs`

Replace the generic `read_fixed<M, F: Fn(&[u8]) -> M>` method and its callers with direct implementations that use Rust's native `from_le_bytes` methods. Each method does its own bounds check against `self.end` (tighter than `bytes.len()` used by `get()`) and calls `from_le_bytes` directly.

- [x] Remove the private `read_fixed<M, F>` generic method
- [x] Implement `read_fixed64` directly: bounds check `self.start + 8 <= self.end`, then `u64::from_le_bytes(bytes[self.start..self.start+8].try_into().unwrap())`
- [x] Implement `read_fixed32` directly with same pattern (4 bytes, u32)
- [x] Implement `read_sfixed64` directly (8 bytes, i64)
- [x] Implement `read_sfixed32` directly (4 bytes, i32)
- [x] Implement `read_float` directly (4 bytes, f32 via `f32::from_le_bytes`)
- [x] Implement `read_double` directly (8 bytes, f64 via `f64::from_le_bytes`)
- [x] Remove `use byteorder::ByteOrder` and `use byteorder::LittleEndian as LE` imports from reader.rs
- [x] Run lib tests (`cargo test --manifest-path quick-protobuf/Cargo.toml --lib`)

### Task 2: Add #[cold] to slow paths

**Files:**
- Modify: `quick-protobuf/src/reader.rs`

- [x] Add `#[cold]` attribute to `read_varint32_slow`
- [x] Add `#[cold]` attribute to `read_varint64_slow`
- [x] Run lib tests

### Task 3: Add size_hint to PackedFixed iterators

**Files:**
- Modify: `quick-protobuf/src/reader.rs`

- [ ] Implement `size_hint()` on `PackedFixedIntoIter`: return `(remaining, Some(remaining))` where `remaining = self.packed_fixed.len() - self.index`
- [ ] Implement `size_hint()` on `PackedFixedRefIter`: same pattern
- [ ] Add `ExactSizeIterator` impl for both iterators (since the count is always known)
- [ ] Add test: verify `size_hint` returns correct values during iteration
- [ ] Add test: verify `collect::<Vec<_>>()` produces correct results (exercises size_hint for pre-allocation)
- [ ] Run lib tests

### Task 4: Benchmark and validate optimizations

**Files:**
- Modify: `quick-protobuf/benches/benches.rs`

- [ ] Run existing benchmarks to establish baseline before changes (if not already done)
- [ ] Run benchmarks after all changes to measure impact
- [ ] Verify no regressions in any benchmark
- [ ] Run full test suite (`cargo test --manifest-path quick-protobuf/Cargo.toml --lib`)
- [ ] Run clippy (`cargo clippy --manifest-path quick-protobuf/Cargo.toml`)

### Task 5: Update documentation

- [ ] Update CLAUDE.md if internal patterns changed
- [ ] Move this plan to `docs/plans/completed/`
