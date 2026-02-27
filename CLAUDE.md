# CLAUDE.md — Developer Knowledge Base

## Project Structure

- `quick-protobuf/src/reader.rs` — core protobuf reader; hot path optimizations live here
- `quick-protobuf/src/writer.rs` — core protobuf writer
- `quick-protobuf/benches/benches.rs` — criterion benchmarks
- `pb-rs/` — code generator (separate crate)

## Build and Test Commands

- Library tests: `cargo test --manifest-path quick-protobuf/Cargo.toml --lib`
- All tests: `cargo test --manifest-path quick-protobuf/Cargo.toml`
- Benchmarks: `cargo bench --manifest-path quick-protobuf/Cargo.toml`
- Linter: `cargo clippy --manifest-path quick-protobuf/Cargo.toml`

## Architecture: BytesReader Performance Patterns

### Fast-path / slow-path split
Hot reader methods (`read_varint32`, `read_varint64`) check whether enough bytes are
available for the maximum possible read, then use direct slice indexing without per-byte
bounds checks. The slow-path fallback is a separate `#[cold]` function. Follow this
pattern for any future varint or variable-length optimizations.

### Fixed-width reads
`read_fixed64`, `read_fixed32`, `read_sfixed64`, `read_sfixed32`, `read_float`,
`read_double` all use `self.end` (the current message-end cursor) as the primary bounds
limit, then `.get()` for a defensive secondary check against `bytes.len()`. Use
`from_le_bytes` directly; do not add `byteorder` back to `reader.rs`.

### Sub-reader boundary (`self.end`)
`self.end` tracks the logical end of the current message context (set by `read_len_varint`
for length-delimited fields). All read methods must check `self.end`, not just
`bytes.len()`, to avoid escaping sub-message boundaries. `read_u8` enforces this.

### Dependencies
- `byteorder` is used only in `writer.rs`. Do not add it to `reader.rs`.

### PackedFixed iterators
Any iterator type over `PackedFixed` must implement both `size_hint()` returning
`(remaining, Some(remaining))` and `ExactSizeIterator`. Test both `Owned` and `Borrowed`
variants.

## Inline Attribute Convention

Use `#[cfg_attr(feature = "std", inline(always))]` for hot single-expression methods and
`#[cfg_attr(feature = "std", inline)]` for larger methods. The bare `#[cfg_attr(std, ...)]`
form is a no-op (never set by the compiler) — always use `feature = "std"`.
