# Spec: canonical state serialization

Design doc reference: section 7 (state as an explicit value) and R8 (flat
files, weights in version control, browser storage). Implemented in
`src/gittins_reference/state.py` (PR 18); mirrored by the Rust core's
`state.rs`; byte strings pinned in `spec/golden.json`, section
`serialization`.

## What it is

`serialize(state) -> bytes` and `deserialize(bytes) -> state` are pure,
exact inverses: one canonical encoding per state, no optional parts, the
same bytes from every platform and every implementation. (This is the
internal byte layer; the *public API* surfaces exactly these bytes as a
lowercase hex string — see `api.md` — so callers never handle bytes.) The byte string is
the deployment artifact — a flat file, a version-controlled blob, a
localStorage value — and the golden corpus pins three of them (a fresh
state, the episode's mid snapshot with two open ledger records, and the
episode's final state), so cross-implementation byte identity is enforced
in CI like every other contract.

## Layout (format version 1)

Everything little-endian, in the style of the candidate-set hash: integers
as 8-byte unsigned, floats as 8-byte IEEE-754 doubles, strings as an 8-byte
length followed by UTF-8 bytes, lists as an 8-byte count followed by the
elements.

| field | encoding |
|---|---|
| magic | 8 bytes, `"gittins\x00"` |
| format version | u64 (this layout is 1) |
| model | dim u64, ridge f64, forgetting f64, scale f64, xx[dim] f64, xy[dim] f64 |
| bandit | next_seq u64, model_version u64, horizon f64, default_reward f64, epsilon f64 |
| ledger | count u64, then each open record (below) in ledger order |
| checksum | u64, FNV-1a over every preceding byte |

Each open decision record, in declaration order:

| field | encoding |
|---|---|
| decision_id | string |
| t | f64 |
| candidate_hash | u64 |
| chosen | u64 |
| features | count u64, then per pair: index u64, value f64 |
| propensity | f64 |
| model_version | u64 |
| salt | string |

## Validation

Deserialization is all-or-nothing; anything malformed raises (ValueError in
the reference, the core's `Error` in Rust) with nothing partially applied:

- magic and format version must match; the checksum must verify (rejecting
  truncation and corruption up front); the payload must be exactly consumed
  (no trailing bytes).
- every constructor invariant is re-checked: `dim >= 1`, `forgetting` in
  (0, 1], `ridge > 0`, `horizon > 0`, `epsilon` in [0, 1], and `scale` in
  (0, 1] (where the update rule keeps it).
- each record's features must be (index, value) pairs in strictly increasing
  index order within the model dimension, values nonzero — the same
  invariant `decide` enforces on candidates.
- strings must be valid UTF-8.

A deserialized state is therefore as trustworthy as a constructed one.
Model sums (`xx`, `xy`) and record floats are deliberately *not*
range-policed beyond structure: the engine itself places no bounds on them,
and serialization must accept any state the engine can produce.

## Versioning

The format version is independent of the golden corpus's `format_version`.
Readers reject unknown versions outright — there is no forward-compatible
parsing. A layout change bumps the version, regenerates the golden section,
and the byte diff is reviewed like code.
