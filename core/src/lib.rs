//! The gittins core: a compiled port of the pure-Python reference
//! (`src/gittins_reference/`), module for module. The reference is the
//! specification; this crate must match it **bit for bit** — every float
//! operation here mirrors the reference's expression order exactly, and
//! `tests/golden/` checks the result against `spec/golden.json`, the same
//! corpus the reference pins itself to.
//!
//! Zero dependencies. Two things are deliberately not ports: state is
//! mutated in place (the reference's immutable tuples were always
//! documented as a reference-only artifact), and `Factorization` recomputes
//! a coordinate instead of memoizing it (see `model`). Both are invisible
//! in the output; everything else — semantics, constants, RNG streams, hash
//! layouts — is unchanged.

pub mod api;
pub mod decide;
pub mod encoding;
pub mod error;
pub mod exploration;
pub mod ledger;
pub mod model;
pub mod rng;
pub mod state;
