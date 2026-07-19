//! The CLI binding's library half: the experience-log and OPE semantics
//! (`ope`), ported from `gittins_reference/ope.py`
//! and gated by the golden `ope` section. The binary (`main.rs`) is the
//! presentation layer over these functions — flags, tables, exit codes —
//! and adds no semantics of its own.

pub mod ope;
