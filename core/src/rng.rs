//! Counter-based deterministic RNG — the port of `rng.py`.
//!
//! Every random number is a pure function of (key, counter): the value at
//! position `counter` of the stream seeded with `key` is
//! `mix64(key + counter * GOLDEN)` (splitmix64). All arithmetic is unsigned
//! 64-bit with wraparound (`wrapping_add` / `wrapping_mul`, the reference's
//! `& MASK64`).

// splitmix64 constants (Steele, Lea & Flood 2014; public domain reference code).
const GOLDEN: u64 = 0x9E3779B97F4A7C15;
const MIX_A: u64 = 0xBF58476D1CE4E5B9;
const MIX_B: u64 = 0x94D049BB133111EB;

// FNV-1a 64-bit constants.
const FNV_OFFSET: u64 = 0xCBF29CE484222325;
const FNV_PRIME: u64 = 0x100000001B3;

/// The FNV-1a accumulator before any byte is folded in.
pub const FNV_START: u64 = FNV_OFFSET;

/// Fold more bytes into an FNV-1a accumulator. FNV-1a is a sequential
/// byte fold, so `fnv1a_extend(fnv1a_extend(FNV_START, a), b)` equals
/// `fnv1a_64` of `a` and `b` concatenated — callers hashing many strings
/// that share a prefix can fold the prefix once and continue from its
/// accumulator, with no intermediate buffer and no change to any hash.
pub fn fnv1a_extend(mut h: u64, data: &[u8]) -> u64 {
    for &byte in data {
        h ^= byte as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// Hash bytes to a 64-bit integer with FNV-1a.
pub fn fnv1a_64(data: &[u8]) -> u64 {
    fnv1a_extend(FNV_START, data)
}

/// The splitmix64 finalizer.
pub fn mix64(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(MIX_A);
    x = (x ^ (x >> 27)).wrapping_mul(MIX_B);
    x ^ (x >> 31)
}

/// The 64-bit stream key for one decision. The decision ID is
/// length-prefixed (8 bytes, little-endian) before the salt is appended, so
/// ("ab", "c") and ("a", "bc") can never produce the same key.
pub fn derive_key(decision_id: &str, salt: &str) -> u64 {
    let id = decision_id.as_bytes();
    // FNV-1a is a sequential byte fold, so folding the length prefix, the id
    // and the salt in turn equals hashing their concatenation — with no
    // per-decision buffer to allocate.
    let h = fnv1a_extend(FNV_START, &(id.len() as u64).to_le_bytes());
    let h = fnv1a_extend(h, id);
    fnv1a_extend(h, salt.as_bytes())
}

/// The uniform 64-bit value at position `counter` of stream `key`.
pub fn random_u64(key: u64, counter: u64) -> u64 {
    mix64(key.wrapping_add(counter.wrapping_mul(GOLDEN)))
}

/// A uniform float in [0, 1) with exactly 53 random bits; every step is
/// exact in IEEE-754, so the result is bit-identical on every platform.
pub fn random_unit(key: u64, counter: u64) -> f64 {
    (random_u64(key, counter) >> 11) as f64 * (2.0f64).powi(-53)
}
