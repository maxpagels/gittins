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

/// FNV_PRIME^k for k = 0..=8.
const FNV_PRIME_POW: [u64; 9] = {
    let mut t = [1u64; 9];
    let mut k = 1;
    while k < 9 {
        t[k] = t[k - 1].wrapping_mul(FNV_PRIME);
        k += 1;
    }
    t
};

/// `fnv1a_extend(h, &w.to_le_bytes())`, bit for bit. A zero byte folds as
/// a bare multiply by the prime, and wrapping multiplication is
/// associative, so each run of k zero bytes at either end of the word
/// folds as one multiply by FNV_PRIME^k.
pub fn fnv1a_extend_u64(mut h: u64, w: u64) -> u64 {
    if w == 0 {
        return h.wrapping_mul(FNV_PRIME_POW[8]);
    }
    let low = w.trailing_zeros() / 8;
    let high = w.leading_zeros() / 8;
    h = h.wrapping_mul(FNV_PRIME_POW[low as usize]);
    let mut x = w >> (low * 8);
    for _ in 0..(8 - low - high) {
        h ^= x & 0xff;
        h = h.wrapping_mul(FNV_PRIME);
        x >>= 8;
    }
    h.wrapping_mul(FNV_PRIME_POW[high as usize])
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The zero-run shortcut must fold exactly the bytes the plain fold
    /// does: all-zero, no zeros, zeros at either end, interior zeros only,
    /// and every single-byte position.
    #[test]
    fn extend_u64_matches_the_byte_fold() {
        let mut words = vec![
            0,
            u64::MAX,
            1,
            1 << 63,
            0x00FF_0000_0000_FF00,
            0xFF00_0000_0000_00FF,
            0x3FF0_0000_0000_0000, // 1.0f64
            (-0.25f64).to_bits(),
        ];
        words.extend((0..64).map(|s| 1u64 << s));
        words.extend((0..1000).map(|c| random_u64(7, c) >> (c % 64)));
        for w in words {
            for h in [FNV_START, 0, u64::MAX, random_u64(1, w)] {
                assert_eq!(fnv1a_extend_u64(h, w), fnv1a_extend(h, &w.to_le_bytes()), "word {w:#x}");
            }
        }
    }
}
