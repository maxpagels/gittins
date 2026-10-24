//! Canonical state serialization — the port of `state.py`; the byte layout
//! and every validation rule are specified there.
//! `serialize` and `deserialize` are exact
//! inverses with one canonical encoding per state; the byte strings are
//! golden-pinned, so this module must produce the reference's bytes
//! exactly. Deserialization is all-or-nothing: the checksum rejects
//! truncation and corruption up front, then every constructor invariant is
//! re-checked, so a loaded state is as trustworthy as a constructed one.
//! The model sums are written sparsely (format version 2): only slots not
//! +0.0 in both sums, bitwise. Version 1 (dense sums) is not accepted.

use crate::decide::{BanditState, DecisionRecord};
use crate::error::Error;
use crate::model::{LinearModel, MAX_DIM};
use crate::rng::fnv1a_64;

pub const MAGIC: &[u8; 8] = b"gittins\x00";
pub const FORMAT_VERSION: u64 = 2;

/// A slot is left out of the serialization exactly when both its sums are
/// +0.0 bitwise; -0.0 is a different double and is stored.
fn is_unstored_slot(xx: f64, xy: f64) -> bool {
    xx.to_bits() == 0 && xy.to_bits() == 0
}

fn put_string(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

/// The canonical byte string for one state.
pub fn serialize(state: &BanditState) -> Vec<u8> {
    let m = &state.model;
    let stored: Vec<usize> = (0..m.dim)
        .filter(|&j| !is_unstored_slot(m.xx[j], m.xy[j]))
        .collect();
    let mut out = Vec::with_capacity(200 + 24 * stored.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&(m.dim as u64).to_le_bytes());
    out.extend_from_slice(&m.ridge.to_le_bytes());
    out.extend_from_slice(&m.forgetting.to_le_bytes());
    out.extend_from_slice(&m.scale.to_le_bytes());
    out.extend_from_slice(&(stored.len() as u64).to_le_bytes());
    for j in stored {
        out.extend_from_slice(&(j as u64).to_le_bytes());
        out.extend_from_slice(&m.xx[j].to_le_bytes());
        out.extend_from_slice(&m.xy[j].to_le_bytes());
    }
    out.extend_from_slice(&state.next_seq.to_le_bytes());
    out.extend_from_slice(&state.model_version.to_le_bytes());
    out.extend_from_slice(&state.horizon.to_le_bytes());
    out.extend_from_slice(&state.default_reward.to_le_bytes());
    out.extend_from_slice(&state.epsilon.to_le_bytes());
    out.extend_from_slice(&(state.ledger.len() as u64).to_le_bytes());
    for r in &state.ledger {
        put_string(&mut out, &r.decision_id);
        out.extend_from_slice(&r.t.to_le_bytes());
        out.extend_from_slice(&r.candidate_hash.to_le_bytes());
        out.extend_from_slice(&(r.chosen as u64).to_le_bytes());
        out.extend_from_slice(&(r.features.len() as u64).to_le_bytes());
        for &(j, v) in &r.features {
            out.extend_from_slice(&(j as u64).to_le_bytes());
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&r.propensity.to_le_bytes());
        out.extend_from_slice(&r.model_version.to_le_bytes());
        put_string(&mut out, &r.salt);
    }
    let checksum = fnv1a_64(&out);
    out.extend_from_slice(&checksum.to_le_bytes());
    out
}

/// A bounds-checked cursor over the payload (checksum already verified and
/// stripped). Reading past the end is "truncated state", never a panic, and
/// allocations are bounded by the input length because every element
/// consumes bytes as it is read.
struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], Error> {
        if self.data.len() - self.pos < n {
            return Err(Error::new("truncated state"));
        }
        let chunk = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(chunk)
    }

    fn u64(&mut self) -> Result<u64, Error> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn f64(&mut self) -> Result<f64, Error> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn string(&mut self) -> Result<String, Error> {
        let length = self.u64()?;
        let length =
            usize::try_from(length).map_err(|_| Error::new("truncated state"))?;
        let bytes = self.take(length)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|_| Error::new("state string is not valid UTF-8"))
    }
}

/// Parse and validate one serialized state.
pub fn deserialize(data: &[u8]) -> Result<BanditState, Error> {
    if data.len() < MAGIC.len() + 16 {
        return Err(Error::new("truncated state"));
    }
    if &data[..MAGIC.len()] != MAGIC {
        return Err(Error::new("not a gittins state (bad magic)"));
    }
    let (payload, trailer) = data.split_at(data.len() - 8);
    if u64::from_le_bytes(trailer.try_into().unwrap()) != fnv1a_64(payload) {
        return Err(Error::new("state checksum mismatch"));
    }
    let mut r = Reader {
        data: payload,
        pos: 0,
    };
    r.take(MAGIC.len())?;
    let version = r.u64()?;
    if version != FORMAT_VERSION {
        return Err(Error::new(format!(
            "unsupported state format version {version}"
        )));
    }

    let dim = r.u64()?;
    let ridge = r.f64()?;
    let forgetting = r.f64()?;
    let scale = r.f64()?;
    if dim < 1 {
        return Err(Error::new("dim must be at least 1"));
    }
    if dim > MAX_DIM as u64 {
        return Err(Error::new("dim must be at most 2**24"));
    }
    if !(0.0 < forgetting && forgetting <= 1.0) {
        return Err(Error::new("forgetting must be in (0, 1] (1.0 = never forget)"));
    }
    if !(ridge > 0.0) {
        return Err(Error::new("ridge must be positive"));
    }
    if !(0.0 < scale && scale <= 1.0) {
        return Err(Error::new("scale must be in (0, 1]"));
    }
    let dim = dim as usize; // at most MAX_DIM
    let mut xx = vec![0.0; dim];
    let mut xy = vec![0.0; dim];
    let mut prev: i64 = -1;
    let slot_count = r.u64()?;
    for _ in 0..slot_count {
        let j = r.u64()?;
        let u = r.f64()?;
        let w = r.f64()?;
        if j >= dim as u64 || j as i64 <= prev || is_unstored_slot(u, w) {
            return Err(Error::new(
                "model slots must be in strictly increasing index order \
                 within the model dimension, never +0.0 in both sums",
            ));
        }
        xx[j as usize] = u;
        xy[j as usize] = w;
        prev = j as i64;
    }

    let next_seq = r.u64()?;
    let model_version = r.u64()?;
    let horizon = r.f64()?;
    let default_reward = r.f64()?;
    let epsilon = r.f64()?;
    if !(horizon > 0.0) {
        return Err(Error::new("horizon must be positive"));
    }
    if !(0.0..=1.0).contains(&epsilon) {
        return Err(Error::new("epsilon must be in [0, 1]"));
    }

    let mut ledger = Vec::new();
    let record_count = r.u64()?;
    for _ in 0..record_count {
        let decision_id = r.string()?;
        let t = r.f64()?;
        let candidate_hash = r.u64()?;
        let chosen = r.u64()? as usize;
        let mut features = Vec::new();
        let mut prev: i64 = -1;
        let feature_count = r.u64()?;
        for _ in 0..feature_count {
            let j = r.u64()?;
            let v = r.f64()?;
            if j >= dim as u64 || j as i64 <= prev || v == 0.0 {
                return Err(Error::new(
                    "record features must be (index, value) pairs in strictly \
                     increasing index order within the model dimension, values nonzero",
                ));
            }
            features.push((j as usize, v));
            prev = j as i64;
        }
        let propensity = r.f64()?;
        let record_version = r.u64()?;
        let salt = r.string()?;
        ledger.push(DecisionRecord {
            decision_id,
            t,
            candidate_hash,
            chosen,
            features,
            propensity,
            model_version: record_version,
            salt,
        });
    }
    if r.pos != r.data.len() {
        return Err(Error::new("trailing bytes in state"));
    }

    Ok(BanditState {
        model: LinearModel {
            dim,
            ridge,
            forgetting,
            scale,
            xx,
            xy,
        },
        next_seq,
        model_version,
        horizon,
        default_reward,
        epsilon,
        ledger,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decide::{decide, new_bandit};
    use crate::exploration::DEFAULT_EPSILON;
    use crate::model::DEFAULT_FORGETTING;

    fn worked_state() -> BanditState {
        let mut state =
            new_bandit(16, 120.0, -0.25, 0.1, DEFAULT_FORGETTING).unwrap();
        let candidates = vec![vec![(0, 1.0), (3, -0.5)], vec![(1, 2.0)]];
        for i in 0..3 {
            let record = decide(&mut state, &candidates, 1000.0 + i as f64, "salty", None, None).unwrap();
            if i == 0 {
                crate::ledger::learn(&mut state, &record.decision_id, 1.0, 1000.0 + i as f64).unwrap();
            }
        }
        state
    }

    fn resealed(mut payload: Vec<u8>) -> Vec<u8> {
        let checksum = fnv1a_64(&payload);
        payload.extend_from_slice(&checksum.to_le_bytes());
        payload
    }

    /// Where the model's slot list starts: after magic, version, dim,
    /// ridge, forgetting and scale.
    const SLOTS_AT: usize = 8 * 6;

    /// A fresh dim-4 state's bytes with its (empty) slot list replaced by
    /// `entries`, written as given.
    fn with_slot_list(entries: &[(u64, f64, f64)]) -> Vec<u8> {
        let fresh = new_bandit(4, 1.0, 0.0, DEFAULT_EPSILON, DEFAULT_FORGETTING).unwrap();
        let data = serialize(&fresh);
        let mut out = data[..SLOTS_AT].to_vec();
        out.extend_from_slice(&(entries.len() as u64).to_le_bytes());
        for &(j, u, w) in entries {
            out.extend_from_slice(&j.to_le_bytes());
            out.extend_from_slice(&u.to_le_bytes());
            out.extend_from_slice(&w.to_le_bytes());
        }
        out.extend_from_slice(&data[SLOTS_AT + 8..data.len() - 8]);
        resealed(out)
    }

    #[test]
    fn size_tracks_learned_slots_not_dim() {
        let small = new_bandit(1 << 4, 1.0, 0.0, DEFAULT_EPSILON, DEFAULT_FORGETTING).unwrap();
        let mut large = new_bandit(1 << 20, 1.0, 0.0, DEFAULT_EPSILON, DEFAULT_FORGETTING).unwrap();
        assert!(serialize(&small).len() == serialize(&large).len());
        let empty = serialize(&large).len();
        large.model.xx[7] = 1.0;
        large.model.xy[1 << 19] = -1.0;
        assert!(serialize(&large).len() == empty + 2 * 24);
    }

    #[test]
    fn negative_zero_is_stored_and_round_trips() {
        let mut state = new_bandit(4, 1.0, 0.0, DEFAULT_EPSILON, DEFAULT_FORGETTING).unwrap();
        let empty = serialize(&state).len();
        state.model.xy[1] = -0.0;
        state.model.xx[2] = -0.0;
        let data = serialize(&state);
        assert!(data.len() == empty + 2 * 24);
        let loaded = deserialize(&data).unwrap();
        assert!(loaded.model.xy[1].to_bits() == (-0.0f64).to_bits());
        assert!(loaded.model.xx[2].to_bits() == (-0.0f64).to_bits());
        assert!(serialize(&loaded) == data);
    }

    #[test]
    fn rejects_version_1() {
        let mut data = serialize(&worked_state());
        data.truncate(data.len() - 8);
        data[8..16].copy_from_slice(&1u64.to_le_bytes());
        let error = deserialize(&resealed(data)).unwrap_err();
        assert!(error.message() == "unsupported state format version 1");
    }

    #[test]
    fn rejects_non_canonical_slot_lists() {
        assert!(deserialize(&with_slot_list(&[(1, 1.0, 1.0)])).is_ok());
        for entries in [
            &[(1, 0.0, 0.0)][..],             // +0.0 in both: must be left out
            &[(2, 1.0, 1.0), (1, 1.0, 1.0)], // decreasing
            &[(1, 1.0, 1.0), (1, 2.0, 2.0)], // repeated
            &[(4, 1.0, 1.0)],                // outside dim 4
        ] {
            let error = deserialize(&with_slot_list(entries)).unwrap_err();
            assert!(error.message().contains("slots"), "{entries:?}: {error}");
        }
    }

    #[test]
    fn rejects_dim_above_the_cap_before_allocating() {
        let fresh = new_bandit(1, 1.0, 0.0, DEFAULT_EPSILON, DEFAULT_FORGETTING).unwrap();
        let mut data = serialize(&fresh);
        data.truncate(data.len() - 8);
        data[16..24].copy_from_slice(&(1u64 << 62).to_le_bytes());
        assert!(deserialize(&resealed(data)).unwrap_err().message().contains("dim"));
    }

    #[test]
    fn round_trips_bit_for_bit() {
        let state = worked_state();
        let data = serialize(&state);
        assert!(deserialize(&data).unwrap() == state);
        assert!(serialize(&deserialize(&data).unwrap()) == data);
    }

    #[test]
    fn rejects_malformed_input() {
        let good = serialize(&worked_state());

        let mut bad_magic = good.clone();
        bad_magic[0] ^= 0xFF;
        assert!(deserialize(&bad_magic).is_err());

        for cut in [0, 7, 23, good.len() / 2, good.len() - 1] {
            assert!(deserialize(&good[..cut]).is_err(), "accepted truncation at {cut}");
        }

        let mut corrupted = good.clone();
        corrupted[40] ^= 0x01;
        assert!(
            deserialize(&corrupted).unwrap_err().message() == "state checksum mismatch"
        );

        // A bumped version with a valid checksum: the version check itself
        // must reject it, not the checksum.
        let mut versioned = good[..good.len() - 8].to_vec();
        versioned[8..16].copy_from_slice(&(FORMAT_VERSION + 1).to_le_bytes());
        let checksum = fnv1a_64(&versioned);
        versioned.extend_from_slice(&checksum.to_le_bytes());
        assert!(deserialize(&versioned)
            .unwrap_err()
            .message()
            .contains("version"));

        // Trailing junk between payload and a recomputed checksum.
        let mut padded = good[..good.len() - 8].to_vec();
        padded.push(0);
        let checksum = fnv1a_64(&padded);
        padded.extend_from_slice(&checksum.to_le_bytes());
        assert!(deserialize(&padded).is_err());
    }

    #[test]
    fn revalidates_constructor_invariants() {
        // Field offsets: dim at 16, ridge 24, forgetting 32, scale 40.
        let fresh = new_bandit(1, 1.0, 0.0, DEFAULT_EPSILON, DEFAULT_FORGETTING).unwrap();
        for (offset, bad, message) in [
            (16, 0u64.to_le_bytes(), "dim"),
            (24, 0.0f64.to_le_bytes(), "ridge"),
            (32, 0.0f64.to_le_bytes(), "forgetting"),
            (40, 2.0f64.to_le_bytes(), "scale"),
        ] {
            let mut data = serialize(&fresh);
            data.truncate(data.len() - 8);
            data[offset..offset + 8].copy_from_slice(&bad);
            let checksum = fnv1a_64(&data);
            data.extend_from_slice(&checksum.to_le_bytes());
            let error = deserialize(&data).unwrap_err();
            assert!(error.message().contains(message), "wanted {message}: {error}");
        }
    }

    #[test]
    fn revalidates_record_features() {
        // One decision at dim 4 with salt "s" puts the record's first
        // feature index at a fixed offset: 6 u64 header/model scalars +
        // an empty slot list (nothing learned yet) + 5 bandit scalars +
        // ledger count + id (8 + 3 bytes) + t + hash + chosen + feature count.
        let mut state = new_bandit(4, 1.0, 0.0, DEFAULT_EPSILON, DEFAULT_FORGETTING).unwrap();
        decide(&mut state, &[vec![(0, 1.0)], vec![(1, 1.0)]], 0.0, "s", None, None).unwrap();
        let offset = 8 * 6 + 8 + 8 * 5 + 8 + (8 + 3) + 8 * 4;
        let mut data = serialize(&state);
        data.truncate(data.len() - 8);
        data[offset..offset + 8].copy_from_slice(&9u64.to_le_bytes()); // 9 >= dim
        let checksum = fnv1a_64(&data);
        data.extend_from_slice(&checksum.to_le_bytes());
        assert!(deserialize(&data)
            .unwrap_err()
            .message()
            .contains("features"));
    }
}
