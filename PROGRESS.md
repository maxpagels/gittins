# gittins — progress and roadmap

This file is the running record of what has been built, what comes next, and the
conventions the work follows. It is updated in every PR. The full design document
(`implementation-plan.md`) is kept out of git deliberately; section references below
point into it for readers who have it locally.

## What this project is

`gittins` is a set-and-forget contextual bandit engine: a zero-dependency, deterministic
core (eventually Rust, with Python/JS bindings) where all state is an explicit value,
decisions are first-class logged records, and non-stationarity is handled by a
self-tuning pool of models instead of tunable knobs. "The SQLite of bandits."

**Current phase: Phase 0 — Specification & pure-Python reference implementation.**
Every core concept is first written as small, readable, dependency-free Python with
tests. The later Rust core must mirror this reference one-to-one and match it
bit-for-bit against golden test vectors.

## Working process

- One concept per PR, roughly 100–300 lines including tests, reviewable in one sitting.
- Each PR description states: the concept, the design-doc section it implements, and
  what the tests prove.
- If a PR is too big to explain, it gets split — that's a feature of the process.
- PRs that introduce semantics (decay, ledger, merge) also add a written spec section
  under `spec/`.
- The reference implementation stays pure Python with zero runtime dependencies
  (pytest is dev-only).

## PR roadmap (Phase 0)

| # | Branch | Concept | Design ref | Status |
|---|--------|---------|-----------|--------|
| 1 | `pr-01-bootstrap` | Repo + Python skeleton, pytest, this file | §12 Phase 0 | Merged |
| 2 | `pr-02` | Counter-based deterministic RNG keyed by decision ID + salt | §8 | Merged |
| 3 | `pr-03` | Decaying sums (decay-on-read) + deterministic exp2 | D3, §8 | Merged |
| 4 | `pr-04` | Incremental ridge regression on decaying sums, predict with uncertainty | D3 | Merged |
| 5 | `pr-05` | Inverse-gap weighting (SquareCB) + probability floor | §5 | Merged |
| 6 | `pr-06` | Decision records; `decide(state, candidates, t, salt)` | D1, D5 | Merged |
| 7 | `pr-07` | Decision ledger; `learn()`; rewarded/expired/censored; late-reward weighting | §6 | Merged |
| 8 | `pr-08` | Feature encoding: everything hashed into 2^bits dims (features, identity, interactions) | D2 | Merged |
| 9 | `pr-09` | Timestamp-aligned state merge; commutativity property tests | D3, §13 risk 3 | Merged |
| 10 | `pr-10` | Golden test vector corpus generated from the reference; GitHub Actions CI | §8 | **In review** |

Phase 0 exit criterion: the spec plus reference is complete enough that an independent
implementation of `decide`/`learn` can match the golden vectors.

## Repository layout

```
src/gittins_reference/   pure-Python reference implementation (Phase 0)
tests/                   pytest suite for the reference
spec/                    written spec sections, grown PR by PR
PROGRESS.md              this file
```

Planned later: `core/` (Rust), `bindings/` (Python native, JS/WASM), `sim/` (simulation
harness).

## Decisions log

- **2026-07-14** — Decay uses the decay-on-read formulation: sums stored pre-scaled to a
  movable origin, decayed only when read, renormalized every 128 half-lives. Chosen over
  decay-on-write because merge commutativity and late-reward correctness become structural
  (design doc §13 risk 3 reduces to a rounding statement, tested in `tests/test_decay.py`).
- **2026-07-14** — Arm identity is hashed into a fixed block of dimensions in the shared
  feature space (design doc D2, v0.4) instead of a separate per-arm correction map. Decay
  is the only cleanup mechanism; no eviction code exists; state memory is fixed under arm
  churn. PR 8 repurposed from per-arm map to feature encoding.

- **2026-07-15** — Feature encoding is fully hashed (design doc open question 2,
  resolved *against* its explicit-schema lean): every feature, categorical value, arm
  identity, the intercept, and every interaction term hashes into one 2^bits space —
  the single up-front declaration is `bits`. One integer replaces schema declaration,
  migration, and unknown-name policy; feature sets vary freely per decision; identity
  stops being a special case (it's one more token). Accepted cost: typo'd names hash
  silently instead of erroring. The encoding stays the outer product
  ([bias]+context) ⊗ ([bias]+action+identity) because with a linear model context-only
  dimensions can never reorder candidates — interactions are the encoding's job (the
  VW `-q SA` move). Hashes use tokens only (no salt), keeping fleets merge-aligned;
  signed hashing makes collisions blur rather than bias.
- **2026-07-14** — `implementation-plan.md` is git-ignored; PROGRESS.md is the in-repo
  source of truth for the roadmap.
- **2026-07-14** — Reference implementation lives at `src/gittins_reference/`, managed
  with `uv`, requires Python ≥3.10, zero runtime dependencies.

## Done

- **PR 1** (2026-07-14) — repo bootstrap: uv-managed Python skeleton, pytest, this file.
- **PR 2** (2026-07-14) — `rng.py`: splitmix64-based counter RNG; FNV-1a key derivation
  from (decision ID, salt); exact-float `random_unit`. First golden vectors pinned in
  `tests/test_rng.py` and `spec/rng.md`.

- **PR 3** (2026-07-14) — `decay.py`: DecayedAccumulator with decay-on-read (contributions
  stored pre-scaled to a per-accumulator origin; renormalization at 128 half-lives), plus
  `detmath.py`: vendored deterministic exp2 (pinned Taylor coefficients, fixed evaluation
  order). Merge commutativity is bit-exact by construction; the exactness contract and
  the "why decay" rationale live in `spec/decay.md`.

- **PR 4** (2026-07-15) — `model.py`: LinearModel, ridge regression on decaying sums.
  State is two DecayedVectors (xx outer-product sums, xy reward-weighted sums; the
  DecayedVector many-sums-one-clock extension was added to `decay.py`). Predict solves
  via fixed-order Cholesky, returns (estimate, uncertainty = sqrt(x·A⁻¹x)); ridge=1.0
  fixed default, not a knob. Uncertainty floor / regrowth during gaps falls out of
  decay — tested. Merge inherits decay-layer exactness. Spec: `spec/model.md`.

- **PR 5** (2026-07-15) — `exploration.py`: inverse-gap weighting (SquareCB) over per-
  candidate reward estimates, first-max tie-breaking, best gets the exact complement
  (sums to 1.0 in float). Probability floor as uniform mixing with fixed
  FLOOR_MASS = 0.05 (never converges, IPS weights bounded by k/0.05); order-preserving.
  Inverse-CDF sampling from the counter RNG; `choose()` returns (index, propensity)
  ready for PR 6's decision records. `gamma` stays a parameter of this pure layer —
  supplying/self-tuning it belongs to the decide layer (D4). Spec: `spec/exploration.md`.

- **PR 6** (2026-07-15) — `decide.py`: BanditState (model + decision counter +
  model_version) and DecisionRecord (id, t, candidate-set hash, chosen index, chosen
  features, propensity, model_version, salt — self-contained for PR 7's `learn`).
  Decision IDs are `"{salt}:{seq}"`: uniqueness structural (per-agent salt + monotone
  counter), no collision math. Gamma schedule: GAMMA_SCALE / mean(candidate
  uncertainty) — fresh model ≈ uniform, gamma grows like sqrt(n) as the posterior
  tightens (SquareCB's schedule from the model's own uncertainty, no clock, no knob),
  bounded because decay bounds effective n; constant provisional until the Phase 2
  battery. RNG counter 0 reserved for the sampling draw. Candidates are pre-encoded
  feature vectors; context folding arrives with PR 8's encoder. Spec: `spec/decide.md`.

- **PR 7** (2026-07-15) — `ledger.py`: the state's ledger of open decision records
  (BanditState gains ledger + the once-up-front declaration horizon/default_reward;
  decide appends its record). Three resolutions, each returning a loggable Resolution
  event: `learn` (rewarded — trains at the *decision's* timestamp, so late/out-of-order
  rewards are bit-identical to on-time ones), `expire(t)` (expired(default) for every
  decision past the horizon, swept with each event's time — the sweep is the ledger
  bound), `censor` (removed without training, exclusion on record). Idempotency is
  structural: resolving spends the one record, so duplicate/conflicting/bogus reports
  and post-expiry rewards are no-ops. No code path learns from an open decision.
  Spec: `spec/ledger.md`.

- **PR 8** (2026-07-15) — `encoding.py`: fully hashed feature encoding. Dicts in,
  2^bits-dim vector out; `bits` is the only declaration. Tokens: string value →
  `ns|name=value` ×1.0, numeric → `ns|name` ×value, None absent, arm identity one
  more token `i|id`. Encoding = hashed outer product ([bias]+context) ⊗
  ([bias]+action+identity): pair token → mix64∘fnv1a → slot (masked) + sign (bit 63);
  collisions add (signed, so they blur not bias); sorted token order makes bits
  independent of dict order. Everything open-world: new names/values/arms need no
  registration, decay recycles dead dimensions. End-to-end personalization test
  through decide + ledger at bits=5, collisions and all. Spec: `spec/encoding.md`.

- **PR 9** (2026-07-15) — `merge.py`: `merge_states(a, b)` pools knowledge (origin-aligned
  model merge, model_version summed) but not operational identity: the ledger does not
  merge (open decisions stay with the agent that will receive their rewards; merging
  them would expire unresolvable copies and double-count on resolution) and next_seq
  resets to 0 (seq belongs to a (salt, agent) pair; a merged state that decides needs a
  fresh salt). Requires equal horizon/default_reward (+ dim/ridge/half-life below).
  Fleet flow is recompute-not-accumulate: shared file rebuilt from agent files each run,
  keeping evidence disjoint. Property tests (§13 risk 3): bit-exact commutativity;
  resolve-then-merge == merge-then-resolve to <1e-12 across half-lives (incl. inf) and
  across a renorm era (empirically exact); 3-agent merge == central model <1e-12;
  associativity <1e-12. Measured: merging evidence >~53 half-lives apart absorbs the
  older side to nothing — exactly decay's "fully forgotten". Spec: `spec/merge.md`.

## Currently in flight

- **PR 10** (`pr-10`) — `golden.py` + `spec/golden.json`: the golden test vector corpus
  generated from the reference (design doc §8). Sections per layer (rng, exp2, decay,
  model, exploration, encoding) plus an end-to-end `episode`: two agents, hashed
  encoding, out-of-order rewards, censor, exact-horizon expiry sweep, merge — every
  record and resolution logged; matching it bit-for-bit is the Phase 0 exit test for an
  independent decide/learn. `tests/test_golden.py` pins the checked-in file to exact
  regeneration, so semantic drift fails CI as a reviewable vector diff. Also adds
  GitHub Actions CI (`.github/workflows/ci.yml`): full pytest on PRs and main pushes
  on Ubuntu × Python 3.10–3.14 — with bit-pinned tests every leg checks bit-identity
  (R6); macOS/Windows legs deferred for cost, to return as a release gate with the
  compiled core. `.gitattributes` forces LF so the corpus compares as exact text
  everywhere. Spec: `spec/golden.md`.
