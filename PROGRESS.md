# gittins — progress and roadmap

The running record of what has been built, what comes next, and the conventions
the work follows; updated in every PR. The full design document
(`implementation-plan.md`) is kept out of git deliberately; section references
point into it for readers who have it locally. Details of completed work are
kept to one entry per PR — this file's own git history has the long versions.

## What this project is

`gittins` is a set-and-forget contextual bandit engine: a zero-dependency,
deterministic core (Rust, with Python/JS bindings) where all state is an
explicit value, decisions are first-class logged records, and
non-stationarity is handled by built-in forgetting with fixed defaults — tuning
lives offline, in replay and off-policy evaluation over the decision log, never
in knobs. "The SQLite of bandits."

**Current phase: Phase 2 — bindings (see Roadmap).** Phase 0 (the
pure-Python reference, which remains the specification), Phase 0.5 (the
benchmarking battery, `sim/`), and Phase 1 (the Rust core) are done. The
core matches the reference bit for bit against the golden vector corpus
(`spec/golden.json`, checked by `cargo test`) and number for number across
the 450k-decision battery.

## Working process

- One concept per PR, roughly 100–300 lines including tests, reviewable in one
  sitting; too big to explain means split.
- PRs that introduce semantics also add a spec section under `spec/`; semantic
  changes regenerate `spec/golden.json` and the vector diff is reviewed like
  code.
- The reference and `sim/` stay pure Python with zero runtime dependencies
  (pytest is dev-only); the Rust core crate stays zero-dependency too —
  binding libraries (PyO3, wasm-bindgen) live in their own crates.

## The engine in one paragraph

Candidates are hashed sparse feature pairs (`encoding.py`: dicts in, (index,
value) pairs out, `bits` the only declaration). The model (`model.py`) is
per-coordinate ridge regression with per-update exponential forgetting, stored
as pre-scaled sums so updates are O(nonzeros). Exploration (`exploration.py`)
is epsilon-greedy with exact-tie splitting (`DEFAULT_EPSILON = 0.05`). `decide`
(`decide.py`) returns a self-contained decision record (id, candidate-set
hash, chosen features, propensity); training happens only by resolving a
record through the ledger (`ledger.py`: rewarded / expired-at-horizon /
censored, exactly-once by construction). Everything is fixed-order IEEE-754
over a counter RNG, bit-identical across platforms. The Rust core (`core/`)
is the same engine module for module, mutated in place.

## Benchmarking harness (Phase 0.5, complete)

`sim/` drives every comparator through the real public path (encode → decide →
ledger) on seeded, paired, exactly-replayable runs. `python -m sim` prints the
environment battery — 10 worlds (linear k5/k20, xor k4/k10, needle,
action-features, shift, drift, churn, dropout) × oracle / uniform / greedy /
epsilon-greedy(0.05, 0.1) / gittins, 1500 rounds × 5 seeds — as a markdown
table with normalized regret (0 = oracle, 1 = uniform), final-window regret,
late-run RMSE, and total decisions/s; `cargo run --release --bin sim` is the
same battery on the Rust core, and CI appends both tables to the job summary
for line-by-line comparison — a diagnostic, not a gate. `python -m sim.sweep`
(by hand) settles engine constants: per-environment bests, pass criteria
(within 1.1x of best on ≥90% of cells, never >2x, beats the epsilon-0.1
baseline overall).

Known limits, recorded: the xor cells are chaotic across RNG salts (regret
0.11–1.03 on identical worlds), so xor comparisons at 5 seeds are noise; the
reward-noise gaussian (Box–Muller over libm log/cos) is the one piece not
bit-pinned *across platforms* — on any one machine Python and Rust call the
same libm, which is why the two tables match exactly there.

Runtime budget: prediction is O(nonzeros) per candidate with lazy per-decision
weight solving. Pure Python: dim 256 × 100 arms ≈ 0.85 ms/decision;
dim 16,384 × 10,000 arms ≈ 87 ms (~82% of it `candidate_set_hash`'s per-byte
FNV). Battery: 450k decisions in ~22s Python (~20k decisions/s), ~2.7s Rust
(~169k decisions/s), with no perf work done on the core yet.

## Repository layout

```
src/gittins_reference/   pure-Python reference implementation (Phase 0)
sim/                     simulation harness (Phase 0.5)
core/                    Rust core (Phase 1): the engine, the golden
                         verifier (cargo test), and the battery rerun
                         (cargo run --release --bin sim)
tests/                   pytest suite for reference + sim
spec/                    written spec sections + golden.json
docs/                    user-facing documentation (usage.md: the public API
                         by example)
PROGRESS.md              this file
```

Planned: `bindings/python` (PyO3), `bindings/wasm` (wasm-bindgen).

## Roadmap (Phase 2 — bindings)

The rule that governs the phase: the dict-shaped public API is specified
once, in the reference, and every binding mirrors it exactly; each binding's
CI gate is the golden episode replayed through the binding, bit for bit.

- **PR 20 — Python binding (`bindings/python`: PyO3 + maturin, abi3).** The
  design rule that decides whether it is fast: one boundary crossing per
  decision — `decide()` takes the context dict and the whole candidate list,
  and encode → score → sample all happen inside Rust; no per-candidate or
  per-feature calls. Golden episode through the binding in pytest; a
  decisions/s benchmark in the CI summary next to the battery tables so
  boundary regressions are visible. Target: dim 256 × 100 arms from 0.85 ms
  (pure Python) to the 10–50 µs range.
- **PR 21 — browser (`bindings/wasm`: wasm-bindgen → npm).** The engine core
  uses only IEEE-exact operations (add/mul/div/sqrt — no libm transcendentals),
  and WASM mandates IEEE-754 semantics, so the corpus should pass in wasm by
  construction; `wasm-pack test --node` in CI becomes the first
  cross-platform bit-identity gate. State serialized to a string makes a
  localStorage-persisted browser bandit the natural demo of R8's ephemeral
  mode.
- **PR 22 — cross-OS CI + packaging.** macOS/Windows legs (the release gate
  the golden spec defers), wheels via maturin-action, npm packaging.

Also outstanding, order flexible:

- the `DEFAULT_EPSILON` sweep (`python -m sim.sweep`) — the 0.05 default is
  provisional until it is run and recorded here (from PR 16);
- the encoding hot spot (per-decision token formatting + pair hashing), worth
  taking only once the binding benchmark exists to price it — the hash
  contract itself stays frozen;
- further out: multislot/ranking (R5) and OPE tooling over the decision
  log (R3).

## Decisions log

- **2026-07-14** — `implementation-plan.md` is git-ignored; PROGRESS.md is the
  in-repo source of truth. Reference lives at `src/gittins_reference/`, managed
  with `uv`, Python ≥3.10, zero runtime dependencies.
- **2026-07-14** — Arm identity hashes into the shared feature space (one more
  token) instead of a per-arm correction map; forgetting is the only cleanup;
  state memory is fixed under arm churn.
- **2026-07-15** — Feature encoding is fully hashed (resolved *against* the
  design doc's explicit-schema lean): every feature, categorical value, arm
  identity, intercept, and interaction hashes into one 2^bits space; `bits` is
  the single declaration. Encoding is the outer product
  ([bias]+context) ⊗ ([bias]+action+identity) because with a linear model
  context-only dimensions can never reorder candidates. No salt in the hashes,
  so fleets encode identically; signed hashing makes collisions blur, not
  bias. Accepted cost: typo'd names hash silently.
- **2026-07-15** — The built-in model keeps only the outer-product matrix's
  *diagonal*: `θ_j = xy_j / (xx_j + ridge)`, per-coordinate uncertainty, no
  linear algebra anywhere (the VW hashed-linear operating point). Known cost,
  pinned in tests: credit is never split — co-firing features double-count;
  disentangling combinations is the encoder's job (interactions).
- **2026-07-15** — The synthetic regret battery is pulled forward to Phase 0.5,
  before the Rust core, so provisional constants get settled while changing
  them is a Python edit plus a golden regeneration.
- **2026-07-16** — Wall-clock decay and state merging are removed (supersedes
  the original decay-on-read decision and PR 9's merge). The model forgets per
  *update* (forgetting factor 0.999 default; an expert override, not a knob);
  late rewards train in arrival order at slightly less weight. Fleet pooling
  is no longer a state merge: agents collect decision logs, logs merge into
  one experience log, and models/configurations are selected offline by
  replay + OPE over it. Chosen after PR 15's experiments: no single forgetting
  rate is right everywhere, and every online self-tuning rule tested had an
  unresolved scoring problem — the tuning question moves offline rather than
  into the engine's semantics.
- **2026-07-16** — Exploration is epsilon-greedy (`DEFAULT_EPSILON = 0.05`,
  exact ties split the greedy mass), replacing SquareCB + floor + gamma
  schedule. Rationale: the permanent exploration floor R2 demands already
  forfeits SquareCB's regret guarantee, so the rules differ only in transient
  exploration — at very different prices (per-candidate uncertainties, a
  distribution build, and a swept schedule constant vs one max-scan).
- **2026-07-16** — The reference is sparse end to end: candidates are (index,
  value) pairs, the model stores pre-scaled sums (O(nonzeros) updates,
  renormalize at scale ≤ 2^-512), weights solve lazily per decision. The
  golden corpus now pins the sparse semantics the compiled core must match.
  The candidate-set hash byte layout is unchanged (zero `candidate_hash`
  values moved); the pre-scaled bookkeeping is the one semantics change
  (last-bit golden drift, no battery decision flipped).
- **2026-07-16** — The Rust core is a bit-exact port, not a reimplementation:
  every float expression keeps the reference's evaluation order, the crate
  has zero dependencies (the golden JSON parser lives in the test tree), and
  state is mutated in place. The reference carries the semantic/property test
  suite once; the core's own tests are the golden corpus plus unit tests for
  exactly what the corpus excludes (never-forget, exactly-once ledger no-ops,
  scale renormalization).

## Done

One line of history per PR; findings that shaped decisions live in the
decisions log above, and this file's git history has the full entries.

- **PR 1–2** (2026-07-14) — repo bootstrap; `rng.py`: splitmix64 counter RNG
  keyed by FNV-1a over (decision id, salt). Spec: `rng.md`.
- **PR 3–4** (2026-07-14/15) — wall-clock decaying sums + ridge regression.
  *Removed in PR 15*; the decay-on-read idea returned, update-clocked, in
  PR 16's pre-scaled sums.
- **PR 5** (2026-07-15) — SquareCB exploration with a uniform floor.
  *Replaced by epsilon-greedy in PR 16*; the inverse-CDF sampling and
  propensity contract survive.
- **PR 6** (2026-07-15) — `decide.py`: BanditState + self-contained
  DecisionRecord; ids `"{salt}:{seq}"`. Spec: `decide.md`.
- **PR 7** (2026-07-15) — `ledger.py`: exactly-once resolution as rewarded /
  expired / censored. Spec: `ledger.md`.
- **PR 8** (2026-07-15) — `encoding.py`: fully hashed feature encoding.
  Spec: `encoding.md`.
- **PR 9** (2026-07-15) — bit-exact state merging. *Removed in PR 15* (fleet
  pooling moved offline).
- **PR 10** (2026-07-15) — the golden corpus (`golden.py`, `spec/golden.json`)
  pinned to exact regeneration; GitHub Actions CI. Spec: `golden.md`.
- **PR 11** (2026-07-16) — `sim/` harness: runner, metrics, stationary
  environments, comparators; `candidate_set_hash` respecced to O(nonzeros).
- **PR 12** (2026-07-16) — non-stationary battery tranche (shift, drift,
  churn, dropout). Headline: churn traps greedy below uniform; explorers
  recover.
- **PR 13** (2026-07-16) — event-time simulation (Poisson traffic, reward
  delays). Findings: next-morning delays put greedy below uniform; the ledger
  holds a full day open, bounded by the horizon as claimed; exp-45m delays
  with a 2h horizon expire ~7% of decisions. *Removed in PR 16*, findings
  recorded; `git log` has the code.
- **PR 14** (2026-07-16) — `sim/sweep.py` sweep driver + pass criteria;
  settled `GAMMA_SCALE = 300` (deleted again in PR 16 with SquareCB itself).
- **PR 15** (2026-07-16) — forgetting evidence (no single rate fits all
  worlds; online self-tuning unresolved) forced the simplification in the two
  2026-07-16 decisions: decay/exp2/merge deleted, per-update forgetting
  shipped. Corpus at format_version 2.
- **PR 16** (2026-07-16) — epsilon-greedy exploration + the sparse reference +
  battery slimming (event-time sim deleted). A/B vs SquareCB@300: at or ahead
  on 9/12 cells, behind on churn and whole-run needle (better final-10%);
  dim 256 × 100 arms 1.32 → 0.85 ms/decision; one less engine constant.

- **PR 17** (2026-07-16) — the Rust core (`core/`), Phase 1: the engine
  ported module for module (rng, encoding, model, exploration, decide,
  ledger), zero dependencies, per the 2026-07-16 bit-exact-port decision.
  `cargo test` verifies every golden section bit for bit and replays the
  end-to-end episode — all passed on the first run, meeting the Phase 0 exit
  criterion. The corpus gained the episode's `rejected` field (additive
  diff): no-op resolution attempts every implementation must reject to match
  the pinned final state. The battery reruns on the core
  (`cargo run --release --bin sim`) into the same CI summary as the Python
  battery: all 60 table rows identical locally, 22s → 2.7s (~8x, untuned).
  Also: README requirements gained R0 (incremental/online).

- **PR 18** (2026-07-16) — canonical state serialization + the core's error
  surface, the prerequisite for both bindings. `serialize`/`deserialize`
  (`state.py`, `state.rs`; spec: `serialization.md`) are exact inverses over
  one canonical little-endian layout — magic, format version 1, model,
  bandit scalars, open ledger records, trailing FNV-1a checksum.
  Deserialization is all-or-nothing and re-checks every constructor
  invariant, so a loaded state is as trustworthy as a constructed one. The
  corpus gained a `serialization` section (fresh state, episode mid
  snapshot with open records, episode final state, as hex) — the Rust core
  produced the reference's bytes exactly. Every ValueError-shaped rejection
  in the core became a returned `Error` with the same message (`error.rs`);
  nothing on the public path panics, so bindings sit directly on it.

## Currently in flight

- **PR 19** (`pr-19`) — the public dict-shaped API (`api.py` in the
  reference, `api.rs` in the core; spec: `api.md`): the complete binding
  surface, specified once. Eight names — `create` (declares `bits` in place
  of a raw dimension; the model spans the 2^bits hashed space), `decide`
  (context dict plus (arm_id, action dict) candidates, encoded inside in
  candidate order), `learn`/`censor`/`expire`/`serialize`/`deserialize`
  passed through unchanged, and `model_bits` (the declaration recovered
  from the dimension — no new state type, no serialization change; states
  whose dimension is not 2^bits in [1, 24] are rejected). The facade adds
  no randomness, reordering, or arithmetic: both test suites pin
  facade-vs-layered bit identity. The corpus gained an `api` section
  (additive): one compact scenario through the public surface only —
  action-feature candidates (numeric + boolean, which the episode never
  exercised), out-of-order rewards, censor, exact-horizon expiry, final
  state hex — the acceptance gate PR 20/21 bindings replay through their
  own public API. The Rust facade reproduced it on the first run.
