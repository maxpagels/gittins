# gittins — progress and roadmap

The running record of what has been built, what comes next, and the conventions
the work follows; updated in every PR. The full design document
(`implementation-plan.md`) is kept out of git deliberately; section references
point into it for readers who have it locally.

## What this project is

`gittins` is a set-and-forget contextual bandit engine: a zero-dependency,
deterministic core (eventually Rust, with Python/JS bindings) where all state is
an explicit value, decisions are first-class logged records, and
non-stationarity is handled by built-in forgetting with fixed defaults — tuning
lives offline, in replay and off-policy evaluation over the decision log, never
in knobs. "The SQLite of bandits."

**Current phase: Phase 1 — the Rust core.** Phase 0 (specification &
pure-Python reference) and Phase 0.5 (the benchmarking battery, `sim/`) are
complete; the reference remains the specification, and the Rust core in
`core/` must match it bit-for-bit against the golden vector corpus
(`spec/golden.json`) — which it does, `cargo test` being the proof.

## Working process

- One concept per PR, roughly 100–300 lines including tests, reviewable in one
  sitting; too big to explain means split.
- PRs that introduce semantics also add a spec section under `spec/`; semantic
  changes regenerate `spec/golden.json` and the vector diff is reviewed like
  code.
- The reference and `sim/` stay pure Python with zero runtime dependencies
  (pytest is dev-only).

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
over a counter RNG, bit-identical across platforms.

## Benchmarking harness (Phase 0.5, complete)

`sim/` drives every comparator through the real public path (encode → decide →
ledger) on seeded, paired, exactly-replayable runs. `python -m sim` prints the
environment battery — 10 worlds (linear k5/k20, xor k4/k10, needle,
action-features, shift, drift, churn, dropout) × oracle / uniform / greedy /
epsilon-greedy(0.05, 0.1) / gittins, 1500 rounds × 5 seeds — as a markdown
table with normalized regret (0 = oracle, 1 = uniform), final-window regret,
late-run RMSE, and total decisions/s; CI appends it to the job summary as a
diagnostic, not a gate. `python -m sim.sweep` (by hand) settles engine
constants: per-environment bests, pass criteria (within 1.1x of best on ≥90%
of cells, never >2x, beats the epsilon-0.1 baseline overall). The event-time
simulation of PR 13 (time-ordered event heap, daily non-homogeneous traffic,
reward-delay models) was deleted in PR 16 after its findings were recorded;
`git log` has the code if delayed-reward studies return.

Known limits, recorded: the xor cells are chaotic across RNG salts (regret
0.11–1.03 on identical worlds), so xor comparisons at 5 seeds are noise; the
reward-noise gaussian (Box–Muller over libm log/cos) is the one non-bit-pinned
piece if the battery is ever wanted as a cross-language differential test.

Runtime budget: prediction is O(nonzeros) per candidate with lazy per-decision
weight solving. Pure Python: dim 256 × 100 arms ≈ 0.85 ms/decision;
dim 16,384 × 10,000 arms ≈ 87 ms (~82% of it `candidate_set_hash`'s per-byte
FNV). Battery: 450k decisions in ~22s (~20k decisions/s).

## Repository layout

```
src/gittins_reference/   pure-Python reference implementation (Phase 0)
sim/                     simulation harness (Phase 0.5)
core/                    Rust core (Phase 1): the engine, the golden
                         verifier (cargo test), and the battery rerun
                         (cargo run --release --bin sim)
tests/                   pytest suite for reference + sim
spec/                    written spec sections + golden.json
PROGRESS.md              this file
```

Planned later: `bindings/` (Python native, JS/WASM).

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
  distribution build, and a swept schedule constant vs one max-scan). Evidence
  in the PR 16 entry.
- **2026-07-16** — The reference is sparse end to end: candidates are (index,
  value) pairs, the model stores pre-scaled sums (O(nonzeros) updates,
  renormalize at scale ≤ 2^-512), weights solve lazily per decision. The
  golden corpus now pins the sparse semantics the compiled core must match.
  The candidate-set hash byte layout is unchanged (zero `candidate_hash`
  values moved); the pre-scaled bookkeeping is the one semantics change
  (last-bit golden drift, no battery decision flipped).

## Done

- **PR 1–2** (2026-07-14) — repo bootstrap; `rng.py`: splitmix64 counter RNG
  keyed by FNV-1a over (decision id, salt), exact-float `random_unit`. First
  golden vectors. Spec: `rng.md`.
- **PR 3–4** (2026-07-14/15) — wall-clock decaying sums (decay-on-read,
  vendored deterministic exp2) and ridge regression over them. *Removed in
  PR 15*; the decay-on-read formulation returned, update-clocked, in PR 16's
  pre-scaled sums.
- **PR 5** (2026-07-15) — `exploration.py`: inverse-gap weighting (SquareCB)
  with a 5% uniform floor, inverse-CDF sampling, propensity returned.
  *Replaced by epsilon-greedy in PR 16*; the sampling and propensity contract
  survive.
- **PR 6** (2026-07-15) — `decide.py`: BanditState + self-contained
  DecisionRecord; ids are `"{salt}:{seq}"` (uniqueness structural); RNG
  counter 0 reserved for the sampling draw. Spec: `decide.md`.
- **PR 7** (2026-07-15) — `ledger.py`: open decisions resolve exactly once as
  rewarded / expired(default at horizon) / censored, each a loggable event;
  duplicates and post-expiry rewards are structural no-ops; no code path
  learns from an open decision. Spec: `ledger.md`.
- **PR 8** (2026-07-15) — `encoding.py`: fully hashed feature encoding per the
  2026-07-15 decision; end-to-end personalization test through decide + ledger
  at bits=5. Spec: `encoding.md`.
- **PR 9** (2026-07-15) — `merge.py`: bit-exact commutative state merging.
  *Removed in PR 15* (fleet pooling moved offline).
- **PR 10** (2026-07-15) — `golden.py` + `spec/golden.json`: the golden
  corpus, one section per layer plus an end-to-end episode (out-of-order
  rewards, censor, exact-horizon expiry); `test_golden.py` pins the checked-in
  file to exact regeneration. GitHub Actions CI, Ubuntu × Python 3.10–3.14,
  LF forced by `.gitattributes`. Spec: `golden.md`.
- **PR 11** (2026-07-16) — `sim/` harness: environment protocol with exact
  oracle regret, runner over the real public path, metrics
  (normalized/final-window regret, RMSE, median/IQR), stationary environments
  (linear, xor, needle, action-features), oracle/uniform/greedy/epsilon
  comparators, `python -m sim` markdown battery in CI. Also from first
  findings: `candidate_set_hash` respecced to the sparse O(nonzeros) encoding
  (golden diff: only `candidate_hash` values), behavior-identical speedups
  (battery 123s → 28s), and evidence that `GAMMA_SCALE = 1.0` was far too low.
- **PR 12** (2026-07-16) — non-stationary battery tranche: shift, drift, churn
  (arms disappear/return/are born mid-run), dropout; `recovery_time` metric.
  Headline readings: churn traps greedy (worse than uniform) while explorers
  recover; everything else uniform-bounded.
- **PR 13** (2026-07-16) — event-time simulation: `traffic.py`
  (non-homogeneous daily Poisson arrivals, bursts; Constant / Exponential /
  NextMorning delay models), `event_runner.py` (one time-ordered heap, expiry
  swept per event, late rewards structurally ignored past the horizon),
  phase-split regret. Findings: with next-morning delays greedy goes worse
  than uniform (learns a day late); the engine holds with a whole day open in
  the ledger, bounded by the horizon as claimed; with exp-45m delays and a 2h
  horizon ~7% of decisions expire (the exponential tail past the horizon).
  *Removed in PR 16* — findings recorded, code retired.
- **PR 14** (2026-07-16) — `sim/sweep.py` sweep driver + pass-criteria report.
  Decision: `GAMMA_SCALE = 300.0` (near-best 9/12 cells, never >1.34x, ahead
  of epsilon-0.1 overall, 0.353 vs 0.408 mean). Goldens regenerated; sim
  bounds recalibrated. (Constant deleted again in PR 16 with SquareCB itself.)
- **PR 15** (2026-07-16) — forgetting evidence and the simplification it
  forced (see the two 2026-07-16 decisions): per-cell sweeps showed hl=50
  halves shift regret and breaks drift open while costing stationary cells
  little — no single rate is right everywhere; a D4 model-pool prototype
  gained on non-stationary cells but its scoring rule was unresolved
  (exploited-arm bias; fixes traded churn against drift). Rather than freeze
  open research into the semantics, decay/exp2/merge were deleted and
  per-update forgetting shipped. Golden corpus at format_version 2; specs
  rewritten.

- **PR 16** (2026-07-16) — epsilon-greedy exploration + the sparse reference.

  **Exploration** (see the 2026-07-16 decision): `p[i] = epsilon/k +
  (1-epsilon)/m` for the `m` exact-tie best candidates, `epsilon/k` otherwise —
  fresh models are uniform by tie-splitting, no schedule, no tie-break draw.
  `epsilon` lives in `BanditState` (`new_bandit(..., epsilon=0.05)`, an expert
  override like `forgetting`); `FLOOR_MASS`, `GAMMA_SCALE`, `choose_gamma`
  deleted; the decision path computes no uncertainty (`estimate_factored`),
  while `predict` keeps returning (estimate, uncertainty). Golden diff was
  exactly the blast radius (propensities + exploration section; every chosen
  episode index unchanged). `sim/sweep.py` repurposed to sweep
  `DEFAULT_EPSILON` — the 0.05 default is provisional until that sweep is run
  and recorded here.

  Same-battery A/B vs SquareCB@300 (median normalized regret): at or ahead on
  9/12 cells (linear-k20 0.204→0.175, dropout 0.299→0.237, drift 0.889→0.828,
  both event cells improved); behind on churn (0.287→0.415, IQRs overlap) and
  needle whole-run (0.129→0.488, but final-10% 0.111→0.067 — slower to find
  the context-free needle, better once found); xor moves are seed noise.
  Decide latency unchanged in the reference (hash-dominated); one less engine
  constant.

  **Sparse representation** (see the 2026-07-16 decision): sparse candidates
  from `encode` (per-slot add order preserved → values bit-identical; hash
  byte layout unchanged → zero `candidate_hash` drift), pre-scaled model sums
  (O(nonzeros) update arithmetic; immutable tuples still copy O(dim) pointers —
  the compiled core mutates in place), lazy factorization (O(touched), never
  O(dim), bit-identical), sparse features in decision records (O(nonzeros) in
  ledger and log). Measured: dim 16,384 × 10,000 arms 87 ms/decision in pure
  Python (dense formulation ~10 s); dim 256 × 100 arms 1.32 → 0.85 ms; battery
  52s → 33s with no regret cell moved. Next known lever at large k: the
  per-nonzero mix64 hash redesign or per-candidate hash memoization —
  deliberately not taken, keeping the hash contract frozen.

  **Battery slimming** (same PR): the event-time simulation is deleted —
  `event_runner.py`, `traffic.py`, their tests, `phase_regret`, and the
  event-time policy primitives (policies are back to plain choose/observe;
  the engine still runs its expiry sweep every round on the real path). Its
  findings stay recorded in the PR 13 entry; `git log` has the code if
  delayed-reward studies return. The battery prints total decisions and
  decisions/s (currently ~450k decisions, ~21k decisions/s in 21s).
  PROGRESS.md compacted: plan prose that had become history is folded into
  the decisions log and PR entries.

## Currently in flight

- **PR 17** (`pr-17`) — the Rust core (`core/`), Phase 1's opening move: the
  engine ported module for module from the reference (rng, encoding, model,
  exploration, decide, ledger), zero dependencies, state mutated in place —
  every float expression keeps the reference's exact order, so the two are
  bit-identical by construction and checked against the corpus.

  **Golden verification**: `cargo test` includes `spec/golden.json` at
  compile time (a ~200-line hand-rolled JSON parser lives in the test tree,
  keeping the crate itself dependency-free; floats parse correctly-rounded
  from the corpus's shortest reprs and compare via `to_bits`), verifies every
  section, and replays the end-to-end episode generator — schedule, reward
  rule, out-of-order resolutions and all. All sections passed on the first
  run: the Phase 0 exit criterion is met.

  The corpus gained an episode field in this PR (additive diff, zero
  existing vectors moved): `rejected`, resolution attempts that must be
  structural no-ops (conflicting duplicate, post-expiry, post-censor,
  censor-after-reward, unknown id), performed before the pinned final
  state so every implementation must reject them to match it. What the
  corpus deliberately excludes is unit-tested in the crate instead:
  `forgetting = 1.0` (plain sums, scale exactly 1.0), exactly-once ledger
  resolution with bitwise state comparison, and the scale-renormalization
  path (which needs 500+ updates to trigger). The crate's test count is
  small because the reference carries the semantic/property suite once and
  the core's contract is only bit-identity — each golden test is hundreds
  of bitwise assertions, and the battery diff is a 450k-decision
  differential test on top.

  **Battery rerun** (`cargo run --release --bin sim`): `sim/` ported as a
  binary in the crate — environments, policies, runner, metrics (including a
  faithful port of CPython's `math.fsum`, exact Shewchuk summation with the
  round-half-even correction), and stream labels formatted exactly as
  Python's f-strings format them. CI appends its table to the same job
  summary as the Python battery, so the two are comparable line by line. On
  one platform they are *identical* — all 60 table rows matched locally
  (macOS), every median/IQR/RMSE digit — because both harnesses call the
  same libm for the Box–Muller log/cos, the one non-bit-pinned piece
  PROGRESS already flags; cross-platform the tables may drift in the last
  digits. Runtime: 22s Python → 2.7s Rust locally (~21k → ~169k decisions/s,
  the same 450k decisions), with no perf work done on the core yet
  (encoding's per-decision token formatting and hashing, the known hot spot,
  is ported straight).
