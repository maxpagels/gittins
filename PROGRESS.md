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
| 10 | `pr-10` | Golden test vector corpus generated from the reference; GitHub Actions CI | §8 | Merged |

Phase 0 exit criterion: the spec plus reference is complete enough that an independent
implementation of `decide`/`learn` can match the golden vectors.

## Benchmarking the reference (Phase 0.5 — before the Rust core)

Before freezing semantics into Rust, measure whether the three provisional pieces of the
engine are competitive and robust with their fixed defaults, across the conditions the
README promises (R1, R2): stationary problems, difficult non-stationarity, arms appearing
and disappearing, and features going missing. The pieces under test, with the constants
currently marked provisional in the source:

1. **Model** — per-coordinate ridge on decaying sums (`model.py`): `ridge = 1.0`, the
   assumption that a linear model on hashed outer-product features degrades gracefully
   when the true reward is nonlinear in them, and the cost of never splitting credit
   (co-firing/redundant features double-count), which makes the misspecified and
   redundant-feature environments doubly important.
2. **Exploration** — SquareCB + floor (`exploration.py`, `decide.py`): `GAMMA_SCALE = 1.0`,
   the *mean* as the uncertainty aggregate in `choose_gamma`, and `FLOOR_MASS = 0.05`
   (is 5% forced exploration too costly on stationary problems with many arms, and is it
   enough to rediscover a recovered arm quickly?).
3. **Forgetting** — the single fixed half-life chosen at `new_bandit`: how large is the
   regret gap between one default half-life and the best per-environment half-life? That
   gap is precisely the case for (or against) prioritizing the D4 self-tuning pool
   before the Rust core.

This pulls the synthetic half of the Phase 2 battery (design doc §10) forward. Out of
scope here: OPE estimators, public logged datasets (Open Bandit, Criteo), the public
benchmark page, and any CI regret gate — those stay in Phases 2/4.

### Harness design

- New `sim/` package, pure Python, zero runtime dependencies like the reference; results
  reported as markdown tables (no plotting deps). Not part of `gittins_reference` and
  never bit-pinned — sims are statistical, but every run is seeded and exactly
  reproducible (environment randomness comes from the same counter RNG, keyed by
  (environment name, seed, t)).
- **Cross-language potential**: a sim run is *nearly* bit-comparable to the same sim
  driven through the future Rust core — the engine side is the bit-pinned path and all
  environment uniforms are counter-RNG — except for one piece: the reward-noise
  gaussian (Box–Muller over libm `log`/`cos`, which have no cross-platform rounding
  guarantee; the same reason the reference vendored `exp2`). If Phase 2 wants the
  battery as a cross-language differential test (diff whole decision streams
  Rust-vs-reference, thousands of seeds, far beyond the golden corpus), the fix is
  contained: vendor deterministic `log`/`cos` in `sim/rand.py`, or switch environment
  noise to a distribution built from exact IEEE-754 operations.
- **Environment protocol**: an environment yields per round `t` a context dict plus
  candidate dicts (arm id + features), and returns a stochastic reward for the chosen
  arm; it also exposes the oracle expected reward of every candidate so regret is exact,
  not estimated.
- **Runner** drives the real public path — encode (PR 8) → `decide` (PR 6) → ledger
  `learn`/`expire` (PR 7) — never the layers in isolation, so hashing, the gamma
  schedule, the floor, and decay are all in the loop together.
- **Event-time runner** (PR 13): the PR 11 runner ticks one decision per second and
  resolves each reward immediately; the event-time runs replace that with a single
  time-ordered event queue — decision arrivals and reward arrivals interleaved,
  `expire` swept at every event — so the system updates in real time exactly as a
  production event loop would, driven at simulated speed. Decision timestamps come
  from a non-homogeneous arrival process and rewards from a delay distribution, so
  learning happens mid-traffic from whatever state exists at that instant, some
  rewards land many decisions late, and some cross the horizon and expire.
- **Comparators**, all run through the same loop on the same seeds (paired):
  - oracle (upper bound) and uniform-random (lower bound), to normalize regret;
  - greedy (gamma → ∞, no floor) and epsilon-greedy (ε ∈ {0.05, 0.1}), the "would
    something dumber beat us" check;
  - gittins itself under swept constants: `GAMMA_SCALE` ∈ {0.25, 1, 4, 16},
    aggregate ∈ {mean, min, max}, `ridge` ∈ {0.1, 1, 10}, half-life ∈ geometric grid
    (plus ∞ = no forgetting). The zero-knob default vs. the best swept point *per
    environment* is the headline comparison.
  - a full-covariance ridge comparator implemented inside `sim/` (a baseline, not part
    of the engine): at small `bits` it prices what credit-splitting would buy over the
    shipped per-coordinate model, keeping the diagonal-only decision evidence-backed.

### Environment battery

Each environment crossed with arm count k ∈ {2, 10, 50}, reward noise ∈ {low, high},
and ~20 seeds; runs of 10k–50k decisions.

- **Stationary, well-specified**: linear expected reward in the encoded features —
  the model's home turf; the floor's cost is measured here.
- **Stationary, misspecified**: nonlinear reward (e.g. XOR-style interactions beyond
  the outer product, thresholds) — graceful-degradation check for the linear model.
- **Abrupt shifts**: the best arm swaps at intervals both long and short relative to
  the half-life; measures recovery (does uncertainty regrow and gamma fall back as
  `decide.py` claims?).
- **Slow drift and seasonal**: reward weights rotate slowly / oscillate with a period;
  the regime where any fixed half-life is a compromise.
- **Arm churn (missing arms)**: arms are born and die mid-run; the best arm disappears
  and later returns (rediscovery time via the floor); cold-start regret of a new
  strong arm; identity-collision stress at small `bits`.
- **Missing features**: per-round feature dropout (each feature absent with
  probability p — hashed encoding treats absent as no token), occasionally empty
  context, and irrelevant/noisy distractor features.
- **Variable event rate (traffic cycles)**: decision arrivals follow a daily
  sinusoid — morning and evening peaks, an overnight trough, rate swinging
  ~10–100× — plus bursty periods, on top of a stationary or shifting reward world.
  The half-life is wall-clock seconds, not events (R4: decisions come at different
  frequencies at different times of day), so quiet stretches decay evidence with
  little replenishment: measures the overnight-forgetting cost (does every morning
  start with regrown uncertainty and a re-exploration tax?), whether gamma's fallback
  during troughs is proportionate, and how the fixed half-life trades off against the
  event rate — direct evidence for the one-default-half-life question. Crossed with
  reward delays (including delays correlated with time of day, e.g. conversions that
  arrive next morning) and run on a representative subset of the reward worlds above
  rather than the full cross.

### Metrics

- Normalized cumulative regret vs. oracle (0 = oracle, 1 = uniform), median and IQR
  over seeds; final-window regret rate.
- Post-shift recovery time: rounds until the rolling reward rate regains 90% of the
  oracle's, after each shift/birth/return event.
- Prediction RMSE vs. oracle expected reward — separates model quality from policy
  quality, so a bad result can be attributed to the model or to exploration.
- Diagnostics logged per run: gamma and mean-uncertainty trajectories, propensity of
  the oracle-best arm over time.
- For event-time runs: regret rate split by traffic phase (peak / trough / the first
  stretch after a trough — the morning re-exploration cost), the fraction of decisions
  that expire unresolved, and ledger occupancy over time (does the horizon bound hold
  under peak load with slow rewards?).

### Pass criteria (what "needs updates" means)

- The zero-knob default lands within ~10% normalized regret of the best swept gittins
  variant on ≥90% of environment cells, and is never catastrophically worse (>2×) on
  any cell — the Phase 0.5 version of the Phase 2 exit criterion.
- The default beats epsilon-greedy overall and is never far behind it on any cell.
- Fallout is recorded as decisions in this file: the settled `GAMMA_SCALE` value and
  aggregate; whether `ridge = 1.0` survives; and whether one default half-life is
  defensible or the D4 pool must be built before (or alongside) the Rust core.

### Runtime budget

Prediction is O(dim) per candidate, with the weights solved once per decision
(`factorize` — dim reciprocals, no matrix anywhere). Measured in pure Python:
dim 64 × 100 arms ≈ 4.5 ms/decision, dim 256 × 100 arms ≈ 46 ms/decision. Every cell
the battery needs is comfortable. The reference's per-candidate cost is O(dim) rather
than O(nonzeros) only because vectors are dense Python lists, kept for readability;
the sparse representation that makes very large `bits` cheap belongs to the Rust core.

### PR slicing

| # | Concept |
|---|---------|
| 11 | `sim/` harness: environment protocol, runner over the real encode→decide→learn path, regret metrics, stationary environments, oracle/uniform/greedy/epsilon baselines |
| 12 | Non-stationary environments (abrupt, drift, seasonal) + arm churn + missing-feature environments |
| 13 | Event-time simulation: event-queue runner (decisions and rewards as one time-ordered stream, expire swept per event), non-homogeneous arrival processes (daily traffic curves, bursts), reward-delay distributions, phase-split metrics |
| 14 | Sweep driver + markdown report generator; battery run; findings written up as decisions here (and any constant changes land as their own follow-up PRs with regenerated golden vectors) |

## Repository layout

```
src/gittins_reference/   pure-Python reference implementation (Phase 0)
sim/                     simulation harness (Phase 0.5): environments, comparators, runner, metrics
tests/                   pytest suite for the reference and the sim harness
spec/                    written spec sections, grown PR by PR
PROGRESS.md              this file
```

Planned later: `core/` (Rust), `bindings/` (Python native, JS/WASM).

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
- **2026-07-15** — The synthetic half of the Phase 2 regret battery is pulled forward to
  Phase 0.5, before the Rust core: the provisional constants (`GAMMA_SCALE`, the gamma
  aggregate, `ridge`, the default half-life, `FLOOR_MASS`) get validated against a
  synthetic environment battery while changing them is still a Python edit plus a golden
  regeneration, not a cross-language semantic change. Plan in "Benchmarking the
  reference" above.
- **2026-07-15** — The built-in model is per-coordinate ridge on decaying sums, and only
  that: `xx` keeps the outer-product matrix's *diagonal* (`Σ x_j²`), prediction is
  `θ_j = xy_j / (xx_j + ridge)` with per-coordinate uncertainty — no Cholesky, no linear
  system, O(dim) state; each weight is that feature's own shrunk, decayed running average
  (the operating point proven at scale by VW's hashed linear learners). PR 4's full
  outer-product matrix was removed: one model keeps the reference, the golden corpus, and
  the future Rust core single-story, and only the diagonal scales to the large hashed
  spaces and candidate sets the engine promises (R5, R6). Known cost, pinned in
  `tests/test_model.py`: credit is never split, so co-firing/redundant features
  double-count; disentangling combinations stays the encoder's job (D2 interactions).
  The Phase 0.5 battery carries a full-covariance comparator in `sim/` to price that
  trade; if credit-splitting proves decisive at small `bits`, reintroducing it will be a
  recorded, evidence-backed change. Golden corpus regenerated. Spec: `spec/model.md`.
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

- **PR 10** (2026-07-15) — `golden.py` + `spec/golden.json`: the golden test vector
  corpus generated from the reference (design doc §8). Sections per layer (rng, exp2,
  decay, model, exploration, encoding) plus an end-to-end `episode`: two agents, hashed
  encoding, out-of-order rewards, censor, exact-horizon expiry sweep, merge — every
  record and resolution logged; matching it bit-for-bit is the Phase 0 exit test for an
  independent decide/learn. `tests/test_golden.py` pins the checked-in file to exact
  regeneration, so semantic drift fails CI as a reviewable vector diff. Also adds
  GitHub Actions CI (`.github/workflows/ci.yml`): full pytest on PRs and main pushes
  on Ubuntu × Python 3.10–3.14 — with bit-pinned tests every leg checks bit-identity
  (R6); macOS/Windows legs deferred for cost, to return as a release gate with the
  compiled core. `.gitattributes` forces LF so the corpus compares as exact text
  everywhere. Spec: `spec/golden.md`.

- **PR 11** (2026-07-16) — `sim/` harness (Phase 0.5 plan above): the environment
  protocol (per-round context dict + candidate dicts in, stochastic reward out, oracle
  expected reward of every candidate exposed so regret is exact), a runner driving the
  real public path — encode → `decide` → ledger `learn`/`expire`, never the layers in
  isolation — and metrics (normalized cumulative regret with 0 = oracle / 1 = uniform,
  final-window regret rate, prediction RMSE vs the oracle means, median/IQR over
  seeds). Stationary environments: well-specified linear and XOR-misspecified.
  Comparators through one identical loop: oracle, uniform, greedy (gamma → ∞, no
  floor), epsilon-greedy (ε ∈ {0.05, 0.1}). All randomness comes from the reference's
  counter RNG keyed by (name, seed, t), so rounds are pure functions of the seed and
  every comparator is paired by construction; runs replay exactly but nothing is
  bit-pinned. `python -m sim` runs the battery and prints a markdown table; CI runs it
  in a single-leg `sim` job and appends the table to the GitHub job summary — a
  diagnostic, not a gate. Expanded before merge with two more stationary
  environments — `needle`
  (context-free, one hidden slightly-better arm among many: exploration isolated from
  the model; greedy provably locks on a lucky arm) and `action-features` (reward
  bilinear in context x action features, so evidence must transfer between arms; the
  first exercise of the encoder's action namespace) — plus larger-k variants in the
  battery. Also from the battery's first findings: `candidate_set_hash` respecced over
  the sparse (dimension, nonzero count, sorted (index, value) pairs) encoding — zero
  entries are absent, so hashing is O(nonzeros) and a sparse core never materializes
  the dense vector; golden vectors regenerated (the diff was exclusively
  `candidate_hash` values, proving decisions untouched). Behavior-identical speedups
  (goldens unchanged): `pair_hash` memoized (bounded lru_cache — the same token pairs
  recur every decision), `predict_factored` single-pass skipping zero entries (±0.0
  never changes a finite sum), decay vector ops via list comprehensions. Battery
  wall time 123s → 28s at 1500 rounds x 5 seeds x 6 environments. Battery data:
  `GAMMA_SCALE = 1.0` plateaus gamma at ~6 (decay bounds uncertainty at ~0.16), so
  the engine picks a 0.3-gap best arm only ~40% of the time forever; sweeping
  GAMMA_SCALE ∈ {10, 30, 100, 300} drops linear-k5 regret 0.68 → 0.10 and xor final-10%
  1.03 → 0.09. The constant change itself is deferred to the battery-findings PR
  (PR 14) per the sweep plan above.

- **PR 12** (2026-07-16) — the non-stationary, churn, and missing-feature environments
  (the battery plan's next tranche): `shift` (a LinearEnvironment world redrawn every
  `period` rounds — post-shift recovery, decide.py's uncertainty-regrowth claim),
  `drift` (parameters rotate continuously between two hidden linear worlds; period
  beyond the run = slow drift, inside it = seasonal), `churn` (context-free arms with
  scripted events in absolute rounds: the best arm disappears, returns, and a strictly
  better stranger is born late — the candidate list changes shape under the policy,
  no registration anywhere), and `dropout` (each true feature absent per round with
  probability p, distractor features always on, oracle means from the full context —
  regret prices the missing information). Runner grows a per-round `best` series (the
  oracle's expected rate); metrics grow `recovery_time(result, event, window,
  fraction)` — rounds after an event until the policy's rolling expected reward rate
  regains `fraction` of the oracle's (meaningful for positive-mean worlds; churn draws
  its means that way). All four join the CI battery (renamed "environment battery":
  10 environments, 41s) and the invariant tests (exact replay, oracle-zero,
  uniform~1). First readings at 1500 rounds: shifts are survivable for every learner
  (greedy 0.62, engine 0.85, uniform-bounded); seasonal drift against the 1000-round
  half-life is everyone's compromise regime (all ~0.83-0.98); churn traps greedy
  (median 1.23, worse than uniform — its locked arm vanishes) while epsilon recovers
  (0.30) and the engine stays uniform-bounded (0.86); dropout degrades gracefully
  (greedy 0.17, engine 0.71 with the best RMSE, 0.24).

- **PR 13** (2026-07-16) — event-time simulation. `sim/traffic.py`: `DailyTraffic`, a
  non-homogeneous Poisson arrival process (Gaussian bumps at 10:00 and 19:00, an
  overnight trough, peak/trough rate swing set by construction, optional burst
  windows multiplying the rate), sampled by thinning; delay models `ConstantDelay`,
  `ExponentialDelay`, and `NextMorningDelay` (the conversion that lands tomorrow at
  09:00 — delay correlated with time of day); a fixed `phase(t)` bucketing
  (trough/morning/peak/day) for the phase-split metrics. `sim/event_runner.py`:
  decisions and rewards interleave in one time-ordered heap, the policy's expiry
  sweep runs with every event's time, reward values are drawn at decision time but
  delivered `delay` later, and a reward landing after its decision expired is a
  structural no-op. `EventRunResult` carries the round runner's series (all round
  metrics apply unchanged) plus timestamps and the plumbing accounting
  (resolved + in_flight == decisions, always; expired; ledger high-water mark).
  Policies gained the event-time primitives sweep/decide_at/resolve —
  choose/observe are now the same primitives composed for the immediate-reward
  case (round behavior unchanged; the zero-delay event run is test-asserted equal
  to the hand-driven round loop). The baselines train on every reward however late
  (no horizon) — deliberately, so the engine's expiry rule is priced against them.
  `metrics.phase_regret` splits normalized regret by traffic phase. The battery
  gained an event-time table (linear-k5, 2 days, half-life 6h): with exp-45m
  delays and a 2h horizon the engine expires ~7% and everyone pays extra in the
  trough; with next-morning delays greedy goes *worse than uniform* (1.10 — it
  learns a day late), epsilon holds 0.75, the engine 0.93 with a whole day open in
  the ledger (high-water ~1875, bounded by the 26h horizon as claimed). The sweep
  driver, full report generator, and battery findings are PR 14.

## Currently in flight

- **PR 14** (`pr-14`) — the sweep driver and the first settled constant.
  `sim/sweep.py` (`python -m sim.sweep`, run by hand, not CI): re-runs every battery
  environment plus both event-time configurations with the engine patched to each
  candidate `GAMMA_SCALE`, and prints the per-cell regret table with per-environment
  bests marked, epsilon-0.1 alongside, and a pass-criteria summary per candidate
  (near-best cells, worst ratio to best, cells lost to epsilon, mean regret).

  **Decision: `GAMMA_SCALE = 300.0`** (was the provisional 1.0). Swept
  {0.25, 1, 3, 10, 30, 100, 300, 1000} over 12 cells: 300 is within 1.1x of the
  per-cell best on 9/12, never worse than 1.34x (criterion: never >2x), beats
  epsilon-0.1 on 9/12 cells and on mean regret (0.353 vs 0.408); the curve turns at
  1000 (needle 0.13 -> 0.31, xor-k4 0.35 -> 0.52). The strict "near-best on >=90% of
  cells" target reads 75% here — recorded honestly; the remaining gap is the drift /
  dropout / shift cells where epsilon stays slightly ahead, and closing it belongs to
  the still-open sweeps (uncertainty aggregate mean vs min/max, `ridge`, the default
  half-life). At 1.0 the engine spent over half its traffic on non-best arms forever
  (gamma plateaued ~6 because decay bounds uncertainty); at 300 it goes from
  worst-in-class to best-in-class on most cells, including both event-time configs
  (fast rewards 0.107 vs epsilon 0.143; next-morning 0.628 vs 0.747, keeping its
  overnight-phase advantage). Golden vectors regenerated for the new constant (the
  diff is propensities and sampled choices — exactly what gamma touches);
  `spec/decide.md`'s schedule section updated; the pinned decision record and
  `choose_gamma` unit tests updated; the sim tests' engine bounds recalibrated (less
  exploration means competitive regret but a coarser late-run model: linear RMSE
  ~0.15 -> ~0.23). Also in this PR: PR-number references scrubbed from code comments
  and docstrings across `sim/`, `src/`, and `tests/` (this file keeps the history) —
  the sim battery work is wrapped for now.
