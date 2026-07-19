# gittins — progress and roadmap

The running record of what has been built and what comes next; updated in
every PR. The full design document (`implementation-plan.md`) is kept out of
git deliberately. Entries here are kept short — this file's own git history
has the long versions.

## What this project is

`gittins` is a set-and-forget contextual bandit engine: a zero-dependency,
deterministic core (Rust, with Python/JS bindings) where all state is an
explicit value, decisions are first-class logged records, and
non-stationarity is handled by built-in forgetting with fixed defaults — tuning
lives offline, in replay and off-policy evaluation over the decision log, never
in knobs. "The SQLite of bandits." The requirements live in the README.

**Current phase: Phase 2 — bindings, nearly done.** Phase 0 (the pure-Python
reference, which remains the specification), Phase 0.5 (the benchmarking
battery, `sim/`), and Phase 1 (the Rust core) are complete; the Python wheel,
the wasm module, and the CLI are built and gated. The core matches the
reference bit for bit against the golden vector corpus (`spec/golden.json`,
checked by `cargo test`) and number for number across the 450k-decision
battery.

## Working process

- One concept per PR, roughly 100–300 lines including tests, reviewable in one
  sitting; too big to explain means split.
- Semantic changes regenerate `spec/golden.json` and the vector diff is
  reviewed like code.
- The reference and `sim/` stay pure Python with zero runtime dependencies
  (pytest is dev-only); the Rust core crate stays zero-dependency too —
  binding libraries (PyO3, wasm-bindgen, serde_json) live in their own crates.

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

## Benchmarking harness

`sim/` drives every comparator through the real public path (encode → decide →
ledger) on seeded, paired, exactly-replayable runs. `python -m sim` prints the
environment battery — 10 worlds × 6 policies, 1500 rounds × 5 seeds — as a
markdown table; `cargo run --release --bin sim` is the same battery on the
Rust core, and CI appends both tables to the job summary for line-by-line
comparison — a diagnostic, not a gate. `python -m sim.sweep` (by hand) settles
engine constants. Known limit: the xor cells are chaotic across RNG salts, so
xor comparisons at 5 seeds are noise. Battery runtime: 450k decisions in ~22s
Python, ~2.7s Rust, with no perf work done on the core yet.

## Repository layout

```
src/gittins_reference/   pure-Python reference implementation (Phase 0)
sim/                     simulation harness (Phase 0.5)
core/                    Rust core (Phase 1): the engine, the golden
                         verifier (cargo test), and the battery rerun
                         (cargo run --release --bin sim)
tests/                   pytest suite for reference + sim
spec/                    golden.json, the golden test corpus
docs/                    user-facing documentation (usage.md: the public API
                         by example)
bindings/python/         Python binding (Phase 2): the public API as a
                         PyO3/maturin wheel, gated on the golden api section
bindings/wasm/           browser/JS binding (Phase 2): the same surface via
                         wasm-bindgen, gated under Node
bindings/cli/            CLI binding: the `gittins` binary — experience-log
                         verify / eval / sweep / replay, gated on the golden
                         ope section
PROGRESS.md              this file
```

## Roadmap

The rule that governs Phase 2: the dict-shaped public API is specified once,
in the reference, and every binding mirrors it exactly; each binding's CI
gate is the golden episode replayed through the binding, bit for bit.

- **PR 22 — cross-OS CI + packaging.** macOS/Windows legs (the release gate
  the golden corpus defers), wheels via maturin-action, npm packaging.

Also outstanding, order flexible:

- the `DEFAULT_EPSILON` sweep (`python -m sim.sweep`) — the 0.05 default is
  provisional until it is run and recorded here;
- the encoding hot spot (per-decision token formatting + pair hashing), now
  priced by the binding benchmark: 210 µs against the roadmap's 10–50 µs
  target (bits=8, 100 arms × 5 context features) — the next perf PR; the
  hash contract itself stays frozen;
- further out: multislot/ranking; estimator growth on the OPE tooling if
  IPS/SNIPS prove insufficient in practice.

## Done

One line per PR; this file's git history has the full entries.

- **PR 1–2** (2026-07-14) — repo bootstrap; `rng.py`: splitmix64 counter RNG
  keyed by FNV-1a over (decision id, salt).
- **PR 3–4** (2026-07-14/15) — wall-clock decaying sums + ridge regression.
  *Removed in PR 15.*
- **PR 5** (2026-07-15) — SquareCB exploration with a uniform floor.
  *Replaced by epsilon-greedy in PR 16.*
- **PR 6** (2026-07-15) — `decide.py`: BanditState + self-contained
  DecisionRecord; ids `"{salt}:{seq}"`.
- **PR 7** (2026-07-15) — `ledger.py`: exactly-once resolution as rewarded /
  expired / censored.
- **PR 8** (2026-07-15) — `encoding.py`: fully hashed feature encoding.
- **PR 9** (2026-07-15) — bit-exact state merging. *Removed in PR 15* (fleet
  pooling moved offline: logs merge, models rebuild by replay).
- **PR 10** (2026-07-15) — the golden corpus (`golden.py`, `spec/golden.json`)
  pinned to exact regeneration; GitHub Actions CI.
- **PR 11–14** (2026-07-16) — the `sim/` battery: runner, metrics, stationary
  + non-stationary worlds, event-time simulation (later removed, findings
  kept), the sweep driver and pass criteria.
- **PR 15** (2026-07-16) — forgetting evidence forced the simplification:
  wall-clock decay, exp2 and merge deleted; per-update forgetting
  (factor 0.999, an expert override, not a knob) shipped.
- **PR 16** (2026-07-16) — epsilon-greedy exploration (exact ties split the
  greedy mass) + the sparse end-to-end reference: (index, value) pairs,
  pre-scaled sums, O(nonzeros) updates, lazy weight solving.
- **PR 17** (2026-07-16) — the Rust core (`core/`), Phase 1: a bit-exact port,
  zero dependencies, every golden section verified bit for bit on the first
  run; the battery reruns on the core into the same CI summary.
- **PR 18** (2026-07-16) — canonical state serialization (one little-endian
  layout, all-or-nothing deserialization) + the core's error surface; nothing
  on the public path panics.
- **PR 19** (2026-07-16) — the public dict-shaped API (`api.py`, `api.rs`):
  the complete binding surface, specified once — create / decide / learn /
  censor / expire / serialize / deserialize / model_bits — with the golden
  `api` section as the binding acceptance gate; `docs/usage.md`.
- **PR 20** (2026-07-16) — the Python binding (`bindings/python`): PyO3 +
  maturin, abi3, one boundary crossing per decision, ValueError with the
  reference's messages; golden-gated in CI plus a decision-cycle benchmark
  (~8–11x over the pure reference on every shape).
- **PR 21** (2026-07-16) — the browser/JS binding (`bindings/wasm`):
  wasm-bindgen over the core's api module, `candidate_hash` as BigInt,
  golden-gated under Node — the project's first cross-platform bit-identity
  check. Same PR: the in-place handle became the uniform convention
  everywhere, and public serialize/deserialize became lowercase hex strings.
- **BYO callbacks** (2026-07-19) — bring-your-own model and exploration as
  three optional per-call callbacks, never stored state: `score`/`explore`
  on `decide`, `train` on `learn`/`expire`, mirrored in the reference, the
  core, and both bindings; outputs validated, `train` fires only after its
  resolution commits (never double-trains); pinned by the corpus's `byo`
  section.
- **OPE + the experience log** (2026-07-19) — offline evaluation over a
  specified JSONL experience log (records with their inputs, plus
  resolutions, gzip transparent) and the CLI binding (`bindings/cli`, the
  `gittins` binary): verify / eval / sweep / replay, single-pass and
  streamed. Estimators are progressive IPS + SNIPS, never printed without
  diagnostics; `verify` makes an invalid evaluation refused, not
  discouraged; replay doubles as the fleet-pooling rebuild. Same PR:
  assembly-free logging — records carry their inputs, and the ninth API
  name, `log_line`, emits canonical log lines. Pinned by the corpus's `ope`
  section.
