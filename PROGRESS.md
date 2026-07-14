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
| 3 | `pr-03-decaying-sums` | Exponential-decay accumulator with timestamp semantics | D3 | Not started |
| 4 | `pr-04-linear-model` | Incremental ridge regression on decaying sums, predict with uncertainty | D3 | Not started |
| 5 | `pr-05-exploration` | Inverse-gap weighting (SquareCB) + probability floor | §5 | Not started |
| 6 | `pr-06-decide` | Decision records; `decide(state, context, candidates, salt)` | D1, D5 | Not started |
| 7 | `pr-07-ledger` | Decision ledger; `learn()`; rewarded/expired/censored; late-reward weighting | §6 | Not started |
| 8 | `pr-08-per-arm` | Per-arm corrections + decayed-count cleanup | D2 | Not started |
| 9 | `pr-09-merge` | Timestamp-aligned state merge; commutativity property tests | D3, §13 risk 3 | Not started |
| 10 | `pr-10-golden-vectors` | Golden test vector generation from the reference | §8 | Not started |

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

- **2026-07-14** — `implementation-plan.md` is git-ignored; PROGRESS.md is the in-repo
  source of truth for the roadmap.
- **2026-07-14** — Reference implementation lives at `src/gittins_reference/`, managed
  with `uv`, requires Python ≥3.10, zero runtime dependencies.

## Done

- **PR 1** (2026-07-14) — repo bootstrap: uv-managed Python skeleton, pytest, this file.
- **PR 2** (2026-07-14) — `rng.py`: splitmix64-based counter RNG; FNV-1a key derivation
  from (decision ID, salt); exact-float `random_unit`. First golden vectors pinned in
  `tests/test_rng.py` and `spec/rng.md`.

## Currently in flight

- Next up: **PR 3** — decaying sums (D3), branch `pr-03`.
