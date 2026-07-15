# gittins

Gittins aims to be a production-ready contextual bandit engine that addresses the practical
considerations with such systems:

- **R1 — Dynamic arms and context.** The set of available actions must change at any moment.
  New arms must be usable immediately; dead arms must be cleaned up automatically.
- **R2 — Non-stationarity, as few hyperparameters as possible ** Learning is incremental.
  If the world changes, the bandit must adapts. The reference implementation must be robust enough
  to work well with just the default settings for most problems. The system must
  never fully converge: it always retains enough exploration to notice that a formerly good
  arm has gone bad, or a bad one has recovered.
- **R3 — Offline policy evaluation (OPE).** It must be possible to estimate how a new policy
  *would have* performed using only logged decisions from an old policy.
- **R4 — Safe reward handling.** Rewards can arrive late, out of order, more than once, or
  never. Constructing invalid training data from logs must be *impossible by design*, not
  merely discouraged. The system must take into account that the order in which experience arrives
  is not the only consideration; for many real world problems, decisions are made are different
  frequencies during different times of day.
- **R5 — Multislot and large action sets.** Ranking / multi-position problems and problems
  with thousands of candidate arms must be practical, fast, and robust.
- **R6 — Speed and determinism.** Sub-microsecond decision cycles. Few to zero dependencies.
  Bit-identical results across platforms and language bindings, enforced by a golden test
  corpus. Every code change must be validated by tens of thousands of simulations in CI.
  If the CI run becomes slow, the core system should be made faster.
- **R7 — Simple algorithms, bring-your-own models.** The built-in algorithms should be
  readable by any competent programmer. Sophistication lives in the layering and the API,
  not in any single component. Users can swap in their own prediction model and inherit
  everything else (exploration, forgetting, logging, OPE) unchanged.
- **R8 — Choose your own complexity.** The same engine needs to be able to runs (a)
  ephemerally in memory, (b) persisted to a flat file, (c) as a shared service, or
  (d) as many bandits whose weights are periodically pooled into a shared weights file.
  There should be no need for databases; indeed, the model weights should be possible to
  check in to version control and deploy as part of normal deployment workflows.


Named after John Gittins, whose index (1974) established that exploration has a
precise, computable value.

**Status: early development.** The current work is a pure-Python reference
implementation of the core, built concept by concept. See [PROGRESS.md](PROGRESS.md)
for the roadmap and current state.

## Development

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run pytest
```
