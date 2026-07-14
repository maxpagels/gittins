# gittins

A set-and-forget contextual bandit engine. Deploy it and walk away: arms can appear and
disappear, the world can drift, rewards can arrive late or never — the system keeps
making good decisions with no windows, learning rates, or forgetting factors to tune.

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
