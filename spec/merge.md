# Spec: timestamp-aligned state merge

Design doc reference: D3 (mergeable weights — the distinctive feature),
section 9 Tier 2 (merged fleet), section 13 risk 3 (decay + late rewards +
merge interact subtly). Implemented in `src/gittins_reference/merge.py`
(PR 9), a thin layer over `merge_models` (`spec/model.md`) and
`vector_merge` (`spec/decay.md`).

## Semantics

`merge_states(a, b)` pools two agents' knowledge as if one agent had seen
both observation streams:

| field | merged value |
|---|---|
| `model` | `merge_models(a, b)` — origin-aligned entry-wise sum |
| `model_version` | `a + b` (observations absorbed in total) |
| `next_seq` | 0 |
| `ledger` | empty |
| `horizon`, `default_reward` | carried over (must be equal) |

Merging requires the same application: equal `horizon` and
`default_reward` (checked here), equal `dim`, `ridge`, and half-life
(checked by the layers below). Anything else is a ValueError.

**A merged state is pooled knowledge, not a running agent.** The ledger
does not merge: an open decision belongs to the agent that made it — that
is where its reward will be reported and where expiry is swept. Once the
owner resolves it, the evidence is in the owner's sums and reaches the
pool at the next merge. (Copied into a shared file, an open decision would
just expire there, unresolvable, and be double-counted when the owner also
resolves it.) `next_seq` resets because sequence numbers belong to a
(salt, agent) pair: a merged state that will make its own decisions must
be given a fresh salt — the same rule that keeps any two agents' decision
IDs and RNG streams distinct — and under a fresh salt, 0 is correct.

**Merge pools disjoint evidence.** `merge(A, A)` counts A's evidence
twice; sums carry no lineage, so overlap is undetectable. The supported
fleet flow is *recompute, not accumulate*: each run rebuilds the shared
file from the current agent files (`gittins merge agent-*.bnd -o
shared.bnd`), whose evidence sets are disjoint by construction.

## Exactness contract (property-tested)

- **Commutative, bit for bit:** `merge_states(a, b) == merge_states(b, a)`
  as whole values. Structural: the decay layer orders by origin
  internally, and float addition of two values is commutative.
- **Merging a fresh same-creation-time state changes no evidence bits**
  (adding zeros at an equal origin is exact).
- **Resolve-then-merge equals merge-then-resolve to rounding (< 1e-12
  relative), across half-lives (including inf) and across a
  renormalization era.** This is section 13 risk 3 reduced to a rounding
  statement: a late reward trains at the decision's timestamp (ledger
  spec), stored sums live on a shared absolute timescale (decay spec), so
  whether evidence enters before or after a merge differs only in float
  summation order — never in weight. In the measured configurations the
  paths are exact; the *contract* is rounding-level, because float
  addition is not associative.
- **Associativity and merge-vs-central agree to rounding (< 1e-12):**
  a three-agent merge in either grouping matches a central model that
  absorbed all events in global time order. Multi-way merges should be
  performed in a fixed order (CLI: argument order) for reproducibility.

One measured subtlety, inherited from decay renormalization: merging
states whose evidence lies more than ~53 half-lives apart absorbs the
older side to nothing — its contributions fall below double precision's
representable ratio, which is exactly the "fully forgotten" semantics the
decay spec promises.

## Golden vectors

Two agents (dim 2, half-life 3600 s, horizon one day, created at
T0 = 1752000000.0), candidates `[[1, 0.5], [0.5, -1]]`. Agent a decides at
T0 (salt `"agent-a"`, chooses arm 0), rewarded 1.0; agent b decides at
T0+1800 (salt `"agent-b"`, chooses arm 0), rewarded −0.5. Merged:
`model_version` 2, and

    predict([1, -1], T0 + 3600) = (0.02918561399511608, 1.3710276201822664)
