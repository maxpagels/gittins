"""Timestamp-aligned state merge (design doc D3, section 13 risk 3).

Because the model's entire memory is decaying *sums*, two bandits' states
combine by aligning their decay origins and adding — `merge_states` is a
thin layer over the model merge (model.py), which is entry-by-entry
DecayedVector merge (decay.py). A fleet of agents pools knowledge by
periodically merging their state files into a shared file; cron plus file
copy is the officially supported federation architecture (R8).

**A merged state is pooled knowledge, not a running agent.** The learned
state (the sums) and the version count merge; the operational fields do
not:

- The **ledger does not merge**. Open decisions stay with the agent that
  made them — that agent is where the reward will be reported, and where
  expiry is being swept. Once an agent resolves a decision, the evidence
  is in its sums and reaches the pool at the next merge. (An open decision
  copied into a shared file would just expire there, unresolvable, and its
  evidence would later be double-counted when the owner also resolves it.)
- **next_seq resets to 0.** Sequence numbers belong to a (salt, agent)
  pair. A merged state that will make its own decisions must be given a
  fresh salt — the same rule that keeps any two agents' decision IDs and
  random streams distinct — and under a fresh salt, 0 is the right start.

Merging requires the states to be the same application: equal horizon and
default_reward (checked here) and equal dim, ridge, and half-life (checked
by the layers below).

**Merge pools disjoint evidence.** merge(A, A) counts A's evidence twice;
nothing in the sums can detect shared history. The supported fleet flow is
*recompute, not accumulate*: each cron run rebuilds the shared file from
the current agent files (`gittins merge agent-*.bnd -o shared.bnd`), whose
evidence sets are disjoint by construction.

Exactness (inherited from the decay layer, property-tested here):

- merge(a, b) == merge(b, a), bit for bit.
- Multi-way merges and resolve-then-merge versus merge-then-resolve agree
  to rounding (~1e-12 relative): float addition is commutative but not
  associative, so evidence *timing* relative to merges never matters
  semantically, only at the last-bit level. This is design doc section 13
  risk 3 reduced to a rounding statement.
"""

from gittins_reference.decide import BanditState
from gittins_reference.model import merge_models


def merge_states(a: BanditState, b: BanditState) -> BanditState:
    """Pool two agents' knowledge as if one agent had seen both streams."""
    if a.horizon != b.horizon:
        raise ValueError("cannot merge states with different horizons")
    if a.default_reward != b.default_reward:
        raise ValueError("cannot merge states with different default rewards")
    return BanditState(
        model=merge_models(a.model, b.model),
        next_seq=0,
        model_version=a.model_version + b.model_version,
        horizon=a.horizon,
        default_reward=a.default_reward,
        ledger=(),
    )
