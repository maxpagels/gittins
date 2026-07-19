# Patterns: what the handle-based API makes possible

Working notes on emergent patterns — things nobody had to build, which fall
out of three properties of the API: handles are cheap references, the whole
state is one serializable string, and every decision is a replayable,
propensity-stamped record. Candidate material for the book (likely "Choose
Your Own Complexity").

The common thread, and arguably the deepest consequence of the API design:
**because the bandit is a value rather than a service, every operational
pattern you know for data — fork, diff, revert, replicate, hash, ship —
applies to the learner.**

## Hierarchical bandits

A hierarchy is nothing more than a dict of handles and two calls per
decision: a router bandit picks a category, a leaf bandit picks the item.

```python
router = gittins.create(bits=8, horizon=3600.0)
leaves = {cat: gittins.create(bits=12, horizon=3600.0) for cat in categories}

r1 = gittins.decide(router, context, category_candidates, t=t, salt="router")
cat = category_candidates[r1.chosen][0]
r2 = gittins.decide(leaves[cat], context, items_in(cat), t=t, salt=f"leaf-{cat}")
# ...outcome arrives:
gittins.learn(router, r1.decision_id, reward)
gittins.learn(leaves[cat], r2.decision_id, reward)
```

The payoff is large action sets: 10,000 items via 100 categories times 100
items means ~200 candidates scored per decision instead of 10,000.

Why it composes safely with this particular engine:

- **The router's problem is non-stationary by construction.** Its reward for
  "category A" depends on how good A's leaf currently is, and leaves improve
  over time. A converging bandit would calcify on whichever leaf learned
  fastest early; permanent epsilon plus forgetting keeps the router
  re-measuring the leaves as they improve.
- **OPE survives composition.** The joint propensity of the final action is
  `r1.propensity * r2.propensity`, and both records carry their propensities
  and ids. Log both records per decision and the log is still a valid
  offline-evaluation dataset for the composed policy.
- **Per-level knobs.** Each handle has its own `bits`, `horizon`, `epsilon`,
  and `forgetfulness`: a tiny slow-moving router over larger fast-adapting
  leaves is natural.
- **Persistence stays trivial.** The whole hierarchy serializes to a JSON
  dict of strings, still version-controllable. Distinct salts keep every
  decision id unique across the tree.

Caveats: it is a greedy factorization, so hierarchy design matters (an item
straddling two categories learns twice, independently); both levels receive
the full reward, the standard simple choice, which means a good category
with a momentarily bad leaf gets blamed jointly; and delayed rewards mean
remembering two decision ids per user event.

The one-level special case is a **meta-bandit choosing between whole
policies** — e.g. routing between a conservative incumbent and an
experimental challenger, which is a safe-rollout story on its own.

## Fork-and-preview (speculative decisions)

`deserialize(serialize(state))` is a cheap fork, and since `decide` mutates
(sequence counter, ledger), a fork is *the* way to ask "what would you pick
right now?" without side effects. This gives you dry-run mode, admin
dashboards showing the current favorite per context, and "tell me before I
commit" flows — none of which the API has to support explicitly.

## Git-revert your model

States are small strings, encouraged to live in version control. If a
reward-pipeline bug poisons a day of learning: roll the state file back to
yesterday's commit and replay today's log from the decision records,
`censor`ing the poisoned resolutions. Model rollback becomes literally the
same operation as code rollback — no ML infrastructure, just git and the
log.

## Ensembles as uncertainty meters

Hold N handles with different salts on the same traffic. They see the same
data but explore differently, so *disagreement between them* is a free
uncertainty signal the engine deliberately does not compute on the decision
path: route to a human when the committee splits, act confidently when it is
unanimous. N bandits cost N small states and no extra machinery.

## A bandit per user, stored anywhere

An 8-bit state is a few KB of string. It fits a KV-store row, a cookie, or
localStorage. Two patterns fall out:

- A completely **stateless decision service**: load string, decide, learn,
  save string. No database, any replica can serve.
- The privacy-flipped version: **personalization runs entirely in the user's
  browser** via the WASM binding, and behavioral data never leaves the
  device. Merge decision logs centrally later if a pooled model is wanted —
  the merge-and-rebuild deployment mode running at the per-user extreme.

## Challenger policies without traffic risk

Every production record carries its propensity, so a proposed new policy
(different hyperparameters, different candidate features, a hierarchy) can
be evaluated on the existing log before it ever serves a request. Offline
policy evaluation turns the log into a test bench for *any* challenger, and
shipping the winner is just deploying a state string.

## Determinism as a production integrity check

Bit-identical behavior is not only a testing feature:

- **Replica integrity.** Replicas of a shared bandit can hash their
  serialized states after syncing; a mismatch proves divergence (a missed
  message, a corrupted store) instantly, with no fuzzy tolerance thresholds,
  because identical inputs must produce identical bytes.
- **Incident forensics.** A state snapshot plus the logged inputs reproduces
  the decision that paged you, bit for bit, on a laptop.

## The audit trail is already written

Records carry `candidate_hash`, `propensity`, and `model_version`: what the
alternatives were, how random the choice was, and which policy made it —
captured at decision time, unforgeable after the fact. Most ML systems bolt
an audit trail on; here it is the return value.

## Learn from things that don't happen

`default_reward` is "what silence means," and nothing says silence must mean
failure. Invert it: for a churn-prevention nudge, create with
`default_reward=1.0` and report `0.0` only when the bad event occurs — now
*the absence of an event* trains the model, with `expire` doing the work at
the horizon you declared. Most systems can't learn from non-events without
building a synthetic-negatives pipeline; here it is a constructor argument.

## Delete your reward dedupe layer

`learn` resolves a decision exactly once and returns nothing on any
duplicate. At-least-once delivery infrastructure — retrying webhooks,
redelivering queues, double-firing analytics — needs no idempotency layer in
front of the bandit. Bonus: the return value doubles as a free "was this
already processed?" primitive for your own bookkeeping.

## An A/B test is epsilon = 1.0

Create a bandit with `epsilon=1.0` and you have a uniform randomizer that
logs audit-grade propensities — a randomized experiment platform. The
kicker: that log is the ideal OPE dataset (uniform propensities, no
weighting pathologies), so a "dumb" collection phase lets you evaluate *any*
candidate policy offline before switching the real bandit on. "An A/B test
is a bandit with the dial at one end" becomes an operational recipe.

## Self-tuning systems code

Sub-microsecond decisions mean the bandit fits inside infrastructure hot
paths, not just UX: arms = upstream replicas / retry backoffs / cache TTLs /
compression levels; context = payload size and time of day; reward = latency
or success. The special case worth naming is **automated canary rollout**:
arms = {old code path, new code path}, reward = error-free completion.
Traffic shifts to the healthy variant on evidence, and permanent exploration
keeps re-probing, so a variant that recovers gets rediscovered — the
"formerly bad arm has recovered" requirement, applied to deploys.

## Right-to-be-forgotten by architecture

A user opts out mid-flight: `censor` their open decisions — an on-record
exclusion that never trains. Their already-trained influence is not forever
either: forgetting decays every observation's weight geometrically, so one
user's contribution is provably negligible after about one effective window
(~1,000 updates at defaults). "The model forgets you, on a schedule, by
construction" is a privacy story most ML systems cannot tell.

## Golden fingerprints for your app

The determinism trick generalizes to user code: an integration test that
replays a fixed decision sequence through *your* feature-encoding code and
asserts a hash of the serialized state. This is the antidote to the hashing
bargain's sharp edge — a typo'd feature name hashes somewhere silently, but
it cannot survive a fingerprint test, because any change to feature names or
values changes the bytes.

## Time is a test fixture

`t` is caller-supplied, so time itself is mockable: simulate a month of
horizon expiries in milliseconds, unit-test "reward arrives 59 vs 61 minutes
late" deterministically, or replay a production log with its original
timestamps years later and get identical behavior. Related freebie:
`candidate_hash` in every record detects catalog drift — two replicas
disagreeing about the candidate set produce different hashes for "the same"
decision.

## The candidate list is a policy lever

Arms don't exist in the model — only in the list you pass to *this* call.
That makes availability logic trivially expressible with zero API support:
drop an out-of-stock item, gate an arm by geography or legal constraints,
introduce a new arm only in low-risk contexts first (staged rollout), pull
an ad when its budget exhausts and reintroduce it tomorrow — the model keeps
its (gently decaying) estimates throughout. Most bandit systems need an "arm
lifecycle" API; here the lifecycle is just list membership, decided fresh
every call.

## Top-k slates by repeated decide

Need a ranking, not a single pick? Call `decide` k times per pageview,
removing the chosen candidate from the list each time. Each slot gets its
own record with its own propensity (conditional on the slots above it), so
the log stays OPE-able per position, and each slot's outcome resolves
independently — slot 3 got the click, slots 1–2 expire to the default. A
five-line slate builder with honest statistics.

## Guaranteed exposure, with receipts

The permanent epsilon floor means *every* candidate has nonzero probability
on every decision — no vendor, article, or product can be silently starved.
And because propensities are logged, you can *prove* delivered exposure
rates to a partner or regulator from the decision log alone. Exploration
re-read as a fairness/SLA guarantee is a framing most marketplaces would pay
for.

## Read the state as an analytics table

The model is deliberately interpretable: each dimension's weight is "shrunk
average reward per unit of this feature," and its evidence sum `xx_j` is
literally an exposure meter. Hash your known feature names to find their
indices in the (documented, canonical) serialized state, and you get a free
report: *what works* (weights on context-action interactions — "discounts
convert on mobile") and *where traffic went* (evidence mass). No BI
pipeline; the model is the dashboard. Caveat: this one needs a little code —
decoding the serialized format per `spec/serialization.md` — so it is
"possible and neat" rather than free.

## Feature importance without experiments

A special case of the challenger trick worth naming: evaluate, via OPE on
your existing log, a challenger policy identical except with a feature
*removed*. The performance drop is that feature's value, measured
counterfactually — feature ablation studies without ever running a live
test or retraining infrastructure.

## The reward is a formula you own

Reward is just a float you compute at resolution time, so multi-objective
trade-offs (clicks + 0.1 · dwell − cost) are one line, and *re-pricing the
objective* is an edit to that line — the forgetting window then migrates the
model to the new objective automatically over ~1,000 decisions, no
retraining ceremony. Practical corollary: log the raw outcome components
alongside the resolution, so future-you can re-derive what any decision
*would* have earned under a new formula.

## Warm starts from synthetic worlds

A state produced by simulation is a real state. Encode your prior beliefs
("discounts probably work on mobile") as a small simulator, train a fresh
bandit against it at engine speed for a few seconds, and ship the resulting
state string as the launch model. Day-one behavior reflects your priors
instead of uniform randomness, and wherever the priors are wrong, forgetting
washes them out within about one effective window of real traffic. Warm
starts without a single real observation, and without any dedicated
warm-start API.

## Power planning by simulation

The mirror image: before launch, simulate the problem with your *expected*
reward gaps (a 12% vs 10% CTR difference, say) and count how many decisions
the bandit needs to reliably separate the arms. Because the engine runs
hundreds of thousands of decisions per second and is deterministic, this is
a seconds-long, repeatable answer to the stakeholder question "how long
until it works?" — sample-size planning for bandits, the way power analysis
serves A/B tests.

## Expire is a smoke alarm

`expire` returns its resolutions, so the expiry rate is a free, live health
metric. A sudden spike in decisions timing out to `default_reward` means one
thing: rewards stopped arriving — a broken analytics pipeline, a dead queue,
a consent-banner change. The model quietly protecting itself (training the
declared default instead of nothing) is also your outage detector, at zero
instrumentation cost.

## Debias any dashboard with logged propensities

Offline policy evaluation is not the propensities' only use. Inverse
propensity weighting turns any per-decision statistic into an unbiased
"as if shown equally" estimate: the true CTR of each banner corrected for
how often the policy actually showed it, revenue per action without
survivorship bias, and so on. Most reporting on top of adaptive systems is
silently biased by the adaptation itself; a propensity-stamped log is the
antidote, and every record already carries the stamp.

## The model fits in a bug report

A few kilobytes of hex string is small enough to paste into a ticket, an
email, or a chat message. "Attach your bandit": a user exports the state
from localStorage, support loads it in a Python notebook, and — because
behavior is bit-identical across bindings — reproduces the exact decisions
the user saw. The gap between "works on my machine" and "works on yours" is
deleted by making the entire learner a copy-pasteable value.
