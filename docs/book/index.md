# Gittins
## By [Max Pagels](https://maxpagels.com)

[VERSION]

[SIM]

Gittins is an opinionated, highly optimised contextual bandit engine that aims to address the practical considerations with such systems, based my experience working with bandit problems. It stands on the shoulders of giants, in particular
[Vowpal Wabbit](https://vowpalwabbit.org/), and adheres strictly to a design in support of real-world production use. Gittins is not a research tool.

1. **Online by nature**. Gittins learns one observation at a time, in O(1) work, and in fixed memory as long as open decisions are regularly resolved.
2. **Non-stationarity is expected**. For many real-world problems, the relationship between context and feedback drifts over time. A contextual bandit engine must learn to adapt over time, and never learn something it cannot eventually unlearn.
3. **Dynamic actions and context.** If you want to choose what banner to display on your website, and the set changes each day, an engine must accept this, and clean up after itself. There should never be a case where you must specify the number of actions beforehand.
4. **Simple algorithms, bring-your-own models.** The built-in algorithms should be readable by any competent programmer, and work in practice. Sophistication lives in the layering and the API, not in any single component. Users can swap in their own prediction model and/or exploration algorithm, and inherit everything else.
5. **Safe reward handling**. Rewards in bandits may arrive late, or not at all. Constructing invalid training data from logs or external sources must be nigh on impossible by design, not merely discouraged.
6. **Speed and determinism**. Fast decision cycles allow for unexpected use cases. Gittins must have best-in-class single core performance, and rely on few to zero dependencies. Bit-identical results across platforms and language bindings must be guaranteed and enforced by a golden test corpus. Code changes must be validated by tens of thousands of simulations and a large test battery.
7. **Multislot and large action sets.** Many problems are ranking / multi-position problems; therefore problems with thousands of candidate actions must be practical, fast, and robust.
8. **Offline policy evaluation.** It must be possible to estimate how a new policy *would have* performed using only logged decisions from an old policy.
9. **Choose your own complexity.** The same model needs to be able to run (a) ephemerally in memory, (b) persisted to a flat file, or (c) used via a shared service. There should be no need for databases; indeed, the model weights should be possible to check in to version control and deploy as part of normal deployment workflows.

These concepts are discussed throughout this document, which serves both as an introduction to bandits and as technical documentation for Gittins. I encourage the reader to read the whole document once; once you have understood the concepts on a broad level, use the table of contents below to jump to sections you need to reference.

[TOC]

---

## The Contextual Bandit Problem

In the contextual bandit setting, you (repeatedly):

1. Observe a **context**
2. Choose an **action**
3. Receive a **reward** for the action you chose

The objective is to learn to select the best action for any given context such that you **maximise total reward over time**. From this problem setting, it follows that contextual bandit algorithms must try out different actions to learn what context/action pairs lead to the best rewards; they must _explore_ the action space using some exploration algorithm, and also exploit their knowledge using some learned model.

Some bandit algorithms explore first, then exploit. Some learn to explore less over time, converging at some point on the optimal action for a given context. Gittins uses a simple algorithm known as epsilon-greedy, where exploration always happens for a small portion of decisions, regardless of how long the bandit has been running. This is a deliberate choice to tackle non-stationarity, an issue I revisit later in this document.

You may be thinking to yourself, "why not use supervised learning instead of bandits?". That is a fair question, and for some problems, it works well. But consider a case of topic recommendation on social media. If you use supervised learning to choose the best topic from a handful of candidates, you must a) have enough cover in your training data to learn the optimal relationships, however obscure, snd b) have some way of being able to recommend entirely new topics that you don't have history for. Bandits offer a natural way of introducing new actions by virtue of its explorative design.

---

## Your First Decision

Gittins comes in three flavours: a (slow) reference Python implementation, which is not packaged and not recommended outside development of the system; Python bindings, for most data science workflows; and WASM bindings, for in-browser decisions. The bindings call a core written in Rust for performance reasons.

Whichever flavour you pick, the interface is the same nine functions with the same
names and the same semantics. A state saved by one loads in the others,
bit for bit. The examples in this document default to Python; use the toggle
above any code block to switch to the WASM (JavaScript) API, and the whole
page follows.

All you need resides in one module, `gittins`. There is no schema to
define, nothing to register, and no background machinery. Start by creating a
bandit. What you get back is a handle: `decide`, `learn`, and `expire` update it in place
and return just their results. The module holds no state of its own — the handle *is* the bandit,
so you can hold several, snapshot one, or ship one across languages.
Passing a handle costs nothing (it is a reference to the one bandit in memory); only `serialize` ever copies the bandit state.

```python
import gittins
import time

state = gittins.create(bits=8, horizon=3600.0)
```

```js
import init, * as gittins from "./pkg/gittins_wasm.js";
await init();

let state = gittins.create(8, 3600.0); // bits, horizon in seconds
```

`bits` sets how much room the model has for features: it learns in a space of
2^bits slots (here 256), and every feature and action is hashed into it. Nothing needs registering: an action or feature the model has never seen
before will be processed without error, and both may vary from decision to decision. `horizon` answers the question
*how long do we wait for a reward?* — a decision that gets no reward within
`horizon` seconds is treated as having earned `default_reward` (0.0 unless
you say otherwise). There are two more optional settings, `epsilon` (how much
to explore) and `forgetfulness` (how fast old evidence fades); the defaults are sensible,
but tunable with off-policy evaluation.

To make your first decision, describe what you know right now (the context),
list what you could do (the candidates: an action id and that action's features),
and supply the time. The engine never reads a clock itself — you pass your
own `t` — so any run can be replayed exactly.

```python
context = {"device": "mobile", "hour": 14}
candidates = [
    ("banner-sale", {"discount": 0.2}),
    ("banner-new", {"discount": 0.0}),
    ("banner-plain", {}),
]

# ask the bandit which banner to show, right now
record = gittins.decide(
    state, context, candidates,
    t=time.time(),
    salt="bandit-1",
)

chosen_action_id = candidates[record.chosen][0]   # act on this
```

```js
const context = { device: "mobile", hour: new Date().getHours() };
const candidates = [
  ["banner-sale", { discount: 0.2 }],
  ["banner-new", { discount: 0.0 }],
  ["banner-plain", {}],
];

// ask the bandit which banner to show, right now
const record = gittins.decide(
  state,
  context,
  candidates,
  Date.now() / 1000, // t
  "browser-1",       // salt
);

const chosenActionId = candidates[record.chosen][0]; // act on this
```

Feature values are strings for categories, numbers for quantities (booleans
count as 1/0), and null for "not available this time". The `salt` is the
bandit's name; if you run several bandits, give each its own so their decision
ids never collide. Note the return value: a **record** of the decision,
not just a pick: its id, which candidate was chosen including features, and the probability the candidate
was chosen with. Gittins will keep this information in a ledger in memory pending for open decisions, but the record is returned to the you in case you wish to inspect it or construct an experience log for offline training.

Gittins learns only when you resolve a decision, and each decision can be
resolved exactly once. When the reward arrives simply report it by id.

In addition to submitting rewards, remember to call `expire` regularly with the current time so that decisions which waited past the horizon are trained as `default_reward`. If you fail to call expire regularly, the memory usage will continue to grow. It is a deliberate choice to pass the expiration responsibility to you, as your application may have reasons not to expire unresolved records at regular intervals. However, for most applications, you will usually want to call expire inside a timer to ensure the records in memory never grow too large.

```python
import threading

# the user clicked the banner we showed — report the
# outcome as a reward for that decision, by its id
resolution = gittins.learn(
    state, record.decision_id, reward=1.0
)

def sweep():
    # resolve every decision that has waited past the
    # horizon with no reward; each one is trained with
    # default_reward (here: 0.0)
    resolutions = gittins.expire(state, t=time.time())
    # sweep again in a minute
    threading.Timer(60.0, sweep).start()

sweep()
```

```js
// the user clicked the banner we showed — report the
// outcome as a reward for that decision, by its id
const resolution = gittins.learn(
  state, record.decision_id, 1.0
);

setInterval(() => {
  // resolve every decision that has waited past the
  // horizon with no reward; each one is trained with
  // default_reward (here: 0.0)
  const resolutions = gittins.expire(state, Date.now() / 1000);
}, 60_000); // sweep again every minute
```

That is the whole loop: create once, then decide, act, and resolve, forever.
The bandit is learning from the very first decision. To keep what it has
learned across restarts (or browser reloads), snapshot the state — the whole
thing is one plain string, identical on every platform, small enough for a
file, a database column, localStorage, or version control:

```python
from pathlib import Path

Path("bandit.state").write_text(gittins.serialize(state))
# ... later, or on another machine:
state = gittins.deserialize(Path("bandit.state").read_text())
```

```js
localStorage.setItem("bandit", gittins.serialize(state));
// ... later, or after a reload:
state = gittins.deserialize(localStorage.getItem("bandit"));
```

`deserialize` validates everything and refuses to load anything corrupt, and
the format is shared across implementations: a state saved in the browser
will load in Python, and vice versa.

The rest of this document unpacks what each of these calls is actually
doing, and introduces the remaining functions as they become relevant.

---

## Anatomy of a Decision

The previous chapter gave you a brief overview on how to make simple decisions with Gittins, but it is worth knowing what is happening under the hood. Gittins makes deliberate choices to offer a high degree of flexibility. For a decision, this is the internal workflow:

**Feature processing.** The context dictionary and each candidate's feature
dict are first broken into tokens: one token is created per (name, value) pair, and tagged by a namespace: `c` for context or `a` for action. Each token is then hashed into the model's space of 2^`bits` dimensions. This is why nothing is ever registered up front: a feature's position in the model *is* its hash, so a brand-new feature name or action simply hashes somewhere and starts accumulating evidence. Pay attention to your feature names: `Afternoon` and `afternoon` will hash into two different indices, learning separate weights inside the model.

There is one additional subtlety worth understanding. With a linear model, a feature derived from the context alone would add the same amount to *every* candidate's score, and so could never change which candidate wins. Context can only matter through its *interaction* with the actions. Gittins encodes each candidate as a hashed outer product: every context token is crossed with every action token, plus main effects for each. "Mobile user" and "discount banner" each get their own dimension, and
*mobile-user-seeing-a-discount-banner* gets a third. It is precisely this third dimension that lets the model learn that discounts work on mobile even if they fail on average.

**Scoring.** Each candidate's feature vector is scored by the built-in
reward model, which is a ridge regression variant: recursive least squares
with a forgetting factor, kept diagonal (forgetting is important for non-stationary problems; diagonality is for performance). Here is the whole model. Each dimension `j` carries two running sums:
`xx_j`, the sum of that feature's squared values, and `xy_j`, the sum of
that feature's value times the observed reward. On every update, both sums
are first multiplied by the `forgetfulness` factor, so recent observations
always outweigh old ones. A dimension's weight is then

```text
weight_j = xy_j / (xx_j + ridge)
```

which you can read as *the reward observed alongside this feature, divided
by how much evidence we have for it*: a running average of reward per unit
of feature, one per dimension. The `ridge` constant in the denominator acts
as a prior: a dimension nobody has seen has `xy_j = 0` and so predicts
exactly 0.0, and a dimension with little evidence is shrunk toward 0.0
because `ridge` dominates its small `xx_j`. Only as evidence accumulates
does the data term drown out the prior and the weight approach the plain
running average.

The word *diagonal* is doing quiet but important work here. Full ridge
regression would also track how every feature co-varies with every other, a
`dim x dim` matrix, and solve a linear system to untangle correlated
features at prediction time. Gittins keeps only the matrix's diagonal, so
each weight is computed from that dimension's two sums alone, independent of
all others. That is why there is nothing to solve: scoring a candidate is
just the dot product of its features with the weights, one multiply-add per
nonzero feature. There are no matrix inversions, making the engine perfomant with high throughput. The cost of this shortcut is that
features which always fire together each take full credit for the same
reward rather than splitting it; disentangling combinations is therefore the
encoder's job (the interaction dimensions above), not the model's.

**From scores to probabilities.** Scores are fed through the
epsilon-greedy algorithm: the best-scoring candidate gets the lion's share of the probability, and every candidate, including the best, gets an equal slice of the `epsilon` mass. The result is a full probability distribution over the candidates, and this distribution, not just the winner, is what the
engine works with. Exact ties split evenly, which is why a fresh bandit
(every estimate 0.0) starts out choosing uniformly at random.

**Exploration.** The choice is sampled from that distribution using a custom random number generator, keyed on decision id and your `salt`. There is no hidden global random state: the same state, the same inputs, and the same salt produce the same choice, bit for bit, on every platform and in every binding. This is what makes a production incident replayable, and is a core design consideration of Gittins. Subtle differences in math libraries across platforms will not cause issues; if Gittins chooses `a` in the browser for a given model state `s`, it will make the same decision in Python if you load that same model.

Following this internal workflow, what you get back from Gittins after calling `decide` is the decision record, and every field in it is there for
a reason:

| field | why it exists |
| --- | --- |
| `decision_id` | `"{salt}:{seq}"`, the id you resolve this decision with later; unique by construction, no collisions to reason about. |
| `t` | the time you supplied, which starts the horizon clock. |
| `chosen` | the index of the winning candidate. |
| `features` | the chosen candidate's hashed features. The engine keeps its own copy in memory until expiration, which is why `learn` needs only an id and a reward; this copy is returned so you can save the decision log for offline training. |
| `propensity` | the probability the choice was made with; the key that unlocks offline policy evaluation. |
| `candidate_hash` | a fingerprint of the whole candidate set, proving what the alternatives were. |
| `model_version` | how many observations the model had absorbed; it identifies exactly which policy made this decision. |
| `salt` | the RNG key that makes the draw itself replayable. |
| `context`, `candidates`, `bits` | the inputs you passed to `decide`, attached to the returned record. The engine's own memory keeps only the compact record above, so these three are `None` on the record a `train` callback receives (see [Bring Your Own Algorithms](#bring-your-own-algorithms)); the log, not the engine, is where inputs persist. |

Notice what the record makes unnecessary. Because `features` is stored at
decision time, and never passed back when calling `learn` there is no API for hand-assembling training data, and so no
way to construct invalid training data post-decision. Because `propensity` is always
recorded, and the record carries its own inputs, offline evaluation needs no additional information (see
[What Would Have Happened?](#what-would-have-happened)).

---

## Learning to Forget

Consider a toy environment where a checkout button has a click-through rate of 6% if the cart is over 50 euros, and 5% otherwise. It is always 6%, regardless of any other factor, save for random noise. Such a problem is, conditional on a simple boolean that states if the cart is valued over 50 euros or not, _stationary_. It doesn't matter what device shoppers use or what time of day it is – next tuesday, `P(click|cart_over_50_euros)` is still 0.06.

Most problems are not that simple. Purchasing behaviour depends on tons of other factors. `P(click|cart_over_50_euros)` will drift over time, making it a non-stationary problem. We can attempt to fix this problem by adding more features we believe to be associated with click-through rate. Indeed, theoretically, if we conditioned on _everything in the universe_, this problem, and all other problems, become stationary.

In practice, it is unfeasible to control for every eventuality. Non-stationary learning algorithms work on the principle of controlled forgetting: given enough time, old training data is discounted, and its contribution to model weights approaches zero.

Gittins works on non-stationary problems by default. Its online learning setting provides a natural foundation: examples are learned on once, and the core model has a forgetfulness factor that discounts old data. Coupled with exploration that never stops, Gittins will eventually retry an action that was previously learned to be poor, and if recent data suggests otherwise, it will learn to resurface it. You can watch the unlearning happen: the simulation below is the same multi-context, multi-action problem as before, with one addition, a button that makes the world do a 180.

[SIM-FORGET]

How quickly to forget is a question without a fixed answer. It depends on the problem at hand, and must be evaluated using data generated by your bandit, as shown in the next chapter.

---

## What Would Have Happened?

[WIP]

---

## Bring Your Own Algorithms

Both of Gittins' built-in algorithms, ridge regression and epsilon-greedy, are deliberately simple, and effective in practice for many problems. However, if your problem outgrows them, you can swap in your own model and/or exploration and keep everything else. Perhaps you want a gradient-boosted model served over the network, a neural network, or a fancier exploration rule from a paper. Gittins fully supports this, and crucially, does not make you rebuild the plumbing that actually makes bandits hard: the feature encoding, the deterministic sampling, the decision records with logged propensities, the exactly-once reward ledger, and offline evaluation will all keep working.

The entire surface for bring-your-own use is three optional callbacks, passed per call and never stored. Two are parameters passed to `decide`: `score`, which receives the exact `context` and `candidates` you passed in and returns one estimated reward per candidate (replacing the built-in model's predictions), and `explore`, which receives those estimates plus the configured `epsilon` if needed and returns one probability per candidate (replacing epsilon-greedy). The third parameter is passed to `learn` and `expire`: `train`, which receives a resolved decision's record and its reward. The engine has already matched the reward to the right decision, exactly once, in place of the built-in model's update.

```python
model = MyModel()  # anything with a predict and an update

def score(context, candidates):
    return [model.predict(context, action) for _, action in candidates]

def train(record, reward):
    # key your own bookkeeping by record.decision_id, or train
    # directly on record.features if your model lives in the
    # hashed space
    model.update(record.decision_id, reward)

record = gittins.decide(
    state, context, candidates,
    t=time.time(), salt="bandit-1",
    score=score,
)

gittins.learn(state, record.decision_id, reward=1.0, train=train)

# give expire the same callback, so timed-out decisions
# train your model too
gittins.expire(state, t=time.time(), train=train)
```

```js
const model = new MyModel(); // anything with a predict and an update

const score = (context, candidates) =>
  candidates.map(([_, action]) => model.predict(context, action));

const train = (record, reward) => {
  // key your own bookkeeping by record.decision_id, or train
  // directly on record.features if your model lives in the
  // hashed space
  model.update(record.decision_id, reward);
};

const record = gittins.decide(
  state, context, candidates,
  Date.now() / 1000, "browser-1",
  score,
);

gittins.learn(state, record.decision_id, 1.0, train);

// give expire the same callback, so timed-out decisions
// train your model too
gittins.expire(state, Date.now() / 1000, train);
```

Swapping the exploration rule looks the same. Note that your callback only produces the *distribution*; the random draw stays inside the engine, on the decision's own counter-based RNG stream, and the record's propensity is your distribution's value at the chosen index. This is what keeps custom decisions exactly as replayable and log-worthy as built-in ones.

```python
import math

def explore(estimates, epsilon):  # e.g. softmax instead of epsilon-greedy
    weights = [math.exp(v) for v in estimates]
    total = sum(weights)
    return [w / total for w in weights]

record = gittins.decide(
    state, context, candidates,
    t=time.time(), salt="bandit-1",
    explore=explore,
)
```

```js
const explore = (estimates, epsilon) => { // e.g. softmax instead of epsilon-greedy
  const weights = estimates.map(Math.exp);
  const total = weights.reduce((a, b) => a + b, 0);
  return weights.map((w) => w / total);
};

const record = gittins.decide(
  state, context, candidates,
  Date.now() / 1000, "browser-1",
  undefined, // score: keep the built-in model
  explore,
);
```

A few rules safety guarantees intact. First, the engine validates what your callbacks return. There must be one finite estimate per candidate, and probabilities must be nonnegative and sum to 1. Anything is is outright rejected, because a malformed distribution would silently poison every logged propensity, and with it the possibility for offline evaluation. Second, `train` fires *after* the decision is marked resolved: if your callback crashes, that one observation is lost. Third, your model's state is your own: the handle serializes only the engine's state (the ledger, the counters, and the untouched built-in model), so you must persist your model alongside it unless it is intentionally ephemeral. Finally, the engine's own arithmetic stays bit-identical across platforms, but your callbacks might not: a `score` that calls a network, or an `explore` using transcendentals, may not work deterministically from platform to platform, depending on how you implemented it. Gittins places the responsibility for bit-identical results on you if you bring your own algorithms. The same goes for performance: make slow predictions and updates and Gittins won't magically speed them up.

---

## Fast, and the Same Everywhere

Gittins is built by implementing a relatively slow Python reference, that is used to create golden example outputs. The reference has then been ported to a fast Rust-based core, which is interfaced with via WASM or Python bindings. The bindings and the core recreate the reference examples (bit-identically) exactly.

Simply put, this means you can save a model in Python, load it in JavaScript, and be confident that given the exact same input, it will make the exact same decision. And thanks to the Rust core, it will make those decisions incredibly fast, on one core, with fixed memory (assuming the ledger's open decisions are expired). Full benchmark information is available on GitHub, but to give you a sense of the throughput, here is an example simulation for the WASM bindings. One cycle is not just a decision, but a decision _and_ a `learn` call.

[BENCH]

As you can see, and I'll hope you agree, these are high performance numbers. High throughput enables architectures not always apparent; for example, if your use case is ephemeral, you can run Gittins single-threaded in one small virtual machine and serve far more users than the server software in front of it is likely to be able to handle, assuming a typical amount of actions.

---

## Choose Your Own Complexity

[WIP]

---

## Stupid? Gittins Tricks

Some interesting implications fall out of the Gittins design. Some are mere curiousities, others may be genuinely useful. Wrapping up the guide,
here are the ones I could think of, in no particular order.

**Hierarchical bandits**. Since Gittins returns handles to bandits, it becomes simple to create _hierarchical bandits_ where you make a top-level decision, that
itself is a choice of bandit. Make a second decision, receive feedback, and roll the reward up the chain.

**Ensembles for uncertainty**. Hold N handles with different salts on the same traffic. They see the same
data but will explore differently, so *disagreement between them* is a low-cost
uncertainty signal. Route this a human when the committee splits, act confidently when it is
unanimous. N bandits cost N small states and no extra machinery.

**One bandit per user**. An 8-bit state is a few KB in size. It fits a KV-store row, a cookie, or
localStorage. You could build a completely stateless service per user: load state, decide, learn,
save state. This flips the personalisation on its head: if every user has their own bandit, the learned
decisions are personalised by default, even if you supply no context.
  
---
