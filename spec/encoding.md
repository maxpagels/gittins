# Spec: feature encoding

Design doc reference: D2 (arms are described by features, not identified by
index), open question 2 (resolved: full feature hashing — see the decision
note below), section 13 (the collision risk). Implemented in
`src/gittins_reference/encoding.py` (PR 8).

## One declaration: `bits`

Everything — feature names, categorical values, arm identity, the
intercept, every interaction term — hashes into one space of `2**bits`
dimensions (`1 <= bits <= 24`; the model dimension for `new_bandit` is
`2**bits`). There is no schema, no field registration, and nothing to
migrate. New field names, new categorical values, and new arms are usable
the moment they first appear; anything that stops appearing is forgotten
out of
the sums and its dimensions are recycled. State memory is fixed under
unbounded feature and arm churn (R1).

**Decision note (open question 2).** The design doc's earlier lean was an
explicit schema for descriptive features. Resolved the other way,
2026-07-15: everything hashed. Rationale: one integer replaces schema
declaration, migration, and unknown-name policy; feature sets may vary
freely per decision; the identity block stops being a special case. The
traded-away safety (a typo'd field name now silently hashes somewhere
instead of erroring) is accepted as part of the hashing bargain.

## Tokens

Each supplied (name, value) pair becomes one token in its namespace —
`c|` for context, `a|` for action:

    string value   ->  "ns|name=value"  contribution 1.0   (categorical)
    numeric value  ->  "ns|name"        contribution float(value)
    bool           ->  numeric (True = 1.0)
    None           ->  skipped (absent)
    anything else  ->  error

The arm's identity is one more token, `"i|arm_id"`, contribution 1.0 —
identity is literally just another feature. Values are expected roughly
unit-scale (the model's ridge prior assumes it); string tokens are ±1 by
construction, numerics are the caller's job.

## The encoding: a hashed outer product

With a linear model, a context-only dimension adds the same amount to every
candidate's score and can never reorder candidates, so context must
interact with action features to matter. Both sides get a leading bias
token (the empty string), and every left x right pair contributes once:

    left  = [bias] + context tokens          (sorted token order)
    right = [bias] + action tokens + identity
    h     = mix64(fnv1a_64(utf8(left_token) + 0x1F + utf8(right_token)))
    x[h & (2**bits - 1)]  +=  sign(h) * left_value * right_value

where `sign(h)` is +1 if bit 63 of `h` is 0, else -1 (hash primitives from
`spec/rng.md`). The bias tokens make the pieces fall out of one rule:
bias x bias is the intercept, bias x action the action main effects,
context x bias the context main effects, and the rest the context-action
interactions.

Hashes depend only on the tokens — no salt, no per-agent state — so every
agent encodes identically and logged experience pools offline (D3). Tokens are
processed in sorted order, so accumulation order and every output bit are
independent of dict insertion order and identical across platforms.

## Collisions

Distinct pairs may share a slot; their contributions **add**. The sign bit
makes colliding features as often opposed as conflated, so collisions blur
estimates rather than bias them, and the cost is bounded by `bits` — size
the space generously, linear state is cheap (design doc section 13). The
golden vector below pins a live collision on purpose: collision behavior
is part of the contract, not an accident.

The reference materializes the dense `2**bits` vector for clarity; the
compiled core will keep the nonzero (index, value) pairs sparse.

## Golden vectors

`bits = 4` (dim 16). Context `{hour: 0.5, device: "mobile"}`, arm
`banner-sale`, action `{color: "red", price: 1}`:

    [1.0, 0.5, 1.5, 0.0, 0.0, 0.0, 0.5, 0.0,
     1.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0]

(the 1.5 cell is two contributions, 1.0 + 0.5, sharing a slot). And
`bits = 1` forces `({c: 2}, "z", {a: 3})`'s six contributions into two
slots: `[-4.0, -5.0]`.
