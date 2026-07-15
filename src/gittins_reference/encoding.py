"""Feature encoding: everything is hashed (design doc D2).

The engine never models "arm number 7" — and after this module, it never
models "the price field" either. Context and action features arrive as
plain dicts of whatever the caller has; every feature, the arm's identity,
the intercept, and every interaction term hashes into one fixed space of
2**bits dimensions. The single up-front declaration is `bits`. There is no
schema, no field registration, and nothing to migrate: new field names, new
categorical values, and new arms are all usable the moment they first
appear, and evidence on anything that stops appearing decays out of the
sums, recycling its dimensions (R1). Fixed state memory under unbounded
feature and arm churn, by construction.

**Tokens.** Each supplied (name, value) becomes one token in its namespace
("c|" context, "a|" action):

    string value          ->  token "ns|name=value", contribution 1.0
    numeric value         ->  token "ns|name",       contribution float(value)
    None                  ->  skipped (absent)

so categoricals need no declared value lists — the value is part of the
token. The arm's identity is one more token, "i|arm_id", contribution 1.0:
identity really is just another feature (the original VW move). Bools count
as numeric (True = 1.0).

**The encoding is an outer product of token lists.** With a linear model, a
context-only dimension adds the same amount to every candidate's score and
can never reorder candidates, so context must interact with action features
to matter. Both sides get a leading bias token (the empty string), and
every left x right pair contributes:

    index = hash(left_token, right_token)  masked to bits
    x[index] += sign(hash) * left_value * right_value

The bias tokens make the pieces fall out of the one rule: bias x bias is the
intercept, bias x action the action main effects, context x bias the context
main effects, the rest the context-action interactions. The pair hash is
FNV-1a over the two token strings joined by a separator byte, finished with
the splitmix64 mixer (both from rng.py); the top hash bit gives the sign.

**Collisions are the accepted cost, not an error.** Distinct pairs may land
in the same slot and their contributions add; the sign makes colliding
features as often opposed as conflated, so collisions blur rather than
bias. The cost is bounded by `bits` — size the space generously; linear
state is cheap. Hashes depend only on the tokens, never on a salt or an
agent, so every agent encodes identically and merged sums stay aligned (D3).

**Determinism.** Features are processed in sorted token order, so the
accumulation order — and therefore every output bit — is independent of
dict insertion order, identical on every platform.

The reference materializes the dense 2**bits vector for clarity; the
compiled core will keep the handful of nonzero (index, value) pairs sparse.
"""

from gittins_reference.rng import fnv1a_64, mix64

# Separator between the two tokens of a pair; no printable token contains it.
PAIR_SEP = b"\x1f"


def feature_tokens(namespace: str, values: dict) -> "list[tuple[str, float]]":
    """(token, contribution) per supplied feature, in sorted token order."""
    out = []
    for name in sorted(values):
        v = values[name]
        if v is None:
            continue
        if isinstance(v, str):
            out.append((f"{namespace}|{name}={v}", 1.0))
        elif isinstance(v, (int, float)):
            out.append((f"{namespace}|{name}", float(v)))
        else:
            raise ValueError(f"feature {name!r} has unsupported type {type(v).__name__}")
    return out


def pair_hash(left_token: str, right_token: str) -> int:
    return mix64(fnv1a_64(left_token.encode("utf-8") + PAIR_SEP + right_token.encode("utf-8")))


def encode(context: dict, arm_id: str, action: dict, bits: int) -> list[float]:
    """The feature vector for one (context, candidate) pair: the hashed
    outer product described in the module docstring. The model dimension is
    2**bits (pass to `new_bandit`)."""
    if not (1 <= bits <= 24):
        raise ValueError("bits must be between 1 and 24")
    mask = (1 << bits) - 1
    left = [("", 1.0)] + feature_tokens("c", context)
    right = [("", 1.0)] + feature_tokens("a", action) + [(f"i|{arm_id}", 1.0)]
    x = [0.0] * (1 << bits)
    for left_token, left_value in left:
        for right_token, right_value in right:
            h = pair_hash(left_token, right_token)
            sign = 1.0 if (h >> 63) == 0 else -1.0
            x[h & mask] += sign * left_value * right_value
    return x
