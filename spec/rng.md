# Spec: counter-based random number generation

## Why counter-based

The engine draws no randomness of its own. Every random number is a pure
function of a **key** (fixed per decision, derived from the decision ID and the
caller-supplied salt) and a **counter** (0, 1, 2, … within that decision). Any
historical decision can therefore be reproduced in isolation, without replaying
the decisions before it. There is no generator state to store, corrupt, or
desynchronize.

## Algorithm

All arithmetic is unsigned 64-bit, wrapping on overflow.

**Key derivation.** Encode the decision ID and salt as UTF-8. Build the message
`len(id_bytes)` as 8 little-endian bytes, then `id_bytes`, then `salt_bytes`
(length-prefixing makes the (id, salt) boundary unambiguous). Hash the message
with FNV-1a 64:

```
h = 0xCBF29CE484222325
for each byte b:  h = (h XOR b) * 0x100000001B3
```

**Draw at position `counter`** (splitmix64, Steele/Lea/Flood 2014):

```
x = key + counter * 0x9E3779B97F4A7C15
x = (x XOR (x >> 30)) * 0xBF58476D1CE4E5B9
x = (x XOR (x >> 27)) * 0x94D049BB133111EB
x = x XOR (x >> 31)              -- this is random_u64(key, counter)
```

**Uniform float in [0, 1).** Take the top 53 bits and scale:

```
random_unit(key, counter) = (random_u64(key, counter) >> 11) * 2^-53
```

Every step is exact in IEEE-754 double precision, so results are bit-identical
across platforms and languages.

## Golden vectors

With `key = derive_key("decision-001", "salt-A")`:

| quantity | value |
|---|---|
| `fnv1a_64(b"")` | `0xCBF29CE484222325` |
| `fnv1a_64(b"abc")` | `0xE71FA2190541574B` |
| `mix64(1)` | `0x5692161D100B05E5` |
| `key` | `0x93A23219EDB6E287` |
| `derive_key("decision-001", "salt-B")` | `0x93A23319EDB6E43A` |
| `random_u64(key, 0)` | `0xC6608EC67B2DA337` |
| `random_u64(key, 1)` | `0x722B4FD724B7F479` |
| `random_u64(key, 2)` | `0x4A3B5D64C3443582` |
| `random_unit(key, 0)` | `0.7749108538220555` |
| `random_unit(key, 1)` | `0.44597338678860843` |
| `random_unit(key, 2)` | `0.2899683352473097` |

## Notes and accepted trade-offs

- `mix64(0) = 0` (the splitmix64 finalizer fixes zero). Harmless here: keys come
  from FNV-1a, which never yields the exact values that would finalize to zero
  in practice, and no security property is claimed.
- This RNG is **not cryptographic** and is not meant to be; it only needs to be
  uniform enough for exploration sampling, and above all reproducible.
