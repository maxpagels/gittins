# Next steps

## Residual updates for the reward model

Keep the forgetting-discounted `xx_j` sums and use them as each dimension's step size. Replace `xy` with a weight vector updated by the residual:

```
ŷ    = Σ v_j w_j
w_j += v_j (y − ŷ) / (scale·xx_j + ridge)
```

When only one feature is active, this gives exactly today's running ridge average, without forgetting: `w_n = w_{n−1} + (y − w_{n−1})/(n + ridge)` reproduces `S_n/(n+ridge)` exactly, and with forgetting it's close. So cold start, ridge shrinkage, scale invariance and the global fading all stay the same. When features fire together, they share credit, which removes the feature-count bias.

One caveat: when k new features fire at once, each takes a step of about 1 and ŷ overshoots by about k. So the step needs dividing by `Σ_j v_j² / (xx_j + ridge)`, the normalised-LMS trick, similar to VW's normalised updates. This still costs O(nonzeros), keeps `forgetfulness` meaning what it means today, and is still deterministic.

### Notes

- **When `xx_j` gets updated.** The exact equivalence to today's running average only holds if `xx_j` already includes the current observation when the step is computed. So the update has to add to `xx_j` first and then compute the step.
- **The normalisation changes single-feature behaviour.** The paragraph above says to divide by `Σ_j v_j² / (xx_j + ridge)`. With one active feature, that makes the update close the whole residual in one go. That no longer matches today's running average. The alternative is to cap the effective step at `max_j 1/(xx_j + ridge)` instead.
