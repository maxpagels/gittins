# gittins

Gittins is an opinionated, highly optimised contextual bandit engine that aims to address the practical considerations with such systems. It stands on the shoulders of giants, in particular
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

Live documentation and user guide:
[docs.getgittins.dev](https://docs.getgittins.dev).

Named after John Gittins, whose index (1974) established that exploration has a precise, computable value.

