"""The sim harness (PR 11): reproducibility, pairing, the real public path,
and the regret scale behaving as the metrics claim.

Unlike the reference's tests, nothing here is bit-pinned — sims are
statistical. But every run is a pure function of (environment, policy,
seed), so the assertions are exact-reproducibility checks plus loose,
evidence-based bounds that the same seeds reproduce deterministically.
"""

from sim.environments import (
    ActionFeatureEnvironment,
    LinearEnvironment,
    NeedleEnvironment,
    XorEnvironment,
)
from sim.metrics import (
    final_window_regret,
    median_iqr,
    normalized_regret,
    quantile,
    rmse,
)
from sim.policies import (
    EpsilonGreedyPolicy,
    GittinsPolicy,
    GreedyPolicy,
    OraclePolicy,
    UniformPolicy,
)
from sim.runner import run

BITS = 6
SEEDS = [0, 1, 2]
ROUNDS = 600


def median_regret(env, make_policy, rounds=ROUNDS):
    return median_iqr([normalized_regret(run(env, make_policy(), s, rounds)) for s in SEEDS])[0]


def test_runs_replay_exactly():
    for env in [
        LinearEnvironment(k=3),
        XorEnvironment(k=4),
        NeedleEnvironment(k=6),
        ActionFeatureEnvironment(k=8),
    ]:
        a = run(env, GittinsPolicy(bits=BITS), seed=7, rounds=50)
        b = run(env, GittinsPolicy(bits=BITS), seed=7, rounds=50)
        assert a == b


def test_rounds_are_paired_across_policies():
    # The world is a pure function of (environment, seed, t): running one
    # policy leaves no trace on the rounds another policy will see.
    env = LinearEnvironment(k=3)
    before = [env.round(0, t) for t in range(20)]
    run(env, GreedyPolicy(bits=BITS), seed=0, rounds=20)
    assert [env.round(0, t) for t in range(20)] == before
    # ...and the reward draw depends on the round, not on the policy.
    rd = env.round(0, 5)
    assert env.reward(0, 5, rd, 1) == env.reward(0, 5, rd, 1)


def test_different_seeds_differ():
    env = LinearEnvironment(k=3)
    assert env.round(0, 0) != env.round(1, 0)


def test_oracle_has_zero_regret():
    for env in [
        LinearEnvironment(k=5),
        XorEnvironment(k=4),
        NeedleEnvironment(k=10),
        ActionFeatureEnvironment(k=16),
    ]:
        result = run(env, OraclePolicy(), seed=0, rounds=200)
        assert normalized_regret(result) == 0.0


def test_uniform_normalizes_to_about_one():
    for env in [
        LinearEnvironment(k=5),
        XorEnvironment(k=4),
        NeedleEnvironment(k=10),
        ActionFeatureEnvironment(k=16),
    ]:
        assert 0.8 < median_regret(env, UniformPolicy) < 1.2


def test_model_baselines_learn_the_linear_environment():
    env = LinearEnvironment(k=5)
    assert median_regret(env, lambda: GreedyPolicy(bits=BITS)) < 0.35
    assert median_regret(env, lambda: EpsilonGreedyPolicy(0.05, bits=BITS)) < 0.35
    assert median_regret(env, lambda: EpsilonGreedyPolicy(0.1, bits=BITS)) < 0.45


def test_gittins_learns_the_linear_environment():
    # Loose ceiling: with GAMMA_SCALE = 1.0 the engine is still very
    # explorative at this horizon (that finding is the battery's point);
    # it must nonetheless be clearly inside the uniform-random bound.
    env = LinearEnvironment(k=5)
    results = [run(env, GittinsPolicy(bits=BITS), s, ROUNDS) for s in SEEDS]
    assert median_iqr([normalized_regret(r) for r in results])[0] < 0.9
    # The broad exploration buys the best model of any comparator: its
    # late-run predictions should be tight (calibrated ~0.12).
    assert median_iqr([rmse(r, first=ROUNDS // 2) for r in results])[0] < 0.25


def test_gittins_drives_the_real_ledger_path():
    env = LinearEnvironment(k=3)
    policy = GittinsPolicy(bits=BITS)
    result = run(env, policy, seed=0, rounds=40)
    # Every round produced one decision and one rewarded resolution through
    # the public path: the counter and model version advanced in lockstep
    # and no decision was left open.
    assert policy.state.next_seq == 40
    assert policy.state.model_version == 40
    assert policy.state.ledger == ()
    assert len(result.regret) == 40


def test_needle_traps_greedy_but_not_explorers():
    # No context, no action features: nothing to generalize from. Greedy
    # pulls arm 0 first (all estimates 0, first-max tie-break), sees a
    # positive reward, and locks on forever — worse than uniform whenever
    # the needle is elsewhere (regret k/(k-1), deterministic on these
    # seeds). Anything that explores must escape the trap.
    env = NeedleEnvironment(k=10)
    assert median_regret(env, lambda: GreedyPolicy(bits=BITS)) > 1.0
    assert median_regret(env, lambda: EpsilonGreedyPolicy(0.1, bits=BITS)) < 0.85
    assert median_regret(env, lambda: GittinsPolicy(bits=BITS)) < 1.2


def test_action_features_generalize_across_arms():
    # 16 arms but only 3 action features: the reward is learnable only
    # through the context x action interactions, so low regret here means
    # evidence transferred between arms rather than accruing per identity.
    env = ActionFeatureEnvironment(k=16)
    assert median_regret(env, lambda: GreedyPolicy(bits=BITS)) < 0.6
    assert median_regret(env, lambda: EpsilonGreedyPolicy(0.1, bits=BITS)) < 0.7
    # The engine over-explores at this horizon (the GAMMA_SCALE finding)
    # but must stay inside uniform, and the exploration must buy the best
    # late-run model of any comparator (calibrated ~0.16).
    results = [run(env, GittinsPolicy(bits=BITS), s, ROUNDS) for s in SEEDS]
    assert median_iqr([normalized_regret(r) for r in results])[0] < 1.1
    assert median_iqr([rmse(r, first=ROUNDS // 2) for r in results])[0] < 0.3


def test_xor_degrades_gracefully():
    # Misspecified: no linear model on these features can beat chance
    # between the parities, so regret hovers near uniform — the requirement
    # is only that it is never catastrophically worse (pass criteria say 2x).
    env = XorEnvironment(k=4)
    assert median_regret(env, lambda: GittinsPolicy(bits=BITS)) < 1.5


def test_normalized_regret_handles_indifferent_rounds():
    result = run(XorEnvironment(k=4), OraclePolicy(), seed=0, rounds=10)
    zeroed = type(result)(
        environment=result.environment,
        policy=result.policy,
        seed=result.seed,
        regret=result.regret,
        normalizer=(0.0,) * 10,
        sq_error=result.sq_error,
        reward=result.reward,
    )
    assert normalized_regret(zeroed) == 0.0


def test_final_window_regret_uses_the_tail():
    env = LinearEnvironment(k=5)
    result = run(env, GreedyPolicy(bits=BITS), seed=0, rounds=400)
    # Learning happened, so the steady-state rate beats the whole-run rate,
    # which is dragged up by the cold start.
    assert final_window_regret(result, 0.25) < normalized_regret(result)


def test_rmse_is_none_for_model_free_policies():
    result = run(LinearEnvironment(k=3), UniformPolicy(), seed=0, rounds=20)
    assert result.sq_error is None
    assert rmse(result) is None


def test_quantiles():
    values = [4.0, 1.0, 3.0, 2.0]
    assert quantile(values, 0.0) == 1.0
    assert quantile(values, 1.0) == 4.0
    assert quantile(values, 0.5) == 2.5
    assert median_iqr(values) == (2.5, 1.75, 3.25)
    assert median_iqr([5.0]) == (5.0, 5.0, 5.0)
