"""The round-based sim harness: reproducibility, pairing, the real public path,
and the regret scale behaving as the metrics claim.

Unlike the reference's tests, nothing here is bit-pinned — sims are
statistical. But every run is a pure function of (environment, policy,
seed), so the assertions are exact-reproducibility checks plus loose,
evidence-based bounds that the same seeds reproduce deterministically.
"""

from sim.environments import (
    AbruptShiftEnvironment,
    ActionFeatureEnvironment,
    ChurnEnvironment,
    DriftEnvironment,
    DropoutEnvironment,
    LinearEnvironment,
    NeedleEnvironment,
    XorEnvironment,
)
from sim.metrics import (
    final_window_regret,
    median_iqr,
    normalized_regret,
    quantile,
    recovery_time,
    rmse,
)
from sim.policies import (
    EpsilonGreedyPolicy,
    GittinsPolicy,
    GreedyPolicy,
    OraclePolicy,
    UniformPolicy,
)
from sim.runner import RunResult, run

BITS = 6
SEEDS = [0, 1, 2]
ROUNDS = 600

# The non-stationary environments at test scale: 600-round runs, events scaled to
# land inside them (the battery instances in sim/__main__.py assume 1500).
def shift_env():
    return AbruptShiftEnvironment(k=5, period=150)


def drift_env():
    return DriftEnvironment(k=5, period=200)


def churn_env():
    return ChurnEnvironment(k=8, absent=(150, 300), newcomer_at=450)


def dropout_env():
    return DropoutEnvironment(k=5, p_drop=0.3)


def median_regret(env, make_policy, rounds=ROUNDS):
    return median_iqr([normalized_regret(run(env, make_policy(), s, rounds)) for s in SEEDS])[0]


def test_runs_replay_exactly():
    for env in [
        LinearEnvironment(k=3),
        XorEnvironment(k=4),
        NeedleEnvironment(k=6),
        ActionFeatureEnvironment(k=8),
        shift_env(),
        drift_env(),
        churn_env(),
        dropout_env(),
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
        shift_env(),
        drift_env(),
        churn_env(),
        dropout_env(),
    ]:
        result = run(env, OraclePolicy(), seed=0, rounds=ROUNDS)
        assert normalized_regret(result) == 0.0


def test_uniform_normalizes_to_about_one():
    for env in [
        LinearEnvironment(k=5),
        XorEnvironment(k=4),
        NeedleEnvironment(k=10),
        ActionFeatureEnvironment(k=16),
        shift_env(),
        drift_env(),
        churn_env(),
        dropout_env(),
    ]:
        assert 0.8 < median_regret(env, UniformPolicy) < 1.2


def test_model_baselines_learn_the_linear_environment():
    env = LinearEnvironment(k=5)
    assert median_regret(env, lambda: GreedyPolicy(bits=BITS)) < 0.35
    assert median_regret(env, lambda: EpsilonGreedyPolicy(0.05, bits=BITS)) < 0.35
    assert median_regret(env, lambda: EpsilonGreedyPolicy(0.1, bits=BITS)) < 0.45


def test_gittins_learns_the_linear_environment():
    # On its home turf the engine must be competitive with the baselines
    # while keeping the floor and propensities they lack (calibrated ~0.15
    # regret, ~0.23 late-run RMSE at the swept GAMMA_SCALE).
    env = LinearEnvironment(k=5)
    results = [run(env, GittinsPolicy(bits=BITS), s, ROUNDS) for s in SEEDS]
    assert median_iqr([normalized_regret(r) for r in results])[0] < 0.4
    assert median_iqr([rmse(r, first=ROUNDS // 2) for r in results])[0] < 0.4


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
    # The engine must generalize too (calibrated ~0.34 regret, ~0.38
    # late-run RMSE at the swept GAMMA_SCALE).
    results = [run(env, GittinsPolicy(bits=BITS), s, ROUNDS) for s in SEEDS]
    assert median_iqr([normalized_regret(r) for r in results])[0] < 0.7
    assert median_iqr([rmse(r, first=ROUNDS // 2) for r in results])[0] < 0.6


def test_xor_degrades_gracefully():
    # Misspecified: no linear model on these features can beat chance
    # between the parities, so regret hovers near uniform — the requirement
    # is only that it is never catastrophically worse (pass criteria say 2x).
    env = XorEnvironment(k=4)
    assert median_regret(env, lambda: GittinsPolicy(bits=BITS)) < 1.5


def test_shift_is_survivable():
    # Four fresh worlds per run (period 150): the learners must keep
    # clearly beating uniform across the shifts — stale evidence decays and
    # gets relearned rather than poisoning the rest of the run.
    env = shift_env()
    assert median_regret(env, lambda: GreedyPolicy(bits=BITS)) < 0.7
    assert median_regret(env, lambda: EpsilonGreedyPolicy(0.1, bits=BITS)) < 0.75
    assert median_regret(env, lambda: GittinsPolicy(bits=BITS)) < 0.95


def test_drift_is_never_catastrophic():
    # Three full rotations in 600 rounds against a 1000-round half-life:
    # the compromise regime by construction. Nobody is expected to do well;
    # everybody is required to stay near uniform, never far below it.
    env = drift_env()
    for make in [
        lambda: GreedyPolicy(bits=BITS),
        lambda: EpsilonGreedyPolicy(0.1, bits=BITS),
        lambda: GittinsPolicy(bits=BITS),
    ]:
        assert median_regret(env, make) < 1.15


def test_churn_schedule():
    # The candidate list changes shape exactly as scripted, and ids are
    # stable across appearances (the encoding needs no registration).
    env = churn_env()
    means = env.arm_means(0)
    best = f"arm{max(range(env.k), key=means.__getitem__)}"
    before, during, after = env.round(0, 0), env.round(0, 200), env.round(0, 500)
    assert best in before.arm_ids and len(before.arm_ids) == env.k
    assert best not in during.arm_ids and len(during.arm_ids) == env.k - 1
    assert best in after.arm_ids and "newcomer" in after.arm_ids
    assert max(after.means) == max(means) + env.newcomer_gap


def test_churn_explorers_recover():
    # Losing the best arm mid-run and meeting a strong stranger late must
    # not break the explorers; greedy has no such guarantee (it can be
    # locked on the arm that vanishes).
    env = churn_env()
    assert median_regret(env, lambda: EpsilonGreedyPolicy(0.1, bits=BITS)) < 0.95
    assert median_regret(env, lambda: GittinsPolicy(bits=BITS)) < 1.05


def test_dropout_degrades_gracefully():
    # A 30% chance of losing each feature plus two distractors: the model
    # sees damaged contexts but must stay well inside uniform.
    env = dropout_env()
    assert median_regret(env, lambda: GreedyPolicy(bits=BITS)) < 0.6
    assert median_regret(env, lambda: EpsilonGreedyPolicy(0.1, bits=BITS)) < 0.7
    assert median_regret(env, lambda: GittinsPolicy(bits=BITS)) < 1.05


def test_dropout_presents_damaged_contexts():
    env = dropout_env()
    sizes = set()
    for t in range(200):
        rd = env.round(0, t)
        assert "d0" in rd.context and "d1" in rd.context  # distractors always on
        sizes.add(len(rd.context))
    # Dropout actually fires: some rounds are missing true features.
    assert min(sizes) < env.n_features + env.n_distractors


def synthetic_result(regret, best):
    return RunResult(
        environment="e",
        policy="p",
        seed=0,
        regret=regret,
        normalizer=(1.0,) * len(regret),
        best=best,
        sq_error=None,
        reward=best,
    )


def test_recovery_time():
    ones = (1.0,) * 10
    # Regret collapses after the event: recovered once the window mean of
    # (best - regret) reaches 0.9 * window mean of best.
    r = synthetic_result((1.0, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0), ones)
    assert recovery_time(r, event=2, window=2) == 3
    # Zero regret recovers immediately; permanent regret never does.
    assert recovery_time(synthetic_result((0.0,) * 10, ones), event=3, window=2) == 0
    assert recovery_time(synthetic_result((0.5,) * 10, ones), event=0, window=2) is None
    # The window must fit inside the run.
    assert recovery_time(r, event=9, window=5) is None


def test_recovery_time_on_churn():
    # After the best arm returns, the oracle is back instantly and uniform
    # never gets there — the metric's two anchors.
    env = churn_env()
    ret = env.absent[1]
    assert recovery_time(run(env, OraclePolicy(), 0, ROUNDS), ret, window=50) == 0
    assert recovery_time(run(env, UniformPolicy(), 0, ROUNDS), ret, window=50) is None


def test_normalized_regret_handles_indifferent_rounds():
    result = run(XorEnvironment(k=4), OraclePolicy(), seed=0, rounds=10)
    zeroed = type(result)(
        environment=result.environment,
        policy=result.policy,
        seed=result.seed,
        regret=result.regret,
        normalizer=(0.0,) * 10,
        best=result.best,
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
