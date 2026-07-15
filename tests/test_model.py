import math

from gittins_reference.decay import RENORM_LIMIT
from gittins_reference.model import (
    LinearModel,
    cholesky,
    dot,
    merge_models,
    new_model,
    predict,
    solve_cholesky,
    update,
)

HOUR = 3600.0
T0 = 1_752_000_000.0


class TestSolver:
    def test_solves_a_known_system(self):
        # A = [[4,2],[2,3]], b = [10,7]  =>  x = [2, 1]
        lower = cholesky([[4.0, 2.0], [2.0, 3.0]])
        x = solve_cholesky(lower, [10.0, 7.0])
        assert math.isclose(x[0], 2.0, rel_tol=1e-12)
        assert math.isclose(x[1], 1.0, rel_tol=1e-12)

    def test_identity_factors_to_itself(self):
        lower = cholesky([[1.0, 0.0], [0.0, 1.0]])
        assert lower == [[1.0, 0.0], [0.0, 1.0]]


class TestPredict:
    def test_fresh_model_predicts_zero_with_unit_uncertainty(self):
        # With ridge = 1 and a unit feature vector, the prior gives
        # estimate 0 and uncertainty sqrt(x . x / ridge) = 1, exactly.
        est, unc = predict(new_model(3, HOUR, T0), [1.0, 0.0, 0.0], T0)
        assert est == 0.0
        assert unc == 1.0

    def test_recovers_a_linear_relationship(self):
        # True weights [2, -1, 0.5]; noiseless rewards; dense repeated data
        # over a few minutes (negligible decay at a one-hour half-life).
        w = [2.0, -1.0, 0.5]
        xs = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.5, -1.0, 1.0],
        ]
        m = new_model(3, HOUR, T0)
        for rep in range(40):
            for k, x in enumerate(xs):
                m = update(m, x, dot(x, w), T0 + rep * len(xs) + k)
        probe = [1.0, 2.0, -1.0]
        est, unc = predict(m, probe, T0 + 200.0)
        assert math.isclose(est, dot(probe, w), rel_tol=0.02)
        assert unc < 0.3

    def test_incremental_sums_match_batch_sums_bit_for_bit(self):
        # The incremental bookkeeping must equal a directly computed batch
        # sum with the same decay weights, summed in the same order.
        from gittins_reference.detmath import exp2

        data = [([1.0, 0.5], 1.0, T0), ([0.5, -1.0], -0.5, T0 + 60), ([1.0, 1.0], 2.0, T0 + 120)]
        m = new_model(2, HOUR, T0)
        for x, r, t in data:
            m = update(m, x, r, t)

        xx = [0.0] * 4
        xy = [0.0] * 2
        for x, r, t in data:
            wt = exp2((t - T0) / HOUR)
            for i in range(2):
                for j in range(2):
                    xx[i * 2 + j] += x[i] * x[j] * wt
                xy[i] += r * x[i] * wt
        assert list(m.xx.values) == xx
        assert list(m.xy.values) == xy


class TestUncertainty:
    def test_uncertainty_shrinks_with_evidence(self):
        m = new_model(2, HOUR, T0)
        _, before = predict(m, [1.0, 0.0], T0)
        for i in range(20):
            m = update(m, [1.0, 0.0], 1.0, T0 + i)
        _, after = predict(m, [1.0, 0.0], T0 + 20)
        assert after < before / 3

    def test_uncertainty_is_directional(self):
        m = new_model(2, HOUR, T0)
        for i in range(50):
            m = update(m, [1.0, 0.0], 1.0, T0 + i)  # evidence only along axis 0
        _, seen = predict(m, [1.0, 0.0], T0 + 50)  # ~ 1/sqrt(51)
        _, unseen = predict(m, [0.0, 1.0], T0 + 50)  # prior: 1.0
        assert unseen > 5 * seen

    def test_uncertainty_grows_back_during_a_data_gap(self):
        # The decay floor on certainty (R2): confidence is confidence in
        # recent evidence, so silence re-opens the error bars.
        m = new_model(2, HOUR, T0)
        for i in range(50):
            m = update(m, [1.0, 0.0], 1.0, T0 + i)
        _, fresh = predict(m, [1.0, 0.0], T0 + 50)
        _, stale = predict(m, [1.0, 0.0], T0 + 50 + 20 * HOUR)
        assert stale > 5 * fresh
        assert math.isclose(stale, 1.0, rel_tol=1e-3)  # back to ~the prior


class TestForgetting:
    def test_adapts_when_the_world_flips(self):
        # An hour-half-life model taught y = +x, then y = -x starting ten
        # half-lives later, ends up predicting the new world.
        m = new_model(1, HOUR, T0)
        for i in range(50):
            m = update(m, [1.0], 1.0, T0 + i)
        flip = T0 + 10 * HOUR
        for i in range(50):
            m = update(m, [1.0], -1.0, flip + i)
        est, _ = predict(m, [1.0], flip + 50)
        assert est < -0.9


class TestMerge:
    def test_merge_equals_central_model(self):
        a = new_model(2, HOUR, T0)
        b = new_model(2, HOUR, T0)
        central = new_model(2, HOUR, T0)
        stream_a = [([1.0, 0.0], 1.0, T0 + 10), ([1.0, 1.0], 0.5, T0 + 30)]
        stream_b = [([0.0, 1.0], -1.0, T0 + 20)]
        for x, r, t in stream_a:
            a = update(a, x, r, t)
            central = update(central, x, r, t)
        for x, r, t in stream_b:
            b = update(b, x, r, t)
            central = update(central, x, r, t)
        merged = merge_models(a, b)
        probe = [1.0, -0.5]
        est_m, unc_m = predict(merged, probe, T0 + 60)
        est_c, unc_c = predict(central, probe, T0 + 60)
        assert math.isclose(est_m, est_c, rel_tol=1e-12)
        assert math.isclose(unc_m, unc_c, rel_tol=1e-12)

    def test_merge_is_commutative_bit_for_bit(self):
        a = update(new_model(2, HOUR, T0), [1.0, 0.5], 1.0, T0 + 10)
        b = update(new_model(2, HOUR, T0), [0.5, -1.0], -1.0, T0 + (RENORM_LIMIT + 9) * HOUR)
        assert merge_models(a, b) == merge_models(b, a)

    def test_merge_rejects_mismatches(self):
        import pytest

        with pytest.raises(ValueError):
            merge_models(new_model(2, HOUR, T0), new_model(3, HOUR, T0))
        with pytest.raises(ValueError):
            merge_models(new_model(2, HOUR, T0), new_model(2, HOUR, T0, ridge=2.0))


class TestLateRewards:
    def test_two_updates_commute_bit_for_bit(self):
        # Mirrors the decay-layer guarantee at the model level: within one
        # renormalization era, a late observation produces identical bits.
        obs1 = ([1.0, 0.5], 1.0, T0)
        obs2 = ([0.5, -1.0], -0.5, T0 + 9 * HOUR)
        m12 = update(update(new_model(2, HOUR, T0), *obs1), *obs2)
        m21 = update(update(new_model(2, HOUR, T0), *obs2), *obs1)
        assert m12 == m21


class TestPinnedVectors:
    # Golden vectors: bit-exact prediction after a fixed little history.
    def test_prediction_bits(self):
        m = new_model(2, HOUR, T0)
        m = update(m, [1.0, 0.0], 1.0, T0)
        m = update(m, [0.0, 1.0], -0.5, T0 + 1800.0)
        m = update(m, [1.0, 1.0], 0.25, T0 + 3600.0)
        est, unc = predict(m, [1.0, -1.0], T0 + 5400.0)
        assert est == 0.4318473976784247
        assert unc == 1.1847437129969538
