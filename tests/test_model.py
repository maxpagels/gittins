import math

from gittins_reference.model import RENORM_THRESHOLD, new_model, predict, update


class TestPredict:
    def test_fresh_model_predicts_zero_with_unit_uncertainty(self):
        # With ridge = 1 and a unit feature, the prior gives estimate 0 and
        # uncertainty sqrt(x . x / ridge) = 1, exactly.
        est, unc = predict(new_model(3), [(0, 1.0)])
        assert est == 0.0
        assert unc == 1.0

    def test_weight_is_a_shrunk_running_average(self):
        # With forgetting 1.0 (never forget), 9 observations of feature 0
        # with reward 1: theta_0 = 9 / (9 + 1), uncertainty 1 / sqrt(9 + 1).
        m = new_model(2, forgetting=1.0)
        for _ in range(9):
            m = update(m, [(0, 1.0)], 1.0)
        est, unc = predict(m, [(0, 1.0)])
        assert math.isclose(est, 9.0 / 10.0, rel_tol=1e-15)
        assert math.isclose(unc, math.sqrt(1.0 / 10.0), rel_tol=1e-15)

    def test_recovers_per_feature_weights(self):
        # True weights [2, -1, 0.5], observed one feature at a time,
        # noiseless. Each weight converges independently.
        w = [2.0, -1.0, 0.5]
        m = new_model(3)
        for _ in range(40):
            for j in range(3):
                m = update(m, [(j, 1.0)], w[j])
        probe = [(0, 1.0), (1, 2.0), (2, -1.0)]
        est, unc = predict(m, probe)
        truth = 1.0 * w[0] + 2.0 * w[1] + -1.0 * w[2]
        assert math.isclose(est, truth, rel_tol=0.05)
        assert unc < 0.5

    def test_coordinates_are_independent(self):
        # Evidence about feature 0 changes nothing about feature 1: its
        # estimate and uncertainty stay exactly at the prior.
        m = new_model(2)
        for _ in range(50):
            m = update(m, [(0, 1.0)], 1.0)
        est, unc = predict(m, [(1, 1.0)])
        assert est == 0.0
        assert unc == 1.0

    def test_cofiring_features_double_count(self):
        # The model's documented blind spot: credit is never split. Two
        # features always seen together with reward 1.0 each converge to the
        # full shrunk average (10/11 after 10 never-forget observations), so
        # together they predict 20/11 ~ 1.8 where the truth is 1.0.
        # Disentangling combinations is the encoder's job (D2).
        m = new_model(2, forgetting=1.0)
        for _ in range(10):
            m = update(m, [(0, 1.0), (1, 1.0)], 1.0)
        est, _ = predict(m, [(0, 1.0), (1, 1.0)])
        assert math.isclose(est, 20.0 / 11.0, rel_tol=1e-12)

    def test_prescaled_sums_match_explicit_bookkeeping_bit_for_bit(self):
        # The update rule's exact contract: scale <- f * scale, entry +=
        # contribution * (1 / scale), replayed here operation by operation.
        f = 0.9
        data = [([(0, 1.0), (1, 0.5)], 1.0), ([(0, 0.5), (1, -1.0)], -0.5), ([(0, 1.0), (1, 1.0)], 2.0)]
        m = new_model(2, forgetting=f)
        for x, r in data:
            m = update(m, x, r)

        scale = 1.0
        xx = [0.0] * 2
        xy = [0.0] * 2
        for x, r in data:
            scale = f * scale
            inv = 1.0 / scale
            for j, v in x:
                xx[j] += (v * v) * inv
                xy[j] += (r * v) * inv
        assert m.scale == scale
        assert list(m.xx) == xx
        assert list(m.xy) == xy

    def test_untouched_coordinates_do_not_move(self):
        # The O(nonzeros) claim, observably: an update leaves every
        # untouched pre-scaled entry bit-identical — forgetting acts on
        # them only through the shared scale.
        m = new_model(4, forgetting=0.9)
        m = update(m, [(1, 1.0)], 1.0)
        before = (m.xx, m.xy)
        m = update(m, [(2, 3.0)], -1.0)
        assert (m.xx[0], m.xx[1], m.xx[3]) == (before[0][0], before[0][1], before[0][3])
        assert (m.xy[0], m.xy[1], m.xy[3]) == (before[1][0], before[1][1], before[1][3])


class TestRenormalization:
    def test_scale_renormalizes_and_predictions_survive(self):
        # forgetting = 0.001 drives the scale below 2^-512 within ~52
        # updates; the sweep multiplies the entries down and resets the
        # scale to 1.0, and the weights are unchanged up to rounding.
        m = new_model(1, forgetting=0.001)
        crossed = False
        for i in range(120):
            m = update(m, [(0, 1.0)], 1.0)
            assert RENORM_THRESHOLD < m.scale <= 1.0
            if m.scale == 1.0 and i > 0:
                crossed = True
        assert crossed
        # Steady state: true xx = xy = s where s = 1/(1 - f), so
        # theta = s / (s + ridge) — evidence and weights survive the sweeps.
        s = 1.0 / (1.0 - 0.001)
        est, _ = predict(m, [(0, 1.0)])
        assert math.isclose(est, s / (s + 1.0), rel_tol=1e-9)

    def test_renormalized_prediction_matches_the_unscaled_math(self):
        # Same history at forgetting 0.5: prediction equals the classic
        # geometric-sum formulation within rounding, across the renorm
        # boundary (0.5^n crosses 2^-512 after 512 updates).
        f = 0.5
        m = new_model(1, forgetting=f)
        xx = 0.0
        xy = 0.0
        for _ in range(600):
            m = update(m, [(0, 1.0)], 1.0)
            xx = f * xx + 1.0
            xy = f * xy + 1.0
        est, unc = predict(m, [(0, 1.0)])
        assert math.isclose(est, xy / (xx + 1.0), rel_tol=1e-9)
        assert math.isclose(unc, math.sqrt(1.0 / (xx + 1.0)), rel_tol=1e-9)


class TestUncertainty:
    def test_uncertainty_shrinks_with_evidence(self):
        m = new_model(2)
        _, before = predict(m, [(0, 1.0)])
        for _ in range(20):
            m = update(m, [(0, 1.0)], 1.0)
        _, after = predict(m, [(0, 1.0)])
        assert after < before / 3

    def test_uncertainty_is_directional(self):
        m = new_model(2)
        for _ in range(50):
            m = update(m, [(0, 1.0)], 1.0)  # evidence only along axis 0
        _, seen = predict(m, [(0, 1.0)])
        _, unseen = predict(m, [(1, 1.0)])  # prior: 1.0
        assert unseen > 5 * seen

    def test_uncertainty_has_a_floor(self):
        # Forgetting bounds the effective sample size at 1/(1 - f), so
        # certainty saturates: the model can never become absolutely sure
        # (R2). With f = 0.9 and unit features, the true xx -> 10 in the
        # limit, so uncertainty can never fall below 1/sqrt(11).
        m = new_model(1, forgetting=0.9)
        for _ in range(500):
            m = update(m, [(0, 1.0)], 1.0)
        _, unc = predict(m, [(0, 1.0)])
        floor = 1.0 / math.sqrt(1.0 / (1.0 - 0.9) + 1.0)
        assert unc >= floor
        assert math.isclose(unc, floor, rel_tol=1e-9)


class TestForgetting:
    def test_adapts_when_the_world_flips(self):
        # A model taught y = +x and then y = -x for a few effective windows
        # ends up predicting the new world.
        m = new_model(1, forgetting=0.99)
        for _ in range(50):
            m = update(m, [(0, 1.0)], 1.0)
        for _ in range(300):
            m = update(m, [(0, 1.0)], -1.0)
        est, _ = predict(m, [(0, 1.0)])
        assert est < -0.9

    def test_dead_dimensions_are_recycled(self):
        # Evidence on a feature that stops appearing fades with every later
        # update (R1): after many updates elsewhere, the dead dimension is
        # back to ~the prior.
        m = new_model(2, forgetting=0.9)
        for _ in range(50):
            m = update(m, [(1, 1.0)], 1.0)
        for _ in range(200):
            m = update(m, [(0, 1.0)], 0.5)
        est, unc = predict(m, [(1, 1.0)])
        assert abs(est) < 1e-6
        assert unc > 0.999

    def test_updates_are_order_dependent(self):
        # The documented trade of per-update forgetting: training order
        # matters (a late reward counts slightly less), so replay requires
        # the ordered sequence.
        obs1 = ([(0, 1.0), (1, 0.5)], 1.0)
        obs2 = ([(0, 0.5), (1, -1.0)], -0.5)
        m12 = update(update(new_model(2), *obs1), *obs2)
        m21 = update(update(new_model(2), *obs2), *obs1)
        assert m12 != m21


class TestPinnedVectors:
    # Golden vectors: bit-exact prediction after a fixed little history.
    def test_prediction_bits(self):
        m = new_model(2, forgetting=0.9)
        m = update(m, [(0, 1.0)], 1.0)
        m = update(m, [(1, 1.0)], -0.5)
        m = update(m, [(0, 1.0), (1, 1.0)], 0.25)
        est, unc = predict(m, [(0, 1.0), (1, -1.0)])
        assert est == 0.4461897165296356
        assert unc == 0.8370779368301933
