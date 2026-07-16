import math

from gittins_reference.model import dot, new_model, predict, update


class TestPredict:
    def test_fresh_model_predicts_zero_with_unit_uncertainty(self):
        # With ridge = 1 and a unit feature vector, the prior gives
        # estimate 0 and uncertainty sqrt(x . x / ridge) = 1, exactly.
        est, unc = predict(new_model(3), [1.0, 0.0, 0.0])
        assert est == 0.0
        assert unc == 1.0

    def test_weight_is_a_shrunk_running_average(self):
        # With forgetting 1.0 (never forget), 9 observations of feature 0
        # with reward 1: theta_0 = 9 / (9 + 1), uncertainty 1 / sqrt(9 + 1).
        m = new_model(2, forgetting=1.0)
        for _ in range(9):
            m = update(m, [1.0, 0.0], 1.0)
        est, unc = predict(m, [1.0, 0.0])
        assert math.isclose(est, 9.0 / 10.0, rel_tol=1e-15)
        assert math.isclose(unc, math.sqrt(1.0 / 10.0), rel_tol=1e-15)

    def test_recovers_per_feature_weights(self):
        # True weights [2, -1, 0.5], observed one feature at a time,
        # noiseless. Each weight converges independently.
        w = [2.0, -1.0, 0.5]
        xs = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        m = new_model(3)
        for _ in range(40):
            for x in xs:
                m = update(m, x, dot(x, w))
        probe = [1.0, 2.0, -1.0]
        est, unc = predict(m, probe)
        assert math.isclose(est, dot(probe, w), rel_tol=0.05)
        assert unc < 0.5

    def test_coordinates_are_independent(self):
        # Evidence about feature 0 changes nothing about feature 1: its
        # estimate and uncertainty stay exactly at the prior.
        m = new_model(2)
        for _ in range(50):
            m = update(m, [1.0, 0.0], 1.0)
        est, unc = predict(m, [0.0, 1.0])
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
            m = update(m, [1.0, 1.0], 1.0)
        est, _ = predict(m, [1.0, 1.0])
        assert math.isclose(est, 20.0 / 11.0, rel_tol=1e-12)

    def test_incremental_sums_match_batch_sums_bit_for_bit(self):
        # The incremental bookkeeping must equal a directly computed batch
        # sum with the same geometric weights: observation i of n carries
        # weight forgetting^(n - 1 - i).
        f = 0.9
        data = [([1.0, 0.5], 1.0), ([0.5, -1.0], -0.5), ([1.0, 1.0], 2.0)]
        m = new_model(2, forgetting=f)
        for x, r in data:
            m = update(m, x, r)

        n = len(data)
        xx = [0.0] * 2
        xy = [0.0] * 2
        for j in range(2):
            for i, (x, r) in enumerate(data):
                # Same order and operations as n successive updates.
                xx[j] = xx[j] * f + x[j] * x[j] if i else x[j] * x[j]
                xy[j] = xy[j] * f + r * x[j] if i else r * x[j]
        assert list(m.xx) == xx
        assert list(m.xy) == xy
        assert n == 3


class TestUncertainty:
    def test_uncertainty_shrinks_with_evidence(self):
        m = new_model(2)
        _, before = predict(m, [1.0, 0.0])
        for _ in range(20):
            m = update(m, [1.0, 0.0], 1.0)
        _, after = predict(m, [1.0, 0.0])
        assert after < before / 3

    def test_uncertainty_is_directional(self):
        m = new_model(2)
        for _ in range(50):
            m = update(m, [1.0, 0.0], 1.0)  # evidence only along axis 0
        _, seen = predict(m, [1.0, 0.0])
        _, unseen = predict(m, [0.0, 1.0])  # prior: 1.0
        assert unseen > 5 * seen

    def test_uncertainty_has_a_floor(self):
        # Forgetting bounds the effective sample size at 1/(1 - f), so
        # certainty saturates: the model can never become absolutely sure
        # (R2). With f = 0.9 and unit features, xx -> 10 exactly in the
        # limit, so uncertainty can never fall below 1/sqrt(11).
        m = new_model(1, forgetting=0.9)
        for _ in range(500):
            m = update(m, [1.0], 1.0)
        _, unc = predict(m, [1.0])
        floor = 1.0 / math.sqrt(1.0 / (1.0 - 0.9) + 1.0)
        assert unc >= floor
        assert math.isclose(unc, floor, rel_tol=1e-12)


class TestForgetting:
    def test_adapts_when_the_world_flips(self):
        # A model taught y = +x and then y = -x for a few effective windows
        # ends up predicting the new world.
        m = new_model(1, forgetting=0.99)
        for _ in range(50):
            m = update(m, [1.0], 1.0)
        for _ in range(300):
            m = update(m, [1.0], -1.0)
        est, _ = predict(m, [1.0])
        assert est < -0.9

    def test_dead_dimensions_are_recycled(self):
        # Evidence on a feature that stops appearing fades with every later
        # update (R1): after many updates elsewhere, the dead dimension is
        # back to ~the prior.
        m = new_model(2, forgetting=0.9)
        for _ in range(50):
            m = update(m, [0.0, 1.0], 1.0)
        for _ in range(200):
            m = update(m, [1.0, 0.0], 0.5)
        est, unc = predict(m, [0.0, 1.0])
        assert abs(est) < 1e-6
        assert unc > 0.999

    def test_updates_are_order_dependent(self):
        # The documented trade of per-update forgetting: training order
        # matters (a late reward counts slightly less), so replay requires
        # the ordered sequence. The former timestamp-weighted design was
        # order-independent; this one buys simplicity instead.
        obs1 = ([1.0, 0.5], 1.0)
        obs2 = ([0.5, -1.0], -0.5)
        m12 = update(update(new_model(2), *obs1), *obs2)
        m21 = update(update(new_model(2), *obs2), *obs1)
        assert m12 != m21


class TestPinnedVectors:
    # Golden vectors: bit-exact prediction after a fixed little history.
    def test_prediction_bits(self):
        m = new_model(2, forgetting=0.9)
        m = update(m, [1.0, 0.0], 1.0)
        m = update(m, [0.0, 1.0], -0.5)
        m = update(m, [1.0, 1.0], 0.25)
        est, unc = predict(m, [1.0, -1.0])
        assert est == 0.4461897165296356
        assert unc == 0.8370779368301933
