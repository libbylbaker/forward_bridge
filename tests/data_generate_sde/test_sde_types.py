import jax.numpy as jnp
import jax.random as jr
import pytest

from src.sdes import (
    sde_bm,
    sde_cell_model,
    sde_interest_rates,
    sde_ornstein_uhlenbeck,
    sde_time_dependent,
    sde_utils,
    time,
)


@pytest.fixture
def times():
    return time.grid(0.0, 1.0, 100)


@pytest.fixture
def times_rev(times):
    return time.reverse(1.0, times=times)


@pytest.fixture
def key():
    return jr.PRNGKey(1)


@pytest.fixture
def keys():
    return jr.split(jr.PRNGKey(1), 5)


class TestDataForward:
    def test_data_forward_langevin(self):
        pass

    @pytest.mark.parametrize(
        "data_forward",
        [
            sde_ornstein_uhlenbeck.data_forward,
            sde_interest_rates.data_forward,
            sde_bm.data_forward,
            sde_time_dependent.data_forward,
        ],
    )
    def test_data_forward_1d(self, data_forward, keys):
        x0 = (1.0,)
        T = 1.0
        N = 100
        data = data_forward(x0, T, N)
        ts, forward, correction = data(keys)
        assert ts.shape == (5, 100, 1)
        assert forward.shape == (5, 100, 1)
        assert correction.shape == (5,)
        assert forward[:, 0, 0].all() == x0[0]

    @pytest.mark.parametrize(
        "data_forward",
        [
            sde_ornstein_uhlenbeck.data_forward,
            sde_interest_rates.data_forward,
            sde_bm.data_forward,
            sde_time_dependent.data_forward,
            sde_cell_model.data_forward,
        ],
    )
    def test_data_forward_2d(self, data_forward, keys):
        x0 = (1.0, 1.0)
        T = 1.0
        N = 100
        data = data_forward(x0, T, N)
        ts, forward, correction = data(keys)
        assert ts.shape == (5, 100, 1)
        assert forward.shape == (5, 100, 2)
        assert correction.shape == (5,)
        assert forward[:, 0, 0].all() == x0[0]
        assert forward[:, 0, 1].all() == x0[1]


class TestDataReverse:
    @pytest.mark.parametrize(
        "data_reverse",
        [
            sde_ornstein_uhlenbeck.data_reverse,
            sde_interest_rates.data_reverse,
            sde_bm.data_reverse,
            sde_time_dependent.data_reverse,
        ],
    )
    def test_data_reverse_1d(self, data_reverse, keys):
        y = (1.0,)
        T = 1.0
        N = 100
        data = data_reverse(y, T, N)
        ts, reverse, correction = data(keys)
        assert ts.shape == (5, 100, 1)
        assert reverse.shape == (5, 100, 1)
        assert correction.shape == (5,)
        assert reverse[:, 0, 0].all() == y[0]

    @pytest.mark.parametrize(
        "data_reverse",
        [
            sde_ornstein_uhlenbeck.data_reverse,
            sde_interest_rates.data_reverse,
            sde_bm.data_reverse,
            sde_time_dependent.data_reverse,
            sde_cell_model.data_reverse,
        ],
    )
    def test_data_reverse_2d(self, data_reverse, keys):
        y = (1.0, 1.0)
        T = 1.0
        N = 100
        data = data_reverse(y, T, N)
        ts, reverse, correction = data(keys)
        assert ts.shape == (5, 100, 1)
        assert reverse.shape == (5, 100, 2)
        assert correction.shape == (5,)
        assert reverse[:, 0, 0].all() == y[0]
        assert reverse[:, 0, 1].all() == y[1]


# class TestDataImportance:
#     @pytest.mark.parametrize(
#         "data_importance",
#         [
#             sde_ornstein_uhlenbeck.data_reverse_importance,
#             sde_interest_rates.data_importance,
#         ],
#     )
#     def test_data_importance_1d(self, data_importance, keys):
#         x0 = (1.0,)
#         y = (1.0,)
#         T = 1.0
#         N = 100
#         data = data_importance(x0, y, T, N)
#         ts, reverse, correction = data(keys)
#         assert ts.shape == (5, 100, 1)
#         assert reverse.shape == (5, 100, 1)
#         assert correction.shape == (5,)
#         assert reverse[:, 0, 0].all() == x0[0]
#
#     @pytest.mark.parametrize(
#         "data_importance",
#         [
#             sde_ornstein_uhlenbeck.data_reverse_importance,
#             sde_interest_rates.data_importance,
#         ],
#     )
#     def test_data_importance_2d(self, data_importance, keys):
#         x0 = (1.0, 1.0)
#         y = (1.0, 1.0)
#         T = 1.0
#         N = 100
#         data = data_importance(x0, y, T, N)
#         ts, reverse, correction = data(keys)
#         assert ts.shape == (5, 100, 1)
#         assert reverse.shape == (5, 100, 2)
#         assert correction.shape == (5,)
#         assert reverse[:, 0, 0].all() == x0[0]
#         assert reverse[:, 0, 1].all() == x0[1]


class TestScores:
    @pytest.mark.parametrize(
        "score_fn",
        [sde_ornstein_uhlenbeck.score, sde_interest_rates.score, sde_bm.score],
    )
    def test_score_1d(self, score_fn):
        t = 0.0
        x = (1.0,)
        T = 1.0
        y = (1.0,)

        score = score_fn(t, x, T, y)
        assert score.ndim == 1

    @pytest.mark.parametrize("score_fn", [sde_ornstein_uhlenbeck.score, sde_bm.score])
    def test_score_2d(self, score_fn):
        t = 0.0
        x = (1.0, 1.0)
        T = 1.0
        y = (1.0, 1.0)

        score = score_fn(t, x, T, y)
        assert score.ndim == 1
