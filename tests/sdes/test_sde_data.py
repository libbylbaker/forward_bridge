import jax
import jax.numpy as jnp
import pytest

from src.sdes import sde_bm, sde_cell_model, sde_data, sde_ornstein_uhlenbeck, sde_time_dependent


@pytest.fixture
def key():
    return jax.random.PRNGKey(1)


@pytest.fixture
def keys():
    k = jax.random.PRNGKey(1)
    return jax.random.split(k, 5)


class TestSDEs1D:
    T = 1.0
    N = 100

    sde_ou_1d = sde_ornstein_uhlenbeck.ornstein_uhlenbeck(T, N, 1)
    sde_bm_1d = sde_bm.brownian_motion(T, N, 1)
    sde_td_1d = sde_time_dependent.simple_time_dependent(T, N, 1)

    @pytest.mark.parametrize(
        "sde_1d",
        [
            sde_ou_1d,
            sde_bm_1d,
            sde_td_1d,
        ],
    )
    def test_data_forward_1d(self, sde_1d, keys):
        x0 = (1.0,)
        data = sde_data.data_forward(x0, sde_1d)
        ts, forward, correction = data(keys)
        assert ts.shape == (5, 100, 1)
        assert forward.shape == (5, 100, 1)
        assert correction.shape == (5,)
        assert forward[:, 0, 0].all() == x0[0]

    @pytest.mark.parametrize(
        "sde_1d",
        [
            sde_ou_1d,
            sde_bm_1d,
            sde_td_1d,
        ],
    )
    def test_data_reverse_1d(self, sde_1d, keys):
        y = (1.0,)
        data = sde_data.data_adjoint(y, sde_1d)
        ts, reverse, correction = data(keys)
        assert ts.shape == (5, 100, 1)
        assert reverse.shape == (5, 100, 1)
        assert correction.shape == (5,)
        assert reverse[:, 0, 0].all() == y[0]

    @pytest.mark.parametrize(
        "sde_1d",
        [
            sde_ou_1d,
            sde_bm_1d,
            sde_td_1d,
        ],
    )
    def test_data_reverse_correction_1d(self, sde_1d, keys):
        y = (1.0,)
        data = sde_data.data_reverse_correction(y, sde_1d)
        ts, reverse, correction = data(keys)
        assert ts.shape == (5, 100, 1)
        assert reverse.shape == (5, 100, 1)
        assert correction.shape == (5,)
        assert reverse[:, 0, 0].all() == y[0]

    @pytest.mark.parametrize(
        "sde_1d",
        [
            sde_ou_1d,
            sde_bm_1d,
            sde_td_1d,
        ],
    )
    def test_data_reverse_variable_y_1d(self, sde_1d, keys):
        y = jnp.ones(shape=(len(keys), 1))
        data = sde_data.data_reverse_variable_y(sde_1d)
        ts, reverse, correction, output_y = data(keys, y)
        assert ts.shape == (5, 100, 1)
        assert reverse.shape == (5, 100, 1)
        assert correction.shape == (5,)
        assert reverse[:, 0, 0].all() == y[0]
        assert jnp.array_equal(output_y, y)

    @pytest.mark.parametrize(
        "sde_1d",
        [
            sde_ou_1d,
            sde_bm_1d,
        ],
    )
    def test_score_1d(self, sde_1d):
        t = 0.0
        x = (1.0,)
        T = 1.0
        y = (1.0,)

        score = sde_1d.params[0](t, x, T, y)
        assert score.ndim == 1


class TestSDEs2D:
    T = 1.0
    N = 100

    sde_ou_2d = sde_ornstein_uhlenbeck.ornstein_uhlenbeck(T, N, 2)
    sde_bm_2d = sde_bm.brownian_motion(T, N, 2)
    sde_td_2d = sde_time_dependent.simple_time_dependent(T, N, 2)
    sde_cell_model_2d = sde_cell_model.cell_model(T, N, 2)

    @pytest.mark.parametrize(
        "sde_2d",
        [
            sde_ou_2d,
            sde_bm_2d,
            sde_td_2d,
            sde_cell_model_2d,
        ],
    )
    def test_data_forward_2d(self, sde_2d, keys):
        x0 = (1.0, 1.0)
        data = sde_data.data_forward(x0, sde_2d)
        ts, forward, correction = data(keys)
        assert ts.shape == (5, 100, 1)
        assert forward.shape == (5, 100, 2)
        assert correction.shape == (5,)
        assert forward[:, 0, 0].all() == x0[0]
        assert forward[:, 0, 1].all() == x0[1]

    @pytest.mark.parametrize(
        "sde_2d",
        [
            sde_ou_2d,
            sde_bm_2d,
            sde_td_2d,
            sde_cell_model_2d,
        ],
    )
    def test_data_reverse_2d(self, sde_2d, keys):
        y = (1.0, 1.0)
        data = sde_data.data_adjoint(y, sde_2d)
        ts, reverse, correction = data(keys)
        assert ts.shape == (5, 100, 1)
        assert reverse.shape == (5, 100, 2)
        assert correction.shape == (5,)
        assert reverse[:, 0, 0].all() == y[0]
        assert reverse[:, 0, 1].all() == y[1]

    @pytest.mark.parametrize(
        "sde_2d",
        [
            sde_ou_2d,
            sde_bm_2d,
            sde_td_2d,
            sde_cell_model_2d,
        ],
    )
    def test_data_reverse_correction_2d(self, sde_2d, keys):
        y = (1.0, 1.0)
        data = sde_data.data_reverse_correction(y, sde_2d)
        ts, reverse, correction = data(keys)
        assert ts.shape == (5, 100, 1)
        assert reverse.shape == (5, 100, 2)
        assert correction.shape == (5,)
        assert reverse[:, 0, 0].all() == y[0]
        assert reverse[:, 0, 1].all() == y[1]

    @pytest.mark.parametrize(
        "sde_2d",
        [
            sde_ou_2d,
            sde_bm_2d,
            sde_td_2d,
            sde_cell_model_2d,
        ],
    )
    def test_data_reverse_variable_y_2d(self, sde_2d, keys):
        y = jnp.ones(shape=(len(keys), 2))
        data = sde_data.data_reverse_variable_y(sde_2d)
        ts, reverse, correction, output_y = data(keys, y)
        assert ts.shape == (5, 100, 1)
        assert reverse.shape == (5, 100, 2)
        assert correction.shape == (5,)
        assert reverse[:, 0, 0].all() == y[0, 0]
        assert reverse[:, 0, 1].all() == y[0, 1]
        assert jnp.array_equal(output_y, y)

    @pytest.mark.parametrize(
        "sde_2d",
        [
            sde_ou_2d,
            sde_bm_2d,
        ],
    )
    def test_score_2d(self, sde_2d):
        t = 0.0
        x = (1.0, 1.0)
        T = 1.0
        y = (1.0, 1.0)

        score = sde_2d.params[0](t, x, T, y)
        assert score.ndim == 1
