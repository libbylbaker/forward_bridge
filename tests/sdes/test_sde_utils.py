import functools

import jax
import jax.numpy as jnp
import pytest

from src.sdes import sde_bm, sde_cell_model, sde_ornstein_uhlenbeck, sde_time_dependent, sde_utils


@pytest.fixture
def key():
    return jax.random.PRNGKey(1)


@pytest.fixture
def keys():
    k = jax.random.PRNGKey(1)
    return jax.random.split(k, 5)


class TestConditioned:
    T = 1.0
    N = 100

    sde_ou_1d = sde_ornstein_uhlenbeck.ornstein_uhlenbeck(T, N, 1)
    sde_bm_1d = sde_bm.brownian_motion(T, N, 1)

    @pytest.mark.parametrize(
        "sde_1d",
        [
            sde_ou_1d,
            sde_bm_1d,
        ],
    )
    def test_conditioned_1d(self, sde_1d, key):
        x0 = (1.0,)
        score_fn = sde_1d.params[0]
        score_T_y = functools.partial(score_fn, T=1.0, y=x0)
        conditioned = sde_utils.conditioned(key, x0, sde_1d, score_T_y)
        assert conditioned.shape == (100, 1)
        assert conditioned[:, 0].all() == x0[0]
