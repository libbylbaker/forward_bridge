import jax.numpy as jnp
import jax.random as jr
import pytest

from src.data_generate_sde import guided_process
from src.data_generate_sde import sde_ornstein_uhlenbeck as ou
from src.data_generate_sde import time


@pytest.fixture
def times():
    return time.grid(0.0, 1.0, 100)


@pytest.fixture
def key():
    return jr.PRNGKey(1)


@pytest.mark.parametrize("dim", [1])
def test_get_guide_fn(dim):
    def beta_aux(t):
        return jnp.zeros(dim)

    def B_aux(t):
        return -jnp.eye(dim)

    def sigma_aux(t):
        return jnp.eye(dim)

    y = 1.0
    T = 1.0
    t0 = 0.0

    guide_fn = guided_process.get_guide_fn(t0, T, y, sigma_aux, B_aux, beta_aux)
    r_zero = guide_fn(0.0, jnp.zeros(dim))
    r_T = guide_fn(T, jnp.zeros(dim))
    assert jnp.isclose(r_zero, ou.score(t0, jnp.zeros(dim), T, y))
    assert r_T == 0.0
