from src.data_generate_sde import sde_interest_rates, time, utils

import jax.numpy as jnp
import jax.random as jr
import pytest


@pytest.fixture
def times():
    return time.grid(0, 1, 10)


@pytest.fixture
def times_rev(times):
    return time.reverse(1.0, times=times)


@pytest.fixture
def key():
    return jr.PRNGKey(1)


def test_reverse_and_correction_drift():
    x = jnp.asarray([1., 1.])
    t = 0.
    drift = sde_interest_rates.reverse_and_correction_drift(t, x)
    assert drift.ndim == 1
    assert drift.shape == (2,)


def test_reverse_and_correction_diffusion():
    x = jnp.asarray([1., 1.])
    t = 0.
    diffusion = sde_interest_rates.reverse_and_correction_diffusion(t, x)
    assert diffusion.ndim == 2
    assert diffusion.shape == (2, 2)
    assert diffusion[1, 0] == 0.
    assert diffusion[1, 1] == 0.
