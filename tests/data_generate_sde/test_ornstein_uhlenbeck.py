from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from src.data_generate_sde import sde_ornstein_uhlenbeck as ou
from src.data_generate_sde import time, utils


@pytest.fixture
def times():
    return time.grid(0.0, 1.0, 100)


@pytest.fixture
def times_rev(times):
    return time.reverse(1.0, times=times)


@pytest.fixture
def key():
    return jr.PRNGKey(1)


def test_reverse(key, times_rev):
    drift, diffusion = ou.vector_fields_reverse()
    sol = utils.solution(key, times_rev, jnp.array([1.0, 1.0]), drift, diffusion).ys
    assert sol.shape == (times_rev.size, 2)


def test_forward(key, times):
    x0 = (-3.0,)
    drift, diffusion = ou.vector_fields()
    sol = utils.solution(key, times, x0, drift, diffusion).ys
    assert sol.shape == (times.size, 1)


def test_reverse_drift(times_rev):
    x = jnp.asarray([1.0, 1.0])
    t = 0.0
    drift, _ = ou.vector_fields_reverse()
    drift_ = drift(t, x)
    assert drift_.ndim == 1
    assert drift_.shape == (2,)


def test_reverse_diffusion_2d(times_rev):
    x = jnp.asarray([1.0, 1.0])
    t = 0.0
    _, diffusion = ou.vector_fields_reverse()
    diffusion_ = diffusion(t, x)
    assert diffusion_.ndim == 2
    assert diffusion_.shape == (2, 2)
    assert diffusion_[1, 0] == 0.0
    assert diffusion_[1, 1] == ou._SIGMA


def test_reverse_diffusion_1d(times_rev):
    x = jnp.asarray([1.0])
    t = 0.0
    drift, diffusion = ou.vector_fields_reverse()
    diffusion_ = diffusion(t, x)
    assert diffusion_.ndim == 2
    assert diffusion_.shape == (1, 1)
    assert diffusion_[0, 0] == ou._SIGMA


def test_correction_drift(times):
    corr = jnp.asarray([1.0])
    t = 0.0
    drift = ou.drift_correction(t, 0.0, corr)
    assert drift.ndim == 1
    assert drift.shape == (1,)


def test_backward(key, times):
    y = (1.0,)
    score = partial(ou.score_forward, 0.0, -1.0)
    drift, diffusion = ou.vector_fields()
    sol = utils.backward(key, times, y, score, drift, diffusion).ys
    assert sol.shape == (times.size, 1)
    assert sol[0, 0] == 1.0


def test_forward_score(key):
    score = ou.score_forward
    t = 0.5
    x = jnp.linspace(-2, 2, 1000)
    x0 = 0.0
    true_score_fn = jax.vmap(score, in_axes=(None, None, None, 0))
    y_true = true_score_fn(0.0, x0, t, x.flatten())
    assert y_true.shape == (1000,)


def test_important_reverse_and_correction_1d(key, times):
    y = jnp.asarray((1.0,))
    x0 = jnp.asarray((1.0,))
    drift, diffusion = ou.vector_fields_reverse()
    sol = utils.important_reverse_and_correction(
        key, times, x0, y, drift, diffusion, ou.drift_correction
    ).ys
    assert sol.shape[1] == 2
    assert sol[0, 0] == 1.0
    assert sol[0, 1] == 1.0


def test_important_reverse_and_correction_2d(key, times):
    y = jnp.asarray((1.0, 1.0))
    x0 = jnp.asarray((1.0, 1.0))
    drift, diffusion = ou.vector_fields_reverse()
    sol = utils.important_reverse_and_correction(
        key, times, x0, y, drift, diffusion, ou.drift_correction
    ).ys
    assert sol.shape[1] == 3
    assert sol[0, 0] == 1.0
    assert sol[0, 1] == 1.0
