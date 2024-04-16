import jax.random as jr
import pytest
from functools import partial
import matplotlib.pyplot as plt

from src.data_generate_sde import time, sde_bm


@pytest.fixture
def times():
    return time.grid(0, 1, 100)


@pytest.fixture
def times_rev(times):
    return time.reverse(1.0, times=times)


@pytest.fixture
def keys():
    return jr.PRNGKey(0), jr.PRNGKey(2)


def test_brownian_motion(keys, times, times_rev):
    bmr = sde_bm.reverse(times_rev, keys[0], (1.0,))
    bmf = sde_bm.forward(times, keys[1], (1.0,))
    bmc = sde_bm.correction(ts=times_rev)


def test_bm_backward(keys, times):
    y = (1.,)
    score = partial(sde_bm.forward_score, 0.)
    bw_keys = jr.split(keys[0], 10)
    for key in bw_keys:
        backward = sde_bm.backward(times, key, score, y)
        plt.plot(times, backward)
    plt.show()
