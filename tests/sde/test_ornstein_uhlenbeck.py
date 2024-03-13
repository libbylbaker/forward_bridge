import jax.random as jr
import jax.numpy as jnp
import jax
import pytest
import matplotlib.pyplot as plt
from functools import partial

from milsteins_loss.sde import time
from milsteins_loss.sde import ornstein_uhlenbeck as ou
from milsteins_loss import plotting


@pytest.fixture
def times():
    return time.grid(0, 2, 100)


@pytest.fixture
def times_rev(times):
    return time.reverse(1.0, times=times)


@pytest.fixture
def key():
    return jr.PRNGKey(1)


def test_ou_rev(key, times, times_rev):
    ou_rev = ou.reverse(times_rev, key, y=jnp.array([1.0, 1.0]))
    assert ou_rev.shape == (times.size, 2)


def test_ou_forward(key, times):
    x0 = (-3.,)
    keys = jr.split(key, 10)
    for _key in keys:
        traj = ou.forward(times, _key, x0)
        plt.plot(times, traj)
    plt.show()


def test_ou_backward(key, times):
    y = (1.,)
    score = partial(ou.forward_score, 0., -1.)
    bw_keys = jr.split(key, 10)
    for _key in bw_keys:
        backward = ou.backward(times, _key, score, y)
        plt.plot(times, backward)
    plt.show()


def test_ou_forward_score(key):
    score = ou.forward_score
    t = jnp.asarray([0.25, 0.5, 0.75, 0.95])
    x = jnp.linspace(-2, 2, 1000)
    x0 = 0.0
    fig, axs = plt.subplots(nrows=1, ncols=t.size, sharey=True)
    for col, ts in enumerate(t):
        true_score_fn = jax.vmap(score, in_axes=(None, None, None, 0))
        y_true = true_score_fn(0., x0, ts, x.flatten())
        axs[col].plot(x.flatten(), y_true.flatten())
        axs[col].set_title(f"Time: {ts}")
    plt.show()