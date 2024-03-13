from milsteins_loss.sde import interest_rates, time

import jax.numpy as jnp
import jax.random as jr
import pytest

import matplotlib.pyplot as plt
import jax


@pytest.fixture
def times():
    return time.grid(0, 1, 10)


@pytest.fixture
def times_rev(times):
    return time.reverse(1.0, times=times)


@pytest.fixture
def keys():
    return jr.PRNGKey(0), jr.PRNGKey(2)


def test_reverse_sol(times, keys):
    y1 = jnp.asarray([1.0])
    sol = interest_rates.reverse_and_correction(ts=times, key=keys[0], y=y1)
    rev = sol[:, :-1]
    assert rev.shape == (times.shape[0], y1.size)
    assert rev[0, 0] == 1.0


def test_correction(times, keys):
    a = 1.0
    y1 = jnp.asarray([1.0])
    sol = interest_rates.reverse_and_correction(ts=times, key=keys[0], y=y1)
    rev = sol[:, :-1]
    correction = sol[:, -1]
    dt = times[1] - times[0]

    print(correction)
    print(rev)

    sol2 = interest_rates.reverse_and_correction(ts=times, key=keys[1], y=y1)
    rev2 = sol2[:, :-1]
    correction2 = sol2[:, -1]

    print(correction2)
    print(rev2)



