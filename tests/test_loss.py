import jax.numpy as jnp

from milsteins_loss.training.loss import score
from milsteins_loss.sde import bm


def test_score():
    t0 = 0
    t1 = 0.5
    Y0 = jnp.array([0.0])
    Y1 = jnp.array([1.0])
    drift = bm.drift
    diffusion = bm.diffusion
    _score = score(t0, t1, Y0, Y1, drift, diffusion)
    assert _score == jnp.array([2.0])
