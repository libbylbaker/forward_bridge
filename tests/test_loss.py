import jax.numpy as jnp

from src.sdes import sde_bm
from src.sdes import sde_ornstein_uhlenbeck
from src.training.train_utils import get_score


def test_score():
    t0 = 0
    t1 = 0.5
    Y0 = jnp.array([0.0])
    Y1 = jnp.array([1.0])
    bm = sde_bm.brownian_motion(T=1., N=10, dim=1)
    score_fn = get_score(bm)
    _score, cov = score_fn(t0, Y0, t1, Y1)
    assert _score == jnp.array([2.0])
