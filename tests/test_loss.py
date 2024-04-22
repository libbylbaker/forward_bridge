import jax.numpy as jnp

from src.data_generate_sde import sde_bm as bm
from src.training.utils import gradient_transition_fn


def test_score():
    t0 = 0
    t1 = 0.5
    Y0 = jnp.array([0.0])
    Y1 = jnp.array([1.0])
    drift, diffusion = bm.vector_fields()
    score_fn = gradient_transition_fn(drift, diffusion)
    _score = score_fn(t0, Y0, t1, Y1)
    assert _score == jnp.array([2.0])
