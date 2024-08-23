import jax
import jax.numpy as jnp

from src.sdes import sde_kunita


def test_reverse():
    t = 0.0
    x = jnp.array([[1.0, 2.0], [3, 4], [5, 6]])
    x = x.flatten()

    drift, diffusion = sde_kunita.vector_fields_reverse()
    drift_ = drift(t, x)
    assert drift_.shape == x.shape

    diffusion_ = diffusion(t, x)
    grid_size = sde_kunita.GRID_SIZE
    assert diffusion_.shape == (6, 2 * grid_size**2)


def test_drift_correction():
    t = 0.0
    corr = 1.0
    rev = jnp.array([[1.0, 2.0], [3, 4], [5, 6]])
    drift_correction = sde_kunita.drift_correction(t, rev, corr)
    assert drift_correction.shape == ()
