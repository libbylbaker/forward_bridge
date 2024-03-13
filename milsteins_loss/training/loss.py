import jax.numpy as jnp

from typing import Callable


def get_score(drift, diffusion) -> Callable:

    def score(t0: float, X0: float, t1: float, X1: float):
        dt = t1 - t0
        drift_last = drift(t0, X0)
        diffusion_last = diffusion(t0, X0)
        inv_cov = invert(diffusion_last, diffusion_last.T)
        _score = 1 / dt * inv_cov @ (X1 - X0 - dt * drift_last)
        return _score

    return score


def invert(mat, mat_transpose):
    """
    Inversion of mat*mat_transpose.
    :param mat: array of shape (n, m) i.e. ndim=2
    :param mat_transpose: array with shape (m, n) with ndim=2
    :return: (mat*mat.T)^{-1} with shape (n, n)
    """
    return jnp.linalg.inv(mat @ mat_transpose)
