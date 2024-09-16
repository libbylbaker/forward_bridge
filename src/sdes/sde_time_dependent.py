import jax.numpy as jnp

from src.sdes import sde_utils


def simple_time_dependent(T, N, dim, C=1):
    def drift(t, x, *args):
        """dX_t = (- X_t) dt + t/C dW_t"""
        return -x

    def diffusion(t, x, *args):
        """dX_t = (a/X_t - X_t) dt + dW_t"""
        assert x.ndim == 1
        dim = x.size
        return (t + 1) / C * jnp.identity(dim)

    def adj_drift(t, y, *args):
        return y

    def adj_diffusion(t, y, *args):
        return (T - t) / C * jnp.identity(y.size)

    def correction(t, y, corr, *args):
        return 1.0 * corr

    return sde_utils.SDE(T, N, dim, drift, diffusion, adj_drift, adj_diffusion, correction, None, (dim,), None)
