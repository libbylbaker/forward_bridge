import jax
import jax.numpy as jnp

from src.sdes import sde_utils


def ornstein_uhlenbeck(T, N, dim, alpha=1.0, sigma=1.0):
    def drift(t, x, *args):
        """dX_t = -alpha X_t dt + sigma dW_t"""
        assert x.ndim == 1
        return -alpha * x

    def diffusion(t, x, *args):
        """dX_t = -alpha X_t dt + sigma dW_t"""
        assert x.ndim == 1
        dim = x.size
        return sigma * jnp.identity(dim)

    def adj_drift(t, y, *args):
        return alpha * y

    def adj_diffusion(t, y, *args):
        return diffusion(T - t, y)

    def correction(t, y, correction, *args):
        return 1.0 * correction

    def score(t, x, T, y):
        x = jnp.asarray(x)
        y = jnp.asarray(y)
        var = _var_score(T - t)
        mean = _mean_score(T - t, x)
        _score = jnp.exp(-alpha * (T - t)) / var * (y - mean)
        return _score

    def score_forward(t0, x0, t, x):
        var = _var_score(t - t0)
        mean = _mean_score(t - t0, x0)
        return (1 / var) * (mean - x)

    def _var_score(t):
        return sigma**2 * (1 - jnp.exp(-2 * alpha * t)) / (2 * alpha)

    def _mean_score(t, x):
        return x * jnp.exp(-1 * alpha * t)

    return sde_utils.SDE(
        T, N, dim, drift, diffusion, adj_drift, adj_diffusion, correction, None, (dim,), (score, score_forward)
    )
