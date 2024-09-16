import jax.numpy as jnp

from src.sdes import sde_utils


def brownian_motion(T, N, dim):
    def drift(t, x, *args):
        assert x.ndim == 1
        dim = x.size
        return jnp.zeros(shape=(dim,))

    def diffusion(t, x, *args):
        assert x.ndim == 1
        dim = x.size
        return jnp.identity(dim)

    def adj_drift(t, x, *args):
        return drift(t, x, *args)

    def adj_diffusion(t, x, *args):
        return diffusion(t, x, *args)

    def correction(t, x, *args):
        return jnp.asarray((1.0,))

    def score(t, x, T, y):
        x = jnp.asarray(x)
        y = jnp.asarray(y)
        return (y - x) / (T - t)

    def forward_score(t0, x0, t, x):
        x0 = jnp.asarray(x0)
        x = jnp.asarray(x)
        return -(x - x0) / (t - t0)

    return sde_utils.SDE(
        T, N, dim, drift, diffusion, adj_drift, adj_diffusion, correction, None, (dim,), (score, forward_score)
    )
