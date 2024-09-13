import jax.numpy as jnp

from src.sdes import sde_utils


def cell_model(T, N, alpha=1 / 16, sigma=0.1):
    def _u(x, alpha):
        return x**4 / (alpha + x**4)

    def drift(t, x, *args):
        dx0 = _u(x[0], alpha) + 1.0 - _u(x[1], alpha) - x[0]
        dx1 = _u(x[1], alpha) + 1.0 - _u(x[0], alpha) - x[1]
        return jnp.array([dx0, dx1])

    def diffusion(t, x, *args):
        return jnp.array([[sigma, 0.0], [0.0, sigma]])

    def adj_drift(t, y, *args):
        return -drift(t, y, *args)

    def adj_diffusion(t, y, *args):
        return diffusion(t, y, *args)

    def correction(t, y, corr, *args):
        term1 = 4 * y**3 / (alpha + y**4)
        term2 = -4 * y**7 / (alpha + y**4) ** 2
        return jnp.sum(-(term1 + term2 - 1)) * corr

    return sde_utils.SDE(T, N, 2, drift, diffusion, adj_drift, adj_diffusion, correction, None, (2,), None)
