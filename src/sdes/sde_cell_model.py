import jax.numpy as jnp

from src.sdes import sde_utils, time

alpha = 1 / 16
sigma = 0.1


def data_forward(x0, T, N):
    return sde_utils.data_forward(x0, T, N, vector_fields())


def data_reverse(
    y,
    T,
    N,
):
    return sde_utils.data_reverse(y, T, N, vector_fields_reverse_and_correction())


def data_reverse_weighted(y, T, N, weight_fn):
    return sde_utils.data_reverse_weighted(y, T, N, vector_fields_reverse_and_correction(), weight_fn)


def vector_fields():
    def _u(x, alpha):
        return x**4 / (alpha + x**4)

    def drift(t, x, *args):
        dx0 = _u(x[0], alpha) + 1.0 - _u(x[1], alpha) - x[0]
        dx1 = _u(x[1], alpha) + 1.0 - _u(x[0], alpha) - x[1]
        return jnp.array([dx0, dx1])

    def diffusion(t, x, *args):
        return jnp.array([[sigma, 0.0], [0.0, sigma]])

    return drift, diffusion


def vector_fields_reverse():
    forward_drift, forward_diffusion = vector_fields()

    def drift(t, rev, *args):
        return -forward_drift(t, rev, *args)

    def diffusion(t, rev, *args):
        return forward_diffusion(t, rev, *args)

    return drift, diffusion


def vector_fields_reverse_and_correction():
    reverse_drift, reverse_diffusion = vector_fields_reverse()

    def drift(t, x):
        reverse = x[:-1]
        correction = x[-1, None]
        d_reverse = reverse_drift(t, reverse)
        d_correction = drift_correction(t, reverse, correction)
        return jnp.concatenate([d_reverse, d_correction])

    def diffusion(t, x):
        reverse = x[:-1]
        correction = x[-1]
        d_reverse = reverse_diffusion(t, reverse)
        reverse_and_correction = jnp.pad(d_reverse, ((0, 1), (0, 1)), mode="constant", constant_values=0.0)
        return reverse_and_correction

    return drift, diffusion


def drift_correction(t, rev, corr, *args):
    term1 = 4 * rev**3 / (alpha + rev**4)
    term2 = -4 * rev**7 / (alpha + rev**4) ** 2
    return jnp.sum(-(term1 + term2 - 1)) * corr


def reverse_guided_auxiliary(dim=2):
    def B_auxiliary(t):
        return jnp.array([[1.0, 0.0], [0.0, 1.0]])

    def beta_auxiliary(t):
        return jnp.array([-1.0, -1.0])

    def sigma_auxiliary(t):
        return sigma * jnp.identity(dim)

    return B_auxiliary, beta_auxiliary, sigma_auxiliary
