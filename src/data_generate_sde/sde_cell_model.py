import jax
import jax.numpy as jnp
import numpy as np

from src.data_generate_sde import guided_process, time, utils

alpha = 1 / 16
sigma = 0.1


def data_forward(x0, T, N):
    ts = time.grid(t_start=0, T=T, N=N)
    drift, diffusion = vector_fields()

    @jax.jit
    @jax.vmap
    def data(key):
        correction_ = 1.0
        forward_ = utils.solution(key, ts, x0, drift, diffusion)
        return ts[..., None], forward_, jnp.asarray(correction_)

    return data


def data_reverse(y, T, N, weight_fn=None):
    ts = time.grid(t_start=0, T=T, N=N)
    ts_reverse = time.reverse(T=T, times=ts)
    y = jnp.asarray(y)
    assert y.ndim == 1
    drift, diffusion = vector_fields_reverse_and_correction()

    # @jax.jit
    @jax.vmap
    def data(key):
        """
        :return: ts,
        reverse process: (t, dim), where t is the number of time steps and dim the dimension of the SDE
        correction process: float, correction process at time T
        """
        start_val = jnp.append(y, 1.0)
        reverse_and_correction_ = utils.solution(
            key, ts_reverse, x0=start_val, drift=drift, diffusion=diffusion
        )
        reverse_ = reverse_and_correction_[:, :-1]
        if weight_fn:
            weight = weight_fn(reverse_[-1])
        else:
            weight = 1.0
        correction_ = reverse_and_correction_[-1, -1]
        return ts[..., None], reverse_, jnp.asarray(correction_ * weight)

    return data


def data_reverse_guided(x0, y, T, N, weight_fn=None):
    x0 = jnp.asarray(x0)
    y = jnp.asarray(y)
    ts = time.grid(t_start=0, T=T, N=N)
    ts_reverse = time.reverse(T, ts)
    start_val = jnp.append(y, 1.0)
    # reverse_drift, reverse_diffusion = vector_fields_reverse()
    # B_auxiliary, beta_auxiliary, sigma_auxiliary = reverse_guided_auxiliary(len(x0))
    #
    # guide_fn = guided_process.get_guide_fn(
    #     0.0, T, x0, sigma_auxiliary, B_auxiliary, beta_auxiliary
    # )
    # guided_drift, guided_diffusion = guided_process.vector_fields_guided(
    #     reverse_drift, reverse_diffusion, guide_fn
    # )

    guided_drift, guided_diffusion = vector_fields_reverse_and_correction_guided(x0, T)

    @jax.jit
    @jax.vmap
    def data(key):
        reverse_and_correction = utils.solution(
            key, ts_reverse, start_val, guided_drift, guided_diffusion
        )
        reverse = reverse_and_correction[:, :-1]
        correction = reverse_and_correction[-1, -1]
        if weight_fn:
            weight = weight_fn(reverse[-1])
        else:
            weight = 1.0
        return ts[..., None], reverse, correction * weight

    return data


def weight_function_gaussian(x0, inverse_covariance):
    x0 = jnp.asarray(x0)
    # Sigma = 1/variance * jnp.identity(x0.size)

    def gaussian_distance(YT):
        YT = jnp.asarray(YT)
        return jnp.exp(-0.5 * (YT - x0).T @ jnp.linalg.solve(inverse_covariance, (YT - x0)))

    return gaussian_distance


def weight_function_euler(x0):
    pass


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
        reverse_and_correction = jnp.pad(
            d_reverse, ((0, 1), (0, 1)), mode="constant", constant_values=0.0
        )
        return reverse_and_correction

    return drift, diffusion


def vector_fields_reverse_and_correction_guided(x0, T):
    reverse_drift, reverse_diffusion = vector_fields_reverse()
    B_auxiliary, beta_auxiliary, sigma_auxiliary = reverse_guided_auxiliary(2)
    guide_fn = guided_process.get_guide_fn(
        0.0, T, x0, sigma_auxiliary, B_auxiliary, beta_auxiliary
    )

    def drift(t, x):
        reverse = x[:-1]
        correction = x[-1, None]
        guided_drift, _ = guided_process.vector_fields_guided(
            reverse_drift, reverse_diffusion, guide_fn
        )
        d_reverse = guided_drift(t, reverse)
        d_correction = drift_correction(t, reverse, correction)
        return jnp.concatenate([d_reverse, d_correction])

    def diffusion(t, x):
        reverse = x[:-1]
        correction = x[-1]
        d_reverse = reverse_diffusion(t, reverse)
        d_correction = -guide_fn(t, reverse).T @ reverse_diffusion(t, reverse) * correction
        diff = jnp.block([[d_reverse, jnp.zeros((2, 1))], [d_correction, jnp.zeros((1, 1))]])
        return diff

    return drift, diffusion


def drift_correction(t, rev, corr, *args):
    term1 = 4 * rev**3 / (alpha + rev**4)
    term2 = -4 * rev**7 / (alpha + rev**4) ** 2
    return jnp.sum(-(term1 + term2 - 1)) * corr


def reverse_guided_auxiliary(dim):
    def B_auxiliary(t):
        return jnp.array([[1.0, 0.0], [0.0, 1.0]])

    def beta_auxiliary(t):
        return jnp.array([-1.0, -1.0])

    def sigma_auxiliary(t):
        return sigma * jnp.identity(dim)

    return B_auxiliary, beta_auxiliary, sigma_auxiliary
