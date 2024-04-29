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


def data_reverse(y, T, N):
    """
    :return: ts,
    reverse process: (t, dim), where t is the number of time steps and dim the dimension of the SDE
    correction process: float, correction process at time T
    """
    ts = time.grid(t_start=0, T=T, N=N)
    ts_reverse = time.reverse(T, ts)
    reverse_drift, reverse_diffusion = vector_fields_reverse()

    @jax.jit
    @jax.vmap
    def data(key):
        reverse_ = utils.solution(key, ts_reverse, y, reverse_drift, reverse_diffusion)
        correction_drift_ = lambda t, corr, *args: drift_correction(t, 0.0, corr)
        correction_ = utils.solution_ode(ts, x0=jnp.asarray([1.0]), drift=correction_drift_)
        return ts[..., None], reverse_, correction_[-1, 0]

    return data


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
        correction = x[-1]
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


def drift_correction(t, rev, corr, *args):
    term1 = 4 * rev**3 / (alpha + rev**4)
    term2 = -4 * rev**7 / (alpha + rev**4) ** 2
    return (term1 + term2 - 1) * corr


def B(t, P):
    return jnp.array([[-1.0, 0.0], [0.0, -1.0]])


def β(t, P):
    return jnp.array([1.0, 1.0])


def σ_aux(t, x, P):
    return jnp.array([[P.σ, 0.0], [0.0, P.σ]])


def σ_aux_no_x(t, P):
    return jnp.array([[P.σ, 0.0], [0.0, P.σ]])


def b_aux(t, x, P):
    return jnp.dot(B(t, P), x) + β(t, P)


def a(t, P):
    return jnp.dot(σ_aux(t, 0, P), σ_aux(t, 0, P))
