import functools

import jax
import jax.numpy as jnp

from src.sdes import guided_process, sde_utils, time

# Constants that are always the same but can be changed in the future

_ALPHA = 1.0
_SIGMA = 1.0


def data_forward(x0, T, N):
    return sde_utils.data_forward(x0, T, N, vector_fields())


def data_reverse_guided(x0, y, T, N):
    x0 = jnp.asarray(x0)
    vf_reverse = vector_fields_reverse()
    vf_aux = reverse_guided_auxiliary(x0.size)
    vf_guided = sde_utils.vector_fields_reverse_and_correction_guided(x0, T, vf_reverse, drift_correction, vf_aux)
    return sde_utils.data_reverse_guided(x0, y, T, N, vf_guided)


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
        reverse_ = sde_utils.solution(key, ts_reverse, y, reverse_drift, reverse_diffusion)
        correction_drift_ = lambda t, corr, *args: drift_correction(t, 0.0, corr)
        correction_ = sde_utils.solution_ode(ts, x0=jnp.asarray([1.0]), drift=correction_drift_)
        return ts[..., None], reverse_, correction_[-1, -1]

    return data


def data_reverse_variable_y(T, N):
    ts = time.grid(t_start=0, T=T, N=N)
    ts_reverse = time.reverse(T, ts)
    reverse_drift, reverse_diffusion = vector_fields_reverse()

    @jax.jit
    @jax.vmap
    def data(key, y):
        reverse_ = sde_utils.solution(key, ts_reverse, y, reverse_drift, reverse_diffusion)
        # correction_drift_ = lambda t, corr, *args: drift_correction(t, 0.0, corr)
        # correction_ = utils.solution_ode(ts, x0=jnp.asarray([1.0]), drift=correction_drift_).ys
        return ts[..., None], reverse_, jnp.asarray(1.0), y

    return data


def vector_fields():
    def drift(t, x, *args):
        """dX_t = -alpha X_t dt + sigma dW_t"""
        assert x.ndim == 1
        return -_ALPHA * x

    def diffusion(t, x, *args):
        """dX_t = -alpha X_t dt + sigma dW_t"""
        assert x.ndim == 1
        dim = x.size
        return _SIGMA * jnp.identity(dim)

    return drift, diffusion


def vector_fields_reverse():
    def drift(t, rev, *args):
        return _ALPHA * rev

    def diffusion(t, rev, *args):
        rev = jnp.asarray(rev)
        assert rev.ndim == 1
        dim = rev.size
        return _SIGMA * jnp.identity(dim)

    return drift, diffusion


def vector_fields_reverse_and_correction():
    reverse_drift, reverse_diffusion = vector_fields_reverse()

    def drift(t, x):
        reverse = x[:-1]
        correction = x[-1, None]
        correction_drift = drift_correction(t, reverse, correction)
        reverse_drift_ = reverse_drift(t, reverse)
        return jnp.concatenate([reverse_drift_, correction_drift])

    def diffusion(t, x):
        reverse = x[:-1]
        correction = x[-1]
        d_reverse = reverse_diffusion(t, reverse)
        reverse_and_correction = jnp.pad(d_reverse, ((0, 1), (0, 1)), mode="constant", constant_values=0.0)
        return reverse_and_correction

    return drift, diffusion


def drift_correction(t, rev, corr, *args):
    assert corr.ndim == 1
    return 1.0 * corr


def reverse_guided_auxiliary(dim):
    def B_auxiliary(t):
        return _ALPHA * jnp.identity(dim)

    def beta_auxiliary(t):
        return jnp.zeros(dim)

    def sigma_auxiliary(t):
        return _SIGMA * jnp.identity(dim)

    return B_auxiliary, beta_auxiliary, sigma_auxiliary


def score(t, x, T, y):
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    var = _var_score(T - t)
    mean = _mean_score(T - t, x)
    _score = jnp.exp(-_ALPHA * (T - t)) / var * (y - mean)
    return _score


def score_forward(t0, x0, t, x):
    var = _var_score(t - t0)
    mean = _mean_score(t - t0, x0)
    return (1 / var) * (mean - x)


def _var_score(t):
    return _SIGMA**2 * (1 - jnp.exp(-2 * _ALPHA * t)) / (2 * _ALPHA)


def _mean_score(t, x):
    return x * jnp.exp(-1 * _ALPHA * t)
