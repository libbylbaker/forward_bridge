import functools

import diffrax
import jax
import jax.numpy as jnp

from src.data_generate_sde import guided_process, time, utils

# Constants that are always the same but can be changed in the future

_ALPHA = 1.0
_SIGMA = 1.0


def data_forward(x0, T, N):
    ts = time.grid(t_start=0, T=T, N=N)
    drift, diffusion = vector_fields()

    @jax.jit
    @jax.vmap
    def data(key):
        correction_ = 1.0
        forward_ = utils.solution(key, ts, x0, drift, diffusion).ys
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
        reverse_ = utils.solution(key, ts_reverse, y, reverse_drift, reverse_diffusion).ys
        # correction_drift_ = lambda t, corr, *args: drift_correction(t, 0.0, corr)
        # correction_ = utils.solution_ode(
        #     ts, x0=jnp.asarray([1.0]), drift=correction_drift_
        # ).ys
        return ts[..., None], reverse_, jnp.asarray(1.0)

    return data


def data_reverse_variable_y(T, N):
    ts = time.grid(t_start=0, T=T, N=N)
    ts_reverse = time.reverse(T, ts)
    reverse_drift, reverse_diffusion = vector_fields_reverse()

    @jax.jit
    @jax.vmap
    def data(key, y):
        reverse_ = utils.solution(key, ts_reverse, y, reverse_drift, reverse_diffusion).ys
        correction_drift_ = lambda t, corr, *args: drift_correction(t, 0.0, corr)
        correction_ = utils.solution_ode(ts, x0=jnp.asarray([1.0]), drift=correction_drift_).ys
        return ts[..., None], reverse_, jnp.asarray(correction_)[-1, 0], y

    return data


def data_reverse_variable_y_guided(x0, T, N):
    x0 = jnp.asarray(x0)
    ts = time.grid(t_start=0, T=T, N=N)
    ts_reverse = time.reverse(T, ts)
    reverse_drift, reverse_diffusion = vector_fields_reverse()
    B_auxiliary, beta_auxiliary, sigma_auxiliary = reverse_guided_auxiliary(len(x0))
    guide_fn = guided_process.get_guide_fn(
        0.0, T, x0, sigma_auxiliary, B_auxiliary, beta_auxiliary
    )
    guided_drift, guided_diffusion = guided_process.vector_fields_guided(
        reverse_drift, reverse_diffusion, guide_fn
    )

    @jax.jit
    @jax.vmap
    def data(key, y):
        reverse_ = utils.solution(key, ts_reverse, y, guided_drift, guided_diffusion).ys
        correction_ = (1.0,)  # Need to work out what this should be (maybe just r.T?)
        return ts[..., None], reverse_, jnp.asarray(correction_)

    return data


def data_reverse_importance(x0, y, T, N):
    ts = time.grid(t_start=0, T=T, N=N)
    ts_reverse = time.reverse(T, ts)
    reverse_drift, reverse_diffusion = vector_fields_reverse()

    @jax.jit
    @jax.vmap
    def data(key):
        rev_corr = utils.important_reverse_and_correction(
            key, ts_reverse, x0, y, reverse_drift, reverse_diffusion, drift_correction
        ).ys
        reverse_ = rev_corr[:, :-1]
        correction_ = rev_corr[-1, -1]
        return ts[..., None], reverse_, jnp.asarray(correction_)

    return data


def data_reverse_guided(x0, y, T, N):
    x0 = jnp.asarray(x0)
    y = jnp.asarray(y)
    ts = time.grid(t_start=0, T=T, N=N)
    ts_reverse = time.reverse(T, ts)
    reverse_drift, reverse_diffusion = vector_fields_reverse()
    B_auxiliary, beta_auxiliary, sigma_auxiliary = reverse_guided_auxiliary(len(x0))
    guide_fn = guided_process.get_guide_fn(
        0.0, T, x0, sigma_auxiliary, B_auxiliary, beta_auxiliary
    )
    guided_drift, guided_diffusion = guided_process.vector_fields_guided(
        reverse_drift, reverse_diffusion, guide_fn
    )

    @jax.jit
    @jax.vmap
    def data(key):
        reverse_ = utils.solution(key, ts_reverse, y, guided_drift, guided_diffusion).ys
        correction_ = (1.0,)  # Need to work out what this should be (maybe just r.T?)
        return ts[..., None], reverse_, jnp.asarray(correction_)

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
    return (1 - jnp.exp(-2 * _ALPHA * t)) / (2 * _ALPHA)


def _mean_score(t, x):
    return x * jnp.exp(-1 * _ALPHA * t)
