import jax.numpy
import jax.numpy as jnp
from jax.scipy.special import i1

from src.data_generate_sde import guided_process, sde_utils, time

# set to 1.5 so that we can use the bessel functions of order 1 in jax.scipy.special, when computing the true score
_A = 1.5


def data_forward(x0, T, N):
    return sde_utils.data_forward(x0, T, N, vector_fields())


def data_importance(x0, y, T, N):
    ts = time.grid(t_start=0, T=T, N=N)
    time_reverse = time.reverse(T=T, times=ts)
    drift, diffusion = vector_fields_reverse()

    @jax.jit
    @jax.vmap
    def data(key):
        """
        :return: ts,
        reverse process: (t, dim), where t is the number of time steps and dim the dimension of the SDE
        correction process: float, correction process at time T
        """
        reverse_and_correction_ = sde_utils.important_reverse_and_correction(
            key, time_reverse, x0, y, drift, diffusion, correction_drift
        )
        reverse_ = reverse_and_correction_[:, :-1]
        correction_ = reverse_and_correction_[-1, -1]
        return ts[..., None], reverse_, jnp.asarray(correction_)

    return data


def data_reverse(y, T, N):
    ts = time.grid(t_start=0, T=T, N=N)
    ts_reverse = time.reverse(T=T, times=ts)
    y = jnp.asarray(y)
    assert y.ndim == 1
    drift, diffusion = vector_fields_reverse_and_correction()

    @jax.jit
    @jax.vmap
    def data(key):
        """
        :return: ts,
        reverse process: (t, dim), where t is the number of time steps and dim the dimension of the SDE
        correction process: float, correction process at time T
        """
        start_val = jnp.append(y, 1.0)
        reverse_and_correction_ = sde_utils.solution(
            key, ts_reverse, x0=start_val, drift=drift, diffusion=diffusion
        )
        reverse_ = reverse_and_correction_[:, :-1]
        correction_ = reverse_and_correction_[-1, -1]
        return ts[..., None], reverse_, jnp.asarray(correction_)

    return data


def data_reverse_guided(x0, y, T, N):
    x0 = jnp.asarray(x0)
    y = jnp.asarray(y)
    ts = time.grid(t_start=0, T=T, N=N)
    ts_reverse = time.reverse(T, ts)
    reverse_drift, reverse_diffusion = vector_fields_reverse()
    B_auxiliary, beta_auxiliary, sigma_auxiliary = reverse_guided_auxiliary(len(x0), y)
    guide_fn = guided_process.get_guide_fn(
        0.0, T, x0, sigma_auxiliary, B_auxiliary, beta_auxiliary
    )
    guided_drift, guided_diffusion = guided_process.vector_fields_guided(
        reverse_drift, reverse_diffusion, guide_fn
    )

    @jax.jit
    @jax.vmap
    def data(key):
        reverse_ = sde_utils.solution(key, ts_reverse, y, guided_drift, guided_diffusion)
        correction_ = (1.0,)  # Need to work out what this should be (maybe just r.T?)
        return ts[..., None], reverse_, jnp.asarray(correction_)

    return data


def vector_fields():
    def drift(t, x, *args):
        """dX_t = (a/X_t - X_t) dt + dW_t"""
        return _A / x - x

    def diffusion(t, x, *args):
        """dX_t = (a/X_t - X_t) dt + dW_t"""
        assert x.ndim == 1
        dim = x.size
        return 1.0 * jnp.identity(dim)

    return drift, diffusion


def vector_fields_reverse():
    def reverse_drift(t, rev, *args):
        return -_A / rev + rev

    def reverse_diffusion(t, rev, *args):
        return jnp.identity(rev.size)

    return reverse_drift, reverse_diffusion


def vector_fields_reverse_and_correction():
    def drift(t, x, *args):
        assert x.ndim == 1
        reverse_drift, _ = vector_fields_reverse()
        reverse = reverse_drift(t, x[:-1])
        correct = correction_drift(t, x[:-1], x[-1, None])
        return jnp.concatenate([reverse, correct])

    def diffusion(t, x, *args):
        assert x.ndim == 1
        _, reverse_diffusion = vector_fields_reverse()
        rev_diffusion = reverse_diffusion(t, x[:-1])
        rev_corr_diff = jnp.pad(
            rev_diffusion, ((0, 1), (0, 1)), mode="constant", constant_values=0.0
        )
        return rev_corr_diff

    return drift, diffusion


def correction_drift(t, rev, corr, *args):
    assert corr.ndim == 1
    c = _A / (rev**2) + 1
    res = jnp.sum(c) * corr
    return res


def reverse_guided_auxiliary(dim, y):
    def B_auxiliary(t):
        return jnp.identity(dim)

    def beta_auxiliary(t):
        return jnp.ones(dim) * -_A / y

    def sigma_auxiliary(t):
        return jnp.identity(dim)

    return B_auxiliary, beta_auxiliary, sigma_auxiliary


def score(t, x, T, y):
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    x = x.squeeze()
    y = y.squeeze()
    arg = x * y / jnp.sinh(T - t)

    if _A == 1.5:
        bessel, i1p = jax.value_and_grad(i1)(arg)
    else:
        raise ValueError("JAX's Bessel function only support a=1.5")

    non_bessel_term = -_A / x + 1.0 / (2.0 * x) - 2.0 * x / (jnp.exp(2 * (T - t)) - 1)
    inv_bessel = 1.0 / bessel
    res = i1p * y / jnp.sinh(T - t)

    return (non_bessel_term + inv_bessel * res)[..., None]


def score_forward(t0, x0, t, x):
    assert _A == 1.5
    x0 = jnp.asarray(x0)
    if x.ndim == 1:
        x = x[0]
    if x0.ndim == 1:
        x0 = x0[0]
    bessel = i1(x * x0 / jnp.sinh(t - t0))
    i1p = jax.grad(i1)
    non_bessel_term = _A / x + 1 / (2 * x) - 2 * x - (2 * x) / (jnp.exp(2 * (t - t0)) - 1)

    inv_bessel = 1 / bessel
    grad_bessel_term = i1p(x * x0 / jnp.sinh(t - t0)) * x0 / jnp.sinh(t - t0)

    return non_bessel_term + inv_bessel * grad_bessel_term
