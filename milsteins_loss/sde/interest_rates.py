import diffrax
import jax.numpy
import jax.numpy as jnp
from jax.scipy.special import i1

from milsteins_loss.sde.utils import solution

# set to this so that we can use the bessel functions of order 1 in jax.scipy.special, when computing the true score
a = 1.5


def drift(t, x, *args):
    """dX_t = (a/X_t - X_t) dt + dW_t"""
    return a / x - x


def diffusion(t, x, *args):
    """dX_t = (a/X_t - X_t) dt + dW_t"""
    assert x.ndim == 1
    dim = x.size
    return 1.0 * jnp.identity(dim)


def forward(ts, key, x0):
    x0 = jnp.asarray(x0)
    sol = solution(ts, key, drift, diffusion, x0)
    return sol.ys


def reverse_and_correction(ts, key, y):
    y = jnp.asarray(y)
    assert y.ndim == 1
    rev_corr = jnp.append(y, 1.0)

    def _drift(t, x, *args):
        assert x.ndim == 1
        reverse = x[:-1]
        correction = x[-1]
        reverse_next = -1 * a / reverse + reverse
        corr_next = (a / (reverse ** 2) + 1) * correction
        return jnp.append(reverse_next, corr_next)

    def _diffusion(t, x, *args):
        assert x.ndim == 1
        dim = x.size
        diff = jnp.identity(dim)
        diff = diff.at[-1, -1].set(0)
        return diff

    sol = solution(ts, key, _drift, _diffusion, x0=rev_corr)
    return sol.ys


def conditioned(ts, key, score_fn, x0):
    x0 = jnp.asarray(x0)

    def _drift(t, x, *args):
        assert x.ndim == 1
        forward_drift = drift(t, x)
        _score = score_fn(t, x)
        diffusion_tx = diffusion(t, x)
        return forward_drift + diffusion_tx @ diffusion_tx.T @ _score.reshape(-1)

    sol = solution(ts, key, _drift, diffusion, x0)
    return sol.ys


def score(t, x, T, y):
    if x.ndim == 1:
        x = x[0]
    bessel = i1(x * y / jnp.sinh(T - t))
    i1p = jax.grad(i1)

    return (-a / x + 1.0 / (2 * x) - 2 * x / (jnp.exp(2 * (T - t)) - 1)
            + (1 / bessel)
            * i1p(x * y / jnp.sinh(T - t)) * y / jnp.sinh(T - t))


def forward_score(t0, x0, t, x):
    if x.ndim == 1:
        x = x[0]
    bessel = i1(x * x0 / jnp.sinh(t - t0))
    i1p = jax.grad(i1)

    return (a/x + 1/(2*x) - 2*x - (2*x)/(jnp.exp(2*(t-t0))-1)
            + (1 / bessel)
            * i1p(x * x0 / jnp.sinh(t - t0)) * x0 / jnp.sinh(t - t0))


def backward(ts, key, score_fn, y):

    y = jnp.asarray(y)
    T = ts[-1]

    def _diffusion(t, x, *args):
        return diffusion(T-t, x)

    def covariance(t, x):
        return jnp.dot(_diffusion(t, x), _diffusion(t, x).T)

    def _drift(t, x, *args):
        forward_drift = -drift(T-t, x)

        _score = score_fn(T-t, x).reshape(-1)
        cov = covariance(T-t, x)

        divergence_cov_fn = jax.jacfwd(covariance)
        divergence_cov = jnp.trace(divergence_cov_fn(T-t, x))

        return forward_drift + cov@_score + divergence_cov

    sol = solution(ts, key, _drift, _diffusion, y)
    return sol.ys
