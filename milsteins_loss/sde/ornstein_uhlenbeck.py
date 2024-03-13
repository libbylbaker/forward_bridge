import jax
import jax.numpy as jnp
import diffrax

from milsteins_loss.sde.utils import solution

alpha = 1.0
sigma = 1.0


def drift(t, x, *args):
    """dX_t = -alpha X_t dt + sigma dW_t"""
    assert x.ndim == 1
    return -alpha * x


def diffusion(t, x, *args):
    """dX_t = -alpha X_t dt + sigma dW_t"""
    assert x.ndim == 1
    dim = x.size
    return sigma * jnp.identity(dim)


def conditioned(ts, key, score_fn, x0):

    x0 = jnp.asarray(x0)

    def _drift(t, x, *args):
        forward_drift = drift(t, x)
        _score = score_fn(t, x)
        diffusion_tx = diffusion(t, x)
        return forward_drift + diffusion_tx @ diffusion_tx.T @ _score.reshape(-1)

    sol = solution(ts, key, _drift, diffusion, x0)
    return sol.ys


def forward(ts, key, x0):
    x0 = jnp.asarray(x0)
    sol = solution(ts, key, drift, diffusion, x0)
    return sol.ys


def reverse(ts, key, y):
    y = jnp.asarray(y)
    assert y.ndim == 1
    dim = y.size

    def _drift(t, x, *args):
        return alpha * x

    def _diffusion(t, x, *args):
        return sigma * jnp.identity(dim)

    sol = solution(ts, key, _drift, _diffusion, y)
    return sol.ys


def correction(ts):
    def _drift(t, x, *args):
        assert x.ndim == 1
        return 1.0 * x

    terms = diffrax.ODETerm(_drift)
    solver = diffrax.Euler()
    saveat = diffrax.SaveAt(ts=jnp.asarray([ts[-1]]))
    sol = diffrax.diffeqsolve(
        terms,
        solver,
        t0=ts[0].astype(float),
        t1=ts[-1].astype(float),
        dt0=0.05,
        y0=jnp.asarray([1.0]),
        saveat=saveat,
    )
    return sol.ys


def var_score(t):
    return (1 - jnp.exp(-2 * alpha * t)) / (2 * alpha)


def mean_score(t, x):
    return x * jnp.exp(-1 * alpha * t)


def score(t, x, T, y):
    var = var_score(T-t)
    mean = mean_score(T-t, x)
    _score = jnp.exp(-alpha * (T - t)) / var * (y - mean)
    return _score


def forward_score(t0, x0, t, x):
    var = var_score(t-t0)
    mean = mean_score(t-t0, x0)
    return (1/var) * (mean - x)


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
