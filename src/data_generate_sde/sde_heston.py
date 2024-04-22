import jax
import jax.numpy as jnp

from src.data_generate_sde import time, utils

mu = 0.05
xi = 0.1
rho = -0.7
kappa = 0.15
theta = -0.3

init = (10, 0.25)


def get_forward(x0, T, N):
    ts = time.grid(t_start=0, T=T, N=N)

    @jax.jit
    def forward_heston(key):
        _correction = 1.0
        _forward = forward(key, ts, x0=x0)
        return ts[..., None], _forward, jnp.asarray(_correction)

    return forward_heston


def get_data_heston(y, T, N):
    ts = time.grid(t_start=0, T=T, N=N)
    time_reverse = time.reverse(T=T, times=ts)

    @jax.jit
    def data_heston(key):
        rev_corr = reverse_correction(key=key, ts=time_reverse, y=y)
        _reverse = rev_corr[:, :-1]
        _correction = rev_corr[-1, -1]
        return ts[..., None], _reverse, jnp.asarray(_correction)

    return data_heston


def forward(key, ts, x0):
    x0 = jnp.asarray(x0)
    sol = utils.solution(key, ts, x0, drift, diffusion)
    return sol.ys


def backward(key, ts, y, score_fn):
    sol = utils.backward(key, ts, y, score_fn, drift, diffusion)
    return sol.ys


def conditioned(key, ts, x0, score_fn):
    sol = utils.conditioned(key, ts, x0, score_fn, drift, diffusion)
    return sol.ys


def drift(t, x, *args):
    assert x.ndim == 1
    assert x.size == 2
    x_0 = mu * x[0]
    x_1 = kappa * (theta - x[1])
    return jnp.asarray([x_0, x_1])


def diffusion(t, x, *args):
    sigma_11 = jnp.sqrt(x[1]) * x[0]
    sigma_12 = 0
    sigma_21 = xi * jnp.sqrt(x[1]) * rho
    sigma_22 = xi * jnp.sqrt(x[1]) * jnp.sqrt(1 - rho**2)
    return jnp.asarray([[sigma_11, sigma_12], [sigma_21, sigma_22]])


def reverse_correction(key, ts, y):
    y = jnp.asarray(y)
    assert y.ndim == 1
    y = jnp.append(y, 1.0)

    def _drift(t, x, *args):
        assert x.ndim == 1
        assert x.size == 3
        alpha_1 = (2 * x[1] + rho * xi - mu) * x[0]
        alpha_2 = (rho * xi + kappa) * x[1] + xi**2 - kappa * theta
        correction = (x[1] + rho * xi - mu + kappa) * x[2]
        return jnp.asarray([alpha_1, alpha_2, correction])

    def _diffusion(t, x, *args):
        diffusion_term = diffusion(t, x, args)
        diffusion_correction = jnp.zeros(shape=(3, 3))
        diffusion_correction = diffusion_correction.at[:-1, :-1].set(diffusion_term)
        return diffusion_correction

    sol = utils.solution(key, ts, y, _drift, _diffusion)
    return sol.ys
