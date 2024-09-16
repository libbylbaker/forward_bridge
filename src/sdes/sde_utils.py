import dataclasses
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True, eq=True)
class SDE:
    T: float
    N: int
    dim: int
    drift: Callable
    diffusion: Callable
    adj_drift: Callable
    adj_diffusion: Callable
    correction: Callable
    weight_fn: Optional[Callable]
    bm_shape: tuple
    params: Any

    @property
    def dt(self):
        return self.T / self.N

    @property
    def time_grid(self):
        return jnp.linspace(0, self.T, self.N)

    @property
    def time_grid_reverse(self):
        T_array = self.T * jnp.ones_like(self.time_grid)
        return T_array - self.time_grid[::-1]


def solution(key, ts, x0, drift, diffusion, bm_shape=None):
    x0 = jnp.asarray(x0)
    if bm_shape is None:
        bm_shape = x0.shape

    def step_fun(key_and_t_and_x, dt):
        k, t, x = key_and_t_and_x
        k, subkey = jax.random.split(k, num=2)
        eps = jax.random.normal(subkey, shape=bm_shape)
        diffusion_ = diffusion(t, x)
        xnew = x + dt * drift(t, x) + jnp.sqrt(dt) * diffusion_ @ eps
        tnew = t + dt

        return (k, tnew, xnew), xnew

    init = (key, ts[0], x0)
    _, x_all = jax.lax.scan(step_fun, xs=jnp.diff(ts), init=init)
    return jnp.concatenate([x0[None], x_all], axis=0)


def conditioned(key, x0, sde: SDE, score_fn):
    x0 = jnp.asarray(x0)

    def _drift(t, x, *args):
        forward_drift = sde.drift(t, x)
        _score = score_fn(t, x)
        diffusion_tx = sde.diffusion(t, x)
        return forward_drift + diffusion_tx @ diffusion_tx.T @ _score.reshape(-1)

    sol = solution(key, sde.time_grid, x0, _drift, sde.diffusion, sde.bm_shape)
    return sol


def time_reversal(key, y, sde: SDE, score_fn):
    y = jnp.asarray(y)
    T = sde.time_grid[-1]

    def _diffusion(t, x, *args):
        return sde.diffusion(T - t, x)

    def covariance(t, x):
        return jnp.dot(_diffusion(t, x), _diffusion(t, x).T)

    def _drift(t, x, *args):
        forward_drift = -sde.drift(T - t, x)

        _score = score_fn(T - t, x).reshape(-1)
        cov = covariance(T - t, x)

        divergence_cov_fn = jax.jacfwd(covariance)
        divergence_cov = jnp.trace(divergence_cov_fn(T - t, x))

        return forward_drift + cov @ _score + divergence_cov

    sol = solution(key, sde.time_grid, y, _drift, _diffusion, bm_shape=sde.bm_shape)
    return sol
