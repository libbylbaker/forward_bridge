import diffrax
import jax
import jax.numpy as jnp

from src.data_generate_sde import utils
from src.data_generate_sde import time


def get_data_bm(y, T, N):
    """
    :return: ts,
    reverse process: (t, dim), where t is the number of time steps and dim the dimension of the SDE
    correction process: float, correction process at time T
    """
    ts = time.grid(t_start=0, T=T, N=N)
    time_reverse = time.reverse(T=T, times=ts)

    @jax.jit
    @jax.vmap
    def data_bm(key):
        _reverse = reverse(ts=time_reverse, key=key, y=y)
        _correction = correction(ts=ts)
        return ts[..., None], _reverse, jnp.asarray(_correction)

    return data_bm


def get_forward_bm(x0, T, N):
    ts = time.grid(t_start=0, T=T, N=N)

    @jax.jit
    @jax.vmap
    def forward_bm(key):
        _correction = 1.0
        _forward = forward(ts, key, x0=x0)
        return ts[..., None], _forward, jnp.asarray(_correction)

    return forward_bm


def forward(ts, key, x0):
    x0 = jnp.asarray(x0)
    assert x0.ndim == 1
    dim = x0.size
    bm = diffrax.VirtualBrownianTree(
        ts[0].astype(float), ts[-1].astype(float), tol=1e-3, shape=(dim,), key=key
    )
    return jax.vmap(bm.evaluate, in_axes=(0,))(ts)


def backward(ts, key, score_fn, y):
    sol = utils.backward(key, ts, y, score_fn, drift, diffusion)
    return sol.ys


def conditioned(ts, key, score_fn, x0):
    sol = utils.conditioned(key, ts, x0, score_fn, drift, diffusion)
    return sol.ys


def reverse(ts, key, y):
    y = jnp.asarray(y)
    bm = forward(ts, key, y)
    return y + bm


def correction(ts):
    return 1.0


def drift(t, val_array, *args):
    assert val_array.ndim == 1
    dim = val_array.size
    return jnp.zeros(shape=(dim,))


def diffusion(t, val_array, *args):
    assert val_array.ndim == 1
    dim = val_array.size
    return jnp.identity(dim)


def score(t, x, T, x_T):
    return (x_T - x) / (T - t)


def forward_score(t0, x0, t, x):
    return -(x - x0) / (t - t0)
