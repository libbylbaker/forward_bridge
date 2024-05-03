import diffrax
import jax
import jax.numpy as jnp

from doobs_score_matching.src.data_generate_sde import time, utils


def data_forward(x0, T, N):
    return utils.data_forward(x0, T, N, vector_fields())


def data_reverse(y, T, N):
    """
    :return: ts,
    reverse process: (t, dim), where t is the number of time steps and dim the dimension of the SDE
    correction process: float, correction process at time T
    """
    ts = time.grid(t_start=0, T=T, N=N)
    time_reverse = time.reverse(T=T, times=ts)
    drift, diffusion = vector_fields()

    @jax.jit
    @jax.vmap
    def data(key):
        _reverse = utils.solution(key, ts, y, drift, diffusion)
        _correction = 1.0
        return ts[..., None], _reverse, jnp.asarray(_correction)

    return data


def vector_fields():
    def drift(t, val_array, *args):
        assert val_array.ndim == 1
        dim = val_array.size
        return jnp.zeros(shape=(dim,))

    def diffusion(t, val_array, *args):
        assert val_array.ndim == 1
        dim = val_array.size
        return jnp.identity(dim)

    return drift, diffusion


def score(t, x, T, y):
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    return (y - x) / (T - t)


def forward_score(t0, x0, t, x):
    x0 = jnp.asarray(x0)
    x = jnp.asarray(x)
    return -(x - x0) / (t - t0)
