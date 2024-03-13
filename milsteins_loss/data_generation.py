import jax
import jax.numpy as jnp
import jax.random as jr

from milsteins_loss.sde import time, bm, interest_rates
from milsteins_loss.sde import ornstein_uhlenbeck as ou


def get_forward_ou(x0, T, N):
    ts = time.grid(t_start=0, T=T, N=N)

    def forward_ou(key):
        correction = 1.0
        forward = ou.forward(ts, key, x0=x0)
        return ts[..., None], forward, jnp.asarray(correction)

    return forward_ou


def get_forward_ir(x0, T, N):
    ts = time.grid(t_start=0, T=T, N=N)

    def forward_ir(key):
        correction = 1.0
        forward = interest_rates.forward(ts, key, x0=x0)
        return ts[..., None], forward, jnp.asarray(correction)

    return forward_ir


def get_forward_bm(x0, T, N):
    ts = time.grid(t_start=0, T=T, N=N)

    def forward_bm(key):
        correction = 1.0
        forward = bm.forward(ts, key, x0=x0)
        return ts[..., None], forward, jnp.asarray(correction)

    return forward_bm


def get_data_ir(y, T, N):

    ts = time.grid(t_start=0, T=T, N=N)
    time_reverse = time.reverse(T=T, times=ts)

    @jax.jit
    def data_ir(key):
        """
        :return: ts,
        reverse process: (t, dim), where t is the number of time steps and dim the dimension of the SDE
        correction process: float, correction process at time T
        """
        rev_corr = interest_rates.reverse_and_correction(ts=time_reverse, key=key, y=y)
        reverse = rev_corr[:, :-1]
        correction = rev_corr[-1, -1]
        return ts[..., None], reverse, jnp.asarray(correction)

    return data_ir


def get_data_bm(y, T, N):
    """
    :return: ts,
    reverse process: (t, dim), where t is the number of time steps and dim the dimension of the SDE
    correction process: float, correction process at time T
    """
    ts = time.grid(t_start=0, T=T, N=N)
    time_reverse = time.reverse(T=T, times=ts)

    @jax.jit
    def data_bm(key):
        reverse = bm.reverse(ts=time_reverse, key=key, y=y)
        correction = bm.correction(ts=ts)
        return ts[..., None], reverse, jnp.asarray(correction)

    return data_bm


def get_data_ou(y, T, N):
    """
    :return: ts,
    reverse process: (t, dim), where t is the number of time steps and dim the dimension of the SDE
    correction process: float, correction process at time T
    """
    ts = time.grid(t_start=0, T=T, N=N)
    time_reverse = time.reverse(T, ts)

    @jax.jit
    def data_ou(key):
        reverse = ou.reverse(ts=time_reverse, key=key, y=y)
        correction = ou.correction(ts=ts)
        return ts[..., None], reverse, jnp.asarray(correction)

    return data_ou


def dataloader(arrays, batch_size, loop, *, key):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    while True:
        perm = jr.permutation(key, indices)
        key = jr.split(key, 1)[0]
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size
        if not loop:
            break
