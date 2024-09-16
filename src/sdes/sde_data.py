from typing import Callable

import jax
import jax.numpy as jnp

from src.sdes import sde_utils


def data_forward(x0, sde: sde_utils.SDE):
    ts = sde.time_grid

    @jax.jit
    @jax.vmap
    def data(key):
        correction_ = jnp.asarray(1.0)
        forward_ = sde_utils.solution(key, ts, x0, sde.drift, sde.diffusion, bm_shape=sde.bm_shape)
        return ts[..., None], forward_, correction_

    return data


def data_adjoint(y, sde: sde_utils.SDE):
    ts = sde.time_grid_reverse
    y = jnp.asarray(y)

    @jax.jit
    @jax.vmap
    def data(key):
        """
        :return: ts,
        reverse process: (t, dim), where t is the number of time steps and dim the dimension of the SDE
        correction process: float, correction process at time T
        """
        reverse_ = sde_utils.solution(
            key, ts, x0=y, drift=sde.adj_drift, diffusion=sde.adj_diffusion, bm_shape=sde.bm_shape
        )
        correction_ = jnp.asarray(1.0)
        return ts[..., None], reverse_, correction_

    return data


def data_reverse_variable_y(sde: sde_utils.SDE):
    ts = sde.time_grid_reverse

    @jax.jit
    @jax.vmap
    def data(key, y):
        reverse_ = sde_utils.solution(key, ts, y, sde.adj_drift, sde.adj_diffusion, bm_shape=sde.bm_shape)
        correction_ = jnp.asarray(1.0)
        return ts[..., None], reverse_, correction_, y

    return data


def data_reverse_distributed_y(sde: sde_utils.SDE, sample_y: Callable):
    ts = sde.time_grid_reverse

    @jax.jit
    @jax.vmap
    def data(key):
        sol_key, y_key = jax.random.split(key)
        y = sample_y(y_key)
        _reverse = sde_utils.solution(sol_key, ts, y, sde.adj_drift, sde.adj_diffusion, bm_shape=sde.bm_shape)
        _correction = 1.0
        return ts[..., None], _reverse, jnp.asarray(_correction)

    return data


def data_reverse_correction(y, sde: sde_utils.SDE):
    ts = sde.time_grid_reverse
    y = jnp.asarray(y)

    def drift(t, x):
        reverse = x[:-1]
        correction = x[-1, None]
        d_reverse = sde.adj_drift(t, reverse)
        d_correction = sde.correction(t, reverse, correction)
        return jnp.concatenate([d_reverse, d_correction])

    def diffusion(t, x):
        reverse = x[:-1]
        correction = x[-1]
        d_reverse = sde.adj_diffusion(t, reverse)
        reverse_and_correction = jnp.pad(d_reverse, ((0, 1), (0, 1)), mode="constant", constant_values=0.0)
        return reverse_and_correction

    @jax.jit
    @jax.vmap
    def data(key):
        """
        :return: ts,
        reverse process: (t, dim), where t is the number of time steps and dim the dimension of the SDE
        correction process: float, correction process at time T
        """
        start_val = jnp.append(y, 1.0)
        bm_shape = (sde.bm_shape[0] + 1, *sde.bm_shape[1:])
        reverse_and_correction_ = sde_utils.solution(
            key, ts, x0=start_val, drift=drift, diffusion=diffusion, bm_shape=bm_shape
        )
        reverse_ = reverse_and_correction_[:, :-1]
        correction_ = reverse_and_correction_[-1, -1]
        return ts[..., None], reverse_, correction_

    return data
