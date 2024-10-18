from functools import partial
from typing import Callable

import jax
import optax
from jax import numpy as jnp

from sdes import sde_utils


def trained_score(model, params, batch_stats) -> Callable:
    @jax.jit
    def score(_t, _x):
        assert _t.ndim == 0
        assert _x.ndim == 1
        result = model.apply(
            {"params": params, "batch_stats": batch_stats}, _x[None, ...], jnp.asarray([_t]), train=False
        )
        return result.flatten()

    return score


def trained_score_variable_y(model, params, batch_stats) -> Callable:
    @jax.jit
    def score(_t, _x, _y):
        assert _t.ndim == 0
        assert _x.ndim == 1
        _y = jnp.asarray(_y)
        result = model.apply(
            {"params": params, "batch_stats": batch_stats}, _x[None, ...], _y[None, ...], jnp.asarray([_t]), train=False
        )
        return result.flatten()

    return score


def create_train_step_reverse(key, model, optimiser, *model_init_sizes, dt, score):
    return _create_train_step(key, model, optimiser, *model_init_sizes, dt=dt, score=score, data_setup=_data_setup)


def create_train_step_forward(key, model, optimiser, *model_init_sizes, dt, score):
    return _create_train_step(
        key,
        model,
        optimiser,
        *model_init_sizes,
        dt=dt,
        score=score,
        data_setup=_data_setup_forward,
    )


def create_train_step_variable_y(key, model, optimiser, *model_init_sizes, dt, score):
    variables = model.init(key, *model_init_sizes, train=True)
    batch_stats = variables["batch_stats"] if "batch_stats" in variables else {}
    init_params = variables["params"]
    init_opt_state = optimiser.init(init_params)

    @jax.jit
    def train_step(params, batch_stats, opt_state, times, trajectory, correction, y):
        y = jnp.repeat(y, (trajectory.shape[1] - 1), axis=0)
        t, traj, correction, true_score, diffusion = _data_setup(times, trajectory, correction, score)

        def loss_fn(params_):
            prediction, updates = model.apply(
                {"params": params_, "batch_stats": batch_stats}, traj, y, t, train=True, mutable=["batch_stats"]
            )

            sqrt_norm = prediction[:, None, :] @ diffusion
            weighted_norm = sqrt_norm @ jnp.moveaxis(sqrt_norm, -1, -2)
            diff_term = -2 * prediction[:, None, :] @ true_score[:, :, None]
            loss = jnp.mean((weighted_norm + diff_term))
            return loss, updates

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (_loss, updates), grads = grad_fn(params)
        opt_updates, opt_state = optimiser.update(grads, opt_state, params)
        params = optax.apply_updates(params, opt_updates)
        batch_stats = updates["batch_stats"]
        return params, batch_stats, opt_state, _loss

    return train_step, init_params, init_opt_state, batch_stats




def _create_train_step_learn_y(key, model, optimiser, *model_init_sizes, dt, score):
    variables = model.init(key, *model_init_sizes, train=True)
    batch_stats = variables["batch_stats"] if "batch_stats" in variables else {}
    init_params = variables["params"]
    init_opt_state = optimiser.init(init_params)

    # @jax.jit
    def train_step(params, batch_stats, opt_state, times, trajectory, correction):
        y = trajectory[:, -1]
        y = jnp.repeat(y, (trajectory.shape[1] - 1), axis=0)
        t, traj, correction, true_score, diffusion = _data_setup_forward(times, trajectory, correction, score)

        def loss_fn(params_):
            prediction, updates = model.apply(
                {"params": params_, "batch_stats": batch_stats}, traj, y, t, train=True, mutable=["batch_stats"]
            )

            sqrt_norm = prediction[:, None, :] @ diffusion
            weighted_norm = sqrt_norm @ jnp.moveaxis(sqrt_norm, -1, -2)
            diff_term = -2 * prediction[:, None, :] @ true_score[:, :, None]
            loss = jnp.mean((weighted_norm + diff_term))
            return loss, updates

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (_loss, updates), grads = grad_fn(params)
        opt_updates, opt_state = optimiser.update(grads, opt_state, params)
        params = optax.apply_updates(params, opt_updates)
        batch_stats = updates["batch_stats"]
        return params, batch_stats, opt_state, _loss

    return train_step, init_params, init_opt_state, batch_stats




def _create_train_step(key, model, optimiser, *model_init_sizes, dt, score, data_setup):
    variables = model.init(key, *model_init_sizes, train=True)
    batch_stats = variables["batch_stats"] if "batch_stats" in variables else {}
    init_params = variables["params"]
    init_opt_state = optimiser.init(init_params)

    @jax.jit
    def train_step(params, batch_stats, opt_state, times, trajectory, correction):
        t, traj, correction, true_score, diffusion = _data_setup_forward(times, trajectory, correction, score)

        def loss_fn(params_):
            prediction, updates = model.apply(
                {"params": params_, "batch_stats": batch_stats}, traj, t, train=True, mutable=["batch_stats"]
            )
            # difference = true_score-prediction
            # weighted_norm = (difference[:, None, :]@covs)@difference[:, :, None]
            # loss = 0.5 * dt * jnp.mean(weighted_norm * correction)
            # return loss, updates
            sqrt_norm = prediction[:, None, :] @ diffusion
            weighted_norm = sqrt_norm @ jnp.moveaxis(sqrt_norm, -1, -2)
            diff_term = -2 * prediction[:, None, :] @ true_score[:, :, None]
            loss = jnp.mean((weighted_norm + diff_term))
            return loss, updates

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (_loss, updates), grads = grad_fn(params)
        opt_updates, opt_state = optimiser.update(grads, opt_state, params)
        params = optax.apply_updates(params, opt_updates)
        batch_stats = updates["batch_stats"]
        return params, batch_stats, opt_state, _loss

    return train_step, init_params, init_opt_state, batch_stats


@partial(jax.jit, static_argnames=["score"])
def _data_setup_variable_y(times, trajectory, correction, y, score):
    reversed_traj = trajectory[:, 1:][:, ::-1]
    reversed_traj_plus1 = trajectory[:, :-1][:, ::-1]
    t = times[:, :-1]
    t_plus1 = times[:, 1:]

    time_step_score = jax.vmap(score, in_axes=(0, 0, 0, 0))
    batched_score = jax.vmap(time_step_score, in_axes=(0, 0, 0, 0))
    true_score, covs = batched_score(t, reversed_traj, t_plus1, reversed_traj_plus1)

    correction = jnp.repeat(correction, reversed_traj.shape[1] * reversed_traj.shape[2])
    correction = jnp.reshape(correction, (-1, reversed_traj.shape[-1]))

    y = jnp.repeat(y, reversed_traj.shape[1] * reversed_traj.shape[2])
    y = jnp.reshape(y, (-1, reversed_traj.shape[-1]))

    reversed_traj = reversed_traj.reshape(-1, reversed_traj.shape[-1])
    t = t.reshape(-1, t.shape[-1])
    true_score = true_score.reshape(-1, reversed_traj.shape[-1])

    return t, reversed_traj, correction, y, true_score, covs


@partial(jax.jit, static_argnames=["score"])
def _data_setup(times, trajectory, correction, score):
    reversed_traj = trajectory[:, 1:][:, ::-1]
    reversed_traj_plus1 = trajectory[:, :-1][:, ::-1]
    t = times[:, :-1]
    t_plus1 = times[:, 1:]

    time_step_score = jax.vmap(score, in_axes=(0, 0, 0, 0))
    batched_score = jax.vmap(time_step_score, in_axes=(0, 0, 0, 0))
    true_score, covs = batched_score(t, reversed_traj, t_plus1, reversed_traj_plus1)

    correction = jnp.repeat(correction, reversed_traj.shape[1] * reversed_traj.shape[2])
    correction = jnp.reshape(correction, (-1, reversed_traj.shape[-1]))

    reversed_traj = reversed_traj.reshape(-1, reversed_traj.shape[-1])
    t = t.reshape(-1, t.shape[-1])
    true_score = true_score.reshape(-1, reversed_traj.shape[-1])
    covs = covs.reshape(t.shape[0], reversed_traj.shape[-1], -1)

    return t, reversed_traj, correction, true_score, covs


def _data_setup_y(times, trajectory, correction, score):
    traj = trajectory[:, :-1]
    traj_plus1 = trajectory[:, 1:]
    t = times[:, :-1]
    t_plus1 = times[:, 1:]

    time_step_score = jax.vmap(score, in_axes=(0, 0, 0, 0))
    batched_score = jax.vmap(time_step_score, in_axes=(0, 0, 0, 0))
    true_score_minus, covs = batched_score(t, traj, t_plus1, traj_plus1)
    true_score = -true_score_minus

    traj_plus1 = traj_plus1.reshape(-1, traj.shape[-1])
    t_plus1 = t_plus1.reshape(-1, t.shape[-1])
    true_score = true_score.reshape(-1, traj.shape[-1])
    covs = covs.reshape(t_plus1.shape[0], traj_plus1.shape[-1], -1)

    return t_plus1, traj_plus1, 1.0, true_score, covs



@partial(jax.jit, static_argnames=["score"])
def _data_setup_forward(times, trajectory, correction, score):
    traj = trajectory[:, :-1]
    traj_plus1 = trajectory[:, 1:]
    t = times[:, :-1]
    t_plus1 = times[:, 1:]

    time_step_score = jax.vmap(score, in_axes=(0, 0, 0, 0))
    batched_score = jax.vmap(time_step_score, in_axes=(0, 0, 0, 0))
    true_score_minus, covs = batched_score(t, traj, t_plus1, traj_plus1)
    true_score = -true_score_minus

    traj_plus1 = traj_plus1.reshape(-1, traj.shape[-1])
    t_plus1 = t_plus1.reshape(-1, t.shape[-1])
    true_score = true_score.reshape(-1, traj.shape[-1])
    covs = covs.reshape(t_plus1.shape[0], traj_plus1.shape[-1], -1)

    return t_plus1, traj_plus1, 1.0, true_score, covs


def get_score(sde: sde_utils.SDE) -> Callable:
    @jax.jit
    def score(t0: float, X0: float, t1: float, X1: float):
        dt = t1 - t0
        drift_last = sde.drift(t0, X0)
        diffusion_last = sde.diffusion(t0, X0)

        # cov = diffusion_last @ diffusion_last.T
        diffusion_step_last = X1 - X0 - dt * drift_last
        # _score = 1/dt * jnp.linalg.solve(cov, diffusion_step_last)

        # return _score, cov
        return 1 / dt * diffusion_step_last, diffusion_last

    return score


def invert(mat, mat_transpose):
    """
    Inversion of mat*mat_transpose.
    :param mat: array of shape (n, m) i.e. ndim=2
    :param mat_transpose: array with shape (m, n) with ndim=2
    :return: (mat*mat.T)^{-1} with shape (n, n)
    """
    return jnp.linalg.inv(mat @ mat_transpose)
