from functools import partial
from typing import Callable

import jax
import optax
import orbax
from flax.training import orbax_utils
from flax.training.train_state import TrainState
from jax import numpy as jnp

from src.data_loader import dataloader


def train(
    key,
    train_step,
    params,
    opt_state,
    train_params,
    network_params,
    sde,
    data_fn,
    gradient_transition,
    data_setup_,
):
    data_key, dataloader_key = jax.random.split(key, 2)
    data_key = jax.random.split(data_key, 1)

    batches_per_epoch = max(train_params["load_size"] // train_params["batch_size"], 1)
    print("Training")

    for load in range(train_params["num_reloads"]):
        # load data
        data = data_fn(jax.random.split(data_key[0], train_params["load_size"]))
        infinite_dataloader = dataloader(
            data, train_params["batch_size"], loop=True, key=jax.random.split(dataloader_key, 1)[0]
        )

        for epoch in range(train_params["epochs_per_load"]):
            total_loss = 0
            for batch, data in zip(range(batches_per_epoch), infinite_dataloader):
                ts, reverse, correction, true_score = data_setup_(*data, gradient_transition)
                params, opt_state, loss = train_step(
                    params, opt_state, ts, reverse, correction, true_score
                )
                total_loss = total_loss + loss
            epoch_loss = total_loss / batches_per_epoch

            print(f"Load: {load}|   Epoch: {epoch}|  Loss: {epoch_loss}")

            last_epoch = (
                load == train_params["num_reloads"] - 1
                and epoch == train_params["epochs_per_load"] - 1
            )
            actual_epoch = load * train_params["epochs_per_load"] + epoch
            if actual_epoch % 100 == 0 or last_epoch:
                ckpt = {
                    "params": params,
                    "opt_state": opt_state,
                    "sde": sde,
                    "training": train_params,
                    "network": network_params,
                }
                orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
                save_args = orbax_utils.save_args_from_target(ckpt)
                orbax_checkpointer.save(
                    train_params["checkpoint_path"], ckpt, save_args=save_args, force=True
                )

    return params, opt_state


def create_train_step(key, model, optimiser, *model_init_sizes, dt) -> (Callable, TrainState):
    init_params = model.init(key, *model_init_sizes, train=True)
    init_opt_state = optimiser.init(init_params)
    # init_state = TrainState.create(apply_fn=model.apply, params=variables["params"], tx=optimiser)

    @jax.jit
    def train_step(params, opt_state, times, trajectory, correction, true_score):
        def loss_fn(params_):
            prediction = model.apply(params_, trajectory, times, train=True)
            # loss = dt * jnp.mean(jnp.square(true_score - prediction)*correction)
            loss = dt * jnp.mean(jnp.square(true_score - prediction))
            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        _loss, grads = grad_fn(params)
        updates, opt_state = optimiser.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, _loss

    return train_step, init_params, init_opt_state


def train_variable_y(
    key,
    train_step,
    params,
    opt_state,
    train_params,
    network_params,
    sde,
    data_fn,
    gradient_transition,
    data_setup_,
    y_sampler,
):
    data_key, dataloader_key, y_key = jax.random.split(key, 3)
    data_key = jax.random.split(data_key, 1)

    batches_per_epoch = max(train_params["load_size"] // train_params["batch_size"], 1)
    print("Training")

    for load in range(train_params["num_reloads"]):
        # load data
        y = y_sampler(y_key)
        data = data_fn(jax.random.split(data_key[0], train_params["load_size"]), y)
        infinite_dataloader = dataloader(
            data, train_params["batch_size"], loop=True, key=jax.random.split(dataloader_key, 1)[0]
        )

        for epoch in range(train_params["epochs_per_load"]):
            total_loss = 0
            for batch, (ts, reverse, correction, y) in zip(
                range(batches_per_epoch), infinite_dataloader
            ):
                ts, reverse, correction, y, true_score = data_setup_(
                    ts, reverse, correction, y, gradient_transition
                )
                params, opt_state, loss = train_step(
                    params, opt_state, ts, reverse, correction, y, true_score
                )
                total_loss = total_loss + loss
            epoch_loss = total_loss / batches_per_epoch

            print(f"Load {load}| Epoch {epoch}| Loss: {epoch_loss}")

            last_epoch = (
                load == train_params["num_reloads"] - 1
                and epoch == train_params["epochs_per_load"] - 1
            )
            actual_epoch = load * train_params["epochs_per_load"] + epoch
            if actual_epoch % 100 == 0 or last_epoch:
                ckpt = {
                    "params": params,
                    "opt_state": opt_state,
                    "sde": sde,
                    "training": train_params,
                    "network": network_params,
                }
                orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
                save_args = orbax_utils.save_args_from_target(ckpt)
                orbax_checkpointer.save(
                    train_params["checkpoint_path"], ckpt, save_args=save_args, force=True
                )

    return params, opt_state


def create_train_step_variable_y(
    key, model, optimiser, *model_init_sizes, dt
) -> (Callable, TrainState):
    init_params = model.init(key, *model_init_sizes, train=True)
    init_opt_state = optimiser.init(init_params)

    @jax.jit
    def train_step(params, opt_state, times, trajectory, correction, y, true_score):
        def loss_fn(params_):
            prediction = model.apply(params_, trajectory, y, times, train=True)
            # loss = dt * jnp.mean(jnp.square(true_score - prediction)*correction)
            loss = dt * jnp.mean(jnp.square(true_score - prediction))
            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        _loss, grads = grad_fn(params)
        updates, opt_state = optimiser.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        return params, opt_state, _loss

    return train_step, init_params, init_opt_state


@partial(jax.jit, static_argnames=["score"])
def data_setup_variable_y(times, trajectory, correction, y, score):
    y = jnp.repeat(y, (trajectory.shape[1] - 1), axis=0)
    t, traj, correction, true_score = data_setup(times, trajectory, correction, score)
    return t, traj, correction, y, true_score


@partial(jax.jit, static_argnames=["score"])
def data_setup(times, trajectory, correction, score):
    reversed_traj = trajectory[:, 1:][:, ::-1]
    reversed_traj_plus1 = trajectory[:, :-1][:, ::-1]
    t = times[:, :-1]
    t_plus1 = times[:, 1:]

    time_step_score = jax.vmap(score, in_axes=(0, 0, 0, 0))
    batched_score = jax.vmap(time_step_score, in_axes=(0, 0, 0, 0))
    true_score = batched_score(t, reversed_traj, t_plus1, reversed_traj_plus1)

    correction = jnp.repeat(correction, reversed_traj.shape[1] * reversed_traj.shape[2])
    correction = jnp.reshape(correction, (-1, reversed_traj.shape[-1]))

    reversed_traj = reversed_traj.reshape(-1, reversed_traj.shape[-1])
    t = t.reshape(-1, t.shape[-1])
    true_score = true_score.reshape(-1, reversed_traj.shape[-1])

    return t, reversed_traj, correction, true_score


@partial(jax.jit, static_argnames=["score"])
def data_setup_forward(times, trajectory, correction, score):
    traj = trajectory[:, :-1]
    traj_plus1 = trajectory[:, 1:]
    t = times[:, :-1]
    t_plus1 = times[:, 1:]

    time_step_score = jax.vmap(score, in_axes=(0, 0, 0, 0))
    batched_score = jax.vmap(time_step_score, in_axes=(0, 0, 0, 0))
    true_score = -batched_score(t, traj, t_plus1, traj_plus1)

    traj_plus1 = traj_plus1.reshape(-1, traj.shape[-1])
    t_plus1 = t_plus1.reshape(-1, t.shape[-1])
    true_score = true_score.reshape(-1, traj.shape[-1])
    correction = jnp.ones(traj_plus1.shape[0])

    return t_plus1, traj_plus1, correction, true_score


def gradient_transition_fn(drift, diffusion) -> Callable:
    @jax.jit
    def score(t: float, x: float, t_next: float, x_next: float):
        dt = t_next - t
        drift_last = drift(t, x)
        diffusion_last = diffusion(t, x)
        inv_cov = invert(diffusion_last, diffusion_last.T)
        _score = 1 / dt * inv_cov @ (x_next - x - dt * drift_last)
        return _score

    return score


def trained_score(model, params) -> Callable:
    @jax.jit
    def score(_t, _x):
        assert _t.ndim == 0
        assert _x.ndim == 1
        result = model.apply(params, _x[None, ...], jnp.asarray([_t]), train=True)
        return result.flatten()

    return score


def trained_score_variable_y(model, params) -> Callable:
    @jax.jit
    def score(_t, _x, _y):
        assert _t.ndim == 0
        assert _x.ndim == 1
        _y = jnp.asarray(_y)
        result = model.apply(params, _x[None, ...], _y[None, ...], jnp.asarray([_t]), train=True)
        return result.flatten()

    return score


def invert(mat, mat_transpose):
    """
    Inversion of mat*mat_transpose.
    :param mat: array of shape (n, m) i.e. ndim=2
    :param mat_transpose: array with shape (m, n) with ndim=2
    :return: (mat*mat.T)^{-1} with shape (n, n)
    """
    return jnp.linalg.inv(mat @ mat_transpose)
