import flax.linen as nn
import matplotlib.pyplot as plt
from flax.training.train_state import TrainState
import jax.numpy as jnp
import jax
import optax
from typing import Any, Callable
from functools import partial

from jax import numpy as jnp


def create_train_state(net, key, learning_rate, x_shape, t_shape):
    """Creates an initial `TrainState`."""
    x = jnp.empty(shape=x_shape)
    t = jnp.empty(shape=t_shape)
    variables = net.init(key, x=x, t=t, train=False)
    tx = optax.adam(learning_rate)
    return TrainState.create(
        apply_fn=net.apply,
        params=variables["params"],
        tx=tx,
    )


def trained_score(state) -> Callable:

    @jax.jit
    def score(_t, _x):
        assert _t.ndim == 0
        assert _x.ndim == 1
        result = state.apply_fn(
            {"params": state.params},
            x=_x[jnp.newaxis, ...],
            t=jnp.asarray([_t]),
            train=False,
        )
        return result.flatten()

    return score


def create_train_step(score, data_setup) -> Callable:

    @jax.jit
    def train_step(state, times, trajectory, correction):
        dt = (times[0, 1] - times[0, 0])[0]
        t, traj, correction, true_score = data_setup(times, trajectory, correction, score)

        def loss_fn(params):
            prediction = state.apply_fn(
                {"params": params},
                x=traj,
                t=t,
                train=True,
            )
            loss = dt * jnp.mean(jnp.square(true_score - prediction)*correction)
            # loss = dt * jnp.mean(jnp.square(true_score - prediction))
            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        _loss, grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state, _loss

    return train_step


@partial(jax.jit, static_argnames=['score'])
def _data_setup(times, trajectory, correction, score):
    reversed_traj = trajectory[:, 1:][:, ::-1]
    reversed_traj_plus1 = trajectory[:, :-1][:, ::-1]
    t = times[:, :-1]
    t_plus1 = times[:, 1:]

    time_step_score = jax.vmap(score, in_axes=(0, 0, 0, 0))
    batched_score = jax.vmap(time_step_score, in_axes=(0, 0, 0, 0))
    true_score = batched_score(t, reversed_traj, t_plus1, reversed_traj_plus1)

    correction = jnp.repeat(correction, trajectory.shape[0])

    reversed_traj = reversed_traj.reshape(-1, reversed_traj.shape[-1])
    t = t.reshape(-1, t.shape[-1])
    true_score = true_score.reshape(-1, reversed_traj.shape[-1])

    return t, reversed_traj, correction, true_score


@partial(jax.jit, static_argnames=['score'])
def _forward_data_setup(times, trajectory, correction, score):
    traj = trajectory[:, :-1]
    traj_plus1 = trajectory[:, 1:]
    t = times[:, :-1]
    t_plus1 = times[:, 1:]

    time_step_score = jax.vmap(score, in_axes=(0, 0, 0, 0))
    batched_score = jax.vmap(time_step_score, in_axes=(0, 0, 0, 0))
    true_score = -batched_score(t, traj, t_plus1, traj_plus1)

    correction = jnp.repeat(correction, trajectory.shape[0])

    traj_plus1 = traj_plus1.reshape(-1, traj.shape[-1])
    t_plus1 = t_plus1.reshape(-1, t.shape[-1])
    true_score = true_score.reshape(-1, traj.shape[-1])

    return t_plus1, traj_plus1, correction, true_score
