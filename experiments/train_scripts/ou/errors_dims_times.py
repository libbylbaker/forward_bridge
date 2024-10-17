import jax
import jax.numpy as jnp
import numpy as np

from experiments.plotting import load_checkpoint_w_batch_stats
from src.sdes import sde_ornstein_uhlenbeck


def score_times(time, seed):
    checkpoint_path = f"../../checkpoints/ou/{seed}/dim_1_T_{time}"
    score, restored = load_checkpoint_w_batch_stats(checkpoint_path)
    return score, restored


def score_dims(dim, seed):
    checkpoint_path = f"../../checkpoints/ou/{seed}/dim_{dim}_T_1.0"
    score, restored = load_checkpoint_w_batch_stats(checkpoint_path)
    return score, restored


def error(ts, true_score, trained_score, sde, y, x0):
    """mean squared error between true and trained score"""
    true = jax.vmap(true_score, in_axes=(0, None, None, None))(ts, x0, sde.T, y)
    trained = jax.vmap(trained_score, in_axes=(0, None))(ts, x0)
    return jnp.mean((true - trained) ** 2)


def error_forward(ts, true_score, trained_score, y):
    """mean squared error between true and trained score"""
    true = jax.vmap(true_score, in_axes=(None, None, 0, None))(0, ts, y)
    trained = jax.vmap(trained_score, in_axes=(0, None))(ts, y)
    return jnp.mean((true - trained) ** 2)


if __name__ == "__main__":

    N = 100

    errors_all = []
    errors_dims_all = []
    for seed in [1, 2, 3, 4, 5]:
        errors = []
        times = np.arange(1, 16)
        dim = 1
        x0 = jnp.ones(shape=(1,))
        for T in times:
            trained_score_, restored_ = score_times(T, seed)
            sde_ = sde_ornstein_uhlenbeck.ornstein_uhlenbeck(T, N, dim)
            true_score = sde_.params[0]
            ts = sde_.time_grid
            error_d_ = error(ts[:-1], true_score, trained_score_, sde_, restored_["training"]["y"], x0)
            errors.append(error_d_)
        errors_all.append(np.asarray(errors))

        errors_dims = []
        dims = np.arange(1, 33)
        T = 1.
        for dim in dims:
            x0 = jnp.ones(shape=(dim,))
            trained_score_, restored_ = score_dims(dim, seed)
            sde_ = sde_ornstein_uhlenbeck.ornstein_uhlenbeck(T, N, dim)
            true_score = sde_.params[0]
            ts = sde_.time_grid
            error_d_ = error(ts[:-1], true_score, trained_score_, sde_, restored_["training"]["y"], x0)
            errors_dims.append(error_d_)
        errors_dims_all.append(np.asarray(errors_dims))

    np.save("ou_errors_dims_1_to_32.npy", errors_dims_all)
    np.save("ou_errors_times_1_to_15.npy", errors_all)