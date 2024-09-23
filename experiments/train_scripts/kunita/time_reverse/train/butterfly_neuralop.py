import os.path

import jax.numpy as jnp
import jax.random as jr
import optax
import orbax.checkpoint

from src import data_boundary_pts
from src.models.neuralop import CTUNO1D
from src.sdes import sde_kunita, sde_data
from src.training import train_loop, train_utils

seed = 1

def main(key):

    load = False

    num_landmarks = 32
    x0 = data_boundary_pts.butterfly1(num_landmarks)

    sigma = 0.04
    kappa = 1 / sigma

    sde_args = {"T": 1., "N": 100, "num_landmarks": num_landmarks, "x0": x0, "sigma": sigma, "kappa": kappa, "grid_size": 50, "grid_range": (-0.5, 1.5)}
    kunita = sde_kunita.kunita(**sde_args)

    checkpoint_path = os.path.abspath(f"../../../../checkpoints/kunita/time_rev/butterfly_neuralop")

    network = {
        "out_co_dim": 2,
        "lifting_dim": 16,
        "act": "gelu",
        "co_dims_fmults": [1, 2, 4, 4],
        "n_modes_per_layer": [8, 6, 4, 4],
        "norm": "batch"
    }

    training = {
        "batch_size": 128,
        "epochs_per_load": 1,
        "lr": 5e-3,
        "num_reloads": 100,
        "load_size": 2048,
    }

    data_gen = sde_data.data_forward(jnp.zeros_like(x0), kunita)
    model = CTUNO1D(**network)
    optimiser = optax.chain(optax.adam(learning_rate=training["lr"]))

    score_fn = train_utils.get_score(kunita)

    x_shape = jnp.empty(shape=(10, num_landmarks, 2))
    t_shape = jnp.empty(shape=(10, 1))
    model_init_sizes = (x_shape, t_shape)

    (loop_key, train_key) = jr.split(key, 2)

    dt = kunita.T/kunita.N
    train_step, params, opt_state, batch_stats = train_utils.create_train_step_forward(
        train_key, model, optimiser, *model_init_sizes, dt=dt, score=score_fn
    )

    if load:
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        restored = orbax_checkpointer.restore(checkpoint_path)
        params = restored["params"]

    train_loop.train(
        loop_key, training, data_gen, train_step, params, batch_stats, opt_state, sde_args, network, checkpoint_path
    )


if __name__ == "__main__":
    main_key = jr.PRNGKey(seed)
    main(main_key)
