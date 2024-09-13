import os.path

import jax.numpy as jnp
import jax.random as jr
import optax

from src.models.score_mlp import ScoreMLP
from src.sdes import sde_data, sde_time_dependent
from src.training import train_loop, train_utils


def main(key):
    td = sde_time_dependent.simple_time_dependent(T=1.0, N=100, dim=1)
    y = 0.0
    dt = td.T / td.N

    checkpoint_path = os.path.abspath(f"../../checkpoints/time_dependent/fixed_y_{y}_d_{td.dim}_T_{td.T}")

    network = {
        "output_dim": td.dim,
        "time_embedding_dim": 16,
        "init_embedding_dim": 16,
        "activation": "leaky_relu",
        "encoder_layer_dims": [16],
        "decoder_layer_dims": [128, 128],
        "batch_norm": False,
    }

    training = {
        "y": y,
        "batch_size": 1000,
        "epochs_per_load": 1,
        "lr": 0.01,
        "num_reloads": 50,
        "load_size": 1000,
    }

    data_gen = sde_data.data_adjoint(y, td)

    model = ScoreMLP(**network)
    optimiser = optax.adam(learning_rate=training["lr"])

    score_fn = train_utils.get_score(td)

    x_shape = jnp.empty(shape=(1, td.dim))
    t_shape = jnp.empty(shape=(1, 1))
    model_shape = (x_shape, t_shape)

    loop_key, train_key = jr.split(key, 2)

    train_step, params, opt_state, batch_stats = train_utils.create_train_step_reverse(
        train_key, model, optimiser, *model_shape, dt=dt, score=score_fn
    )

    train_loop.train(
        loop_key, training, data_gen, train_step, params, batch_stats, opt_state, td, network, checkpoint_path
    )


if __name__ == "__main__":
    seed = 1
    main_key = jr.PRNGKey(seed)
    main(main_key)
