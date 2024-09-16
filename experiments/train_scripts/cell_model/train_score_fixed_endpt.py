import os.path

import jax.numpy as jnp
import jax.random as jr
import optax

from src.models.score_mlp import ScoreMLP
from src.sdes import sde_cell_model, sde_data
from src.training import train_loop, train_utils

seed = 1


def main(key, T=2.0):
    cell_sde = sde_cell_model.cell_model(T, N=50)

    dt = cell_sde.T / cell_sde.N

    y = [1.5, 0.2]
    checkpoint_path = os.path.abspath(f"../../checkpoints/cell/fixed_y_{y}_T_{T}_N_{cell_sde.N}")

    network = {
        "output_dim": cell_sde.dim,
        "time_embedding_dim": 32,
        "init_embedding_dim": 16,
        "activation": "leaky_relu",
        "encoder_layer_dims": [16],
        "decoder_layer_dims": [128, 128],
    }

    training = {
        "y": [1.5, 0.2],
        "batch_size": 100,
        "epochs_per_load": 1,
        "lr": 1e-2,
        "num_reloads": 100,
        "load_size": 10000,
    }

    data_fn = sde_data.data_adjoint(y, cell_sde)

    model = ScoreMLP(**network)
    optimiser = optax.chain(optax.adam(learning_rate=training["lr"]))

    score_fn = train_utils.get_score(cell_sde)

    x_shape = jnp.empty(shape=(1, cell_sde.dim))
    t_shape = jnp.empty(shape=(1, 1))
    model_init_sizes = (x_shape, t_shape)

    (loop_key, train_key) = jr.split(key, 2)

    train_step, params, opt_state, batch_stats = train_utils.create_train_step_reverse(
        train_key, model, optimiser, *model_init_sizes, dt=dt, score=score_fn
    )

    train_loop.train(
        loop_key, training, data_fn, train_step, params, batch_stats, opt_state, cell_sde, network, checkpoint_path
    )


if __name__ == "__main__":
    main_key = jr.PRNGKey(seed)
    main(main_key)
