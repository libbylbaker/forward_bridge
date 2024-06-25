import os.path

import jax.numpy as jnp
import jax.random as jr
import optax

from src.models.score_mlp import ScoreMLP
from src.sdes import sde_ornstein_uhlenbeck as ou
from src.training import train_loop, train_utils

seed = 1


def main(key):
    y = (0.0,)
    sde = {"N": 50, "dim": 1, "T": 1.0, "y": y}
    dt = sde["T"] / sde["N"]

    checkpoint_path = os.path.abspath(f"../../checkpoints/ou/y_{y}_alpha_-5_sigma_1")

    network = {
        "output_dim": sde["dim"],
        "time_embedding_dim": 64,
        "init_embedding_dim": 64,
        "activation": "leaky_relu",
        "encoder_layer_dims": [64, 64, 64],
        "decoder_layer_dims": [64, 64, 64],
        "batch_norm": False,
    }

    training = {
        "batch_size": 64,
        "epochs_per_load": 1,
        "lr": 5e-3,
        "num_reloads": 300,
        "load_size": 64 * 50,
    }

    drift, diffusion = ou.vector_fields()
    data_fn = ou.data_reverse(sde["y"], sde["T"], sde["N"])

    model = ScoreMLP(**network)
    optimiser = optax.chain(optax.adam(learning_rate=training["lr"]))

    score_fn = train_utils.get_score(drift=drift, diffusion=diffusion)

    x_shape = jnp.empty(shape=(1, sde["dim"]))
    t_shape = jnp.empty(shape=(1, 1))
    model_init_sizes = (x_shape, t_shape)

    (data_key, train_key) = jr.split(key, 2)

    train_step, params, opt_state, batch_stats = train_utils.create_train_step_reverse(
        train_key, model, optimiser, *model_init_sizes, dt=dt, score=score_fn
    )

    train_loop.train(
        data_key, training, data_fn, train_step, params, batch_stats, opt_state, sde, network, checkpoint_path
    )


if __name__ == "__main__":
    main_key = jr.PRNGKey(seed)
    main(main_key)
