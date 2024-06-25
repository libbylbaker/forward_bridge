import os

import jax.numpy as jnp
import jax.random as jr
import optax

from src import data_boundary_pts as data
from src.models.score_mlp import ScoreMLP
from src.sdes import sde_ornstein_uhlenbeck as ou
from src.training import train_loop, train_utils

seed = 1


def main(key):
    num_pts = 10
    x0 = data.sample_circle(num_pts)
    sde = {"x0": x0, "N": 100, "dim": num_pts * 2, "T": 1.0}
    dt = sde["T"] / sde["N"]

    checkpoint_path = os.path.abspath(f"../../checkpoints/ou/time_discretisation_forward_{num_pts}_pts")

    network = {
        "output_dim": sde["dim"],
        "time_embedding_dim": 64,
        "init_embedding_dim": 64,
        "activation": "leaky_relu",
        "encoder_layer_dims": [256, 128, 64],
        "decoder_layer_dims": [64, 128, 256],
        "batch_norm": True,
    }

    training = {
        "batch_size": 64,
        "epochs_per_load": 1,
        "lr": 5e-3,
        "num_reloads": 300,
        "load_size": 64 * 50,
    }

    drift, diffusion = ou.vector_fields()
    data_fn = ou.data_forward(sde["x0"], sde["T"], sde["N"])

    model = ScoreMLP(**network)
    optimiser = optax.chain(optax.adam(learning_rate=training["lr"]))

    score_fn = train_utils.get_score(drift=drift, diffusion=diffusion)

    x_empty = jnp.empty(shape=(1, sde["dim"]))
    t_empty = jnp.empty(shape=(1, 1))
    model_init_xt = (x_empty, t_empty)

    (key, train_key) = jr.split(key, 2)

    train_step, params, opt_state, batch_stats = train_utils.create_train_step_forward(
        train_key, model, optimiser, *model_init_xt, dt=dt, score=score_fn
    )

    train_loop.train(key, training, data_fn, train_step, params, batch_stats, opt_state, sde, network, checkpoint_path)


if __name__ == "__main__":
    main_key = jr.PRNGKey(seed)
    main(main_key)
