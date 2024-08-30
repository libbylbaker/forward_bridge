import os.path

import jax.numpy as jnp
import jax.random as jr
import optax

from src.models.score_unet import ScoreUNet
from src.sdes import sde_kunita
from src.training import train_loop, train_utils
from src import data_boundary_pts

seed = 1


def main(key, T=1.0):
    num_landmarks = 200

    y = data_boundary_pts.sample_circle(num_landmarks)

    sde = {"N": 100, "dim": y.size, "T": T, "y": y}
    dt = sde["T"] / sde["N"]

    checkpoint_path = os.path.abspath(f"../../checkpoints/kunita/circ_r1_lms_{num_landmarks}")

    network = {
        "output_dim": sde["dim"],
        "time_embedding_dim": 64,
        "init_embedding_dim": 64,
        "activation": "leaky_relu",
        "encoder_layer_dims": [8*num_landmarks, 4*num_landmarks, 2*num_landmarks, num_landmarks],
        "decoder_layer_dims": [num_landmarks, 2*num_landmarks, 4*num_landmarks, 8*num_landmarks],
        "batch_norm": True,
    }

    training = {
        "batch_size": 100,
        "load_size": 5000,
        "num_reloads": 300,
        "lr": 5e-3,
        "epochs_per_load": 1,
    }

    drift, diffusion = sde_kunita.vector_fields()
    data_fn = sde_kunita.data_reverse(sde["y"], sde["T"], sde["N"])

    model = ScoreUNet(**network)
    optimiser = optax.chain(optax.adam(learning_rate=training["lr"]))

    score_fn = train_utils.get_score(drift=drift, diffusion=diffusion)

    x_shape = jnp.empty(shape=(1, sde["dim"]))
    t_shape = jnp.empty(shape=(1, 1))
    model_init_sizes = (x_shape, t_shape)

    (loop_key, train_key) = jr.split(key, 2)

    train_step, params, opt_state, batch_stats = train_utils.create_train_step_reverse(
        train_key, model, optimiser, *model_init_sizes, dt=dt, score=score_fn
    )

    train_loop.train(
        loop_key, training, data_fn, train_step, params, batch_stats, opt_state, sde, network, checkpoint_path
    )


if __name__ == "__main__":
    main_key = jr.PRNGKey(seed)
    main(main_key)
