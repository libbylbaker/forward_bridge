import os.path

import jax.numpy as jnp
import jax.random as jr
import optax

from src import data_boundary_pts
from src.models.score_unet import ScoreUNet
from src.sdes import sde_kunita, sde_data
from src.training import train_loop, train_utils

seed = 1


def main(key):
    num_eye = 5
    num_brow = 5
    num_mouth = 5
    num_outline = 25

    num_landmarks = num_mouth + num_outline + 2*num_eye + 2*num_brow

    fns = data_boundary_pts.smiley_face_fns(num_eye, num_brow, num_mouth, num_outline)
    x0 = data_boundary_pts.flattened_array_from_faces(fns)

    sigma = 0.5
    kappa = 1 / (sigma * jnp.sqrt(2 * jnp.pi))

    kunita = sde_kunita.kunita(T=1., N=100, num_landmarks=num_landmarks, sigma=sigma, kappa=kappa, grid_size=25)

    # sde = {"N": 100, "dim": x0.size, "T": T, "x0": x0}
    dt = kunita.T/kunita.N

    checkpoint_path = os.path.abspath(f"../../checkpoints/kunita/fw/emoji")

    network = {
        "output_dim": kunita.dim,
        "time_embedding_dim": 64,
        "init_embedding_dim": 64,
        "activation": "leaky_relu",
        "encoder_layer_dims": [8*num_landmarks, 4*num_landmarks, 2*num_landmarks, num_landmarks],
        "decoder_layer_dims": [num_landmarks, 2*num_landmarks, 4*num_landmarks, 8*num_landmarks],
        "batch_norm": True,
    }

    training = {
        "batch_size": 64,
        "epochs_per_load": 4,
        "lr": 5e-3,
        "num_reloads": 3000,
        "load_size": 256,
    }

    data_gen = sde_data.data_forward(x0, kunita)
    model = ScoreUNet(**network)
    optimiser = optax.chain(optax.adam(learning_rate=training["lr"]))

    score_fn = train_utils.get_score(kunita)

    x_shape = jnp.empty(shape=(1, num_landmarks, 2))
    t_shape = jnp.empty(shape=(1, 1))
    model_init_sizes = (x_shape, t_shape)

    (loop_key, train_key) = jr.split(key, 2)

    train_step, params, opt_state, batch_stats = train_utils.create_train_step_forward(
        train_key, model, optimiser, *model_init_sizes, dt=dt, score=score_fn
    )

    train_loop.train(
        loop_key, training, data_gen, train_step, params, batch_stats, opt_state, kunita, network, checkpoint_path
    )


if __name__ == "__main__":
    main_key = jr.PRNGKey(seed)
    main(main_key)
