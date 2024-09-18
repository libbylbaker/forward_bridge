import os.path

import jax.numpy as jnp
import jax.random as jr
import optax
import orbax.checkpoint

from src import data_boundary_pts
from src.models.score_unet import ScoreUNet
from src.sdes import sde_kunita, sde_data
from src.training import train_loop, train_utils

seed = 1

def main(key):

    load = False

    num_landmarks = 64
    x0 = data_boundary_pts.butterfly1(num_landmarks)

    sigma = 1.0
    kappa = 1 / (sigma * jnp.sqrt(2 * jnp.pi))
    kappa = 0.1

    sde_args = {"T": 1., "N": 100, "num_landmarks": num_landmarks, "sigma": sigma, "kappa": kappa, "grid_size": 25}
    kunita = sde_kunita.kunita(**sde_args)

    dt = kunita.T/kunita.N

    checkpoint_path = os.path.abspath(f"../../../checkpoints/kunita/fw/butterfly1_{sigma}_{num_landmarks}")

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
        "batch_size": 128,
        "epochs_per_load": 1,
        "lr": 5e-3,
        "num_reloads": 3000,
        "load_size": 2048,
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
