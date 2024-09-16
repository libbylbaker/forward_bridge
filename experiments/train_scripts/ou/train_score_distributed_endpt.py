import os.path

import jax.numpy as jnp
import jax.random as jr
import optax

from src.models.score_mlp import ScoreMLPDistributedEndpt
from src.sdes import sde_data, sde_ornstein_uhlenbeck
from src.training import train_loop, train_utils


def main(key):
    y_min = -1.0
    y_max = 1.0

    def y_sampler(key, shape):
        return jr.uniform(key, shape, minval=y_min, maxval=y_max)

    ou = sde_ornstein_uhlenbeck.ornstein_uhlenbeck(T=1.0, N=100, dim=1)

    dt = ou.T / ou.N
    checkpoint_path = os.path.abspath(f"../../checkpoints/ou/varied_y_{y_min}_to_{y_max}")

    network = {
        "output_dim": ou.dim,
        "time_embedding_dim": 16,
        "init_embedding_dim": 16,
        "activation": "leaky_relu",
        "encoder_layer_dims": [16],
        "decoder_layer_dims": [128, 128],
    }

    training = {
        "y_min": y_min,
        "y_max": y_max,
        "batch_size": 100,
        "epochs_per_load": 1,
        "lr": 0.01,
        "num_reloads": 1000,
        "load_size": 2000,
    }

    data_gen = sde_data.data_reverse_variable_y(ou)

    model = ScoreMLPDistributedEndpt(**network)
    optimiser = optax.adam(learning_rate=training["lr"])

    score_fn = train_utils.get_score(ou)

    x_shape = jnp.empty(shape=(1, ou.dim))
    t_shape = jnp.empty(shape=(1, 1))
    model_init_sizes = (x_shape, x_shape, t_shape)

    (loop_key, train_key) = jr.split(key, 2)

    train_step, params, opt_state, batch_stats = train_utils.create_train_step_variable_y(
        train_key, model, optimiser, *model_init_sizes, dt=dt, score=score_fn
    )

    train_loop.train_variable_y(
        loop_key,
        training,
        data_gen,
        train_step,
        params,
        batch_stats,
        opt_state,
        ou,
        network,
        checkpoint_path,
        y_sampler,
    )


if __name__ == "__main__":
    seed = 1

    main_key = jr.PRNGKey(seed)
    main(main_key)
