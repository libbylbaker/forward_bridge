import os.path

import jax.numpy as jnp
import jax.random
import jax.random as jr
import optax

from src.models.score_mlp import ScoreMLP
from src.sdes import sde_bm
from src.training import train_loop, train_utils

seed = 1


def main(key):
    r = 1.0

    def y_uniform_circle(key):
        theta = jax.random.uniform(key, (1,), minval=0, maxval=2 * jax.numpy.pi)
        return jnp.concatenate([r * jax.numpy.cos(theta), r * jax.numpy.sin(theta)], axis=-1)

    sde = {"N": 100, "dim": 2, "T": 1.0}
    dt = sde["T"] / sde["N"]

    checkpoint_path = os.path.abspath(f"../../checkpoints/bm/circle_uniformly_distributed_endpt_r_{r}")

    network = {
        "output_dim": sde["dim"],
        "time_embedding_dim": 16,
        "init_embedding_dim": 16,
        "activation": "leaky_relu",
        "encoder_layer_dims": [16],
        "decoder_layer_dims": [128, 128],
    }

    training = {
        "batch_size": 1000,
        "epochs_per_load": 1,
        "lr": 0.01,
        "num_reloads": 1000,
        "load_size": 1000,
    }

    drift, diffusion = sde_bm.vector_fields()
    data_fn = sde_bm.data_reverse_distributed_y(sde["T"], sde["N"], y_uniform_circle)

    model = ScoreMLP(**network)
    optimiser = optax.adam(learning_rate=training["lr"])

    score_fn = train_utils.get_score(drift=drift, diffusion=diffusion)

    x_shape = jnp.empty(shape=(1, sde["dim"]))
    t_shape = jnp.empty(shape=(1, 1))
    model_init_sizes = (x_shape, t_shape)

    (data_key, train_key) = jr.split(key, 2)

    train_step, params, opt_state, batch_stats = train_utils.create_train_step_reverse(
        train_key, model, optimiser, *model_init_sizes, dt=dt, score=score_fn
    )

    train_loop.train(key, training, data_fn, train_step, params, batch_stats, opt_state, sde, network, checkpoint_path)


if __name__ == "__main__":
    main_key = jr.PRNGKey(seed)
    main(main_key)
