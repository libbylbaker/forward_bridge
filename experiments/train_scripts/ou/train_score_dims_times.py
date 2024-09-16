import jax.numpy as jnp
import jax.random as jr
import optax

from src.models.score_mlp import ScoreMLP
from src.sdes import sde_data, sde_ornstein_uhlenbeck
from src.training import train_loop, train_utils


def main(key, checkpt_path, dim=1, T=1.0):
    y = jnp.ones(shape=(dim,))

    ou = sde_ornstein_uhlenbeck.ornstein_uhlenbeck(T=T, N=100, dim=dim)

    dt = ou.T / ou.N

    network = {
        "output_dim": ou.dim,
        "time_embedding_dim": 32,
        "init_embedding_dim": 16,
        "activation": "leaky_relu",
        "encoder_layer_dims": [16],
        "decoder_layer_dims": [128, 128],
        "batch_norm": False,
    }

    training = {
        "y": y,
        "batch_size": 100,
        "epochs_per_load": 1,
        "lr": 0.01,
        "num_reloads": 100,
        "load_size": 1000,
    }

    data_gen = sde_data.data_adjoint(y, ou)

    model = ScoreMLP(**network)
    optimiser = optax.chain(optax.adam(learning_rate=training["lr"]))

    score_fn = train_utils.get_score(ou)

    x_shape = jnp.empty(shape=(1, ou.dim))
    t_shape = jnp.empty(shape=(1, 1))
    model_init_sizes = (x_shape, t_shape)

    (loop_key, train_key) = jr.split(key, 2)

    train_step, params, opt_state, batch_stats = train_utils.create_train_step_reverse(
        train_key, model, optimiser, *model_init_sizes, dt=dt, score=score_fn
    )

    train_loop.train(
        loop_key, training, data_gen, train_step, params, batch_stats, opt_state, ou, network, checkpt_path
    )


if __name__ == "__main__":
    import os.path

    seeds = [1, 2, 3, 4, 5]
    dims = jnp.arange(1, 33)
    Ts = jnp.arange(1, 16)
    for seed in seeds:
        main_key = jr.PRNGKey(seed)

        for dim in dims:
            T = 1.0
            checkpoint_path = os.path.abspath(f"../../checkpoints/ou/{seed}/dim_{dim}_T_{T}")
            main(main_key, checkpoint_path, dim=dim, T=T)

        for T in Ts:
            dim = 1
            checkpoint_path = os.path.abspath(f"../../checkpoints/ou/{seed}/dim_{dim}_T_{T}")
            main(main_key, checkpoint_path, dim=dim, T=T)
