import jax.numpy as jnp
import jax.random as jr
import optax

from src.models.score_mlp import ScoreMLP
from src.sdes import sde_ornstein_uhlenbeck as ou
from src.training import train_loop, train_utils


def main(key, checkpt_path, dim=1, T=1.0):
    y = jnp.ones(shape=(dim,))

    sde = {"N": 100, "dim": dim, "T": T, "y": y}
    dt = sde["T"] / sde["N"]

    network = {
        "output_dim": sde["dim"],
        "time_embedding_dim": 16,
        "init_embedding_dim": 16,
        "activation": "leaky_relu",
        "encoder_layer_dims": [16],
        "decoder_layer_dims": [128, 128],
        "batch_norm": False,
    }

    training = {
        "batch_size": 100,
        "epochs_per_load": 1,
        "lr": 0.01,
        "num_reloads": 100,
        "load_size": 1000,
    }

    drift, diffusion = ou.vector_fields()
    data_fn = ou.data_reverse(sde["y"], sde["T"], sde["N"])

    model = ScoreMLP(**network)
    optimiser = optax.chain(optax.adam(learning_rate=training["lr"]))

    score_fn = train_utils.get_score(drift=drift, diffusion=diffusion)

    x_shape = jnp.empty(shape=(1, sde["dim"]))
    t_shape = jnp.empty(shape=(1, 1))
    model_init_sizes = (x_shape, t_shape)

    (loop_key, train_key) = jr.split(key, 3)

    train_step, params, opt_state, batch_stats = train_utils.create_train_step_reverse(
        train_key, model, optimiser, *model_init_sizes, dt=dt, score=score_fn
    )

    train_loop.train(
        loop_key, training, data_fn, train_step, params, batch_stats, opt_state, sde, network, checkpt_path
    )


if __name__ == "__main__":
    seeds = [1, 2, 3]
    dims = jnp.arange(1, 33)
    Ts = jnp.arange(1, 16)
    for seed in seeds:
        main_key = jr.PRNGKey(seed)

        for dim in dims:
            T = 1.0
            checkpoint_path = f"../../checkpoints/ou/{seed}/dim_{dim}_T_{T}"
            main(main_key, checkpoint_path, dim=dim, T=T)

        for T in Ts:
            dim = 1
            checkpoint_path = f"../../checkpoints/ou/{seed}/dim_{dim}_T_{T}"
            main(main_key, checkpoint_path, dim=dim, T=T)
