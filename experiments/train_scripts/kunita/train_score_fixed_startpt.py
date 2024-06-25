import os

import jax.numpy as jnp
import jax.random as jr
import optax
import orbax
from flax.training import orbax_utils

from src.models.score_mlp import ScoreMLP
from src.sdes import sde_kunita
from src.training import train_utils
from src.training.data_loader import dataloader

seed = 1


def main(key, T=1.0):
    def _save(params, opt_state, batch_stats):
        ckpt = {
            "params": params,
            "batch_stats": batch_stats,
            "opt_state": opt_state,
            "sde": sde,
            "network": network,
            "training": training,
        }
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(ckpt)
        orbax_checkpointer.save(checkpoint_path, ckpt, save_args=save_args, force=True)

    num_landmarks = 5

    def sample_circle(num_landmarks: int, radius=1.0, centre=jnp.asarray([0, 0])) -> jnp.ndarray:
        theta = jnp.linspace(0, 2 * jnp.pi, num_landmarks, endpoint=False)
        x = jnp.cos(theta)
        y = jnp.sin(theta)
        return (radius * jnp.stack([x, y], axis=1) + centre).flatten()

    x0 = sample_circle(num_landmarks)

    sde = {"x0": x0, "N": 100, "dim": x0.size, "T": T}
    dt = sde["T"] / sde["N"]

    checkpoint_path = os.path.abspath(f"../../checkpoints/kunita/fixed_x0_lms_{num_landmarks}")

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

    # weight_fn = sde_cell_model.weight_function_gaussian(x0, 1.)
    drift, diffusion = sde_kunita.vector_fields()
    data_fn = sde_kunita.data_forward(sde["x0"], sde["T"], sde["N"])

    model = ScoreMLP(**network)
    optimiser = optax.adam(learning_rate=training["lr"])

    score_fn = train_utils.get_score(drift=drift, diffusion=diffusion)

    x_shape = jnp.empty(shape=(1, sde["dim"]))
    t_shape = jnp.empty(shape=(1, 1))
    model_init_sizes = (x_shape, t_shape)

    (data_key, dataloader_key, train_key) = jr.split(key, 3)
    data_key = jr.split(data_key, 1)

    train_step, params, opt_state, batch_stats = train_utils.create_train_step_forward(
        train_key, model, optimiser, *model_init_sizes, dt=dt, score=score_fn
    )

    # training
    batches_per_epoch = max(training["load_size"] // training["batch_size"], 1)

    print("Training")

    for load in range(training["num_reloads"]):
        # load data
        data_key = jr.split(data_key[0], training["load_size"])
        data = data_fn(data_key)
        infinite_dataloader = dataloader(data, training["batch_size"], loop=True, key=jr.split(dataloader_key, 1)[0])

        for epoch in range(training["epochs_per_load"]):
            total_loss = 0
            for batch, (ts, reverse, correction) in zip(range(batches_per_epoch), infinite_dataloader):
                params, batch_stats, opt_state, _loss = train_step(
                    params, batch_stats, opt_state, ts, reverse, correction
                )
                total_loss = total_loss + _loss
            epoch_loss = total_loss / batches_per_epoch

            actual_epoch = load * training["epochs_per_load"] + epoch
            print(f"Epoch: {actual_epoch}, Loss: {epoch_loss}")

            last_epoch = load == training["num_reloads"] - 1 and epoch == training["epochs_per_load"] - 1
            if actual_epoch % 100 == 0 or last_epoch:
                _save(params, opt_state, batch_stats)


if __name__ == "__main__":
    main_key = jr.PRNGKey(seed)
    main(main_key)
