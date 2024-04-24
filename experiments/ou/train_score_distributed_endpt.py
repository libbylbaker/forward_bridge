import functools

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as jr
import orbax
from flax.training import orbax_utils

from src.data_generate_sde import sde_ornstein_uhlenbeck as ou
from src.data_loader import dataloader
from src.models.score_mlp import ScoreMLPDistributedEndpt
from src.training import utils

seed = 1
y_min = -1.0
max_val = 1.0
checkpoint_path = f"/Users/libbybaker/Documents/Python/doobs-score-project/doobs_score_matching/checkpoints/ou/varied_y_{y_min}_to_{max_val}"

sde = {"x0": (1.0,), "N": 100, "dim": 1, "T": 1.0, "y": (1.0,)}

drift, diffusion = ou.vector_fields()
score_fn = utils.get_score(drift=drift, diffusion=diffusion)
train_step = utils.create_train_step_variable_y(score_fn)
data_fn = ou.data_reverse_variable_y(sde["T"], sde["N"])


network = {
    "output_dim": sde["dim"],
    "time_embedding_dim": 16,
    "init_embedding_dim": 16,
    "activation": nn.leaky_relu,
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


def main(key):
    (data_key, dataloader_key, train_key) = jr.split(key, 3)
    data_key = jr.split(data_key, 1)

    # initialise model and train_state
    num_samples = training["batch_size"] * sde["N"]
    x_shape = jnp.empty(shape=(num_samples, sde["dim"]))
    t_shape = jnp.empty(shape=(num_samples, 1))
    model = ScoreMLPDistributedEndpt(**network)
    train_state = utils.create_train_state(
        model, train_key, training["lr"], x_shape, x_shape, t_shape
    )

    # training

    batches_per_epoch = max(training["load_size"] // training["batch_size"], 1)

    print("Training")

    for load in range(training["num_reloads"]):
        # load data
        data_key = jr.split(data_key[0], training["load_size"])
        y_key = jr.split(data_key[0], 1)[0]

        y = jr.uniform(
            y_key,
            shape=(training["load_size"], sde["dim"]),
            minval=y_min,
            maxval=max_val,
        )
        data = data_fn(data_key, y)
        infinite_dataloader = dataloader(
            data, training["batch_size"], loop=True, key=jr.split(dataloader_key, 1)[0]
        )

        for epoch in range(training["epochs_per_load"]):
            total_loss = 0

            for batch, (ts, reverse, correction, y) in zip(
                range(batches_per_epoch), infinite_dataloader
            ):
                train_state, _loss = train_step(train_state, ts, reverse, correction, y)
                total_loss = total_loss + _loss
            epoch_loss = total_loss / batches_per_epoch

            actual_epoch = load * training["epochs_per_load"] + epoch
            print(f"Epoch: {actual_epoch}, Loss: {epoch_loss}")

            last_epoch = (
                load == training["num_reloads"] - 1 and epoch == training["epochs_per_load"] - 1
            )
            # if actual_epoch % 100 == 0 or last_epoch:
            #     _plot(train_state, actual_epoch, ts)

            if actual_epoch % 100 == 0 or last_epoch:
                _save(train_state)


def _save(train_state):
    ckpt = {"state": train_state, "sde": sde, "network": network, "training": training}
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(ckpt)
    orbax_checkpointer.save(checkpoint_path, ckpt, save_args=save_args, force=True)


if __name__ == "__main__":
    main_key = jr.PRNGKey(seed)
    main(main_key)
