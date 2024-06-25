import flax.linen as nn
import jax.numpy as jnp
import jax.random as jr
import orbax
from flax.training import orbax_utils

from src.models.score_mlp import ScoreMLP
from src.sdes import sde_ornstein_uhlenbeck as ou
from src.training import utils
from src.training.data_loader import dataloader

seed = 1

sde = {"x0": jnp.ones(shape=(1,)), "N": 100, "dim": 1, "T": 1.0, "y": (2.0,)}

sigma = 0.2

y = sde["y"]
dim = sde["dim"]
T = sde["T"]
checkpoint_path = f"/Users/libbybaker/Documents/Python/doobs-score-project/doobs_score_matching/checkpoints/ou/normally_distributed_mean_{y}_sigma_{sigma}"


drift, diffusion = ou.vector_fields()
score_fn = utils.get_score(drift=drift, diffusion=diffusion)
train_step = utils.create_train_step_reverse(score_fn)
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
    model = ScoreMLP(**network)
    model_shape = (x_shape, t_shape)
    train_state = utils.create_train_state(model, train_key, training["lr"], *model_shape)

    # training

    batches_per_epoch = max(training["load_size"] // training["batch_size"], 1)

    print("Training")

    for load in range(training["num_reloads"]):
        # load data
        data_key = jr.split(data_key[0], training["load_size"])

        y_key = jr.split(data_key[0], 1)[0]

        eps = jr.normal(
            y_key,
            shape=(training["load_size"], sde["dim"]),
        )
        y = jnp.asarray(sde["y"]) + sigma * eps
        data = data_fn(data_key, y)

        infinite_dataloader = dataloader(data, training["batch_size"], loop=True, key=jr.split(dataloader_key, 1)[0])

        for epoch in range(training["epochs_per_load"]):
            total_loss = 0
            for batch, (ts, reverse, correction, _) in zip(range(batches_per_epoch), infinite_dataloader):
                train_state, _loss = train_step(train_state, ts, reverse, correction)
                total_loss = total_loss + _loss
            epoch_loss = total_loss / batches_per_epoch

            actual_epoch = load * training["epochs_per_load"] + epoch
            print(f"Epoch: {actual_epoch}, Loss: {epoch_loss}")

            last_epoch = load == training["num_reloads"] - 1 and epoch == training["epochs_per_load"] - 1
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
