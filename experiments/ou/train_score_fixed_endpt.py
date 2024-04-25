import flax.linen as nn
import jax.numpy as jnp
import jax.random
import jax.random as jr
import optax
import orbax
from flax.training import orbax_utils

from src.data_generate_sde import sde_ornstein_uhlenbeck as ou
from src.data_loader import dataloader
from src.models.score_mlp import ScoreMLP
from src.training import utils

seed = 1

sde = {"x0": jnp.ones(shape=(8,)), "N": 100, "dim": 8, "T": 1.0, "y": jnp.ones(shape=(8,))}
dt = 0.01

y = sde["y"]
dim = sde["dim"]
T = sde["T"]
checkpoint_path = f"/Users/libbybaker/Documents/Python/doobs-score-project/doobs_score_matching/checkpoints/ou/fixed_y_{y}_d_{dim}_T_{T}"

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


drift, diffusion = ou.vector_fields()
data_fn = ou.data_reverse(sde["y"], sde["T"], sde["N"])

model = ScoreMLP(**network)
optimiser = optax.adam(learning_rate=training["lr"])

score_fn = utils.get_score(drift=drift, diffusion=diffusion)

num_samples = training["batch_size"] * sde["N"]
x_shape = jnp.empty(shape=(num_samples, sde["dim"]))
t_shape = jnp.empty(shape=(num_samples, 1))
model_init_sizes = (x_shape, t_shape)


def main(key):
    (data_key, dataloader_key, train_key) = jr.split(key, 3)
    data_key = jr.split(data_key, 1)
    train_step, params, opt_state = utils.create_train_step_reverse(
        train_key, model, optimiser, *model_init_sizes, dt=dt, score=score_fn
    )

    # initialise model and train_state

    # train_state = utils.create_train_state(model, train_key, training["lr"], *model_shape)

    # training

    batches_per_epoch = max(training["load_size"] // training["batch_size"], 1)

    print("Training")

    for load in range(training["num_reloads"]):
        # load data
        data_key = jr.split(data_key[0], training["load_size"])
        data = data_fn(data_key)
        infinite_dataloader = dataloader(
            data, training["batch_size"], loop=True, key=jr.split(dataloader_key, 1)[0]
        )

        for epoch in range(training["epochs_per_load"]):
            total_loss = 0
            for batch, (ts, reverse, correction) in zip(
                range(batches_per_epoch), infinite_dataloader
            ):
                params, opt_state, _loss = train_step(params, opt_state, ts, reverse, correction)
                total_loss = total_loss + _loss
            epoch_loss = total_loss / batches_per_epoch

            actual_epoch = load * training["epochs_per_load"] + epoch
            print(f"Epoch: {actual_epoch}, Loss: {epoch_loss}")

            last_epoch = (
                load == training["num_reloads"] - 1 and epoch == training["epochs_per_load"] - 1
            )
            if actual_epoch % 100 == 0 or last_epoch:
                _save(params, opt_state)


def _save(params, opt_state):
    ckpt = {
        "params": params,
        "opt_state": opt_state,
        "sde": sde,
        "network": network,
        "training": training,
    }
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(ckpt)
    orbax_checkpointer.save(checkpoint_path, ckpt, save_args=save_args, force=True)


if __name__ == "__main__":
    main_key = jr.PRNGKey(seed)
    main(main_key)
