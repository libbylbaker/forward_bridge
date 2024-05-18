import jax.numpy as jnp
import jax.random as jr
import optax
import orbax
from flax.training import orbax_utils

from src.data_generate_sde import sde_langevin_landmarks
from src.data_loader import dataloader
from src.models.score_mlp import ScoreMLP
from src.training import train_utils

seed = 1


def main(key, n=2, T=1.0):
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

    num_landmarks = 5

    p0 = jnp.ones(shape=(num_landmarks, 2))
    q0_x = jnp.linspace(0, 5, num_landmarks)
    q0_y = jnp.zeros(num_landmarks)
    q0 = jnp.stack([q0_x, q0_y], axis=1)
    y = jnp.stack([p0, q0], axis=0)

    sde = {"x0": [0.1, 0.1], "N": 600, "dim": n, "T": T, "y": y}
    dt = sde["T"] / sde["N"]

    checkpoint_path = f"/Users/libbybaker/Documents/Python/doobs-score-project/doobs_score_matching/checkpoints/cell/fixed_y_{num_landmarks}"

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

    # weight_fn = sde_cell_model.weight_function_gaussian(x0, 1.)
    drift, diffusion = sde_langevin_landmarks.vector_fields()
    data_fn = sde_langevin_landmarks.data_reverse(sde["y"], sde["T"], sde["N"])

    model = ScoreMLP(**network)
    optimiser = optax.adam(learning_rate=training["lr"])

    score_fn = train_utils.get_score(drift=drift, diffusion=diffusion)

    x_shape = jnp.empty(shape=(1, *y.shape))
    t_shape = jnp.empty(shape=(1, 1))
    model_init_sizes = (x_shape, t_shape)

    (data_key, dataloader_key, train_key) = jr.split(key, 3)
    data_key = jr.split(data_key, 1)

    train_step, params, opt_state = train_utils.create_train_step_reverse(
        train_key, model, optimiser, *model_init_sizes, dt=dt, score=score_fn
    )

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


if __name__ == "__main__":
    main_key = jr.PRNGKey(seed)
    main(main_key)
