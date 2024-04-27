import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax

from src import plotting
from src.data_generate_sde import sde_bm
from src.data_loader import dataloader
from src.models.score_mlp import ScoreMLP
from src.training import utils

seed = 1

sde = {"x0": (1.0,), "N": 100, "dim": 1, "T": 1.0, "y": (1.0,)}
dt = 0.01

x0 = sde["x0"]
checkpoint_path = f"/Users/libbybaker/Documents/Python/doobs-score-project/doobs_score_matching/checkpoints/bm/fixed_x0_{x0}"

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
    "lr": 5e-3,
    "num_reloads": 1000,
    "load_size": 1000,
}

drift, diffusion = sde_bm.vector_fields()
data_fn = sde_bm.data_forward(sde["x0"], sde["T"], sde["N"])

model = ScoreMLP(**network)
optimiser = optax.adam(learning_rate=training["lr"])

score_fn = utils.get_score(drift=drift, diffusion=diffusion)

x_shape = jnp.empty((1, sde["dim"]))
t_shape = jnp.empty((1, 1))
model_init_sizes = (x_shape, t_shape)


def main(key):
    (data_key, dataloader_key, train_key) = jr.split(key, 3)
    data_key = jr.split(data_key, 1)

    train_step, params, opt_state = utils.create_train_step_forward(
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

            if actual_epoch % 100 == 0:
                trained_score = utils.trained_score(model, params)
                _ = plotting.plot_forward_score(
                    sde_bm.forward_score,
                    trained_score,
                    sde["x0"],
                    x=jnp.linspace(-3, 5, 1000)[..., None],
                    t=jnp.asarray([0.25, 0.5, 0.75]),
                )
                plt.show()


if __name__ == "__main__":
    main_key = jr.PRNGKey(seed)
    main(main_key)
