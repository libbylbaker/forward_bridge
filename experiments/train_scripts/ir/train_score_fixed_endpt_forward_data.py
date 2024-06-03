import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from src import plotting
from src.data_generate_sde import sde_interest_rates as ir
from src.data_loader import dataloader
from src.models.score_mlp import ScoreMLP
from src.training import utils

seed = 1

sde = {"x0": (3.0,), "N": 1000, "dim": 1, "T": 1.0, "y": (5.0,)}

score_fn = utils.get_score(drift=ir.drift, diffusion=ir.diffusion)
train_step = utils.create_train_step_reverse(score_fn)
data_fn = ir.data_forward(sde["y"], sde["T"], sde["N"])

network = {
    "output_dim": sde["dim"],
    "time_embedding_dim": 64,
    "init_embedding_dim": 64,
    "activation": nn.leaky_relu,
    "encoder_layer_dims": [32, 32],
    "decoder_layer_dims": [32, 32],
}

training = {
    "batch_size": 128,
    "epochs_per_load": 50,
    "lr": 5e-3,
    "num_reloads": 5,
    "load_size": 1024,
}


def main(key):
    (data_key, dataloader_key, train_key) = jr.split(key, 3)
    data_key = jr.split(data_key, 1)

    # initialise model

    num_samples = training["batch_size"] * sde["N"]
    x_shape = (num_samples, sde["dim"])
    t_shape = (num_samples, 1)
    model = ScoreMLP(**network)

    # initialise train_state, score_fn and train_step

    train_state = utils.create_train_state(model, train_key, training["lr"], x_shape=x_shape, t_shape=t_shape)

    # training

    batches_per_epoch = max(training["load_size"] // training["batch_size"], 1)

    print("Training")

    for load in range(training["num_reloads"]):
        # load data
        data_key = jr.split(data_key[0], training["load_size"])
        data = data_fn(data_key)
        infinite_dataloader = dataloader(data, training["batch_size"], loop=True, key=jr.split(dataloader_key, 1)[0])

        plotting.visualise_data(data)

        for epoch in range(training["epochs_per_load"]):
            total_loss = 0
            for batch, (ts, reverse, correction) in zip(range(batches_per_epoch), infinite_dataloader):
                train_state, _loss = train_step(train_state, ts, reverse, correction)
                total_loss = total_loss + _loss
            epoch_loss = total_loss / batches_per_epoch

            actual_epoch = load * training["epochs_per_load"] + epoch
            print(f"Epoch: {actual_epoch}, Loss: {epoch_loss}")

            if actual_epoch % 20 == 0:
                trained_score = utils.trained_score(train_state)

                min_range = min(sde["y"][0], sde["x0"][0])
                max_range = max(sde["y"][0], sde["x0"][0])
                extra_range = (max_range - min_range) * 0.3

                _ = plotting.plot_score(
                    ir.score,
                    trained_score,
                    epoch,
                    sde["T"],
                    sde["y"],
                    x=jnp.linspace(min_range - extra_range, max_range + extra_range, 1000)[..., None],
                )
                plt.show()

                traj_keys = jax.random.split(jax.random.PRNGKey(70), 20)
                conditioned_traj = jax.vmap(ir.conditioned, in_axes=(0, None, None, None))

                trajs = conditioned_traj(traj_keys, ts[0].flatten(), sde["x0"], trained_score)

                plt.title(f"Trajectories at Epoch {actual_epoch}")
                for traj in trajs:
                    plt.plot(ts[0], traj)
                plt.scatter(x=sde["T"], y=sde["y"])
                plt.show()


if __name__ == "__main__":
    main_key = jr.PRNGKey(seed)
    main(main_key)
