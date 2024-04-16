import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from src import plotting
from src.data_loader import dataloader
from src.models.score_mlp import ScoreMLP
from src.data_generate_sde import sde_ornstein_uhlenbeck as ou
from src.training import utils

seed = 1

sde = {"x0": (1.,), "N": 1000, "dim": 1, "T": 1., "y": (10.,)}

score_fn = utils.get_score(drift=ou.drift, diffusion=ou.diffusion)
train_step = utils.create_train_step_forward(score_fn)
data_fn = ou.data_forward(sde["x0"], sde["T"], sde["N"])

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
        infinite_dataloader = dataloader(
            data, training["batch_size"], loop=True, key=jr.split(dataloader_key, 1)[0]
        )

        plotting.visualise_data(data)

        for epoch in range(training["epochs_per_load"]):
            total_loss = 0
            for batch, (ts, reverse, correction) in zip(
                    range(batches_per_epoch), infinite_dataloader
            ):
                train_state, _loss = train_step(train_state, ts, reverse, correction)
                total_loss = total_loss + _loss
            epoch_loss = total_loss / batches_per_epoch

            actual_epoch = load * training["epochs_per_load"] + epoch
            print(f"Epoch: {actual_epoch}, Loss: {epoch_loss}")

            if actual_epoch % 20 == 0:
                trained_score = utils.trained_score(train_state)
                _ = plotting.plot_forward_score(
                    ou.score_forward,
                    trained_score,
                    actual_epoch,
                    sde["x0"],
                    x=jnp.linspace(-1, 15, 1000)[..., None],
                    t=jnp.asarray([0.05, 0.5, 0.75, 1.0]),
                )
                plt.show()

                traj_keys = jax.random.split(jax.random.PRNGKey(70), 20)
                backward_traj = jax.vmap(ou.backward, in_axes=(0, None, None, None))
                trajs = backward_traj(traj_keys, ts[0].flatten(), sde["y"], trained_score)

                plt.title(f"Trajectories at Epoch {actual_epoch}")
                for traj in trajs:
                    plt.plot(ts[0], traj)
                plt.scatter(x=sde["T"], y=sde["x0"])
                plt.show()


if __name__ == "__main__":
    main_key = jr.PRNGKey(seed)
    main(main_key)
