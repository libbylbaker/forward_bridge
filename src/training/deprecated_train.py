import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from src import plotting
from src.data_generation import dataloader
from src.models.score_mlp import ScoreMLP
from src.data_generate_sde import interest_rates as ir
from src.data_generate_sde import ornstein_uhlenbeck as ou
from src.data_generate_sde import heston, bm
from src.training import utils

seed = 1

sde = {"x0": (1.,), "N": 1000, "dim": 1, "T": 1., "y": (1.5,)}

heston = {
    "get_data": heston.get_data_heston,
    "conditioned": heston.conditioned,
    "score": None,
    "diffusion": heston.diffusion,
    "drift": heston.drift,
}

ir = {
    "get_data": ir.data_reverse_importance,
    "conditioned": ir.conditioned,
    "score": ir.score,
    "diffusion": ir.diffusion,
    "drift": ir.drift,
}

bm = {
    "get_data": bm.get_data_bm,
    "conditioned": bm.conditioned,
    "score": bm.score,
    "diffusion": bm.diffusion,
    "drift": bm.drift,
}

ou = {
    "get_data": ou.data_reverse,
    "conditioned": ou.conditioned,
    "score": ou.score,
    "diffusion": ou.diffusion,
    "drift": ou.drift,
}

sde_fns = ou

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


def main(key, plot_load=None, plot_epoch=None):
    
    (data_key, dataloader_key, train_key) = jr.split(key, 3)
    data_key = jr.split(data_key, 1)

    # initialise model

    num_samples = training["batch_size"] * sde["N"]
    x_shape = (num_samples, sde["dim"])
    t_shape = (num_samples, 1)
    model = ScoreMLP(**network)

    # initialise train_state, score_fn and train_step

    train_state = utils.create_train_state(model, train_key, training["lr"], x_shape=x_shape, t_shape=t_shape)

    score_fn = utils.get_score(drift=sde_fns["drift"], diffusion=sde_fns["diffusion"])

    train_step = utils.create_train_step_reverse(score_fn)

    # data setup
    data_fn = sde_fns["get_data"](sde["x0"], sde["y"], sde["T"], sde["N"])
    batched_data_fn = jax.vmap(data_fn)

    # training

    batches_per_epoch = max(training["load_size"] // training["batch_size"], 1)

    print("Training")

    for load in range(training["num_reloads"]):

        # load data
        data_key = jr.split(data_key[0], training["load_size"])
        data = batched_data_fn(data_key)
        infinite_dataloader = dataloader(
            data, training["batch_size"], loop=True, key=jr.split(dataloader_key, 1)[0]
        )

        plotting.visualise_data(data)
        # if plot_load is not None:
        #     plot_load(data)

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
                # plot_epoch(trained_score)
                _ = plotting.plot_score(
                    sde_fns["score"],
                    trained_score,
                    epoch,
                    sde["T"],
                    sde["y"],
                    x=jnp.linspace(0.1, 4, 1000)[..., None],
                )
                plt.show()

                traj_keys = jax.random.split(jax.random.PRNGKey(70), 20)
                conditioned_traj = jax.vmap(
                    sde_fns["conditioned"], in_axes=(0, None, None, None)
                )

                trajs = conditioned_traj(traj_keys,
                    ts[0].flatten(), sde["x0"], trained_score
                )

                plt.title(f"Trajectories at Epoch {actual_epoch}")
                for traj in trajs:
                    plt.plot(ts[0], traj)
                plt.scatter(x=sde["T"], y=sde["y"])
                plt.show()

                # plt.title(f"Trajectories at Epoch {actual_epoch}")
                # for traj in trajs:
                #     plt.plot(ts[0], traj)
                # plt.scatter(x=sde["T"], y=sde["y"])
                # plt.show()


if __name__ == "__main__":
    main_key = jr.PRNGKey(seed)
    main(main_key)
