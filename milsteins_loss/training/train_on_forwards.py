import flax.linen as nn
import matplotlib.pyplot as plt
import jax.random as jr
import jax
import jax.numpy as jnp

from milsteins_loss.data_generation import dataloader, get_forward_ir, get_forward_ou, get_forward_bm
from milsteins_loss.models.score_mlp import ScoreMLP
from milsteins_loss.sde import interest_rates as ir
from milsteins_loss.sde import ornstein_uhlenbeck as ou
from milsteins_loss.sde import bm
from milsteins_loss import plotting
from milsteins_loss.training import utils, loss

seed = 1

sde = {"x0": (1.,), "N": 100, "dim": 1, "T": 1., "y": (1.5,)}

sde_fns = {"get_data": get_forward_ir, "backward": ir.backward, "score": ir.forward_score, "diffusion": ir.diffusion,
           "drift": ir.drift}

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
    "epochs_per_load": 1,
    "learning_rate": 5e-3,
    "num_reloads": 50,
    "load_size": 1024
}


def main():
    key = jr.PRNGKey(seed)
    (data_key, dataloader_key, train_key) = jr.split(key, 3)
    data_key = jr.split(data_key, 1)

    # initialise model, score function and train state

    num_samples = training["batch_size"] * sde["N"]
    x_shape = (num_samples, sde["dim"])
    t_shape = (num_samples, 1)
    model = ScoreMLP(**network)
    train_state = utils.create_train_state(
        model, train_key, training["learning_rate"], x_shape=x_shape, t_shape=t_shape
    )

    score_fn = loss.get_score(drift=sde_fns["drift"], diffusion=sde_fns["diffusion"])
    # score_fn = ou.forward_score

    train_step = utils.create_train_step(score_fn, utils._forward_data_setup)

    # data setup
    data_fn = sde_fns["get_data"](sde["x0"], sde["T"], sde["N"])
    batched_data_fn = jax.vmap(data_fn)

    # training

    batches_per_epoch = max(training["load_size"] // training["batch_size"], 1)

    print("Training")

    for load in range(training["num_reloads"]):

        # load data
        data_key = jr.split(data_key[0], training["load_size"])
        data = batched_data_fn(data_key)
        infinite_dataloader = dataloader(data, training["batch_size"], loop=True, key=jr.split(dataloader_key, 1)[0])

        if load == 0:
            plotting.visualise_data(data)

        for epoch in range(training["epochs_per_load"]):
            total_loss = 0
            for batch, (ts, traj, correction) in zip(
                    range(batches_per_epoch), infinite_dataloader
            ):
                train_state, _loss = train_step(train_state, ts, traj, correction)
                total_loss = total_loss + _loss
            epoch_loss = total_loss / batches_per_epoch

            actual_epoch = (load + 1)*training["epochs_per_load"] + epoch
            print(f"Epoch: {actual_epoch}, Loss: {epoch_loss}")

            if actual_epoch % 10 == 0:
                trained_score = utils.trained_score(train_state)
                _ = plotting.plot_forward_score(sde_fns["score"], trained_score, epoch, sde["x0"],
                                                x=jnp.linspace(0.1, 3, 1000)[..., None],
                                                t=jnp.asarray([0.05, 0.5, 0.75, 1.]))
                plt.show()

                traj_keys = jax.random.split(jax.random.PRNGKey(70), 5)

                backward_traj = jax.vmap(sde_fns["backward"], in_axes=(None, 0, None, None))

                trajs = backward_traj(ts[0].flatten(), traj_keys, trained_score, sde["y"])
                plt.title(f"Trajectories at Epoch {actual_epoch}")
                for traj in trajs:
                    plt.plot(ts[0], traj)
                plt.scatter(x=sde["T"], y=sde["x0"])
                plt.show()


if __name__ == "__main__":
    main()
