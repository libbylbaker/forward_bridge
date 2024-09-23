import os.path

import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

from experiments.plotting import checkpoint_neural
from src import data_boundary_pts
from src.sdes import sde_kunita, sde_utils

seed = 1

def main(key):

    test_landmarks = 100
    y = data_boundary_pts.butterfly1(test_landmarks)
    x0 = data_boundary_pts.butterfly2(test_landmarks)

    checkpoint_path = os.path.abspath(f"../../../../checkpoints/kunita/adjoint/butterfly_neuralop_full")
    trained_score, restored = checkpoint_neural(checkpoint_path)

    sde_args = restored["sde"]
    sde_args["num_landmarks"] = test_landmarks
    kunita = sde_kunita.kunita(**sde_args)

    keys = jax.random.split(key, 10)

    trajs = jax.vmap(sde_utils.conditioned, in_axes=(0, None, None, None))(keys, y, kunita, trained_score)

    traj = trajs[0]
    traj = traj.reshape(-1, sde_args["num_landmarks"], 2)

    # for lm in range(sde_args["num_landmarks"]):
    #     plt.plot(traj[:, lm, 0], traj[:, lm, 1])
    plt.scatter(traj[0, :, 0], traj[0, :, 1], label="start")
    plt.scatter(traj[-1, :, 0], traj[-1, :, 1], label="end")
    plt.scatter(y[::2], y[1::2], label="y")
    plt.scatter(x0[::2], x0[1::2], label="x0")
    plt.legend()
    plt.savefig(f"../figs/butterfly_neuralop_full.png")
    plt.show()


if __name__ == "__main__":
    main_key = jax.random.PRNGKey(seed)
    main(main_key)
