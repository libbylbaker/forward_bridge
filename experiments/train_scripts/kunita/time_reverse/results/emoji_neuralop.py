import os.path

import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

from experiments.plotting import checkpoint_neural
from src import data_boundary_pts
from src.sdes import sde_kunita, sde_utils

seed = 1

def main(key):

    num_eye = 10
    num_brow = 10
    num_mouth = 10
    num_outline = 20
    num_landmarks = num_mouth + num_outline + 2*num_eye + 2*num_brow

    test_landmarks = num_landmarks

    fns_smile = data_boundary_pts.smiley_face_fns(num_eye, num_brow, num_mouth, num_outline)
    x0 = data_boundary_pts.flattened_array_from_faces(fns_smile)

    fns_pensive = data_boundary_pts.pensive_face_fns(num_eye, num_brow, num_mouth, num_outline)
    y = data_boundary_pts.flattened_array_from_faces(fns_pensive)

    target = y-x0

    checkpoint_path = os.path.abspath(f"../../../../../checkpoints/kunita/time_rev/emoji_neuralop")
    trained_score, restored = checkpoint_neural(checkpoint_path)

    sde_args = restored["sde"]
    sde_args["num_landmarks"] = test_landmarks
    sde_args["x0"] = x0
    kunita = sde_kunita.kunita(**sde_args)

    keys = jax.random.split(key, 10)

    trajs = jax.vmap(sde_utils.time_reversal, in_axes=(0, None, None, None))(keys, target, kunita, trained_score)

    traj = trajs[0]
    traj = traj.reshape(-1, sde_args["num_landmarks"], 2)

    traj = x0.reshape(-1, sde_args["num_landmarks"], 2) + traj

    for lm in range(sde_args["num_landmarks"]):
        plt.plot(traj[:, lm, 0], traj[:, lm, 1])
    plt.scatter(traj[0, :, 0], traj[0, :, 1], label="start")
    plt.scatter(traj[-1, :, 0], traj[-1, :, 1], label="end")
    plt.scatter(y[::2], y[1::2], label="y")
    plt.scatter(x0[::2], x0[1::2], label="x0")
    plt.legend()
    plt.savefig(f"../figs/emoji_neuralop.png")
    plt.show()


if __name__ == "__main__":
    main_key = jax.random.PRNGKey(seed)
    main(main_key)
