import os.path

import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

from experiments.plotting import checkpoint_unet
from src import data_boundary_pts
from src.sdes import sde_kunita, sde_data, sde_utils
from training import train_utils

seed = 1

def main(key):
    num_eye = 5
    num_brow = 5
    num_mouth = 5
    num_outline = 25

    num_landmarks = num_mouth + num_outline + 2*num_eye + 2*num_brow

    fns = data_boundary_pts.smiley_face_fns(num_eye, num_brow, num_mouth, num_outline)
    x0 = data_boundary_pts.flattened_array_from_faces(fns)
    x0 = x0.reshape(num_landmarks, 2)

    fns_y = data_boundary_pts.pensive_face_fns(num_eye, num_brow, num_mouth, num_outline)
    y = data_boundary_pts.flattened_array_from_faces(fns_y)

    sigma = 0.5
    kappa = 1 / (sigma * jnp.sqrt(2 * jnp.pi))

    kunita = sde_kunita.kunita(T=1., N=100, num_landmarks=num_landmarks, sigma=sigma, kappa=kappa, grid_size=25)

    checkpoint_path = os.path.abspath(f"../../checkpoints/kunita/fw/emoji")

    trained_score, restored = checkpoint_unet(checkpoint_path)

    keys = jax.random.split(key, 10)

    trajs = jax.vmap(sde_utils.time_reversal, in_axes=(0, None, None, None))(keys, y, kunita, trained_score)

    traj = trajs[0]
    traj = traj.reshape(-1, num_landmarks, 2)

    # for lm in range(num_landmarks):
    #     plt.plot(traj[:, lm, 0], traj[:, lm, 1])
    plt.scatter(traj[0, :, 0], traj[0, :, 1])
    plt.scatter(traj[-1, :, 0], traj[-1, :, 1])
    plt.scatter(x0[:, 0], x0[:, 1])

    plt.savefig("kunita_fw_emoji.png")


if __name__ == "__main__":
    main_key = jax.random.PRNGKey(seed)
    main(main_key)
