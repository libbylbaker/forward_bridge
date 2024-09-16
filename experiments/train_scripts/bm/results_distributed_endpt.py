import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tueplots import bundles, axes, cycler
from tueplots.constants.color import palettes

from experiments.plotting import load_checkpoint_w_batch_stats
from src.sdes import sde_utils, sde_bm


def circle(theta, r=5.):
    return r * jax.numpy.cos(theta), r * jax.numpy.sin(theta)

def plot_formatting():
    bundle = bundles.aistats2023(column="full", nrows=1, ncols=2)
    plt.rcParams.update(bundle)
    plt.rcParams.update(axes.lines())
    plt.rcParams.update(cycler.cycler(color=palettes.paultol_muted))
    cyc = plt.rcParams['axes.prop_cycle']
    return cyc


if __name__ == "__main__":

    formatting = False

    if formatting:
        plot_formatting()

    key = jax.random.PRNGKey(1)

    path_r3 = "../../checkpoints/bm/circle_uniformly_distributed_endpt_r_3.0"

    bm = sde_bm.brownian_motion(T=1.0, N=100, dim=2)

    trained_score_r3, restored_r3 = load_checkpoint_w_batch_stats(path_r3)

    num_trajectories = 50

    x0 = jax.vmap(circle)(jnp.linspace(0, 2 * jnp.pi, num_trajectories))
    x1 = jnp.zeros((num_trajectories, 2))

    keys = jax.random.split(key, num_trajectories)

    trajs_centre = jax.vmap(sde_utils.conditioned, in_axes=(0, 0, None, None))(keys, x1, bm, trained_score_r3)

    trajs_circ = jax.vmap(sde_utils.conditioned, in_axes=(0, 0, None, None))(keys, x0, bm, trained_score_r3)

    thetas_50 = jax.numpy.linspace(0, 2 * jax.numpy.pi, 20)
    x, y = circle(thetas_50, 3.0)

    fig, axs = plt.subplot_mosaic([["centre", "circ"]], sharex=True, sharey=True)

    scatter_kwargs = {"s": 10}
    plot_kwargs = {"alpha": 0.5, "linewidth": 0.5}

    for i, traj in enumerate(trajs_centre):
        c = "C0"
        axs["centre"].plot(traj[:, 0], traj[:, 1], **plot_kwargs, color=c)
        axs["centre"].scatter(traj[0, 0], traj[0, 1], **scatter_kwargs, color=c)
        axs["centre"].scatter(traj[-1, 0], traj[-1, 1], **scatter_kwargs, color=c)
    axs["centre"].plot(x, y, color="grey", **plot_kwargs)

    for traj in trajs_circ:
        c = "C1"
        axs["circ"].plot(traj[:, 0], traj[:, 1], **plot_kwargs, color=c)
        axs["circ"].scatter(traj[0, 0], traj[0, 1], **scatter_kwargs, color=c)
        axs["circ"].scatter(traj[-1, 0], traj[-1, 1], **scatter_kwargs, color=c)
    axs["circ"].plot(x, y, color="grey", alpha=0.5, linewidth=0.7)

    plt.savefig("bm_endpt_distribution_r_3.pdf")
    plt.savefig("bm_endpt_distribution_r_3.png")
