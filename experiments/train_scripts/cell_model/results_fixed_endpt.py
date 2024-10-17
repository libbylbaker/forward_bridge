import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from experiments.plotting import load_checkpoint_w_batch_stats
from src.sdes import sde_cell_model, sde_utils, sde_data

import matplotlib.pyplot as plt
from tueplots import bundles, axes, cycler, figsizes
from tueplots.constants.color import palettes


def plot_trajectories(trajs, ts, axis, axs, color, **plot_kwargs):
    for traj in trajs:
        axs.plot(ts, traj[:, axis], color=color, **plot_kwargs)
    return axs


def mean_and_std(trajs):
    mean = jnp.mean(trajs, axis=0)
    std = jnp.std(trajs, axis=0)
    return mean, std


def plot_mean_and_std(mean, std, ts, ax, fig, axs):
    axs.plot(ts, mean[:, ax])
    axs.fill_between(ts, mean[:, ax] - std[:, ax], mean[:, ax] + std[:, ax], alpha=0.5)
    return fig, axs


def format_plots():
    bundle = bundles.aistats2023()
    plt.rcParams.update(bundle)
    plt.rcParams.update(cycler.cycler(color=palettes.paultol_muted))
    plt.rcParams.update(axes.lines())
    plt.rcParams.update(figsizes.aistats2022_half(nrows=4, ncols=1))


if __name__ == "__main__":

    formatting = True
    if formatting:
        format_plots()

    x0 = [0.1, 0.1]
    y = [1.5, 0.2]
    T = 2.0
    N = 50

    # Make forward unconditioned trajectories
    keys_fw = jax.random.split(jax.random.PRNGKey(10), 1000)
    cell_500 = sde_cell_model.cell_model(T=T, N=50)
    data_gen_500 = sde_data.data_forward(x0, cell_500)
    _, forward_trajs, _ = data_gen_500(keys_fw)

    # Conditioned trajectories via learned score
    path = f"../../checkpoints/cell/fixed_y_{y}_T_{T}_N_50"
    keys_cond = jax.random.split(jax.random.PRNGKey(20), 1000)
    cell_50 = sde_cell_model.cell_model(T=T, N=50)
    trained_score, restored = load_checkpoint_w_batch_stats(path)
    conditioned_trajs = jax.vmap(sde_utils.conditioned, (0, None, None, None))(keys_cond, x0, cell_50, trained_score)

    # Conditioned trajectories via DiffusionBridge trained on forward data
    diffusion_bridge_fw = np.load(
        "comparisons/heng_cell_traj_T_2.0_X0_tensor([0.1000, 0.1000])_XT_tensor([1.5000, 0.2000])_M_50_50.npy")
    time_steps = diffusion_bridge_fw.shape[1]
    ts_fw = np.linspace(0, T, time_steps)

    # Conditioned trajectories via DiffusionBridge trained on backward data
    diffusion_bridge_bw = np.load(
        "comparisons/heng_cell_traj_T_2.0_X0_tensor([0.1000, 0.1000])_XT_tensor([1.5000, 0.2000])_M_50_bw.npy")
    time_steps = diffusion_bridge_bw.shape[1]
    ts_bw = np.linspace(0, T, time_steps)

    # Conditioned trajectories via MCMC Guiding method
    df = pd.read_csv('/Users/libbybaker/Documents/Bridge.jl/example/cell_model_runs/cell_T_2_x_0.1_y_[1.5, 0.2].csv')

    iterations = []
    for i in range(2000):
        iteration = df[df['iteration'] == i]
        vals = iteration['value'].values
        vals = np.reshape(vals, (-1, 2))
        if len(vals) > 0:
            iterations.append(vals)

    guided_traj = jnp.asarray(iterations)
    time_steps = guided_traj.shape[1]
    ts_gt = np.linspace(0, T, time_steps)

    fig, axs = plt.subplot_mosaic([["proposed"], ["forward"], ["guided"], ["backward"]], sharex=True, sharey=True)

    upper_max = 1.7
    upper_min = 1.3
    lower_max = 0.4
    lower_min = 0.0

    plot_kwargs_true = {"linewidth": 0.8, "alpha": 0.1}
    plot_kwargs = {"linewidth": 0.8, "alpha": 0.2}

    for ax in axs["proposed"], axs["guided"], axs["forward"], axs["backward"]:
        i = 0
        for traj in forward_trajs:
            if upper_max > traj[-1, 0] > upper_min and lower_min < traj[-1, 1] < lower_max and i<100:
                i += 1
                ax.plot(cell_500.time_grid, traj[:, 0], color='grey', **plot_kwargs_true)
                ax.plot(cell_500.time_grid, traj[:, 1], color='grey', **plot_kwargs_true)
            # if traj[-1, 1] > upper_min and lower_min < traj[-1, 0] < lower_max:
            #     ax.plot(ts, traj[:, 1], color='grey', **plot_kwargs_true)
            #     ax.plot(ts, traj[:, 0], color='grey', **plot_kwargs_true)

        ax.set_xlim(-0.05, 2.05)
        ax.set_xlabel("Time $t$")
        ax.plot([0, 2, 2], [0.1, 1.5, 0.2], marker="X", markersize=5, markeredgewidth=0.5, linestyle="None", zorder=10,
                color="black", markeredgecolor="white")

    axs["guided"] = plot_trajectories(guided_traj[:100], ts_gt, 0, axs["guided"], 'C0', **plot_kwargs)
    axs["guided"] = plot_trajectories(guided_traj[:100], ts_gt, 1, axs["guided"], 'C1', **plot_kwargs)
    axs["guided"].set_title("Guided MCMC")

    axs["forward"] = plot_trajectories(diffusion_bridge_fw[:100], ts_fw, 0, axs["forward"], 'C0', **plot_kwargs)
    axs["forward"] = plot_trajectories(diffusion_bridge_fw[:100], ts_fw, 1, axs["forward"], 'C1', **plot_kwargs)
    axs["forward"].set_title("Time-reversed bridge (DB)")

    axs["proposed"] = plot_trajectories(conditioned_trajs[:100], cell_50.time_grid, 0, axs["proposed"], 'C0',
                                        **plot_kwargs)
    axs["proposed"] = plot_trajectories(conditioned_trajs[:100], cell_50.time_grid, 1, axs["proposed"], 'C1',
                                        **plot_kwargs)
    axs["proposed"].set_title("Proposed")

    axs["backward"] = plot_trajectories(diffusion_bridge_bw[:100], ts_bw, 0, axs["backward"], 'C0', **plot_kwargs)
    axs["backward"] = plot_trajectories(diffusion_bridge_bw[:100], ts_bw, 1, axs["backward"], 'C1', **plot_kwargs)
    axs["backward"].set_title("Forward bridge (DB)")

    plt.savefig("cell_comparison.pdf")
    # fig, axs = plot_trajectories(diffusion_bridge_fw, ts_gt, 0, fig, axs, 'C2', alpha=0.5)
