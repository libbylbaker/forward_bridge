import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def plot_emoji_traj(parts, true_parts=None, time_step=1):
    fig, ax = plt.subplots()
    cmap = plt.get_cmap("Spectral")
    for part in parts:  # part has shape (T, N_part, 2)
        for t in jnp.arange(1, len(part)-1, time_step):
            linewidth = 0.7
            alpha = 0.5
            col = cmap(t / len(part))
            plt.plot(part[t, :, 0], part[t, :, 1], c=col, linewidth=linewidth, alpha=alpha)

        for t in [0, len(part)-1]:
            linewidth = 2
            alpha = 1
            col = cmap(t / len(part))
            plt.plot(part[t, :, 0], part[t, :, 1], c=col, linewidth=linewidth, alpha=alpha)

        if true_parts is not None:
            for part in true_parts:
                plt.plot(part[:, 0], part[:, 1])

    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax)
    cbar.set_label("Time")
    return fig, ax


def plot_kunita_traj(n_landmarks, traj, ax_scale=0.1):
    fig, ax = plt.subplots()
    for landmark in jnp.arange(0, n_landmarks, 1):
        x = traj[:, landmark, 0]
        y = traj[:, landmark, 1]
        N = len(x)
        points = jnp.array([x, y]).T.reshape(-1, 1, 2)
        segments = jnp.concatenate([points[:-1], points[1:]], axis=1)
        plt.Normalize(0, N)
        lc = LineCollection(segments, cmap="viridis")

        lc.set_array(jnp.linspace(0, 1, N))
        lc.set_linewidth(1)
        ax.add_collection(lc)
    # add color bar for time, with scale from 0 to 1
    cbar = fig.colorbar(lc, ax=ax)
    cbar.set_label("Time")
    ax.set_xlim([traj[:, :, 0].min() - ax_scale, traj[:, :, 0].max() + ax_scale])
    ax.set_ylim([traj[:, :, 1].min() - ax_scale, traj[:, :, 1].max() + ax_scale])
    return fig, ax


def visualise_data(data):
    _ = data_heatmap(data[0], data[1][..., 0])
    plt.show()

    for i in range(min(10, data[1].shape[0])):
        plt.plot(data[0][0], data[1][i])
    plt.show()


def data_heatmap(times, trajectories):
    traj_plt = trajectories.flatten()
    t_plot = times.flatten()
    H, yedges, xedges = jnp.histogram2d(traj_plt, t_plot, bins=(20, 100))

    fig, ax = plt.subplots()
    ax.pcolormesh(xedges, yedges, H)
    ax.set_xlim(times.min(), times.max())

    ax.pcolormesh(xedges, yedges, H, cmap="rainbow")
    ax.set_ylim(trajectories.min(), trajectories.max())

    return fig


def plot_score(
        true_score,
        learned_score,
        T,
        y,
        t=jnp.asarray([0.0, 0.25, 0.5, 0.75]),
        x=jnp.linspace(-1, 3, 1000)[..., None],
):
    y = y[0]
    fig, axs = plt.subplots(nrows=1, ncols=t.size, sharey=True)

    for col, ts in enumerate(t):
        if true_score is not None:
            true_score_fn = jax.vmap(true_score, in_axes=(None, 0, None, None))
            y_true = true_score_fn(ts, x.flatten(), T, y)
            axs[col].plot(x.flatten(), y_true.flatten())

        batch_learn_score = jax.vmap(learned_score, in_axes=(None, 0))
        y_pred = batch_learn_score(ts, x)
        axs[col].plot(x.flatten(), y_pred.flatten())

        axs[col].set_title(f"Time: {ts:.2f}")

    return fig, axs


def plot_forward_score(
        true_score,
        learned_score,
        x0,
        t=jnp.asarray([0.0, 0.5, 0.75, 0.95]),
        x=jnp.linspace(-2, 2, 1000)[..., None],
):
    x0 = x0[0]
    fig, axs = plt.subplots(nrows=1, ncols=t.size, sharey=True)
    for col, ts in enumerate(t):
        batch_learn_score = jax.vmap(learned_score, in_axes=(None, 0))
        y_pred = batch_learn_score(ts, x)

        true_score_fn = jax.vmap(true_score, in_axes=(None, None, None, 0))
        y_true = true_score_fn(0.0, x0, ts, x.flatten())

        axs[col].plot(x.flatten(), y_pred.flatten())
        axs[col].plot(x.flatten(), y_true.flatten())
        axs[col].set_title(f"Time: {ts:.2f}")

    return fig, axs


def plot_score_2d(learned_score, t=jnp.asarray([0.0, 1.0, 2.0])):
    x = jnp.linspace(0, 4, 100)
    y = jnp.linspace(0, 2, 100)
    x, y = jnp.meshgrid(x, y)
    x = x[..., None]
    y = y[..., None]

    fig, axs = plt.subplots(nrows=2, ncols=t.size, sharey=True)
    for col, ts in enumerate(t):
        def vectorised_score(ts, x, y):
            xy = jnp.concatenate([x, y], axis=-1)
            return jax.vmap(jax.vmap(learned_score, in_axes=(None, 0)), in_axes=(None, 0))(ts, xy)

        score_pred = vectorised_score(ts, x, y)
        pc = axs[0, col].pcolormesh(x.squeeze(), y.squeeze(), score_pred[:, :, 0])
        pc1 = axs[1, col].pcolormesh(x.squeeze(), y.squeeze(), score_pred[:, :, 1])
        fig.colorbar(pc, ax=axs[0, col])
        fig.colorbar(pc1, ax=axs[1, col])
        axs[0, col].set_title(f"Time: {ts:.2f}")
    plt.show()


def plot_score_variable_y(true_score, learned_score, epoch):
    x = jnp.linspace(2, 6, 100)
    y = jnp.linspace(0, 4, 100)
    t = jnp.asarray([0.0, 0.5, 0.75, 0.9])
    x, y = jnp.meshgrid(x, y)
    x = x[..., None]
    y = y[..., None]

    fig, axs = plt.subplots(nrows=1, ncols=t.size, sharey=True)
    fig.suptitle(f"Epoch: {epoch}")
    for col, ts in enumerate(t):
        vectorised_learnt_score = jax.vmap(jax.vmap(learned_score, in_axes=(None, 0, 0)), in_axes=(None, 0, 0))
        score_pred = vectorised_learnt_score(ts, x, y)
        pc = axs[col].pcolormesh(x.squeeze(), y.squeeze(), score_pred.squeeze())
        fig.colorbar(pc, ax=axs[col])
        axs[col].set_title(f"Time: {ts:.2f}")
    plt.show()

    fig2, axs2 = plt.subplots(nrows=1, ncols=t.size, sharey=True)
    fig2.suptitle(f"Epoch: {epoch}")
    for col, ts in enumerate(t):
        vectorised_score = jax.vmap(jax.vmap(true_score, in_axes=(None, 0, None, 0)), in_axes=(None, 0, None, 0))
        score_true = vectorised_score(ts, x.squeeze(), 1.0, y.squeeze())
        pc = axs2[col].pcolormesh(x.squeeze(), y.squeeze(), score_true.squeeze())
        fig2.colorbar(pc, ax=axs2[col])
        axs2[col].set_title(f"Time: {ts:.2f}")
    plt.show()

    return fig, axs
