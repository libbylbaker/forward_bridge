import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


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
    epoch,
    T,
    y,
    t=jnp.asarray([0.0, 0.5, 0.75, 0.95]),
    x=jnp.linspace(-1, 3, 1000)[..., None],
):
    y = y[0]
    fig, axs = plt.subplots(nrows=1, ncols=t.size, sharey=True)
    fig.suptitle(f"Epoch: {epoch}")

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
    epoch,
    x0,
    t=jnp.asarray([0.0, 0.5, 0.75, 0.95]),
    x=jnp.linspace(-2, 2, 1000)[..., None],
):
    x0 = x0[0]
    fig, axs = plt.subplots(nrows=1, ncols=t.size, sharey=True)
    fig.suptitle(f"Epoch: {epoch}")
    for col, ts in enumerate(t):
        batch_learn_score = jax.vmap(learned_score, in_axes=(None, 0))
        y_pred = batch_learn_score(ts, x)

        true_score_fn = jax.vmap(true_score, in_axes=(None, None, None, 0))
        y_true = true_score_fn(0.0, x0, ts, x.flatten())

        axs[col].plot(x.flatten(), y_pred.flatten())
        axs[col].plot(x.flatten(), y_true.flatten())
        axs[col].set_title(f"Time: {ts:.2f}")

    return fig, axs


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
        vectorised_learnt_score = jax.vmap(
            jax.vmap(learned_score, in_axes=(None, 0, 0)), in_axes=(None, 0, 0)
        )
        score_pred = vectorised_learnt_score(ts, x, y)
        pc = axs[col].pcolormesh(x.squeeze(), y.squeeze(), score_pred.squeeze())
        fig.colorbar(pc, ax=axs[col])
        axs[col].set_title(f"Time: {ts:.2f}")
    plt.show()

    fig2, axs2 = plt.subplots(nrows=1, ncols=t.size, sharey=True)
    fig2.suptitle(f"Epoch: {epoch}")
    for col, ts in enumerate(t):
        vectorised_score = jax.vmap(
            jax.vmap(true_score, in_axes=(None, 0, None, 0)), in_axes=(None, 0, None, 0)
        )
        score_true = vectorised_score(ts, x.squeeze(), 1.0, y.squeeze())
        pc = axs2[col].pcolormesh(x.squeeze(), y.squeeze(), score_true.squeeze())
        fig2.colorbar(pc, ax=axs2[col])
        axs2[col].set_title(f"Time: {ts:.2f}")
    plt.show()

    return fig, axs
