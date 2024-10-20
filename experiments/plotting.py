import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import orbax.checkpoint

import src.models
from sdes import sde_utils
from src.training import train_utils


def load_checkpoint_wo_batch_stats(checkpoint_path):
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    restored = orbax_checkpointer.restore(checkpoint_path)
    model = src.models.ScoreMLP(**restored["network"])
    params = restored["params"]["params"]
    batch_stats = {}
    trained_score = train_utils.trained_score(model, params, batch_stats)
    return trained_score, restored


def load_checkpoint_w_batch_stats(checkpoint_path):
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    restored = orbax_checkpointer.restore(checkpoint_path)
    model = src.models.ScoreMLP(**restored["network"])
    params = restored["params"]
    batch_stats = restored["batch_stats"]
    trained_score = train_utils.trained_score(model, params, batch_stats)
    return trained_score, restored


def checkpoint_unet(checkpoint_path):
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    restored = orbax_checkpointer.restore(checkpoint_path)
    model = src.models.ScoreUNet(**restored["network"])
    params = restored["params"]
    batch_stats = restored["batch_stats"]
    trained_score = train_utils.trained_score(model, params, batch_stats)
    return trained_score, restored

def checkpoint_neural(checkpoint_path):
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    restored = orbax_checkpointer.restore(checkpoint_path)
    model = src.models.CTUNO1D(**restored["network"])
    params = restored["params"]
    batch_stats = restored["batch_stats"]
    trained_score = train_utils.trained_score(model, params, batch_stats)
    return trained_score, restored


def plot_score_variable_y(true_score, learned_score, x_min, x_max, y_min, y_max, cmap="viridis"):
    x = jnp.linspace(x_min, x_max, 100)
    y = jnp.linspace(y_min, y_max, 100)
    t = jnp.asarray([0.25, 0.5, 0.75])
    x, y = jnp.meshgrid(x, y)
    x = x[..., None]
    y = y[..., None]

    fig, axs = plt.subplots(nrows=2, ncols=t.size, sharey=True, sharex=True)
    for col, ts in enumerate(t):
        vectorised_learnt_score = jax.vmap(jax.vmap(learned_score, in_axes=(None, 0, 0)), in_axes=(None, 0, 0))
        score_pred = vectorised_learnt_score(ts, x, y)
        pc = axs[0, col].pcolormesh(x.squeeze(), y.squeeze(), score_pred.squeeze(), cmap=cmap)
        axs[0, col].set_title(f"Time: {ts:.2f}")
        axs[1, col].set_xlabel(r"$x$")
    axs[0, 0].set_ylabel(r"$y$")
    axs[1, 0].set_ylabel(r"$y$")
    # axs[0, 0].text('Learned Score', rotation='vertical', x=-0.4, y=0.1)
    # axs[1, 0].text('True Score', rotation='vertical', x=-0.4, y=0.1)
    axs[0, 0].text(-1.9, 0.0, 'Learned Score', verticalalignment='center', rotation=90)
    axs[1, 0].text(-1.9, 0.0, 'True Score', verticalalignment='center', rotation=90)

    for col, ts in enumerate(t):
        vectorised_score = jax.vmap(jax.vmap(true_score, in_axes=(None, 0, None, 0)), in_axes=(None, 0, None, 0))
        score_true = vectorised_score(ts, x.squeeze(), 1.0, y.squeeze())
        pc = axs[1, col].pcolormesh(x.squeeze(), y.squeeze(), score_true.squeeze(), cmap=cmap)
    fig.colorbar(pc, ax=axs.ravel().tolist())

    return fig, axs


def plot_score_error_variable_y(true_score, learned_score, x_min, x_max, y_min, y_max, cmap="viridis"):
    x = jnp.linspace(x_min, x_max, 100)
    y = jnp.linspace(y_min, y_max, 100)
    t = jnp.asarray([0.25, 0.5, 0.75])
    x, y = jnp.meshgrid(x, y)
    x = x[..., None]
    y = y[..., None]

    fig, axs = plt.subplots(nrows=1, ncols=t.size, sharey=True, sharex=True)
    for col, ts in enumerate(t):
        vectorised_learnt_score = jax.vmap(jax.vmap(learned_score, in_axes=(None, 0, 0)), in_axes=(None, 0, 0))
        score_pred = vectorised_learnt_score(ts, x, y)
        vectorised_score = jax.vmap(jax.vmap(true_score, in_axes=(None, 0, None, 0)), in_axes=(None, 0, None, 0))
        score_true = vectorised_score(ts, x.squeeze(), 1.0, y.squeeze())
        # mse = jnp.mean((score_pred.squeeze() - score_true.squeeze()) ** 2)
        log_abs_error = jnp.log(abs(score_pred.squeeze() - score_true.squeeze()) + jnp.finfo(score_pred.dtype).eps)
        abs_error = abs(score_pred.squeeze() - score_true.squeeze())
        # eps = 1e-1
        # rel_error = abs_error / (abs(score_true.squeeze()) + eps)

        pc = axs[col].pcolormesh(x.squeeze(), y.squeeze(), abs_error, cmap=cmap)
        axs[col].set_title(f"Time: {ts:.2f}")
        axs[col].set_xlabel(r"$x$")
    axs[0].set_ylabel(r"$y$")
    fig.colorbar(pc, ax=axs.ravel().tolist(), label="Absolute Error", aspect=5)
    # fig.supxlabel(r"$x$ value")
    # fig.supylabel(r"$y$ value")
    return fig, axs


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
    t=jnp.asarray([0.25, 0.5, 0.75]),
    x=jnp.linspace(-1, 5, 1000)[..., None],
):
    y = y[0]
    fig, axs = plt.subplots(nrows=1, ncols=t.size, sharey=True)
    # fig.suptitle(f"T: {t}")

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


def plot_score_error(
    true_score,
    learned_score,
    T,
    y,
    t=jnp.asarray([0.25, 0.5, 0.75]),
    x=jnp.linspace(-1, 5, 1000)[..., None],
):
    y = y[0]
    fig, axs = plt.subplots(nrows=1, ncols=t.size, sharey=True)

    for col, ts in enumerate(t):
        true_score_fn = jax.vmap(true_score, in_axes=(None, 0, None, None))
        y_true = true_score_fn(ts, x.flatten(), T, y)
        batch_learn_score = jax.vmap(learned_score, in_axes=(None, 0))
        y_pred = batch_learn_score(ts, x)
        abs_error = abs(y_pred.flatten() - y_true.flatten())
        eps = 10e-5
        rel_error = abs_error / abs(y_true.flatten() + eps)
        axs[col].plot(x.flatten(), rel_error)
        axs[col].set_title(f"Time: {ts:.2f}")

    return fig, axs


def plot_forward_score(
    true_score,
    learned_score,
    x0,
    t=jnp.asarray([0, 0.25, 0.5, 0.75, 0.9]),
    x=jnp.linspace(-1, 5, 1000)[..., None],
):
    x0 = x0[0]
    fig, axs = plt.subplots(nrows=1, ncols=t.size, sharey=True)
    fig.suptitle(f"Time: {t}")
    for col, ts in enumerate(t):
        batch_learn_score = jax.vmap(learned_score, in_axes=(None, 0))
        y_pred = batch_learn_score(ts, x)

        true_score_fn = jax.vmap(true_score, in_axes=(None, None, None, 0))
        y_true = true_score_fn(0.0, x0, ts, x.flatten())

        axs[col].plot(x.flatten(), y_pred.flatten())
        axs[col].plot(x.flatten(), y_true.flatten())
        axs[col].set_title(f"Time: {ts:.2f}")

    return fig, axs
