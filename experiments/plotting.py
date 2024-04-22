import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


def plot_score_variable_y(true_score, learned_score, x_min, x_max, y_min, y_max):
    x = jnp.linspace(x_min, x_max, 100)
    y = jnp.linspace(y_min, y_max, 100)
    t = jnp.asarray([0.25, 0.5, 0.75])
    x, y = jnp.meshgrid(x, y)
    x = x[..., None]
    y = y[..., None]

    fig, axs = plt.subplots(nrows=2, ncols=t.size, sharey=True, sharex=True)
    for col, ts in enumerate(t):
        vectorised_learnt_score = jax.vmap(
            jax.vmap(learned_score, in_axes=(None, 0, 0)), in_axes=(None, 0, 0)
        )
        score_pred = vectorised_learnt_score(ts, x, y)
        pc = axs[0, col].pcolormesh(x.squeeze(), y.squeeze(), score_pred.squeeze())
        axs[0, col].set_title(f"Time: {ts:.2f}")

    for col, ts in enumerate(t):
        vectorised_score = jax.vmap(
            jax.vmap(true_score, in_axes=(None, 0, None, 0)), in_axes=(None, 0, None, 0)
        )
        score_true = vectorised_score(ts, x.squeeze(), 1.0, y.squeeze())
        pc = axs[1, col].pcolormesh(x.squeeze(), y.squeeze(), score_true.squeeze())
    fig.colorbar(pc, ax=axs.ravel().tolist())
    plt.show()

    return fig, axs


def plot_score_error_variable_y(true_score, learned_score, x_min, x_max, y_min, y_max):
    x = jnp.linspace(x_min, x_max, 100)
    y = jnp.linspace(y_min, y_max, 100)
    t = jnp.asarray([0.25, 0.5, 0.75])
    x, y = jnp.meshgrid(x, y)
    x = x[..., None]
    y = y[..., None]

    fig, axs = plt.subplots(nrows=1, ncols=t.size, sharey=True, sharex=True)
    for col, ts in enumerate(t):
        vectorised_learnt_score = jax.vmap(
            jax.vmap(learned_score, in_axes=(None, 0, 0)), in_axes=(None, 0, 0)
        )
        score_pred = vectorised_learnt_score(ts, x, y)
        vectorised_score = jax.vmap(
            jax.vmap(true_score, in_axes=(None, 0, None, 0)), in_axes=(None, 0, None, 0)
        )
        score_true = vectorised_score(ts, x.squeeze(), 1.0, y.squeeze())
        abs_error = abs(score_pred.squeeze() - score_true.squeeze())
        rel_error = abs_error / abs(score_true.squeeze())

        pc = axs[col].pcolormesh(x.squeeze(), y.squeeze(), abs_error)
        axs[col].set_title(f"Time: {ts:.2f}")
    fig.colorbar(pc, ax=axs.ravel().tolist())
    plt.show()
