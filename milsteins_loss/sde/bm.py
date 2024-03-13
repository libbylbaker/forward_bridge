import diffrax
import jax
import jax.numpy as jnp

from milsteins_loss.sde.utils import solution


def forward(ts, key, x0):
    x0 = jnp.asarray(x0)
    assert x0.ndim == 1
    dim = x0.size
    bm = diffrax.VirtualBrownianTree(
        float(ts[0]), float(ts[-1]), tol=1e-3, shape=(dim,), key=key
    )
    return jax.vmap(bm.evaluate, in_axes=(0,))(ts)


def reverse(ts, key, y):
    y = jnp.asarray(y)
    bm = forward(ts, key, y)
    return y + bm


def correction(ts):
    return 1.0


def drift(t, val_array, *args):
    assert val_array.ndim == 1
    dim = val_array.size
    return jnp.zeros(shape=(dim,))


def diffusion(t, val_array, *args):
    assert val_array.ndim == 1
    dim = val_array.size
    return jnp.identity(dim)


def conditioned(ts, key, score_state, x0):
    x0 = jnp.asarray(x0)

    def _drift(t, x, *args):
        _score = score_state.apply_fn(
            {"params": score_state.params, "batch_stats": score_state.batch_stats},
            x=x[..., None],
            t=jnp.asarray([t]),
            train=False,
        )
        return _score.reshape(-1)

    sol = solution(ts, key, _drift, diffusion, x0)
    return sol.ys


def score(t, x, T, x_T):
    return (x_T - x) / (T - t)


def forward_score(x0, t, x):
    return -(x-x0)/t


def backward(ts, key, score_fn, y):
    y = jnp.asarray(y)
    T = ts[-1]

    def _drift(t, x, *args):
        _score = score_fn(T-t, x).reshape(-1)
        return _score

    sol = solution(ts, key, _drift, diffusion, y)
    return sol.ys
