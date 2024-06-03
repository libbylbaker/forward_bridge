import jax
import jax.numpy as jnp

from src.data_generate_sde import guided_process, sde_utils, time

_SIGMA = 1.0
_KAPPA = 0.1
GRID_SIZE = 5


def data_forward(x0, T, N):
    return sde_utils.data_forward(x0, T, N, vector_fields(), bm_shape=(2 * GRID_SIZE**2,))


def data_reverse(y, T, N):
    """
    :return: ts,
    reverse process: (t, dim), where t is the number of time steps and dim the dimension of the SDE
    correction process: float, correction process at time T
    """
    ts = time.grid(t_start=0, T=T, N=N)
    ts_reverse = time.reverse(T, ts)
    reverse_drift, reverse_diffusion = vector_fields_reverse()

    @jax.jit
    @jax.vmap
    def data(key):
        reverse_ = sde_utils.solution(
            key, ts_reverse, y, reverse_drift, reverse_diffusion, bm_shape=(2 * GRID_SIZE**2,)
        )
        return ts[..., None], reverse_, 1

    return data


def vector_fields(grid_range=(-1, 2), dim=2, eps=1e-10):
    grid_ = jnp.linspace(*grid_range, GRID_SIZE)
    grid_ = jnp.stack(jnp.meshgrid(grid_, grid_, indexing="xy"), axis=-1)
    grid_ = grid_.reshape(-1, dim)

    def drift(t, x, *args):
        return jnp.zeros_like(x)

    def diffusion(t: float, x: jnp.ndarray) -> jnp.ndarray:
        x = x.reshape(-1, dim)

        gauss_kernel = gaussian_kernel_2d(_KAPPA, _SIGMA)
        matern_kernel = matern_kernel_5_2(_KAPPA, _SIGMA)
        batch_over_grid = jax.vmap(gauss_kernel, in_axes=(None, 0))
        batch_over_vals = jax.vmap(batch_over_grid, in_axes=(0, None))
        Q_half = batch_over_vals(x, grid_)

        Q_half = Q_half.reshape(-1, GRID_SIZE**2)
        Q_half = jnp.kron(Q_half, jnp.eye(2))

        return Q_half

    return drift, diffusion


def vector_fields_reverse():
    forward_drift, forward_diffusion = vector_fields()

    def drift(t, x, *args):
        assert x.ndim == 1
        covariance = lambda y: forward_diffusion(t, y) @ forward_diffusion(t, y).T
        columns = jnp.arange(0, x.size)
        identity = jnp.identity(x.size)

        def jac_column(col_i):
            return jax.jvp(covariance, (x,), (identity[:, col_i],))[1][:, col_i]

        matrix = jax.vmap(jac_column)(columns)
        return matrix @ jnp.ones_like(x)

    def diffusion(t, x, *args):
        return forward_diffusion(t, x)

    return drift, diffusion


def drift_correction(t, rev, corr, *args):
    _, diffusion = vector_fields()
    covariance = lambda x: diffusion(t, x) @ diffusion(t, x).T
    c = jax.hvp(covariance, (rev,), (jnp.ones(rev.shape),))
    return c


def matern_kernel_5_2(alpha, sigma):
    def k(x, y):
        dist = jnp.sum(jnp.abs(x - y), axis=-1)
        return (
            alpha
            * (1 + jnp.sqrt(5) * dist / sigma + 5 * dist**2 / (3 * sigma**2))
            * jnp.exp(-jnp.sqrt(5) * dist / sigma)
        )

    return k


def gaussian_kernel_2d(alpha: float, sigma: float) -> callable:
    def k(x, y):
        return alpha * jnp.exp(-0.5 * jnp.sum(jnp.square(x - y), axis=-1) / (sigma**2))

    return k
