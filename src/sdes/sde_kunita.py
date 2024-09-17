import jax
import jax.numpy as jnp

from src.sdes import sde_utils


def kunita(T, N, num_landmarks, dim=2, sigma=1.0, kappa=0.1, grid_size=5, grid_range=(-2, 2)):
    grid_ = jnp.linspace(*grid_range, grid_size)
    grid_ = jnp.stack(jnp.meshgrid(grid_, grid_, indexing="xy"), axis=-1)
    grid_ = grid_.reshape(-1, dim)

    gauss_kernel = gaussian_kernel_2d(kappa, sigma)
    # matern_kernel = matern_kernel_5_2(kappa, sigma)

    def drift(t, x, *args):
        return jnp.zeros_like(x)

    def diffusion(t: float, x: jnp.ndarray) -> jnp.ndarray:
        x = x.reshape(-1, dim)
        batch_over_grid = jax.vmap(gauss_kernel, in_axes=(None, 0))
        batch_over_vals = jax.vmap(batch_over_grid, in_axes=(0, None))
        Q_half = batch_over_vals(x, grid_)
        Q_half = Q_half.reshape(-1, grid_size**2)
        Q_half = jnp.kron(Q_half, jnp.eye(2))

        scaling = (grid_range[1] - grid_range[0]) / grid_size

        return scaling**2 * Q_half

    def adj_drift(t, x, partials=None):
        if partials is None:
            partials = _partial_derivatives_by_row(t, x)
        kernel_matrix = diffusion(t, x)
        sum_of_partials = partials.sum(axis=0)
        return 1 / x.size * kernel_matrix @ sum_of_partials

    def adj_diffusion(t, x, *args):
        return diffusion(t, x)

    def correction(t, y, corr, partials=None):
        if partials is None:
            partials = _partial_derivatives_by_row(t, y)
        sum_rows = jnp.sum(partials, axis=0)
        total_sum = sum_rows @ sum_rows.T
        c = 0.5 * total_sum
        return c * corr

    def _partial_derivatives_by_row(t, x):
        """Computes matrix A_ij = d/dx_i k(x_i, y_j)"""
        identity = jnp.identity(x.size)

        def diffusion_row(y, row_i: int):
            """Computes sigm, i.e. the ith row of the diffusion matrix"""
            return diffusion(t, y)[row_i, :]

        rows = jnp.arange(0, x.size)

        def partial_derivative_per_row(i):
            dxi = identity[i, :]
            diffusion_i = lambda y: diffusion_row(y, i)
            _diff, d_diff = jax.jvp(diffusion_i, (x,), (dxi,))
            return d_diff

        matrix = jax.vmap(partial_derivative_per_row)(rows)
        return matrix

    return sde_utils.SDE(
        T,
        N,
        num_landmarks * dim,
        drift,
        diffusion,
        adj_drift,
        adj_diffusion,
        correction,
        None,
        (2 * grid_size**2,),
        None,
    )


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
