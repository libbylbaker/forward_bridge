import jax
import jax.numpy as jnp

from src.sdes import sde_utils, time

_SIGMA = 1.0
_KAPPA = 0.1
GRID_SIZE = 5


def data_forward(x0, T, N):
    return sde_utils.data_forward(x0, T, N, vector_fields(), bm_shape=(2 * GRID_SIZE**2,))


def data_reverse(y, T, N):
    ts = time.grid(t_start=0, T=T, N=N)
    ts_reverse = time.reverse(T=T, times=ts)
    y = jnp.asarray(y)
    assert y.ndim == 1
    drift, diffusion = vector_fields_reverse_and_correction()

    @jax.jit
    @jax.vmap
    def data(key):
        """
        :return: ts,
        reverse process: (t, dim), where t is the number of time steps and dim the dimension of the SDE
        correction process: float, correction process at time T
        """
        start_val = jnp.append(y, 1.0)
        reverse_and_correction_ = sde_utils.solution(
            key, ts_reverse, x0=start_val, drift=drift, diffusion=diffusion, bm_shape=(2 * GRID_SIZE**2 + 1,)
        )
        reverse_ = reverse_and_correction_[:, :-1]
        correction_ = reverse_and_correction_[-1, -1]
        return ts[..., None], reverse_, jnp.asarray(correction_)

    return data


def vector_fields(grid_range=(-2, 2), dim=2, eps=1e-10):
    grid_ = jnp.linspace(*grid_range, GRID_SIZE)
    grid_ = jnp.stack(jnp.meshgrid(grid_, grid_, indexing="xy"), axis=-1)
    grid_ = grid_.reshape(-1, dim)

    def drift(t, x, *args):
        return jnp.zeros_like(x)

    def diffusion(t: float, x: jnp.ndarray) -> jnp.ndarray:
        x = x.reshape(-1, dim)

        gauss_kernel = gaussian_kernel_2d(_KAPPA, _SIGMA)
        # matern_kernel = matern_kernel_5_2(_KAPPA, _SIGMA)
        batch_over_grid = jax.vmap(gauss_kernel, in_axes=(None, 0))
        batch_over_vals = jax.vmap(batch_over_grid, in_axes=(0, None))
        Q_half = batch_over_vals(x, grid_)

        Q_half = Q_half.reshape(-1, GRID_SIZE**2)
        Q_half = jnp.kron(Q_half, jnp.eye(2))

        return Q_half

    return drift, diffusion


def partial_derivatives_by_row(t, x):
    """Computes matrix A_ij = d/dx_i k(x_i, y_j)"""
    forward_drift, forward_diffusion = vector_fields()
    identity = jnp.identity(x.size)

    def diffusion_row(y, row_i: int):
        """Computes sigm, i.e. the ith row of the diffusion matrix"""
        return forward_diffusion(t, y)[row_i, :]

    rows = jnp.arange(0, x.size)

    def partial_derivative_per_row(i):
        dxi = identity[i, :]
        diffusion_i = lambda y: diffusion_row(y, i)
        _diff, d_diff = jax.jvp(diffusion_i, (x,), (dxi,))
        return d_diff

    matrix = jax.vmap(partial_derivative_per_row)(rows)
    return matrix


def vector_fields_reverse():
    forward_drift, forward_diffusion = vector_fields()

    def drift(t, x, partials=None):
        if partials is None:
            partials = partial_derivatives_by_row(t, x)
        kernel_matrix = forward_diffusion(t, x)
        sum_of_partials = partials.sum(axis=0)
        return 1/x.size*kernel_matrix @ sum_of_partials

    def diffusion(t, x, *args):
        return forward_diffusion(t, x)

    return drift, diffusion


def vector_fields_reverse_and_correction():
    reverse_drift, reverse_diffusion = vector_fields_reverse()

    def drift(t, x, *args):
        assert x.ndim == 1
        partials = partial_derivatives_by_row(t, x[:-1])
        reverse = reverse_drift(t, x[:-1], partials=partials)
        correct = drift_correction(t, x[:-1], x[-1, None], partials=partials)
        return jnp.concatenate([reverse, correct])

    def diffusion(t, x, *args):
        assert x.ndim == 1
        rev_diffusion = reverse_diffusion(t, x[:-1])
        rev_corr_diff = jnp.pad(rev_diffusion, ((0, 1), (0, 1)), mode="constant", constant_values=0.0)
        return rev_corr_diff

    return drift, diffusion


def drift_correction(t, rev, corr, partials=None):
    if partials is None:
        partials = partial_derivatives_by_row(t, rev)
    sum_rows = jnp.sum(partials, axis=0)
    total_sum = sum_rows @ sum_rows.T
    c = 0.5 * total_sum
    return c * corr


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
