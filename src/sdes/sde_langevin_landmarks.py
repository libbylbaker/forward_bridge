import jax
import jax.numpy as jnp

from src.sdes import sde_utils, time

kernel_sigma, diffusion_constant, dissipation_constant = 0.5, jnp.sqrt(0.1), 0.5


def data_forward(x0, T, N):
    return sde_utils.data_forward(x0, T, N, vector_fields(2))


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
        reverse_ = sde_utils.solution(key, ts_reverse, y, reverse_drift, reverse_diffusion)
        return ts[..., None], reverse_, 1

    return data


def vector_fields(dims):
    def drift(t, x, *args):
        x.reshape(2, -1, dims)
        p = x[0]  # x is of shape (2, #lms, dim)
        q = x[1]
        dhq = dH_q(p, q, kernel_sigma)
        dhp = dH_p(p, q, kernel_sigma)
        dq = dhp
        dp = -dissipation_constant * dhp - dhq
        x_full = jnp.stack([dp, dq], axis=0).flatten()
        return x_full

    def diffusion(t, x, *args):
        x.reshape(2, -1, dims)
        p = x[0]
        q = x[1]
        diffusion_q = 0 * jnp.identity(q.shape[0])
        diffusion_p = diffusion_constant * jnp.identity(p.shape[0])
        return jnp.stack([diffusion_p, diffusion_q], axis=0).flatten()

    return drift, diffusion


def vector_fields_reverse(dims):
    forward_drift, forward_diffusion = vector_fields(dims)

    def drift(t, x, *args):
        return -forward_drift(t, x, *args)

    def diffusion(t, x, *args):
        return forward_diffusion(t, x, *args)

    return drift, diffusion


def vector_fields_reverse_and_correction(dim=2):
    reverse_drift, reverse_diffusion = vector_fields_reverse()

    def drift(t, x, *args):
        reverse = x[:-1].reshape(2, -1, dim)
        correction = x[-1, None]
        d_reverse = reverse_drift(t, reverse)
        d_correction = drift_correction(t, reverse, correction)
        return jnp.concatenate([d_reverse.flatten(), d_correction])

    def diffusion(t, x, *args):
        return reverse_diffusion(t, x, *args)

    return drift, diffusion


def drift_correction(t, rev, corr, *args):
    drift, _ = vector_fields()
    drift_dr = jax.grad(drift, argnums=1)(t, rev)
    sum = jnp.sum(drift_dr)
    return -sum * corr


def dH_p(p, q, sigma):
    kernel_row_fn = jax.vmap(kernel, in_axes=(0, None, None), out_axes=-1)
    kernel_col_fn = jax.vmap(kernel_row_fn, in_axes=(None, 0, None), out_axes=0)
    mat = kernel_col_fn(q, q, sigma)
    result = jnp.dot(mat, p)
    return result


def dH_q(p, q, sigma):
    def summand(pi, pj, qi, qj):
        p_ip_j = jnp.dot(pi, pj)
        diff_qij = (qi - qj) / sigma**2
        k_ij = kernel(qi, qj, sigma)
        return p_ip_j * k_ij * diff_qij

    summand_j = jax.vmap(summand, in_axes=(None, 0, None, 0))
    sum_over_j = lambda pi, qi: jnp.sum(summand_j(pi, p, qi, q), axis=0)
    result = jax.vmap(sum_over_j, in_axes=(0, 0))(p, q)
    return result


def kernel(x, y, sigma):
    assert x.ndim == y.ndim == 1
    dist = jnp.linalg.norm(x - y)
    cov = sigma**2
    tmp = 1 / (2 * cov) ** (x.size / 2)
    # return tmp * jnp.exp(-dist ** 2 / (2 * cov))
    return jnp.exp(-(dist**2) / cov)
