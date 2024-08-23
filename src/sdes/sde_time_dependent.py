import jax.numpy
import jax.numpy as jnp

from src.sdes import sde_utils, time

C = 1


def data_forward(x0, T, N):
    ts = time.grid(t_start=0, T=T, N=N)
    drift, diffusion = vector_fields()

    @jax.jit
    @jax.vmap
    def data(key):
        correction_ = 1.0
        forward_ = sde_utils.solution(key, ts, x0=x0, drift=drift, diffusion=diffusion)
        return ts[..., None], forward_, jnp.asarray(correction_)

    return data


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
        reverse_and_correction_ = sde_utils.solution(key, ts_reverse, x0=start_val, drift=drift, diffusion=diffusion)
        reverse_ = reverse_and_correction_[:, :-1]
        correction_ = reverse_and_correction_[-1, -1]
        return ts[..., None], reverse_, jnp.asarray(correction_)

    return data


def vector_fields():
    def drift(t, x, *args):
        """dX_t = (- X_t) dt + t/C dW_t"""
        return -x

    def diffusion(t, x, *args):
        """dX_t = (a/X_t - X_t) dt + dW_t"""
        assert x.ndim == 1
        dim = x.size
        return (t + 1) / C * jnp.identity(dim)

    return drift, diffusion


def vector_fields_reverse(T=1.0):
    def reverse_drift(t, rev, *args):
        return rev

    def reverse_diffusion(t, rev, *args):
        return (T - t) / C * jnp.identity(rev.size)

    return reverse_drift, reverse_diffusion


def vector_fields_reverse_and_correction():
    def drift(t, x, *args):
        reverse_drift, _ = vector_fields_reverse()
        assert x.ndim == 1
        reverse = reverse_drift(t, x[:-1])
        correct = correction_drift(t, x[:-1], x[-1, None])
        return jnp.concatenate([reverse, correct])

    def diffusion(t, x, *args):
        assert x.ndim == 1
        _, rev_diff = vector_fields_reverse()
        rev_diffusion = rev_diff(t, x[:-1])
        rev_corr_diff = jnp.pad(rev_diffusion, ((0, 1), (0, 1)), mode="constant", constant_values=0.0)
        return rev_corr_diff

    return drift, diffusion


def correction_drift(t, rev, corr, *args):
    return 1.0 * corr
