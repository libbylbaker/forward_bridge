import diffrax
import jax
import jax.numpy as jnp

from src.data_generate_sde import guided_process, time


def data_forward(x0, T, N, vector_fields, bm_shape=None):
    ts = time.grid(t_start=0, T=T, N=N)
    drift, diffusion = vector_fields

    @jax.jit
    @jax.vmap
    def data(key):
        correction_ = 1.0
        forward_ = solution(key, ts, x0, drift, diffusion, bm_shape=bm_shape)
        return ts[..., None], forward_, jnp.asarray(correction_)

    return data


def data_reverse(y, T, N, vector_fields_reverse_and_correction, weight_fn=None):
    ts = time.grid(t_start=0, T=T, N=N)
    ts_reverse = time.reverse(T=T, times=ts)
    y = jnp.asarray(y)
    assert y.ndim == 1
    drift, diffusion = vector_fields_reverse_and_correction

    @jax.jit
    @jax.vmap
    def data(key):
        """
        :return: ts,
        reverse process: (t, dim), where t is the number of time steps and dim the dimension of the SDE
        correction process: float, correction process at time T
        """
        start_val = jnp.append(y, 1.0)
        reverse_and_correction_ = solution(key, ts_reverse, x0=start_val, drift=drift, diffusion=diffusion)
        reverse_ = reverse_and_correction_[:, :-1]
        correction_ = reverse_and_correction_[-1, -1]
        return ts[..., None], reverse_, correction_

    return data


def data_reverse_weighted(y, T, N, vector_fields_reverse_and_correction, weight_fn):
    unweighted_data = data_reverse(y, T, N, vector_fields_reverse_and_correction)

    @jax.jit
    @jax.vmap
    def data(key):
        ts, reverse_, unweighted_correction = unweighted_data(key)
        weight = weight_fn(reverse_[-1])
        return ts, reverse_, unweighted_correction * weight

    return data


def data_reverse_guided(x0, y, T, N, vector_fields_reverse_and_correction_guided):
    x0 = jnp.asarray(x0)
    y = jnp.asarray(y)
    ts = time.grid(t_start=0, T=T, N=N)
    ts_reverse = time.reverse(T, ts)
    start_val = jnp.append(y, 1.0)

    guided_drift, guided_diffusion = vector_fields_reverse_and_correction_guided

    @jax.jit
    @jax.vmap
    def data(key):
        reverse_and_correction = solution(key, ts_reverse, start_val, guided_drift, guided_diffusion, y.shape)
        reverse = reverse_and_correction[:, :-1]
        correction = reverse_and_correction[:, -1]
        return ts[..., None], reverse, correction

    return data


def weight_function_gaussian(x0, inverse_covariance):
    x0 = jnp.asarray(x0)

    def gaussian_distance(YT):
        YT = jnp.asarray(YT)
        return jnp.exp(-0.5 * (YT - x0).T @ jnp.linalg.solve(inverse_covariance, (YT - x0)))

    return gaussian_distance


def vector_fields_reverse_and_correction_guided(
    x0, T, vector_fields_reverse, drift_correction, reverse_guided_auxiliary
):
    reverse_drift, reverse_diffusion = vector_fields_reverse
    B_auxiliary, beta_auxiliary, sigma_auxiliary = reverse_guided_auxiliary
    guide_fn = guided_process.get_guide_fn(0.0, T, x0, sigma_auxiliary, B_auxiliary, beta_auxiliary)
    guided_drift, _ = guided_process.vector_fields_guided(reverse_drift, reverse_diffusion, guide_fn)

    def drift(t, x):
        reverse = x[:-1]
        correction = x[-1, None]
        d_reverse = guided_drift(t, reverse)
        d_correction = drift_correction(t, reverse, correction)
        return jnp.concatenate([d_reverse, d_correction])

    def diffusion(t, x):
        reverse = x[:-1]
        correction = x[-1]
        d_reverse = reverse_diffusion(t, reverse)
        d_correction = -guide_fn(t, reverse).T @ reverse_diffusion(t, reverse) * correction
        diff = jnp.vstack((d_reverse, d_correction))
        return diff

    return drift, diffusion


def conditioned(key, ts, x0, score_fn, drift, diffusion):
    x0 = jnp.asarray(x0)

    def _drift(t, x, *args):
        forward_drift = drift(t, x)
        _score = score_fn(t, x)
        diffusion_tx = diffusion(t, x)
        return forward_drift + diffusion_tx @ diffusion_tx.T @ _score.reshape(-1)

    sol = solution(key, ts, x0, _drift, diffusion)
    return sol


def backward(key, ts, y, score_fn, drift, diffusion, bm_shape=None):
    y = jnp.asarray(y)
    T = ts[-1]

    def _diffusion(t, x, *args):
        return diffusion(T - t, x)

    def covariance(t, x):
        return jnp.dot(_diffusion(t, x), _diffusion(t, x).T)

    def _drift(t, x, *args):
        forward_drift = -drift(T - t, x)

        _score = score_fn(T - t, x).reshape(-1)
        cov = covariance(T - t, x)

        divergence_cov_fn = jax.jacfwd(covariance)
        divergence_cov = jnp.trace(divergence_cov_fn(T - t, x))

        return forward_drift + cov @ _score + divergence_cov

    sol = solution(key, ts, y, _drift, _diffusion, bm_shape=bm_shape)
    return sol


def solution(key, ts, x0, drift, diffusion, bm_shape=None):
    x0 = jnp.asarray(x0)
    if bm_shape is None:
        bm_shape = x0.shape

    def step_fun(key_and_t_and_x, dt):
        k, t, x = key_and_t_and_x
        k, subkey = jax.random.split(k, num=2)
        eps = jax.random.normal(subkey, shape=bm_shape)
        diffusion_ = diffusion(t, x)
        xnew = x + dt * drift(t, x) + jnp.sqrt(dt) * diffusion_ @ eps
        tnew = t + dt

        return (k, tnew, xnew), xnew

    init = (key, ts[0], x0)
    _, x_all = jax.lax.scan(step_fun, xs=jnp.diff(ts), init=init)
    return jnp.concatenate([x0[None], x_all], axis=0)


def important_reverse_and_correction(key, ts, x0, y, reverse_drift, reverse_diffusion, correction_drift):
    y = jnp.asarray(y)
    x0 = jnp.asarray(x0)
    assert y.ndim == 1
    assert x0.ndim == 1
    rev_corr = jnp.append(y, 1.0)

    def h(t, x, *args):
        T = ts[-1]
        assert x.ndim == 1
        zero_array = jnp.zeros(shape=(x.size,))
        return jax.lax.cond(t == T, lambda f: zero_array, lambda f: -(x0 - f) / (T - t), x)

    def drift_reverse(t, rev, corr, *args):
        return reverse_drift(t, rev) - reverse_diffusion(t, rev) @ h(t, rev)

    def drift_correction(t, rev, corr, *args):
        return correction_drift(t, rev, corr)

    def diffusion_correction(t, rev, corr, *args):
        h_transpose = h(t, rev)[..., None].T
        return h_transpose * corr

    def _drift(t, x, *args):
        rev = x[:-1]
        corr = x[-1][..., None]
        reverse_next = drift_reverse(t, rev, corr)
        corr_next = drift_correction(t, rev, corr)
        return jnp.concatenate([reverse_next, corr_next])

    def _diffusion(t, x, *args):
        rev = x[:-1]
        corr = x[-1][..., None]
        reverse_next = reverse_diffusion(t, rev)
        corr_next = diffusion_correction(t, rev, corr)
        top_block_zeros = jnp.zeros(shape=(reverse_next.shape[0], corr_next.shape[1]))
        bottom_block_zeros = jnp.zeros(shape=(corr_next.shape[0], reverse_next.shape[1]))
        return jnp.block([[reverse_next, top_block_zeros], [corr_next, bottom_block_zeros]])

    sol = solution(key, ts, x0=rev_corr, drift=_drift, diffusion=_diffusion, bm_shape=(2 * y.size,))
    return sol


def solution_ode(ts, x0, drift):
    terms = diffrax.ODETerm(drift)
    solver = diffrax.Dopri5()
    saveat = diffrax.SaveAt(ts=ts)
    sol = diffrax.diffeqsolve(
        terms,
        solver,
        t0=ts[0].astype(float),
        t1=ts[-1].astype(float),
        dt0=0.05,
        y0=x0,
        saveat=saveat,
    )
    return sol.ys


def solution_ode_dense(t0, t1, x0, drift):
    terms = diffrax.ODETerm(drift)
    solver = diffrax.Dopri5()
    saveat = diffrax.SaveAt(dense=True)
    sol = diffrax.diffeqsolve(
        terms,
        solver,
        t0=t0,
        t1=t1,
        dt0=0.05,
        y0=x0,
        saveat=saveat,
    )
    return sol
