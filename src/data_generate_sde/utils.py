import diffrax
import jax
import jax.numpy as jnp


def conditioned(key, ts, x0, score_fn, drift, diffusion):
    x0 = jnp.asarray(x0)

    def _drift(t, x, *args):
        forward_drift = drift(t, x)
        _score = score_fn(t, x)
        diffusion_tx = diffusion(t, x)
        return forward_drift + diffusion_tx @ diffusion_tx.T @ _score.reshape(-1)

    sol = solution(key, ts, x0, _drift, diffusion)
    return sol


def backward(key, ts, y, score_fn, drift, diffusion):
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

    sol = solution(key, ts, y, _drift, _diffusion)
    return sol


def solution(key, ts, x0, drift, diffusion, bm_shape=None):
    x0 = jnp.asarray(x0)
    assert x0.ndim == 1

    def step_fun(key_and_t_and_x, dt):
        k, t, x = key_and_t_and_x
        k, subkey = jax.random.split(k, num=2)
        eps = jax.random.normal(subkey, shape=x.shape)

        xnew = x + dt * drift(t, x) + jnp.sqrt(dt) * diffusion(t, x) @ eps
        tnew = t + dt

        return (k, tnew, xnew), xnew

    init = (key, ts[0], x0)
    _, x_all = jax.lax.scan(step_fun, xs=jnp.diff(ts), init=init)
    return jnp.concatenate([x0[None], x_all], axis=0)

    # if not bm_shape:
    #     bm_shape = (x0.size,)
    #
    # bm = diffrax.VirtualBrownianTree(
    #     ts[0].astype(float), ts[-1].astype(float), tol=1e-3, shape=bm_shape, key=key
    # )
    # # bm = diffrax.UnsafeBrownianPath(shape=bm_shape, key=key)
    # terms = diffrax.MultiTerm(diffrax.ODETerm(drift), diffrax.ControlTerm(diffusion, bm))
    # solver = diffrax.Euler()
    # saveat = diffrax.SaveAt(ts=ts)
    # sol = diffrax.diffeqsolve(
    #     terms,
    #     solver,
    #     ts[0].astype(float),
    #     ts[-1].astype(float),
    #     dt0=0.01,
    #     y0=x0,
    #     saveat=saveat,
    #     # adjoint=diffrax.DirectAdjoint(),
    # ).ys
    # return sol


def important_reverse_and_correction(
    key, ts, x0, y, reverse_drift, reverse_diffusion, correction_drift
):
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

    sol = solution(
        key, ts, x0=rev_corr, drift=_drift, diffusion=_diffusion, bm_shape=(2 * y.size,)
    )
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
    return sol


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
