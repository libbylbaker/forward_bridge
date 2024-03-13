import jax.numpy as jnp
import diffrax


def solution(ts, key, drift, diffusion, x0):
    assert x0.ndim == 1
    dim = x0.size
    bm = diffrax.VirtualBrownianTree(
        ts[0].astype(float), ts[-1].astype(float), tol=1e-3, shape=(dim,), key=key
    )
    terms = diffrax.MultiTerm(
        diffrax.ODETerm(drift), diffrax.ControlTerm(diffusion, bm)
    )
    solver = diffrax.Euler()
    saveat = diffrax.SaveAt(ts=ts)
    sol = diffrax.diffeqsolve(
        terms, solver, ts[0].astype(float), ts[-1].astype(float), dt0=0.05, y0=x0, saveat=saveat
    )
    return sol
