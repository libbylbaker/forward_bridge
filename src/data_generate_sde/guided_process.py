import diffrax
import jax.numpy as jnp
import jax

from typing import Callable
from src.data_generate_sde import utils


def vector_fields_guided(drift, diffusion, guide_fn):

    def guided_drift(t, x, *args):
        guiding_term = (diffusion(t, x) @ (diffusion(t, x).T @ guide_fn(t, x)))
        return drift(t, x) + guiding_term

    return guided_drift, diffusion


def get_guide_fn(t0, T, y, sigma_auxiliary: Callable, B_auxiliary: Callable, beta_auxiliary: Callable):
    ode_sol = _backward_ode(t0, T, sigma_auxiliary, B_auxiliary, beta_auxiliary)
    r = _define_r(ode_sol, T, y)
    return r


def _define_r(sol_L_M_mu: Callable, T, y: jax.Array):

    def r_end_pt(t, x):
        return jnp.zeros_like(x)

    def r_standard(t, x):
        L_t, M_t, mu_t = sol_L_M_mu(t)
        L_t_transpose = L_t.T

        Minv_L_x = jnp.linalg.solve(M_t, L_t @ x)
        Minv_y_minus_mu = jnp.linalg.solve(M_t, y - mu_t)
        H = L_t_transpose @ Minv_L_x
        F = L_t_transpose @ Minv_y_minus_mu
        return F - H

    def r(t, x):
        return jax.lax.cond(t == T, lambda f: r_end_pt(t, x), lambda f: r_standard(t, x), (t, x))

    return r


def _backward_ode(t0, T, sigma_auxiliary: Callable, B_auxiliary: Callable, beta_auxiliary: Callable):
    dim = beta_auxiliary(0).shape[0]

    mu_T = jnp.zeros(shape=(dim,))
    sigma_T = 1e-10 * jnp.eye(dim)
    L_T = 1. * jnp.eye(dim)
    x0 = L_T, sigma_T, mu_T

    def drift_backward_ode_system(t, x, *args):
        L, M, mu = x
        dL = -L @ B_auxiliary(T - t)
        L_sigma = L @ sigma_auxiliary(T - t)
        dM = -L_sigma @ L_sigma.T
        d_mu = -L @ beta_auxiliary(T - t)
        d_x = -dL, -dM, -d_mu
        return d_x

    sol_rev = utils.solution_ode_dense(t0, T, x0, drift_backward_ode_system)
    sol_L_M_mu = lambda t: sol_rev.evaluate(T - t)
    return sol_L_M_mu
