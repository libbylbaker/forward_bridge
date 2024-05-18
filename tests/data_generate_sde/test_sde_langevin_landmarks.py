import jax
import jax.numpy as jnp
import pytest

from doobs_score_matching.src.data_generate_sde.sde_langevin_landmarks import dH_p, dH_q, kernel


@pytest.fixture
def p_3_landmarks_2_dim():
    p = jnp.array([[1.0, 2.0], [3.0, 4.0], [5, 6]])
    return p


@pytest.fixture
def q_3_landmarks_2_dim():
    q = jnp.array([[1.0, 2.0], [3.0, 4.0], [5, 6]])
    return q


def test_kernel(q_3_landmarks_2_dim):
    q = q_3_landmarks_2_dim
    sigma = 1.0
    kernel_row_fn = jax.vmap(kernel, in_axes=(0, None, None), out_axes=-1)
    kernel_col_fn = jax.vmap(kernel_row_fn, in_axes=(None, 0, None), out_axes=0)
    mat = kernel_col_fn(q, q, sigma)
    assert mat.shape == (3, 3)
    assert mat[0, 0] == 1 / (2 * jnp.pi)


def test_dhamiltonian_p(p_3_landmarks_2_dim, q_3_landmarks_2_dim):
    p = p_3_landmarks_2_dim
    q = q_3_landmarks_2_dim
    sigma = 1.0

    kernel_row_fn = jax.vmap(kernel, in_axes=(0, None, None), out_axes=-1)
    kernel_col_fn = jax.vmap(kernel_row_fn, in_axes=(None, 0, None), out_axes=0)
    mat = kernel_col_fn(q, q, sigma)

    sigma = 1.0
    dhp = dH_p(p, q, sigma)
    assert dhp.shape == p.shape
    assert dhp[0, 0] == p[0, 0] * mat[0, 0] + p[1, 0] * mat[0, 1] + p[2, 0] * mat[0, 2]


def test_dhamiltonian_q(p_3_landmarks_2_dim, q_3_landmarks_2_dim):
    p = p_3_landmarks_2_dim
    q = q_3_landmarks_2_dim
    sigma = 1.0

    result = dH_q(p, q, sigma)
    assert result.shape == p.shape


def test_data_forward(p_3_landmarks_2_dim, q_3_landmarks_2_dim):
    p0 = p_3_landmarks_2_dim
    q0 = q_3_landmarks_2_dim
    T = 1.0
    N = 10
