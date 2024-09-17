import jax
import jax.numpy as jnp
import pytest

from src.sdes import sde_data, sde_kunita, sde_utils


@pytest.fixture
def kunita_sde_1_landmarks():
    grid_size = 5
    kunita = sde_kunita.kunita(T=1.0, N=10, num_landmarks=1, grid_size=grid_size)
    return kunita


@pytest.fixture
def kunita_sde_3_landmarks():
    grid_size = 5
    kunita = sde_kunita.kunita(T=1.0, N=10, num_landmarks=3, grid_size=grid_size)
    return kunita


def test_kunita_drift(kunita_sde_3_landmarks):
    t = 0.0
    x = jnp.array([[1.0, 2.0], [3, 4], [5, 6]])
    x = x.flatten()
    drift_ = kunita_sde_3_landmarks.drift(t, x)
    assert drift_.shape == (3 * 2,)


def test_kunita_diffusion(kunita_sde_3_landmarks):
    t = 0.0
    x = jnp.array([[1.0, 2.0], [3, 4], [5, 6]])
    x = x.flatten()
    diffusion_ = kunita_sde_3_landmarks.diffusion(t, x)
    assert diffusion_.shape == (2 * 3, 2 * 5**2)


# def test_drift_correction():
#     t = 0.0
#     corr = 1.0
#     rev = jnp.array([[1.0, 2.0], [3, 4], [5, 6]])
#     kunita = sde_kunita.kunita(T=1., N=10, num_landmarks=2)
#     drift_correction = kunita.correction(t, rev, corr)
#     assert drift_correction.shape == ()


def test_data_forward_1_landmark(kunita_sde_1_landmarks):
    x0 = jnp.array([[1.0, 2.0]])
    x0 = x0.flatten()
    data_gen = sde_data.data_forward(x0, kunita_sde_1_landmarks)
    keys = jax.random.split(jax.random.PRNGKey(1), 5)
    ts, forward, correction = data_gen(keys)
    assert ts.shape == (5, 10, 1)
    assert forward.shape == (5, 10, 2)
    assert correction.shape == (5,)
    assert jnp.all(forward[:, 0] == jnp.asarray(x0))


def test_data_adjoint_1_landmark(kunita_sde_1_landmarks):
    y = jnp.array([[1.0, 2.0]])
    y = y.flatten()
    data_gen = sde_data.data_adjoint(y, kunita_sde_1_landmarks)
    keys = jax.random.split(jax.random.PRNGKey(1), 5)
    ts, reverse, correction = data_gen(keys)
    assert ts.shape == (5, 10, 1)
    assert reverse.shape == (5, 10, 2)
    assert correction.shape == (5,)
    assert jnp.all(reverse[:, 0] == jnp.asarray(y))
