import flax.linen as nn
import jax.numpy as jnp
import jax.random
import pytest
from optax import adam

from src.training.utils import create_train_step, trained_score


class Model(nn.Module):
    output_dim: int

    @nn.compact
    def __call__(self, x, t, train):
        x = nn.Dense(self.output_dim)(x)
        return x


@pytest.fixture
def model():
    return Model(output_dim=2)


@pytest.fixture
def x_data(dim=2):
    return jnp.ones((8, dim), dtype=jnp.float32)


@pytest.fixture
def t_data():
    return jnp.linspace(0, 1, 8, dtype=jnp.float32)


@pytest.fixture
def train_step(x_data, t_data, model):
    key = jax.random.PRNGKey(0)
    model_init_sizes = (x_data.shape, t_data.shape)
    dt = 0.01
    optimiser = adam(0.1)
    return create_train_step(key, model, optimiser, *model_init_sizes, dt=dt)


def test_create_train_state(train_step):
    step, state = train_step
    assert isinstance(state.params, dict)


def test_trained_score(train_step, x_data, t_data):
    step, state = train_step
    score_fn = trained_score(state)
    score = score_fn(t_data[0], x_data[0])
    assert score.ndim == 1
    assert score.size == 2
