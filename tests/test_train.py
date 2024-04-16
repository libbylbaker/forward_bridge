import flax.linen as nn
import jax.numpy as jnp
import jax.random
import pytest
from flax.training.train_state import TrainState

from src.training.utils import create_train_state, trained_score


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


def test_create_train_state(model, x_data, t_data):
    state = create_train_state(
        net=model,
        key=jax.random.PRNGKey(0),
        learning_rate=0.1,
        x_shape=x_data.shape,
        t_shape=t_data.shape,
    )
    assert isinstance(state.params, dict)


def test_trained_score(model, x_data, t_data):
    state = create_train_state(
        net=model,
        key=jax.random.PRNGKey(0),
        learning_rate=0.1,
        x_shape=x_data.shape,
        t_shape=t_data.shape,
    )
    score_fn = trained_score(state)
    score = score_fn(t_data[0], x_data[0])
    assert score.ndim == 1
    assert score.size == 2
