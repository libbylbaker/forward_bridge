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
