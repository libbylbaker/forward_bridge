import jax
import jax.numpy as jnp
import flax.linen as nn
import pytest

from src.models import ScoreMLP, ScoreUNet


@pytest.fixture
def x_data():
    return jnp.ones((8, 2), dtype=jnp.float32)


@pytest.fixture
def t_data():
    return jnp.ones((8, 1), dtype=jnp.float32)


@pytest.fixture
def model_setup(output_dim=2):
    setup = {"output_dim": output_dim,
             "time_embedding_dim": 32,
             "init_embedding_dim": 32,
             "activation": nn.elu,
             "encoder_layer_dims": [16, 8, 4],
             "decoder_layer_dims": [4, 8, 16],}
    return setup


def test_ScoreMLP(x_data, t_data, model_setup):
    net = ScoreMLP(**model_setup)
    key = jax.random.PRNGKey(0)
    variables = net.init(key, x=x_data, t=t_data, train=False)
    params = variables["params"]
    score = net.apply(
        {"params": params},
        x=x_data,
        t=t_data,
        train=True,
    )
    assert score.shape[0] == x_data.shape[0]
    assert score.shape[1] == model_setup["output_dim"]
    assert score.ndim == 2


def test_ScoreUNet(x_data, t_data, model_setup):
    net = ScoreUNet(**model_setup)
    key = jax.random.PRNGKey(0)
    variables = net.init(key, x=x_data, t=t_data, train=False)
    params, batch_stats = variables["params"], variables["batch_stats"]
    score, updates = net.apply(
        {"params": params, "batch_stats": batch_stats},
        x=x_data,
        t=t_data,
        train=True,
        mutable=["batch_stats"]
    )
    assert score.shape[0] == x_data.shape[0]
    assert score.shape[1] == model_setup["output_dim"]
    assert score.ndim == 2
