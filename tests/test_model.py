import flax.linen as nn
import jax
import jax.numpy as jnp
import pytest

from src.models import ScoreMLP, ScoreMLPDistributedEndpt, ScoreUNet


@pytest.fixture
def x_data():
    return jnp.ones((8, 2), dtype=jnp.float32)


@pytest.fixture
def t_data():
    return jnp.ones((8, 1), dtype=jnp.float32)


@pytest.fixture
def model_setup(output_dim=2):
    setup = {
        "output_dim": output_dim,
        "time_embedding_dim": 16,
        "init_embedding_dim": 32,
        "activation": "leaky_relu",
        "encoder_layer_dims": [16],
        "decoder_layer_dims": [128, 128],
    }
    return setup


@pytest.mark.parametrize("net_type", [ScoreMLP])
def test_ScoreMLP(x_data, t_data, model_setup, net_type):
    net = net_type(**model_setup)
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


def testScoreMLPDistributedEndpt(x_data, t_data, model_setup):
    net = ScoreMLPDistributedEndpt(**model_setup)
    key = jax.random.PRNGKey(0)
    variables = net.init(key, x=x_data, y=x_data, t=t_data, train=False)
    params = variables["params"]
    score = net.apply(
        {"params": params},
        x=x_data,
        y=x_data,
        t=t_data,
        train=True,
    )
    assert score.shape[0] == x_data.shape[0]
    assert score.shape[1] == model_setup["output_dim"]
    assert score.ndim == 2
