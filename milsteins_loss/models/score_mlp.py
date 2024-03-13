from functools import partial

import jax
import jax.numpy as jnp
from flax import linen as nn

from milsteins_loss.models.time_embedding import get_time_embedding


class ScoreMLP(nn.Module):
    output_dim: int
    time_embedding_dim: int
    init_embedding_dim: int
    activation: nn.activation
    encoder_layer_dims: list
    decoder_layer_dims: list

    @nn.compact
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray, train: bool) -> jnp.ndarray:
        # assert len(x.shape) == len(t.shape) == 2
        time_embedding = get_time_embedding(self.time_embedding_dim)
        t = jax.vmap(time_embedding, in_axes=0)(t)
        t = MLP(
            output_dim=self.init_embedding_dim,
            activation=self.activation,
            layer_dims=self.encoder_layer_dims,
        )(t, train)
        x = MLP(
            output_dim=self.init_embedding_dim,
            activation=self.activation,
            layer_dims=self.encoder_layer_dims,
        )(x, train)
        xt = jnp.concatenate([x, t], axis=-1)
        score = MLP(
            output_dim=self.output_dim,
            activation=self.activation,
            layer_dims=self.decoder_layer_dims,
        )(xt, train)
        return score


class MLP(nn.Module):
    output_dim: int
    activation: nn.activation
    layer_dims: list

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool) -> jnp.ndarray:
        for dim in self.layer_dims:
            x = self.activation(nn.Dense(dim)(x))
        x = nn.Dense(self.output_dim)(x)
        return x
