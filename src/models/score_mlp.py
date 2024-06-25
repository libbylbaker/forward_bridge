from functools import partial

import jax
import jax.numpy as jnp
from flax import linen as nn

from src.models.time_embedding import get_time_embedding


class ScoreMLP(nn.Module):
    output_dim: int
    time_embedding_dim: int
    init_embedding_dim: int
    activation: str
    encoder_layer_dims: list
    decoder_layer_dims: list
    batch_norm: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray, train: bool) -> jnp.ndarray:
        # assert len(x.shape) == len(t.shape) == 2
        time_embedding = get_time_embedding(self.time_embedding_dim)
        t = jax.vmap(time_embedding, in_axes=0)(t)
        t = MLP(
            output_dim=self.init_embedding_dim,
            activation=self.activation,
            layer_dims=self.encoder_layer_dims,
            batch_norm=self.batch_norm,
        )(t, train)
        x = MLP(
            output_dim=self.init_embedding_dim,
            activation=self.activation,
            layer_dims=self.encoder_layer_dims,
            batch_norm=self.batch_norm,
        )(x, train)
        xt = jnp.concatenate([x, t], axis=-1)
        score = MLP(
            output_dim=self.output_dim,
            activation=self.activation,
            layer_dims=self.decoder_layer_dims,
            batch_norm=self.batch_norm,
        )(xt, train)
        return score


class ScoreMLPDistributedEndpt(nn.Module):
    output_dim: int
    time_embedding_dim: int
    init_embedding_dim: int
    activation: str
    encoder_layer_dims: list
    decoder_layer_dims: list
    batch_norm: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, y: jnp.ndarray, t: jnp.ndarray, train: bool) -> jnp.ndarray:
        time_embedding = get_time_embedding(self.time_embedding_dim)
        t = jax.vmap(time_embedding, in_axes=0)(t)
        t = MLP(
            output_dim=self.init_embedding_dim,
            activation=self.activation,
            layer_dims=self.encoder_layer_dims,
            batch_norm=self.batch_norm,
        )(t, train)
        xy = jnp.concatenate([x, y], axis=-1)
        xy = MLP(
            output_dim=self.init_embedding_dim,
            activation=self.activation,
            layer_dims=self.encoder_layer_dims,
            batch_norm=self.batch_norm,
        )(xy, train)
        xyt = jnp.concatenate([xy, t], axis=-1)
        score = MLP(
            output_dim=self.output_dim,
            activation=self.activation,
            layer_dims=self.decoder_layer_dims,
            batch_norm=self.batch_norm,
        )(xyt, train)
        return score


class MLP(nn.Module):
    output_dim: int
    activation: str
    layer_dims: list
    batch_norm: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool) -> jnp.ndarray:
        for dim in self.layer_dims:
            x = nn.Dense(dim)(x)
            x = get_activation(self.activation)(x)
            if self.batch_norm:
                x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.Dense(self.output_dim)(x)
        return x


def get_activation(activation: str):
    if activation == "relu":
        return nn.relu
    elif activation == "leaky_relu":
        return nn.leaky_relu
    elif activation == "tanh":
        return nn.tanh
    elif activation == "sigmoid":
        return nn.sigmoid
    elif activation == "gelu":
        return nn.gelu
    else:
        raise ValueError(f"Activation {activation} not supported")
