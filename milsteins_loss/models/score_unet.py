from functools import partial
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn

from milsteins_loss.models.time_embedding import TimeEmbeddingMLP, get_time_embedding


class ScoreUNet(nn.Module):
    output_dim: int
    time_embedding_dim: int
    init_embedding_dim: int
    activation: nn.activation
    encoder_layer_dims: Sequence[int]
    decoder_layer_dims: Sequence[int]
    batch_norm: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray, train: bool) -> jnp.ndarray:
        assert (
            self.encoder_layer_dims[-1] == self.decoder_layer_dims[0]
        ), "Bottleneck dim does not match"
        time_embedding = get_time_embedding(embedding_dim=self.time_embedding_dim)
        t_emb = jax.vmap(time_embedding)(t)
        x = InputDense(self.init_embedding_dim, self.activation)(x)

        # downsample
        down = []
        for dim in self.encoder_layer_dims:
            x = Downsample(
                output_dim=dim,
                activation=self.activation,
                batch_norm=self.batch_norm,
            )(x, t_emb, train)
            down.append(x)

        # bottleneck
        bottleneck_dim = self.encoder_layer_dims[-1]
        x_out = Dense(bottleneck_dim)(x)
        x = self.activation(x_out) + x

        # upsample
        for dim, x_skip in zip(self.decoder_layer_dims, down[::-1]):
            x = Upsample(
                output_dim=dim,
                activation=self.activation,
                batch_norm=self.batch_norm,
            )(x, x_skip, t_emb, train)

        score = Dense(self.output_dim)(x)
        return score


class InputDense(nn.Module):
    output_dims: int
    activation: nn.activation
    kernel_init: Callable = nn.initializers.xavier_normal()

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.output_dims, kernel_init=self.kernel_init)(x)
        x = self.activation(x)
        return x


class Dense(nn.Module):
    output_dims: int
    kernel_init: Callable = nn.initializers.xavier_normal()

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.output_dims, kernel_init=self.kernel_init)(x)
        return x


class Downsample(nn.Module):
    output_dim: int
    activation: nn.activation
    batch_norm: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, t_emb: jnp.ndarray, train: bool) -> jnp.ndarray:
        input_dim = x.shape[-1]
        x_in = x
        x = Dense(input_dim)(x)
        scale, shift = TimeEmbeddingMLP(input_dim, self.activation)(t_emb)
        x = x * (1.0 + scale) + shift
        x = self.activation(x) + x_in
        x = Dense(self.output_dim)(x)
        x = self.activation(x)
        if self.batch_norm:
            x = nn.BatchNorm(use_running_average=not train)(x)
        else:
            x = nn.LayerNorm()(x)
        return x


class Upsample(nn.Module):
    output_dim: int
    activation: nn.activation
    batch_norm: bool = True

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, x_skip: jnp.ndarray, t_emb: jnp.ndarray, train: bool
    ) -> jnp.ndarray:
        x_in = jnp.concatenate([x, x_skip], axis=-1)
        input_dim = x_in.shape[-1]
        x = Dense(input_dim)(x_in)
        scale, shift = TimeEmbeddingMLP(input_dim, self.activation)(t_emb)
        x = x * (1.0 + scale) + shift
        x = self.activation(x) + x_in
        x = Dense(self.output_dim)(x)
        x = self.activation(x)
        if self.batch_norm:
            x = nn.BatchNorm(use_running_average=not train)(x)
        else:
            x = nn.LayerNorm()(x)
        return x
