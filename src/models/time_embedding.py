import math

import flax.linen as nn
import jax
import jax.numpy as jnp


def get_time_embedding(
    embedding_dim: int, max_period: float = 128.0, scaling: float = 100.0
):
    div_term = jnp.exp(
        jnp.arange(0, embedding_dim, 2, dtype=jnp.float32)
        * (-math.log(max_period) / embedding_dim)
    )

    def time_embedding(t: float) -> jnp.ndarray:
        """Embed scalar time steps into a vector of size `embedding_dim`"""
        emb = jnp.empty((embedding_dim,), dtype=jnp.float32)
        emb = emb.at[0::2].set(jnp.sin(scaling * t * div_term))
        emb = emb.at[1::2].set(jnp.cos(scaling * t * div_term))
        return emb

    return time_embedding


class TimeEmbeddingMLP(nn.Module):
    output_dim: int
    activation: nn.activation

    @nn.compact
    def __call__(self, t_emb: jax.Array) -> (jax.Array, jax.Array):
        kernel_init = nn.initializers.xavier_normal()
        scale_shift = nn.Dense(2 * self.output_dim, kernel_init=kernel_init)(t_emb)
        scale, shift = jnp.array_split(scale_shift, 2, axis=-1)
        return scale, shift
