import jax
import jax.numpy as jnp


def grid(t_start, T, N):
    return jnp.linspace(t_start, T, N)


def reverse(T: float, times: jax.Array):
    T_array = T * jnp.ones_like(times)
    return T_array - times[::-1]
