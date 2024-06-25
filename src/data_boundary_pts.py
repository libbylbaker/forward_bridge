import jax
import jax.numpy as jnp


def sample_circle(num_landmarks: int, radius=1.0, centre=jnp.asarray([0, 0])) -> jnp.ndarray:
    theta = jnp.linspace(0, 2 * jnp.pi, num_landmarks, endpoint=False)
    x = jnp.cos(theta)
    y = jnp.sin(theta)
    return (radius * jnp.stack([x, y], axis=1) + centre).flatten()


def y_uniform_circle(key, num_samples, r=1.0):
    theta = jax.random.uniform(key, (num_samples, 1), minval=0, maxval=2 * jax.numpy.pi)
    return jnp.concatenate([r * jax.numpy.cos(theta), r * jax.numpy.sin(theta)], axis=-1)
