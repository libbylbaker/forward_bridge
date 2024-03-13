import dataclasses
from typing import Callable, Any, NamedTuple
import jax.numpy as jnp
import jax


@dataclasses.dataclass
class SDEDef:
    drift: Callable[[float, jax.Array, Any], jax.Array]   # (scalar, array with ndim 1) -> array with ndim 1
    diffusion: Callable


class SDESol(NamedTuple):
    forward: jax.Array
    conditioned: jax.Array
    backward: jax.Array

def ornstein_uhlenbeck() -> SDEDef:
    def drift(t, x, p):
        return -x * p

    def diffusion(t, x):
        return jnp.ones_like(x)

    return SDE(drift, diffusion)


def sde_solve(sde) -> Callable:

    def solve(key, u0, *params) -> SDESol:
        return diffrax.diffeqsolve(...)

    return solve
