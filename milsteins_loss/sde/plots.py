import jax.numpy as jnp
import jax.random
import matplotlib.pyplot as plt
from functools import partial

from milsteins_loss import plotting
import milsteins_loss.sde.interest_rates as ir
import milsteins_loss.sde.time as time

key = jax.random.PRNGKey(0)
num_trajectories = 20

x0 = (1.,)
y = 2.

traj_keys = jax.random.split(key, num_trajectories)
time_grid = time.grid(0, 1, 100)
trajectories = jax.vmap(ir.forward, (None, 0, None))(time_grid, traj_keys, x0)

for traj in trajectories:
    plt.plot(time_grid, traj)
plt.title("forward")
plt.show()


rev_keys = jax.random.split(key, num_trajectories)
rev_corr = jax.vmap(ir.reverse_and_correction, (None, 0, None))(time_grid, traj_keys, (y,))
rev = rev_corr[:, :, :-1]
corr = rev_corr[:, :, -1]

for traj in rev:
    plt.plot(time_grid, traj)
plt.title("reverse")
plt.show()

for traj in corr:
    plt.plot(time_grid, traj)
plt.title("correction")
plt.show()


cond_keys = jax.random.split(key, num_trajectories)
score = partial(ir.score, T=1, y=2.)

conditioned_fn = jax.vmap(ir.conditioned, (None, 0, None, None))
conditioned_trajs = conditioned_fn(time_grid, cond_keys, score, x0)

for traj in conditioned_trajs:
    plt.plot(time_grid, traj)
plt.title("conditioned")
plt.scatter(1.0, y)
plt.scatter(1., -y)
plt.show()
