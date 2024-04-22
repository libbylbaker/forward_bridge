import functools

import flax.linen as nn
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import orbax
from flax.training import orbax_utils

from src import plotting
from src.data_generate_sde import sde_ornstein_uhlenbeck as ou
from src.data_generate_sde import utils as sde_utils
from src.data_loader import dataloader
from src.models.score_mlp import ScoreMLPDistributedEndpt
from src.training import utils

seed = 1
min_val = -1.0
max_val = 1.0

sde = {"x0": (1.0,), "N": 100, "dim": 1, "T": 1.0, "y": (1.0,)}
dt = 0.01


def y_sampler(key):
    return jax.random.uniform(
        key, shape=(training["load_size"], sde["dim"]), minval=min_val, maxval=max_val
    )


network = {
    "output_dim": sde["dim"],
    "time_embedding_dim": 16,
    "init_embedding_dim": 16,
    "activation": "leaky_relu",
    "encoder_layer_dims": [16],
    "decoder_layer_dims": [128, 128],
}

y = sde["y"]
training = {
    "batch_size": 1000,
    "epochs_per_load": 1,
    "lr": 0.01,
    "num_reloads": 1000,
    "load_size": 1000,
    "checkpoint_path": f"/Users/libbybaker/Documents/Python/doobs-score-project/doobs_score_matching/checkpoints/ou/varied_y_{min_val}_to_{max_val}",
}

drift, diffusion = ou.vector_fields()
data_fn = ou.data_reverse_variable_y(sde["T"], sde["N"])

model = ScoreMLPDistributedEndpt(**network)
optimiser = optax.adam(learning_rate=training["lr"])

gradient_transition = utils.gradient_transition_fn(drift=drift, diffusion=diffusion)

num_samples = training["batch_size"] * sde["N"]
x_shape = jnp.empty(shape=(num_samples, sde["dim"]))
t_shape = jnp.empty(shape=(num_samples, 1))
model_init_sizes = (x_shape, x_shape, t_shape)

(train_step_key, train_loop_key) = jax.random.split(jax.random.PRNGKey(seed), 2)

train_step, params, opt_state = utils.create_train_step_variable_y(
    train_step_key, model, optimiser, *model_init_sizes, dt=dt
)

if __name__ == "__main__":
    state = utils.train_variable_y(
        train_loop_key,
        train_step,
        params,
        opt_state,
        training,
        network,
        sde,
        data_fn,
        gradient_transition,
        utils.data_setup_variable_y,
        y_sampler,
    )
