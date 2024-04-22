import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from src.data_generate_sde import sde_interest_rates as ir
from src.models.score_mlp import ScoreMLP
from src.training import utils

seed = 1

sde = {"x0": (1.0,), "N": 1000, "dim": 1, "T": 1.0, "y": (1.5,)}
dt = 0.01

network = {
    "output_dim": sde["dim"],
    "time_embedding_dim": 16,
    "init_embedding_dim": 16,
    "activation": nn.leaky_relu,
    "encoder_layer_dims": [16],
    "decoder_layer_dims": [128, 128],
}

x0 = sde["x0"]
training = {
    "batch_size": 1000,
    "epochs_per_load": 100,
    "lr": 0.01,
    "num_reloads": 10,
    "load_size": 1000,
    "checkpoint_path": f"/Users/libbybaker/Documents/Python/doobs-score-project/doobs_score_matching/checkpoints/ir/forward_x0_{x0}",
}


drift, diffusion = ir.vector_fields()
reverse_data = ir.data_forward(sde["x0"], sde["T"], sde["N"])

model = ScoreMLP(**network)
optimiser = optax.adam(learning_rate=training["lr"])

gradient_transition = utils.gradient_transition_fn(drift=drift, diffusion=diffusion)

num_samples = training["batch_size"] * sde["N"]
x_shape = jnp.empty(shape=(num_samples, sde["dim"]))
t_shape = jnp.empty(shape=(num_samples, 1))
model_init_sizes = (x_shape, t_shape)

(train_step_key, train_loop_key) = jax.random.split(jax.random.PRNGKey(seed), 2)

train_step, train_state = utils.create_train_step(
    train_step_key, model, optimiser, *model_init_sizes, dt=dt
)


if __name__ == "__main__":
    state = utils.train(
        train_loop_key,
        train_step,
        train_state,
        training,
        sde,
        reverse_data,
        gradient_transition,
        utils.data_setup_forward,
    )
