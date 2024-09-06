import jax.random as jr
import orbax
from flax.training import orbax_utils
import time

from src.training.data_loader import dataloader

seed = 1


def save(path, params, opt_state, batch_stats, sde, network, training, epoch_time):
    ckpt = {
        "params": params,
        "batch_stats": batch_stats,
        "opt_state": opt_state,
        "sde": sde,
        "network": network,
        "training": training,
        "train_time": epoch_time
    }
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(ckpt)
    orbax_checkpointer.save(path, ckpt, save_args=save_args, force=True)


def train(key, training, data_fn, train_step, params, batch_stats, opt_state, sde, network, checkpoint_path):
    (data_key, dataloader_key, train_key) = jr.split(key, 3)
    data_key = jr.split(data_key, 1)

    batches_per_epoch = max(training["load_size"] // training["batch_size"], 1)

    print("Training")

    for load in range(training["num_reloads"]):
        # load data
        data_key = jr.split(data_key[0], training["load_size"])
        data = data_fn(data_key)
        infinite_dataloader = dataloader(data, training["batch_size"], loop=True, key=jr.split(dataloader_key, 1)[0])

        for epoch in range(training["epochs_per_load"]):
            total_loss = 0
            total_time = 0
            start = time.process_time()
            for batch, (ts, reverse, correction) in zip(range(batches_per_epoch), infinite_dataloader):
                params, batch_stats, opt_state, _loss = train_step(
                    params, batch_stats, opt_state, ts, reverse, correction
                )
                total_loss = total_loss + _loss
            epoch_loss = total_loss / batches_per_epoch
            end = time.process_time()
            epoch_time = end - start
            total_time += epoch_time
            actual_epoch = load * training["epochs_per_load"] + epoch
            print(f"Epoch: {actual_epoch}, Loss: {epoch_loss}")

            last_epoch = load == training["num_reloads"] - 1 and epoch == training["epochs_per_load"] - 1
            if actual_epoch % 100 == 0 or last_epoch:
                average_time = total_time / (actual_epoch + 1)
                save(checkpoint_path, params, opt_state, batch_stats, sde, network, training, average_time)


def train_variable_y(
    key, training, data_fn, train_step, params, batch_stats, opt_state, sde, network, checkpoint_path, sample_y_fn
):
    (data_key, dataloader_key, train_key) = jr.split(key, 3)
    data_key = jr.split(data_key, 1)

    batches_per_epoch = max(training["load_size"] // training["batch_size"], 1)

    print("Training")

    for load in range(training["num_reloads"]):
        # load data
        data_key = jr.split(data_key[0], training["load_size"])
        y_key = jr.split(data_key[0], 1)[0]
        y = sample_y_fn(y_key, shape=(training["load_size"], sde["dim"]))
        data = data_fn(data_key, y)

        infinite_dataloader = dataloader(data, training["batch_size"], loop=True, key=jr.split(dataloader_key, 1)[0])
        total_time = 0
        for epoch in range(training["epochs_per_load"]):
            total_loss = 0
            start = time.process_time()
            for batch, (ts, reverse, correction, y) in zip(range(batches_per_epoch), infinite_dataloader):
                params, batch_stats, opt_state, _loss = train_step(
                    params, batch_stats, opt_state, ts, reverse, correction, y
                )
                total_loss = total_loss + _loss
            end = time.process_time()
            epoch_time = end - start
            total_time += epoch_time
            epoch_loss = total_loss / batches_per_epoch

            actual_epoch = load * training["epochs_per_load"] + epoch
            print(f"Epoch: {actual_epoch}, Loss: {epoch_loss}")

            last_epoch = load == training["num_reloads"] - 1 and epoch == training["epochs_per_load"] - 1
            if actual_epoch % 100 == 0 or last_epoch:
                average_time = total_time/(actual_epoch+1)
                save(checkpoint_path, params, opt_state, batch_stats, sde, network, training, average_time)
