import src.models
from experiments.plotting import plot_score_error_variable_y, load_checkpoint_w_batch_stats

from src.sdes import sde_ornstein_uhlenbeck
from tueplots import bundles, axes, cycler, figsizes
from tueplots.constants.color import palettes
import matplotlib.pyplot as plt

import orbax.checkpoint
from src.training import train_utils


def format_plt():
    bundle = bundles.aistats2023()
    plt.rcParams.update(bundle)
    plt.rcParams.update(axes.lines())
    plt.rcParams.update(cycler.cycler(color=palettes.paultol_muted))
    plt.rcParams.update(figsizes.aistats2023_full(nrows=1, ncols=3))


if __name__=="__main__":

    formatting = False

    if formatting:
        format_plt()

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

    y_min = -1.0
    max_val = 1.0
    checkpoint_path = f"../../checkpoints/ou/varied_y_{y_min}_to_{max_val}"

    _, restored = load_checkpoint_w_batch_stats(checkpoint_path)
    model = src.models.score_mlp.ScoreMLPDistributedEndpt(**restored["network"])
    trained_score = train_utils.trained_score_variable_y(model, restored["params"], {})

    ou = sde_ornstein_uhlenbeck.ornstein_uhlenbeck(T=1.0, N=100, dim=1)

    true_score = ou.params[0]

    cmap = "PuRd"

    fig, axs = plot_score_error_variable_y(true_score, trained_score, -1, 1, -1, 1, cmap=cmap)
    plt.savefig('figs/ou_score_varied_y_-1.0_to_1.0_error.png')
