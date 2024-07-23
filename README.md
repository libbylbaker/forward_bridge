# Score matching for diffusion bridges

We learn the score function for stochastic differential equations (SDEs) which via Doob's h-transform, we can use to sample from bridged SDEs, that is SDEs conditioned on their endpoints.

## Installation

To install the package, run the following command in the root directory of the repository:

```bash
pip install .
```
Alternatively, to run examples and tests too, install everything with the following command:

```bash
pip install 'forwardbridge[full]'
```

## Usage

The package provides some predefined SDEs.
For these the drift and diffusion of the SDEs are already defined, for both the original SDE and the adjoint SDE.
The adjoint SDE can be used to with the train loop in `src.training` to learn the score function for the SDE.
For examples see the `experiments` directory, particularly `train_scripts`.

[//]: # (## Citation)

[//]: # ()
[//]: # (Please cite our article as follows)

[//]: # ()
[//]: # (```bibtex)

[//]: # (@article{baker2024scorematching,)

[//]: # (      title={Score matching for bridges without time-reversals},)

[//]: # (      author={Elizabeth Louise Baker and Moritz Schauer and Stefan Sommer},)

[//]: # (      year={2024},)

[//]: # (      eprint={},)

[//]: # (})

[//]: # (```)
