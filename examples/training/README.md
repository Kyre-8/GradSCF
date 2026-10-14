# Training examples

Run examples from the repository root with `PYTHONPATH=src`, or use an editable
GradSCF installation. Each example states its dependencies and numerical scope.

| Example | Purpose |
| --- | --- |
| [Neural XC implicit GPU memory](neural_xc/README.md) | H2 dissociation training with the built-in neural-XC module; measured forward, VJP and Adam memory; includes a Slurm launcher and validated results. |
| [H2 FCI training](h2_fci_training.py) | FCI total-energy supervision of a density-matrix network in fixed-density, explicit and implicit modes. |
| [External functional](external_functional.py) | An external Flax XC network connected through `dft.Functional`, with synthetic teacher targets. |
| [Periodic external XC](periodic_xc.py) | A four-layer external JAX network using the shared Sample/Trainer in all three modes; internal-energy labels per cell and a separate static periodic DFPT calculation. |
| [Density-matrix MLP](density_matrix_mlp.py) | A density-matrix energy functional used by the FCI training example. |
| [Plot H2 FCI training](plot_h2_fci_training.py) | Plot the recorded fixed-density, explicit and implicit loss curves. |
