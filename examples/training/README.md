# Training examples

Run examples from the repository root with `PYTHONPATH=src`, or use an editable
GradSCF installation. Each example states its dependencies and numerical scope.

| Example | Purpose |
| --- | --- |
| [Neural XC implicit GPU memory](neural_xc/README.md) | H2 dissociation training with the built-in neural-XC module; measured forward, VJP and Adam memory; includes a Slurm launcher and validated results. |
| [H2 FCI training](h2_fci_training.py) | FCI total-energy supervision of a density-matrix network in fixed-density, explicit and implicit modes. |
| [External functional](external_functional.py) | An external Flax XC network connected through `dft.custom_xc`, with synthetic teacher targets. |
| [DFPT response supervision](dfpt_response.py) | External functional parameters trained against fixed-geometry AO density response; explicit/implicit include the relaxed reference. |
| [Periodic external XC](periodic_xc.py) | A four-layer external JAX network using the shared Sample/Trainer in all three modes; internal-energy labels per cell and a separate static periodic DFPT calculation. |
| [Density-matrix MLP](density_matrix_mlp.py) | A density-matrix energy functional used by the FCI training example. |
| [Neural SCF forces](train_neural_scf_forces.py) | Force-supervision gradients through native integrals and a small neural XC model. |
| [Plot H2 FCI training](plot_h2_fci_training.py) | Plot the recorded fixed-density, explicit and implicit loss curves. |

`dft.custom_xc(input_fn=..., energy_fn=...)` returns the existing
`dft.Functional`; training modes and model-owned trainers are unchanged.
For a handwritten formula using the same port, see
[custom_energy.py](../xc/custom_energy.py). Traditional expressions use
`dft.custom_xc(spec)` and retain their complete exact-exchange metadata.

[A07 showcase evidence](../../reproducibility/ad_showcase/A07_external_functional/README.md)
records current `jy_gpu_1` three-mode and response-target checks, the eight
focused derivative regressions, and the separate provenance/accuracy limits
of the historical FCI-supervised curves. Synthetic teacher targets demonstrate
the derivative path; they do not establish chemical accuracy.
