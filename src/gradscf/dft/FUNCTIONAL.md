# Generic XC functional interface

`dft.Functional` connects an external feature builder and external network
energy to the existing GradSCF SCF, AD, response and training code. It does not
choose the network architecture, prescribe a feature vector, or add an XC
baseline automatically. It introduces no separate SCF solver or trainer.

The built-in `model.neural_xc.Functional` is a separate concrete model. It
implements the same consumer-facing energy/SCF/response contracts and can be
passed directly to the shared trainer. This module does not import that model.

## Module ownership

```text
gradscf/
  dft/
    functional.py      # generic energy-callback adapter and current-density inputs
    derivatives.py     # shared XC potential and Hessian actions
    hfx.py, pt2.py      # shared physical ingredients and bounded-memory caches
  training/            # Sample / Trainer, losses and optimization
  model/neural_xc/     # one concrete built-in model, its input transforms and presets
```

The generic DFT, SCF, response, data and training paths do not import the
concrete model. The built-in model imports shared physical implementations
from DFT. A subprocess test blocks every `gradscf.model` import while computing
an external functional's potential/kernel and taking an implicit SCF training
step. The old locations have been removed, with no forwarding files.

```python
from gradscf import dft

functional = dft.Functional(
    input_fn=make_inputs,         # DensityInputs -> array PyTree
    energy_fn=energy,             # (params, inputs) -> scalar XC energy, Ha
    init_fn=initialize,           # optional (key, inputs) -> parameter mapping
)
```

`make_inputs` and `energy` must be JAX differentiable to the order required by
the requested loss. Flax, Haiku or pure-JAX networks can be called inside
`energy`; no inheritance from a GradSCF neural network is required. Trainable
weights are explicit parameters, not mutable object state or captured traced
closure variables. The current Flax TrainState-based trainer expects a mapping
at the parameter root; nested parameter values can be PyTrees. Supply existing
weights using `training.Trainer(functional, params=...)`, or provide `init_fn`.

## SCF-current physical inputs

For every trial density D, GradSCF constructs a `DensityInputs` view. The
external input function is evaluated **inside** the XC energy derivative:

```
trial D -> DensityInputs -> make_inputs -> network energy -> dE_xc/dD
```

Thus derivatives include the feature construction and features are updated
on each SCF evaluation. There is no cross-iteration density-feature cache.
The view reuses existing contractions in `gradscf.tools.features`; quantities
are computed on access, so density-only networks do not require AO gradients.
Missing AO derivatives/Laplacians raise an error rather than supplying zeros.

| Property | Meaning and layout |
|---|---|
| `density_matrix` | Current symmetric alpha/beta matrices, `(2, nao, nao)` |
| `total_density_matrix` | Spin sum, `(nao, nao)` |
| `rho`, `rho_spin` | Total `(ngrid,)` / spin `(2, ngrid)` density |
| `grad_rho`, `grad_rho_spin` | Total `(ngrid, 3)` / spin `(2, ngrid, 3)` gradient |
| `tau`, `tau_spin` | Total / spin kinetic-energy density; tau = 1/2 sum occupied orbital-gradient squares |
| `laplacian_rho`, `laplacian_rho_spin` | Total / spin density Laplacian |
| `weights`, `coordinates` | Grid weights and coordinates |
| `ao`, `ao_deriv1` | AO values; AO values plus three first derivatives |
| `atom_coords`, `atom_charges`, `overlap_matrix` | Geometry, charges and AO metric |

All quantities use atomic units. Density-independent arrays are reused at
fixed geometry; if geometry/grid/AO data are differentiated, their input
arrays must remain on the JAX graph. The feature builder may produce nested
dictionaries/tuples and may couple grid points. Use differentiable smooth
operations where potentials or higher responses are required.

Canonical orbital coefficients/energies, frozen reference HFX and PT2 caches
are not exposed as density ingredients. Treating them as independent constants
while varying D would omit their response. Arbitrary architecture does not
mean an arbitrary black-box runtime or arbitrary orbital-dependent physics.
Complex/current-dependent functionals and a separate exact-exchange fraction
require an explicit adapter with the corresponding physics; they are not
inferred by this real-density interface.

## One energy, potential and response

`energy_fn` returns the **total scalar XC energy**, not per-electron energy,
energy per volume, a potential, or network coefficients. For per-electron
network output epsilon, explicitly return `sum(weights*rho*epsilon)`. For a
learned correction, explicitly add the baseline in that same energy function.

- `energy_for_density(params, molecule, density)` evaluates trial D.
- `energy_from_molecule(params, molecule)` uses the state's current D.
- `potential(params, molecule, density)` calls the shared
  `dft.xc_energy_and_potential_from_density`.
- `kernel_action(params, molecule, density, tangent)` calls the shared
  `dft.xc_kernel_action`, a JVP of the potential. The tangent has D's shape.

The kernel is an AO density-matrix Hessian action, including external feature
and network derivatives. It is **not** a scalar pointwise f_xc approximation;
nonlocal grid coupling is retained. No complete four-index Hessian is built.
Both spin-stacked and spin-summed densities are supported. TDA/TDDFT use the
same adiabatic real-density Hessian through their existing AO transition and
projection routines. Parameter derivatives remain available through the kernel.

## Training API

```python
from gradscf import training

data = [training.Sample(reference, energy=e_fci)]
trainer = training.Trainer(functional, params=params)
trainer.mode = 'implicit'
trainer.loss = {'energy': {'mse': 1., 'mae': 1.}}
trainer.learning_rate = .002
trainer.run(data, steps=100)
print(trainer.history['loss'])
print(trainer.params)
```

`Sample` constructs the existing validated PyTree record; `Trainer` uses the
existing loss and optimizer-step functions. There is only one high-level
trainer, with no external-network-specific training implementation.

| `trainer.mode` | Behavior |
|---|---|
| `fixed_density` | Evaluate at the supplied fixed density |
| `explicit` | Differentiate the existing JAX SCF computation, implemented using `lax.scan` |
| `implicit` | Differentiate the SCF fixed point using the shared adjoint solver |

The name `explicit` replaces the earlier public label. This does not change
the SCF algorithm or gradient: it does not manually expand each iteration
into a separate computation graph, and it does not freeze the final density.
Historical `unrolled`/`expl` inputs remain aliases only at the low-level shared
SCF policy boundary. New examples and Trainer use `explicit`.

`trainer.scf` contains numerical SCF settings such as `max_cycle`,
`conv_tol_energy`, `conv_tol_density` and `require_converged`.
`trainer.adjoint` accepts `tolerance`, `max_iter` and `regularization`.
The default external-functional policy requires SCF convergence. Failed or
nonfinite updates leave parameters and optimizer state unchanged.

`trainer.evaluate(data)` returns the metrics in its current mode, while
`trainer.predict(molecule)` returns energy and electronic state. Either
supports `mode='implicit'` for a separate self-consistent diagnostic;
`predict(..., params=teacher_params)` can evaluate a different parameter set
without replacing the trainable weights.

History includes step 0 and metrics after each attempted update. Repeated
`run()` calls continue the optimizer/history. Replacing `params` or changing
`learning_rate` restarts them. `history['update_accepted'][0]` is None; remaining
entries correspond to the preceding update. `optimizer_step` counts accepted
updates and can differ from the attempted `step` after a rejection.

The new facade rejects density/orbital-energy supervision in fixed-density
mode instead of silently running SCF. Low-level loss helpers retain their
existing behavior for advanced callers. See
[the training API guide](../training/README.md) for migration and metric names.

## Executed demonstration and validation

For a direct matrix-to-energy example, see
[density_matrix_mlp.py](../../../examples/training/density_matrix_mlp.py):
`D.flatten() -> 16 -> 16 -> 16 -> 1`, three tanh hidden layers plus a scalar
output layer. The external MLP defines the entire XC energy, without an
additional baseline. It demonstrates the shared potential/kernel derivatives
and implicit SCF training with synthetic energy and density targets. Its input
is tied to a fixed AO basis and ordering.

[external_functional.py](../../../examples/training/external_functional.py)
uses external Flax features/architecture and GradSCF H2/6-31G* native integrals.
It adds a neural correction to Dirac exchange, with correlation omitted.
Targets come from a second parameter set and are a training demonstration,
not a chemical-accuracy benchmark. No CLI or PySCF calculation is involved.

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  python examples/training/external_functional.py
PYTHONPATH=src JAX_PLATFORMS=cpu \
  python -m pytest -q tests/test_external_neural_xc.py
```

Measured on 2026-09-28, Apple M4 Pro CPU, float64, JAX 0.8.1, seed 0.
Three Adam steps, learning rate 0.002, SCF maximum 80 cycles, density tolerance
1e-9, energy tolerance 1e-11 Ha, implicit tolerance 1e-9.

| Mode | Initial energy MSE / Ha² | Final energy MSE / Ha² | Seconds |
|---|---:|---:|---:|
| Fixed density | 4.1417802781e-6 | 9.2387785597e-7 | 1.69 |
| Explicit SCF | 5.6129071029e-6 | 1.6771517640e-6 | 4.80 |
| Implicit SCF | 5.6129071029e-6 | 1.6771517640e-6 | 8.23 |

Timings include different compilation/cache states and are not a performance
comparison. CPU tests check current-density features, nonlocal feature
coupling, potential/HVP finite differences, mixed parameter/kernel derivatives,
R/U SCF density responses, three-mode training, TDA/TDDFT A/B actions,
initialization, and failure rejection. GPU and complex-orbital execution have
not been validated.

Validation scope: the initial expanded run recorded 102 passed, 2 skipped,
12 failures caused by missing jax-xc, and one deliberately deselected legacy
orbital-energy fixture failure. All 12 dependency failures and the legacy
fixture failure were independently reproduced on unchanged base fced3c4.
The legacy fixture lacks bind_to_molecule_for_scf. They were not repaired or
reclassified as passing by this change. The final targeted results are listed
below; no full-repository or GPU validation is claimed.

Before the short-API refactor, the targeted run after review corrections recorded: **54 passed, 1 deselected**,
58.59 seconds, CPU float64. The deselected case is the independently confirmed
legacy fixture failure above. The external-functional suite alone has 15 tests.

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu python -m pytest -q \
  tests/test_external_neural_xc.py tests/test_scf_implicit_and_xc_energy.py \
  tests/test_molecular_training_api.py tests/test_scf_autodiff.py \
  tests/test_training.py \
  --deselect=tests/test_training.py::test_orbital_energy_loss_uses_explicit_target_and_weights
```


## FCI-supervised 100-step comparison

[h2_fci_training.py](../../../examples/training/h2_fci_training.py) reuses
that same density-matrix network and trains all three modes for 100 steps
against the GradSCF FCI total energy. Loss is energy MSE + MAE, with unit
weights and energies numerically expressed in Hartree. Results, CSVs, plots,
FCI cross-checks and limitations are in
[h2_fci_results/README.md](../../../examples/training/h2_fci_results/README.md).

Earlier short-API/explicit-name validation: 106 passed, 3 skipped, 1 known
baseline failure deselected; see [training/README.md](../training/README.md#validation).

## Module-layout validation

After relocation: **123 regression tests passed** (43.55 s), plus **9 targeted
HFX/PT2 tests passed** (4.97 s; 79 unrelated runtime tests deselected).
CPU float64, JAX 0.8.1. Existing deprecation/version warnings were retained;
no full-repository or GPU run is claimed. Package discovery includes
`gradscf.training` and excludes the removed `gradscf.model.training` package.

The 9 HFX and 3 PT2 function/class definitions have identical ASTs to the
previous implementation. Seven training modules are unchanged except imports,
and the shared derivative module is byte-identical to its previous location.
All three external-network examples ran from `examples/training`. The three
100-step FCI histories match the `b5af8ed` baseline exactly for total loss,
MSE, MAE and total energy. See `layout_equivalence.json` in the example results.

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu python -m pytest -q \
  tests/test_functional_module_boundaries.py tests/test_external_neural_xc.py \
  tests/test_training_facade.py tests/test_neural_xc_public_api.py \
  tests/test_molecular_training_api.py tests/test_scf_implicit_and_xc_energy.py \
  tests/test_pyscf_style_namespace.py tests/test_reference_boundaries.py \
  tests/test_data_reference.py tests/test_workflows_config.py \
  tests/test_workflow_core.py tests/test_neural_d.py tests/test_excited_state_trainer.py
PYTHONPATH=src JAX_PLATFORMS=cpu python -m pytest -q \
  tests/test_neural_xc_runtime.py -k 'chunked or hdf5 or local_pt2'
```
