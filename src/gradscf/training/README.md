# Training API

```python
from gradscf import training

data = [training.Sample(mf, energy=e_fci)]
trainer = training.Trainer(functional, params=params)
trainer.mode = 'implicit'
trainer.loss = {'energy': {'mse': 1.0, 'mae': 1.0}}
trainer.learning_rate = 0.002
trainer.run(data, steps=100)

print(trainer.params)
print(trainer.history['loss'])
print(trainer.evaluate(data))
```

The functional can be built in or a generic `dft.Functional`. `mf` is a solved
RKS/UKS object; `Sample` eagerly captures its prepared reference without rerunning
SCF. An explicit `reference = mf.to_reference()` is also accepted. This state
contains integrals, grid and initial density/orbitals. Recreate samples after
changing the source. See [public API](../../../API.md).
Energies use Hartree; density targets use that state's grid. Parameters may
be supplied explicitly or initialized by the functional using `trainer.seed`
(default 0). This facade retains the existing loss, SCF and differentiation
implementations. It owns JIT, Adam state, update rejection, and history.

This module is shared framework code, independent of `gradscf.model.neural_xc`.
Use `dft.Functional` for user-defined energy callbacks, or pass the built-in
`model.neural_xc.Functional` directly. The old `gradscf.model.training` import
path has been removed. See [DFT module ownership](../dft/FUNCTIONAL.md#module-ownership).

## Three modes

| `trainer.mode` | Forward and derivative |
|---|---|
| `fixed_density` | Evaluate at the supplied fixed state |
| `explicit` | Run the existing SCF computation and differentiate it with JAX |
| `implicit` | Run SCF and differentiate its fixed-point condition |

`explicit` is the canonical name. SCF is represented by `jax.lax.scan`, not
manually expanded into separate iteration blocks. The rename does not freeze
its density response or change the executed numerical algorithm. The shared
low-level SCF normalizer accepts historical `expl`/`unrolled` aliases and
maps them to `explicit`; new high-level Trainer configurations use only the
three names above. `impl` remains a historical low-level alias for `implicit`.

## Inputs and loss

`Sample` is a constructor for the existing validated PyTree sample record,
not a second storage format. Common labels:

```python
training.Sample(reference, energy=energy_hartree, density=density_on_grid)
training.Sample(reference, excitation_energies=gaps_hartree,
                oscillator_strengths=strengths)
```

Other optional labels are `s1_energy`, `orbital_energies`,
`orbital_occupations`, `spectrum=(grid_ev, curve)`, `xc_potential`, `xc_kernel`,
and `weight`. The physical capabilities of the chosen functional still apply;
adding a label does not create an unsupported potential/response definition.

Loss is a mapping from target to metric weights. Supported target names are
`energy`, `density`, `orbital_energy`, `s1_energy`, `excitation`,
`oscillator_strength`, `spectrum`, `xc_potential`, and `xc_kernel`.
`energy`, `orbital_energy`, `s1_energy`, `excitation` and `oscillator_strength`
support MSE and MAE; the other targets support MSE. Unknown keys fail before
training. Default loss is `{'energy': {'mse': 1.0}}`.

The new facade rejects density/orbital-energy supervision in fixed-density
mode, since those objectives otherwise request SCF through the low-level
training API. Select `explicit` or `implicit` deliberately for those targets.

## SCF controls and failure handling

```python
trainer.scf = dict(max_cycle=80, damping=.2, conv_tol_energy=1e-11,
                   conv_tol_density=1e-9, require_converged=True)
trainer.adjoint = dict(tolerance=1e-9, max_iter=40, regularization=0.)
```

Options are translated into the existing numerical configuration. For external
functionals, convergence is required by default. Use
`trainer.scf['require_converged'] = False` only when finite-iterate explicit
training is intended. NaN/inf loss or gradient rejects the whole update,
including Adam state. Learning rate must be a finite positive scalar.

## State, history and evaluation

- `run(data, steps=N)` returns the same trainer. `kernel(data, steps=N)` returns
  the resulting parameters. Repeated `run` calls preserve Adam state.
- `params` contains the final weights. Replacing it, or changing learning rate,
  starts a new optimizer and history on the next operation using trainer state.
- `history` contains step 0 and metrics after each attempted update. Running
  100 steps produces 101 rows, not 100 pre-update values.
- `history['update_accepted'][0]` is None; row k>0 records acceptance of the
  preceding update. `step` counts attempts; `optimizer_step` counts accepted
  updates. A rejection leaves parameters unchanged but still produces a row.
- `loss`, `energy_mse`, `energy_mae`, `density_mse` are scalar histories;
  `energy` retains one prediction per sample. Metrics respect sample weights.
- `scf_converged` is None and `scf_cycles` is zero when no SCF was performed.
  For multiple samples, convergence requires all SCF states and cycles reports
  the largest cycle count.
- `evaluate(data)` returns current metrics without optimizing.
  `evaluate(data, mode='implicit')` gives a separate self-consistent diagnostic.
- `predict(mf)` returns `(energy, electronic_state)`.
  `predict(mf, params=other_weights)` does not replace trainable weights.

Registered array PyTrees use JIT. Legacy opaque molecule-like test/workflow
objects retain the eager shared path; the mathematical loss is the same.

## Migration and reuse

The old `NeuralXCTrainer` implementation and separate `TrainingResult` carrier
were removed. Use `Trainer` and its `params`, `history`, and `metrics` attributes.
There is no parallel old/new training loop. Low-level helpers such as
`molecular_loss` and `make_molecular_train_step`, and their detailed configuration
records, remain available to advanced workflows; simple examples need not
assemble them individually.

The H2/6-31G* FCI example now imports only the `training` namespace. Three
100-update runs reproduce every recorded old total loss and energy exactly;
MSE roundoff differences are below 3e-17. The `explicit.csv` series replaces
the old label. The four-layer network and scalar objective are unchanged.
See `examples/training/h2_fci_training.py` and its recorded results.

## Validation

2026-09-28, CPU float64: **106 passed, 3 skipped, 1 deselected**, 150.50 s.
The skipped tests require unavailable jax-xc. The deselected orbital-energy
fixture failure was reproduced on the unchanged base before this refactor.
Periodic complex-to-real AD emitted existing JAX warnings; the periodic
geometry derivatives still passed their finite-difference assertions.
No full-repository or GPU execution is claimed.

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu python -m pytest -q \
  tests/test_training_facade.py tests/test_neural_xc_public_api.py \
  tests/test_external_neural_xc.py tests/test_molecular_training_api.py \
  tests/test_scf_autodiff.py tests/test_scf_implicit_and_xc_energy.py \
  tests/test_training.py tests/ofdft/test_response.py \
  tests/ofdft/test_representations.py tests/pbc/test_gradients.py \
  --deselect=tests/test_training.py::test_orbital_energy_loss_uses_explicit_target_and_weights
```

All three neural-XC examples were executed. The 100-step FCI comparison is
recorded in `examples/training/h2_fci_results/api_equivalence.json`, using
commit `a9f15f4` as its reference. Every total loss and total energy matches
exactly; the largest MSE roundoff difference is 2.78e-17.
