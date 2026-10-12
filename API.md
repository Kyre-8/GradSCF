# Public API

Import domain namespaces, then construct calculation objects:

```python
from gradscf import gto, scf, fci
from gradscf.model.neural_xc import training

mol = gto.M(atom='H 0 0 0; H 0 0 .74', basis='6-31g*')
mf = scf.RHF(mol).run()
e_fci = fci.FCI(mf, solver='dense').run().e_tot

data = [training.Sample(mf, energy=e_fci)]
# functional is an externally defined dft.Functional or a built-in model.
trainer = training.Trainer(functional, params=params)
trainer.mode = 'implicit'
trainer.loss = {'energy': {'mse': 1., 'mae': 1.}}
trainer.run(data, steps=100)
```

For a complete external-network example see
[the four-layer density-matrix MLP](examples/training/density_matrix_mlp.py).
`fixed_density`, `explicit`, and `implicit` retain their numerical definitions.

## Ownership and naming

The root exports domain namespaces. The previous 144 flat symbol exports
have been removed. Use the owning module instead of adding a second name for
the same implementation. The [migration table](API_MIGRATION.csv) lists every
removed root export and the additional facade relocations.

| Task | Public entry |
|---|---|
| Molecule | `gto.M(...)` |
| Ground state | `scf.RHF(mol)` / `scf.UHF(mol)` for HF; `dft.RKS(mol)` / `dft.UKS(mol)` for DFT |
| AO density | `mf.make_rdm1()` |
| Reusable AO/grid reference | `mf.to_reference()` |
| External XC architecture | `dft.Functional(inputs, apply, init_fn=...)` |
| Training data and optimization | `training.Sample(mf, ...)`, `training.Trainer(functional)` |
| Prediction | `trainer.predict(mf)` |
| Excited states | `tdscf.TDA(mf)`, `tdscf.TDDFT(mf)` |
| CI / CC / FCI | `ci.CIS(mf)`, `cc.CCSD(mf)`, `fci.FCI(mf)` |
| Moller–Plesset correlation | `mp.MP2(mf)`, `mp.MP3(mf)`, `mp.MP(mf, order=6)`; see [reference restrictions](src/gradscf/mp/README.md) |
| Configured workflow | `workflows.ExperimentPipeline(config)` |

Meaningful mathematical array APIs remain in their domains, including
Hamiltonians/RDMs, GW self energies, OFDFT functionals and shared solvers.
Their names are not shortened merely because they are long. Implementation
choices such as `with_jax` and chained construction steps do not belong in
ordinary calculation entry names.

Four forwarding files (`dft.rks`, `dft.uks`, `dft.roks`, `dft.gks`) are removed;
import their classes through `dft` and advanced array kernels through `scf`.
The six workflow `_from_spec` / `_from_molecule_spec` forwarding functions are
removed. Their base functions already accept the same keyword inputs. Two
preset implementations use `water_experiment_config` and
`benzene_experiment_config`, without separate `strict_jax` aliases.
`tools.api` remains an explicit legacy workflow module: its helpers have
particular return and file-writing contracts, so they are not renamed to
ordinary molecule constructors.

`scf.RHF` fixes full exact exchange and zero semilocal XC while reusing the
RKS solver, integral backends, reference conversion and differentiation rules.
Its defaults match the existing `dft.RKS(mol, xc="hf")` object (including
`max_cycle=80`), not the separate low-level `RHFConfig` defaults. The generic
RKS HF configuration remains valid; examples use `scf.RHF` to identify the
physical reference explicitly. Use RKS for configurable XC.

## Solved references

`mf.to_reference()` reuses the converged RHF/UHF or RKS/UKS solution and integral inputs,
completing missing grid/response data on demand. It does not rerun SCF.
`training.Sample(mf, ...)` and `trainer.predict(mf)` use this same conversion.
The shared eager adapter is `scf.as_reference`; post-HF MO Hamiltonians and
GW/BSE provenance retain their separate scientific contracts.

Unsolved, unconverged or stale SCF sources are rejected. After changing the
molecule, SCF settings or solved `mo_coeff` / `mo_occ` / `mo_energy`, rerun
`kernel()` before requesting a reference. Value fingerprints detect in-place
orbital edits as well as replacement arrays. A UKS device-only change reuses
the solved reference without rebuilding integrals or SCF.

`Sample` captures a prepared state at construction; later changes to the SCF
object do not refresh existing samples automatically. Rebuild the sample when
you want to train on a new state. RKS `make_rdm1()` returns a total AO density;
UKS returns alpha/beta AO densities. Reference containers retain their existing
spin-stacked representation. Conversion is eager; array kernels remain the
entry points inside `jax.jit` / `jax.grad`.

## Migration scope

This is an intentional import-path break. Repository examples, workflows and
tests have been migrated, without compatibility forwarding files or a second
training implementation. See [MIGRATION.md](MIGRATION.md) for earlier module
ownership changes, and [training](src/gradscf/model/neural_xc/training/README.md) for loss,
convergence and derivative contracts.

## Validation of this cleanup

On 2026-09-28, Apple M4 Pro, CPU/float64, Python 3.12.2 and JAX 0.8.1:
**155 tests passed, 2 deselected in 59.02 s**. The two PBE smoke tests need
`jax_xc`, which is unavailable in this environment. Existing workflow/JAX
warnings were reported. This is a focused API/SCF/training/GW-BSE regression,
not a complete test-suite or GPU run.

```bash
PYTHONPATH=src:tests:tests/bse JAX_PLATFORMS=cpu python -m pytest -q --import-mode=importlib \
  tests/test_scf_public_reference.py tests/test_scf_reference_reuse.py \
  tests/test_pyscf_style_ground_state_api.py tests/test_pyscf_style_excited_state_api.py \
  tests/test_pyscf_style_namespace.py tests/test_simplified_api.py \
  tests/test_workflows_public_api_usage.py tests/test_workflows_config.py tests/test_workflow_core.py \
  tests/test_reference_boundaries.py tests/test_external_neural_xc.py tests/test_training_facade.py \
  tests/test_functional_module_boundaries.py tests/test_scf_implicit_and_xc_energy.py \
  tests/test_scf_autodiff.py tests/test_molecular_training_api.py tests/bse/test_api.py \
  --deselect=tests/test_pyscf_style_ground_state_api.py::test_real_rks_kernel_smoke_sto3g_h2 \
  --deselect=tests/test_pyscf_style_ground_state_api.py::test_real_rks_kernel_smoke_sto3g_water
```

The three H2/6-31G* FCI-supervised 100-step runs were repeated through the
short API. All 101 loss/energy/MSE/MAE rows per mode match the recorded
`517b943` baseline exactly; see
[the equivalence record](examples/training/h2_fci_results/public_api_equivalence.json).
This verifies the refactor, not quantitative accuracy of the small network.
The final energy errors remain about 0.29–0.33 Ha.


## RHF entry validation (2026-09-30)

CPU float64: 62 focused SCF/API tests passed, including the new RHF checks.
Two PBE tests were deselected because `jax_xc` is unavailable. RHF and
RKS-with-HF agree for full, DF and direct integral backends; cached training
references and CI/CC/FCI/GW-BSE consumers are covered. The four-layer MLP
example also ran with unchanged output after the constructor migration.
The full repository suite and GPU backends were not run.
