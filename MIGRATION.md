# Migrating from GradTDDFT to GradSCF

GradSCF is the current project and package name. The original package rename preserved numerical methods and signatures.
Subsequent API changes are listed below; some old import paths are removed.

## 2026-09-28 — Public entry points use domain ownership

Use `from gradscf import gto, dft, training` and
`mf = dft.RKS(gto.M(...), xc=...).run()`. The former
`restricted_molecule_from_spec_with_jax_rks` / unrestricted builder names
are private implementations. Obtain their prepared AO/grid state through
`mf.to_reference()`, or pass the solved `mf` directly to `training.Sample`
and `trainer.predict`.

The root no longer reexports 144 individual symbols. This is a breaking
import cleanup, not a numerical-method change. See [API.md](API.md) and the
complete [old-to-new table](API_MIGRATION.csv). DFT forwarding modules and
six duplicate workflow builder wrappers are removed. Advanced numerical
APIs remain available in their owning modules.

## 2026-09-28 — Generic functionals and training are framework modules

Concrete neural architectures remain under `gradscf.model`. The generic
functional adapter, physical ingredients and derivatives belong to DFT;
training is shared infrastructure at the package root.

| Previous import | Current import |
|---|---|
| `gradscf.model.neural_xc.ExternalFunctional` | `gradscf.dft.Functional` |
| `gradscf.model.neural_xc.DensityInputs` | `gradscf.dft.DensityInputs` |
| `gradscf.model.training` | `gradscf.model.neural_xc.training` |
| `gradscf.scf.xc_energy` | `gradscf.dft.derivatives` |
| HFX physical/cache helpers in `model.neural_xc.inputs` | `gradscf.dft.hfx` |
| PT2 physical helpers in `model.neural_xc.inputs` | `gradscf.dft.pt2` |

Use `from gradscf import dft, training` for a user-defined architecture.
The built-in model remains `from gradscf.model import neural_xc`, followed
by `neural_xc.Functional(...)`. Its model-specific feature transforms and
presets remain in that package. The ambiguous root `gradscf.Functional`
export is removed; it is not silently redirected to a different class.
Old module paths are removed rather than retained as forwarding files.

Generic examples and their recorded results moved from `examples/neural_xc/`
to `examples/training/`. The shared SCF modes remain `fixed_density`,
`explicit`, and `implicit`; this ownership change does not alter numerical
methods, losses, or gradients.

## DifferentiableSCF XC contracts

`DifferentiableSCF.run(molecule, functional, params)` retains the current Neural
XC/NeuralD/force-training interfaces. Historical SCF-only fallbacks were removed:

- Binding requires `bind_to_molecule_for_scf(params, molecule)`. Generic `bind`,
  `bind_to_molecule`, or an implicitly already-bound object is not substituted.
- Direct restricted potentials return seven entries: `(v_rho, v_grad, v_tau,
  v_lapl, kind, alpha, extra_fock)`. Old four/six-entry forms are rejected.
- Bound grid potentials use `grid_potential_components(molecule)`, returning
  the current three/four-entry form. Two-entry forms, `grid_potential`, and
  `local_potential` fallbacks are no longer accepted here. Grid gradients have
  shape `(ngrids, 3)`; transposed legacy arrays are not repaired automatically.
- Density-energy models expose `scf_xc_energy_and_alpha_for_density`, returning
  `(energy, alpha)` together. The old scalar-only callback and separate alpha
  fallback are not used by DifferentiableSCF.

The optimized direct-Fock path, SCF-specific bound path (including frozen
functional diagnostics), and eight-entry unrestricted potential path remain
active. Public response APIs on bound XC objects are unchanged. This cleanup
does not alter SCF iterations, density layouts, or implicit/explicit semantics.

## SCF backward configuration

`gradscf.scf.SCFDifferentiationConfig` is shared by the functional/molecule SCF
adapter and the fixed-occupation orbital solvers. Use `mode="implicit"` or
`mode="explicit"`; `impl`, `expl`, and the historical `unrolled` name remain accepted aliases. Pass this object
as `DifferentiableSCFConfig(differentiation=...)` or as the orbital solver's
`differentiation=...` argument. An explicit object takes precedence over the
legacy DFT backward fields. Existing DFT defaults are retained when it is absent.
Training workflows now honor their configured backward mode instead of forcing
implicit mode; the existing nonfinite-step recovery option may retry with implicit.

Orbital solvers now use JAX/Optax and return array-valued PyTrees, including
energy, convergence flags, and fixed-shape diagnostic histories. Convert arrays
to Python scalars only when reporting outside JIT/autodiff. The separate
`orbital-optimization` SciPy extra is no longer required.
Orbital updates use a Cayley retraction to preserve the metric while avoiding
large higher-order matrix-exponential derivative graphs; iteration trajectories
may differ from the former exponential parametrization.

The shared implicit policy requires a converged state by default and checks
the adjoint residual. Failed backward solves return nonfinite cotangents while
preserving forward diagnostics. Explicit AD differentiates the existing JAX loop computation
and can be used before convergence. Fixed 0/1 occupations are static topology;
continuous integrals and overlap remain differentiable. Use
`orthonormalize_initial=True` to transport a fixed orbital seed as overlap changes.

The default implicit path now composes JVP/VJP rules through its root and linear
solves. `training.energy_and_forces`, `force_matching_loss`, and
`make_force_loss_and_grad` provide force supervision for differentiable energy
callbacks whose full derivative chain is supported. Native derivative support
is operator-, layout- and parameter-specific: documented dense operators have
coordinate Hessian products, while one-electron and fixed-auxiliary three-center
paths support coefficient products and first-order exponent derivatives.
Native exponent Hessians and basis/geometry mixed second derivatives remain
unsupported. See the [native derivative contract](src/gradscf/integrals/_native/README.md);
integral-level rules alone do not establish complete molecular Hessian or
force-loss coverage.
Overlap orthogonalization now differentiates the inverse square root with a
Sylvester solve above the eigenvalue cutoff, avoiding spurious loss of response
at repeated overlap eigenvalues. Clipped overlaps retain regularized derivatives.

## SCF convergence, results and reference reuse

RKS now uses its public density tolerance and independent `conv_tol_grad=1e-7`,
matching the shared convergence policy. It no longer derives a gradient threshold
from `sqrt(conv_tol)` or relaxes the final level-shift check with an OR condition.
Runs may take additional iterations to satisfy the requested criteria.

`RKSResult` is the common internal PyTree; `TraceableRKSResult` remains an alias.
The eager entry retains Python scalar outputs. Cached RKS results/inputs are reused
for TD references; changing geometry or ground-state configuration requires
`kernel()` again. Preparing a reference preserves facade orbital layout/status.

Private test-reference helpers formerly imported from `gradscf.scf.features`
have moved to `tests/reference_scf_features.py`; no production compatibility
module is retained for those private helpers. HF wrappers and method-specific
RO/spinor logic remain separate.

## Name mapping

| Previous name | Current name |
| --- | --- |
| Distribution `td-graddft` | Distribution `gradscf` |
| `import td_graddft` | `import gradscf` |
| `from td_graddft import gto, scf, dft` | `from gradscf import gto, scf, dft` |
| `td_graddft_tools` | `gradscf_tools` |
| `src/td_graddft/` | `src/gradscf/` |
| `src/td_graddft_tools/` | `src/gradscf_tools/` |

Replace the package prefix in notebook imports, Python scripts, `python -m`
commands, dynamic import strings, and test monkeypatch targets. For example,
`td_graddft.scf.GHF` becomes `gradscf.scf.GHF`.

No `td_graddft` or `td_graddft_tools` compatibility namespace is shipped.
An old installation may still provide those names independently; importing it
does not select this checkout.

## Install or run the current checkout

With the required dependencies already available:

```sh
python -m pip install --no-deps -e .
python -c 'import gradscf; print(gradscf.__file__)'
```

For source-based execution without changing the environment:

```sh
PYTHONPATH=src python -c 'from gradscf import gto, scf, dft; print(scf.__file__)'
python -m pytest -q tests/test_gradscf_package.py
```

On c20, the working directory remains `/home/yjiao/GradSCF`, with the existing
`/home/yjiao/opt/miniconda3/envs/jax_scf/bin/python` interpreter. The package
layout is now `src/gradscf`; the environment name does not need to change.

## Scientific names and existing data

Keep `tdscf`, `tddft`, TDA/TDDFT result names, and upstream GradDFT model or
adapter names: they identify methods or external projects. SCF class names
such as `UHF`, `ROHF`, `GHF`, `RKS`, `UKS`, `ROKS`, and `GKS` are unchanged.

The historical `reproducibility/v1.0.0/` files, their hashes, and third-party
basis data remain unchanged. NPZ target bundles retain the original embedded
metadata key so existing bundles remain readable. This does not provide
compatibility for arbitrary Python pickles containing old module paths.

The Git remote and repository URL still identify the existing GradTDDFT
repository; this API migration does not rename the GitHub repository.

## Consolidated integral modules

All integral implementations, basis resources and numerical grids now live
under `gradscf.integrals`. The following legacy files/exports are removed;
update imports rather than relying on compatibility aliases:

| Previous path | Canonical path |
| --- | --- |
| `gradscf.data.integrals` | `gradscf.integrals` |
| `gradscf.data.integrals.jax` | `gradscf.integrals.backends.jax_reference` |
| `gradscf.data.integrals.libcint.mol` | Removed; use native integral plans |
| `gradscf.data.integrals.libcint.autodiff` | Removed; native plans provide geometry AD |
| `gradscf.data.basis` | `gradscf.integrals.basis` |
| `gradscf.data.pyscf_basis_loader` | `gradscf.integrals.basis.data` |
| `gradscf.data.grid` / `data.grid_ao` | `gradscf.integrals.grids` / `integrals.grids.ao` |
| `gradscf.data.ris_auxbasis` | `gradscf.integrals.basis.auxiliary` |
| `gradscf.integrals.assembly` / `integrals.input_*` | `gradscf.scf.inputs` |
| `gradscf.df` J/K functions | `gradscf.integrals.molecular.jk` |
| `gradscf.df` spectral ERI factorization | `gradscf.integrals.molecular.factorization` |
| `gradscf.integrals.mo` / `gradscf.integrals.molecular.ao2mo` | `gradscf.ao2mo` |
| `gradscf.integrals.density_fitting` | `gradscf.integrals.molecular.density_fitting` |
| `gradscf.integrals.contraction` basis transforms | `gradscf.integrals.basis.contraction` |
| `gradscf.integrals.contraction.exchange_matrix` | `gradscf.integrals.molecular.jk.exchange_matrix` |
| `gradscf._native` | `gradscf.integrals._native` |

Basis/grid functions previously exported from `gradscf.data` are exported from
`gradscf.integrals`. Public `gradscf.gto` facades remain available. All 321
original basis snapshot files and the supplementary JSON basis bundle are
retained; resource lookup now uses `gradscf.integrals.basis.data`. Angular
momenta, general contractions and kappa labels are preserved by the loader.
Loading data does not imply native ECP/spinor/operator support.

Native Python bindings now live under `integrals.backends.native`: `ffi` owns
raw FFI, `packing` owns ATM/BAS/ENV, `autodiff` owns geometry/coefficient/exponent
rules, and `eri`, `density_fitting`, `jk` separate execution roles. The old
`native_geometry`, `native_coefficients` and `native_compact` files are removed.
Current public `prepare_basis`, `make_plan`, `make_auxiliary_plan` and
`backend_capabilities` imports remain unchanged. See the
[integral ownership guide](src/gradscf/integrals/README.md) for remaining
reference-kernel and dense-ERI migration dependencies.

## External chemistry dependency removal

Production PySCF/gpu4pyscf bridges and object adapters are removed. Reference
converters now live in test-only `pyscf_data_reference`/`pyscf_adapters` helpers.
Comparative CLIs moved into `tests/comparisons/`. The `comparison-tests` extra
is the only dependency group that installs PySCF.

Use `init_guess="hcore"` or explicit density matrices. Legacy external guesses
are not silently approximated. Native full-ERI spectral factorization replaces
external auxiliary DF in production. Basis resource bytes are unchanged, but
Python-format resources now have the inert `.pydata` suffix.

## 2026-09-18 — Neural model code moves to `gradscf.model`

Historical migration; the 2026-09-28 section above supersedes the training
location and ambiguous top-level `Functional` export.

Hard switch, no compatibility aliases:

| Old path | New path |
|---|---|
| `gradscf.neural_xc` | `gradscf.model.neural_xc` |
| `gradscf.neural_d` | `gradscf.model.neural_d` |
| `gradscf.model.neural_xc.training` | `gradscf.model.training` |
| `nnao` (top level) | `gradscf.model.nnao` |

Top-level `gradscf` symbols re-exported via `gradscf/__init__.py`
(`Functional`, `make_neural_xc_functional`, `MolecularTrainingConfig`, ...)
are unchanged.  The vendored mace-jax tree ships inside
`src/gradscf/model/nnao/mace_jax` but remains a separate top-level `mace_jax`
package at runtime (install via the `nnao` extra / git pin); its sources are
untouched, and `UPSTREAM.json` hash keys stay valid relative to the moved
`nnao` root.

| `gradscf.traditional_xc` | removed — use `gradscf.dft` / `gradscf.dft.xc` |
| `gradscf.xc_backend` | `gradscf.dft.libxc_jax` |
