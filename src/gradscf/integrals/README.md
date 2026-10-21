# Integral module ownership

The public entry points remain `prepare_basis`, `make_plan`,
`make_auxiliary_plan` and `backend_capabilities`. Static plans bind current
JAX parameter arrays; they must not cache stale exponents or coefficients.

## Directory map

| Owner | Responsibilities |
|---|---|
| `basis/types.py` | Static shell topology, differentiable raw parameters and backend-independent shape validation |
| `basis/library.py`, `basis/data/` | Named/raw basis construction and complete bundled resources |
| `basis/cartesian.py` | Cartesian AO/shell views and grouping used by grids and reference kernels |
| `basis/normalization.py`, `basis/contraction.py` | JAX normalization and primitive-to-contracted basis transforms |
| `basis/ecp.py`, `basis/auxiliary.py` | ECP data/adapter and auxiliary-basis specifications |
| `molecular/one_electron.py`, `molecular/eri.py` | Existing Cartesian operator entry points and packed AO-pair metadata |
| `molecular/jk.py` | DF, packed and dense-reference density contractions; legacy direct dispatch |
| `molecular/density_fitting.py` | Native auxiliary 2c/3c plans, fixed metric caches, whitening and projections |
| `molecular/factorization.py` | Spectral ERI factorization, distinct from auxiliary-basis DF |
| `grids/`, `periodic/` | Molecular grids/AO evaluation and periodic FFT/Coulomb/pseudopotential operations |
| `backends/native/packing.py` | Native table construction, normalization binding and storage checks |
| `backends/native/ffi.py` | Float64 CPU FFI value calls and shape/dtype validation |
| `backends/native/autodiff/evaluation.py` | Combined differentiable value dispatch and independent parameter storage |
| `backends/native/autodiff/` | Geometry, coefficient and exponent derivative products; shared batching |
| `_native/csrc/` | C++ shell drivers, integral products and immediate tangent/cotangent reductions |
| `_native/vendor/` | Pinned upstream sources, licenses and checksums; unchanged by this reorganization |

AO-to-MO transformations belong to the sibling `gradscf.ao2mo` package:
`blocks.py` handles selected dense, packed and DF blocks, `native.py` contracts
native shell slabs, `reference.py` builds complete small-system MO tensors,
and `response.py` supplies occupied/virtual response slices. Integral evaluation
and its derivative primitives remain in `gradscf.integrals`.

SCF configuration, XC/grid input assembly and RKS/UKS input payloads are owned
by `gradscf.scf.inputs`. Its grid cache lives with the grid implementation;
electron-count validation lives in `scf.core`. SCF implicit/unrolled response
belongs to SCF/solver modules, not the integral backend. Existing public
integrals-level SCF-builder re-exports remain lazy and contain no implementation.

## Native execution and differentiation

`IntegralPlan` dispatches to native packing and mathematical derivative rules,
which call FFI and C++ kernels. Raw coefficients are normalized in JAX. C++
derivatives act on the normalized native coefficients and bare Gaussians; JAX
adds the normalization chain. Fixed auxiliary metric whitening is also JAX.

The C++ compact ABI dispatcher delegates packed ERI, auxiliary DF, direct J/K
and packed J/K to separate translation units. Shared Coulomb shell traversal
is in `coulomb_shells.h`; library exports and vendor bytes are unchanged.

Current AD limits remain explicit: geometry first/second derivatives for the
existing dense operators; coefficient first/second products for one-electron
and fixed-auxiliary 3c integrals; exponent first-order products for those same
basis-parameter paths. Exponent and coefficient/geometry mixed second rules,
RI geometry/auxiliary AD and direct-J/K basis AD are not supplied by moving files.
See [_native/README.md](_native/README.md) for the exact numerical contract.

## Remaining production migrations

This reorganization preserves calculations. The following dependencies prevent
moving every JAX molecular kernel into tests or removing every dense ERI API:

| Current consumer | Required replacement before removal |
|---|---|
| `dft.hfx` grid `rinv_matrices`, including range separation | Native batched grid-Coulomb operator and matching parameter/geometry AD |
| Native SCF assembly with traced geometry | Packed/block geometry JVP/VJP and second products; current path forms dense ERI before packing |
| CartesianBasis RHF/UHF/ROHF/GHF entry points | Native plan/packed input migration with the same derivative contracts |
| Post-HF full MO and spin-orbital transformations | Consumer-by-consumer block/DF interfaces in MP/CC/CI/FCI |

`backends/jax_reference` is retained only where the existing interfaces still
need these kernels or for explicit reference comparisons. Shared basis types
and primitive normalization have been removed from that backend; native plans
and periodic AO normalization no longer import it. Complete basis resources,
angular momenta, contraction columns and representations are preserved.

## Validation

Run from the repository root with CPU float64:

```sh
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=src python -m pytest -q \
  tests/integrals tests/test_integral_namespace.py tests/test_scf_inputs_api.py \
  tests/nnao/test_df_rhf_solver.py tests/nnao/test_native_exponent_scf.py \
  tests/nnao/test_joint_native_training.py
```

Build native code with `python -m gradscf.integrals._native.build` before native
tests. Basis resources and native sources must also be checked in a built
wheel. Remote experiments use their own frozen source/native snapshots; do not
overwrite those snapshots to apply a local module reorganization.
