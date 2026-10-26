# GW/BSE merge validation, 2026-10-02

The merge review corrected two issues before integration:

1. Finite-eta real-axis screening now preserves the existing CD convention.
   With d=Delta-i*eta and z=omega+i*eta, the complex symmetric transition
   matrix and coupling reproduce `rho_response_real` exactly. The imaginary
   axis retains the real, zero-broadening RPA matrix. Cached eigenvectors are
   numerical factors only; implicit JVP/VJP follow the matrix equation. A
   defective or poorly conditioned complex basis falls back to a dense solve.
2. Cached Cholesky screening uses the shared checked linear solver. Both
   primal and transposed solves check the actual residual inside the opaque
   solve, including rejection of stale/inaccurate factors. Matrix response
   is retained while the cached numerical factor is stopped for AD.

## Completed checks

All checks used CPU float64. No GPU backend or full-repository suite is claimed.

- The affected regression below passed **174 tests**, with one large auxiliary
  stress case explicitly deselected, in 753.15 seconds. The 25 JAX
  ComplexWarnings arise when complex intermediates return cotangents to real
  inputs; independent derivative comparisons passed.
- After adding the fixed JIT entry point to `solve_shifted`, the shifted-solve,
  RPA-pole and evGW subset passed **18 tests** in 25.92 seconds (3 such warnings).
  These tests overlap the larger regression and must not be added to its count.
- The standalone periodic q0 residue consistency selection passed **2 tests**
  in 1.75 seconds; 15 unrelated tests in that file were deselected.
- New Cholesky failure/adjoint cases and existing screened-action transpose
  cases separately passed 6 tests; these overlap the affected regression.
- Script syntax and code/document whitespace checks passed. The archived
  `auxiliary_cutoff.patch` preserves its two original blank context lines;
  these carry the space prefix required by its unified-diff representation.
  The source review found no
  remaining blocking issue after the numerical corrections.

```sh
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python -m pytest -q \
  tests/gw/test_contour_pole_limit.py tests/gw/test_gw_poles.py \
  tests/gw/test_gw_evgw.py tests/gw/test_gw_qsgw.py \
  tests/gw/test_gw_cd_rgw.py tests/gw/test_gw_cd_ugw.py \
  tests/gw/test_gw_differentiability.py tests/gw/test_gw_qp.py \
  tests/gw/test_molecular_controls.py tests/gw/test_qp_methods.py \
  tests/bse tests/solvers/test_shifted.py tests/solvers/test_block_linear.py \
  tests/solvers/test_gmres_accuracy.py \
  -k 'not large_auxiliary_screening_checks_physical_column_residuals'

PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python -m pytest -q tests/solvers/test_shifted.py \
  tests/gw/test_gw_poles.py tests/gw/test_gw_evgw.py

PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python -m pytest -q tests/gw/pbc/test_kpoint_invariants.py \
  -k standalone_q0_residue_matches_multichannel_self_energy
```

## Benchmark provenance

The four-size scaling archive is explicitly a pre-review timing snapshot:
its finite-eta representation and Cholesky guard differ from the final code.
Its measurements must not be reported as final-code scaling. The
[serial resolvent recheck](cycloalkane_scaling/resolvent_reuse.md) compares the
final implementation with a dense reference of the same finite-eta equation.
Raw experimental tables and MolGW stdout/control outputs remain local;
source URLs, checksums, selected numerical references, input files and plots
are committed, with download/reproduction instructions.
