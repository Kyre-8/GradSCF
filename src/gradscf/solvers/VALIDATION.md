# Shared solver migration validation — 2026-09-21

The first section records the initial migration, before removal of the
compatibility export files. The removal is validated separately below.
See `README.md` for current canonical import paths.

Environment: local arm64 CPU, JAX 0.8.1, float64 enabled. Commands were run
inside the isolated `refactor/shared-solvers` worktree with `PYTHONPATH=src`,
`JAX_PLATFORMS=cpu`, `JAX_ENABLE_X64=1`, and `OMP_NUM_THREADS=1`.

| Test group | Result | Elapsed |
| --- | --- | --- |
| `tests/solvers` | 17 passed | 23.84 s |
| CI, TD-SCF eigen/API, SCF implicit/higher-order, DIIS state/scaling | 95 passed | 81.72 s |
| Orbital optimization/gradients, UHF/UKS stability | 21 passed, 3 skipped | 75.96 s |
| scGW response and periodic TD-SCF | 26 passed, 8 skipped | 144.20 s |

Total: **159 passed, 11 skipped**. Skips require the unavailable optional
`jax_xc` package (three UKS and eight periodic DFT cases). The scGW tests emit
JAX complex-to-real cotangent warnings also present before this migration.
This is targeted migration validation, not the complete repository test suite.
GPU execution and generic non-Hermitian EOM-CC are not validated here.

Exact test selections:

```bash
python -m pytest -q tests/solvers

python -m pytest -q tests/ci tests/test_tddft_eigensolvers.py \
  tests/test_pyscf_style_excited_state_api.py \
  tests/test_scf_implicit_and_xc_energy.py tests/test_scf_higher_order.py \
  tests/test_scf_diis_state.py tests/test_scf_diis_scaling.py

python -m pytest -q tests/test_scf_orbital_optimization.py \
  tests/test_scf_orbital_gradients.py tests/test_uhf_stability.py \
  tests/test_uks_stability.py

python -m pytest -q -rs tests/gw/test_gw_scgw_response.py tests/pbc/test_tdscf.py
```

`examples/solvers/shared_solvers.py` also ran successfully. Its matrix-free
eigenvector-dependent observable gave derivative -0.23383077651091166 versus
central finite difference -0.2338307764748748 (step 1e-5); the nonsymmetric
linear solve had residual 1.11e-16.

Additional checks:

- Architecture regression: historical solver files contain exports only;
  shared modules do not import electronic-structure modules or JAX private APIs.
- DIIS and the callback minimizer were relocated without changing their source;
  the spectral forward/JVP implementations retain identical ASTs.
- `git diff --check` passed.
- Original-workspace CI baseline files retain their recorded SHA-256 hashes.

Migration boundaries: domain-specific residuals, Hamiltonian construction,
physical root selection and charge-resolution criteria remain in method modules.
RPA exposes eigenvalue response only; its X/Y derivatives are not supplied.
Davidson eigenvector response assumes isolated converged roots and is first-order.
SCF's existing tested higher-order root and matrix-function response is retained.

## Compatibility export removal

After removing the six forwarding files and the fixed-point solver exports
from `gradscf.scf`, all repository clients were migrated to `gradscf.solvers`.
TD-SCF numerical defaults retain their values in `tddft/defaults.py`.
The numerical implementations were not changed in this cleanup.

With the same CPU/float64 environment:

```bash
python -m pytest -q tests/solvers tests/ci tests/test_tddft_eigensolvers.py \
  tests/test_pyscf_style_excited_state_api.py \
  tests/test_scf_implicit_and_xc_energy.py tests/test_scf_higher_order.py \
  tests/test_scf_autodiff.py tests/test_scf_diis_state.py \
  tests/test_scf_diis_scaling.py
# 128 passed, 2 PySCF OpenMP warnings, 105.80 s

python -m pytest -q tests/pbc/test_tdscf.py tests/test_scf_orbital_optimization.py \
  -k 'cached_response or lowers_energy'
# 3 passed, 22 deselected, 12.23 s
```

Architecture tests now require the old files to be absent and reject imports
of removed paths across source, tests, tools and examples. The two pre-existing
dirty files touched by the migration were checked against snapshots: only their
intended import lines changed. `git diff --check` also passed.

## Isolated degenerate spectral subspaces (2026-09-21)

The former `solve_spectral_projector` entry (now migrated to the unified
`solve_hermitian` subspace target) was validated on real symmetric toy matrices on
arm64 CPU (`TFRT_CPU_0`), JAX 0.8.1, float64, `OMP_NUM_THREADS=1`.
The reference and mathematical contract are in [DEGENERACY.md](DEGENERACY.md).
No generalized metric, non-Hermitian/RPA, GPU or higher-order subspace-response
claim is made by these tests.

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=1 \
  python -m pytest -q tests/solvers/test_subspace.py
```

Focused result: **21 passed**, 40.11 s. The coverage includes:

- Exact internal degeneracy, splitting perturbations, and small/nonzero internal
  gaps; projector and trace derivatives against analytic small-matrix values
  and independent NumPy eigendecomposition finite differences.
- JIT, vmap, vector/block probes and probe derivatives; JVP/VJP agreement.
- Frame rotations within the selected span, including a nondiagonal Ritz block;
  projector symmetry/idempotence, `P dP P = 0` and `dP P + P dP = dP`.
- A 24-dimensional matrix-free operator with an eight-column Davidson bound;
  no full operator block is requested in forward or backward evaluation.
- A cut degenerate boundary, unresolved small boundary gap, unconverged guard
  root, failed primal solve and failed adjoint solve; finite primal diagnostics
  but invalid derivatives where required, including a direct response solver.
- The full-space identity projector; reduction to the existing isolated-root
  response for a nondegenerate rank-one subspace.
- Bounded restart capacity, arbitrary starting vectors for a diagonal operator,
  and disconnected invariant sectors whose low eigenvalues are missed by
  diagonal-only guesses (dimensions 6 and 20).

The shared Davidson forward fixes were independently reviewed. Accepted append
counts are capped by remaining capacity, so the restarted basis cannot write
past its last column. Vanishing projected preconditioner directions use the
projected original residual rather than a normalized roundoff direction.

The final shared-solver and client regression command was:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=1 \
  python -m pytest -q tests/solvers tests/ci tests/test_tddft_eigensolvers.py \
  tests/test_scf_higher_order.py
```

Result: **115 passed**, no skips, 166.03 s. Two PySCF warnings state that
OpenMP is unavailable. This covers the new subspace suite and existing
isolated-root, CI, TDDFT and higher-order SCF regression tests. The latter
validate preservation of their existing contracts, not higher-order projector AD.

### Executable example

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  python examples/solvers/degenerate_subspace.py
```

The dimensionless four-state example selects the twofold eigenvalue 1, with
excluded eigenvalues 3 and 5. A perturbation mixes/splits the degenerate pair
and couples it to the excluded states. For `loss = probe.T @ P @ probe`:

| Quantity | Result |
| --- | ---: |
| Loss | 0.2500000000000002 |
| JVP | -0.11499999999999991 |
| VJP | -0.11500000000000000 |
| Central finite difference, step 1e-4 | -0.11499999993655474 |
| Absolute derivative error | 6.35e-11 |
| Boundary gap | 1.9999999999999973 |
| Maximum Ritz residual | 1.82e-15 |
| Cut-cluster status with one selected root | 2 (invalid response) |
| Elapsed time including compilation | 6.15 s |

## Unified Hermitian response (2026-09-23)

The P1–P3 refactor in [UNIFIED_EIGEN.md](UNIFIED_EIGEN.md) starts from
`bef1be3`, in `/private/tmp/gradscf-eigen-unified`. Environment: macOS arm64,
Python 3.12, JAX 0.8.1 CPU, float64 unless an existing float32 regression
explicitly specifies otherwise; `OMP_NUM_THREADS=1`.

Before implementation, `tests/solvers/test_shared.py` and `test_subspace.py`
passed **38 tests in 64.15 seconds**. The unified core and migrated versions of
these tests subsequently passed **43 tests in 70.40 seconds**. Those checks
cover isolated-vector equivalence, exact/near internal degeneracies, rotated
frames, variable probes, boundary guards, matrix-free memory bounds and
one-forward-solve behavior. These are overlapping checkpoints, not additive
coverage counts.

Independent review found two migration regressions and both were reproduced
before correction:

- TDA must select the lowest roots *above* its physical threshold. The common
  primal now supports explicit `value_min` intervals rather than merely clearing
  a lowest negative root after the solve. Both the excluded and included side
  of the cutoff are checked. Missing upper guards are accepted only after full
  spectral coverage; missing requested roots remain unconverged.
- The dense RPA inner problem has squared-frequency units. Its default absolute
  Hermitian gap tolerance incorrectly rejected separated low-frequency roots.
  The inner check now uses relative machine resolution and the RPA layer keeps
  its physical frequency-gap criterion. For A=diag(1e-5,2e-5), B=0, the first
  frequency's scaling derivative is 1e-5, not NaN. RPA explicitly rejects the
  new Hermitian interval option until RPA window semantics are defined.

The updated migration/interval/low-frequency regressions plus TDDFT eigensolver
and BSE-limit tests passed **34 tests in 51.45 seconds**. The first broad pass
had 219 passes and one obsolete underconvergence-fixture failure: its
`max_subspace=3` no longer had room for a guard and two correction slots.
The fixture was enlarged to four slots while retaining a six-dimensional
problem and a one-cycle limit; it still tests genuine nonconvergence.

### Runnable degenerate-subspace example

```sh
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  /opt/anaconda3/bin/python examples/solvers/degenerate_subspace.py
```

Dimensionless 4-by-4 real symmetric matrix, rank-2 internally degenerate
subspace. The scalar parameter multiplies the example's perturbation matrix;
there is no SCF or nuclear-coordinate derivative in this example.

- Projected-probe loss: 0.2500000000000002.
- JVP: -0.11499999999999991; VJP: -0.115.
- Central difference, step 1e-4: -0.11499999993655474.
- Absolute AD/FD difference: 6.35e-11.
- Requested residuals below 9.2e-16; boundary gap approximately 2.
- Cutting the cluster returns status 2 (invalid response prerequisites).
- Total script time, including compilation and checks: 5.60 seconds.

The old standalone response exports and forwarding module are absent. The
first-order fixed-Euclidean-metric contract remains unchanged in scope. RPA
cluster response, aggregate optical APIs, generalized variable metrics,
non-Hermitian EOM response and higher-order spectral derivatives are not
implemented by this refactor.

### Unified-response final regression gate

```sh
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  /opt/anaconda3/bin/python -m pytest -q tests/bse tests/solvers tests/ci \
  tests/test_tddft_eigensolvers.py tests/test_pyscf_style_excited_state_api.py --tb=short
```

**226 passed**, no skips, 3 warnings, 329.22 seconds. The warnings are the
existing GW complex-to-real reverse-pass warning and two PySCF OpenMP-availability
warnings. Independent review rechecked both original migration failures, absent
interval guards, insufficient requested roots, and cutoff-boundary behavior;
no remaining blocking issue was found. `git diff --check` passed.

This command includes the final RPA interval-option rejection tests. Earlier
checkpoint counts overlap with it. The complete repository suite, remote/GPU
backends, higher derivatives and the deferred P4/P5 interfaces were not run or
implemented by this increment.

## PySCF forward correspondence (2026-09-23)

A dedicated suite now records 31 successful comparisons against PySCF 2.9.0,
using CPU float64: matched response operators (HF/PBE/B3LYP and open-shell HF),
complete Be degenerate subspaces, CISD/UCISD with common MO integrals, and two
independently converged native HF/TDA chains. See [PYSCF_FORWARD.md](PYSCF_FORWARD.md)
for the numerical tables, reference-initialization controls, reproducible script,
raw artifact and scope limits. The production version measured is `8084557`.

The comparison uncovered a native UHF HF-string routing error; the exact zero
semilocal HF branch was added without removing nonlocal exchange. Its focused
regressions passed 13 tests. Two separately attempted pre-existing GGA-HVP tests
could not execute their kernels because optional jax_xc is missing; native
PBE/B3LYP kernel equivalence is not claimed. The report also retains an OH case
where default PySCF guesses miss a qualifying low root, rather than silently
reporting that default settings always give the same requested spectrum.
