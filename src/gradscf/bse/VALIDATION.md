# Static molecular TDA-BSE validation

Executed on 2026-09-22, macOS 26.6.2 arm64, Python 3.12, JAX 0.8.1 CPU,
float64, PySCF 2.9.0. These checks validate the P0/P1 scope in
[README.md](README.md); they do not establish full-BSE or GPU support.

## Automated checks

From the repository root, using `/opt/anaconda3/bin/python`:

```sh
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 python -m pytest -q \
  tests/bse tests/solvers \
  tests/gw/test_gw_qp.py tests/gw/test_gw_differentiability.py \
  tests/gw/test_gw_evgw.py tests/gw/test_gw_cd_rgw.py \
  tests/gw/test_gw_cd_ugw.py tests/test_pyscf_style_excited_state_api.py
```

Result: **106 passed**, no skips, 8 warnings, 182.58 seconds. This broader
regression preceded the final cache and auxiliary-block boundary refinements.
After those refinements, the final affected-area run was:

```sh
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 python -m pytest -q \
  tests/bse tests/solvers/test_block_linear.py --tb=short
```

Result: **29 passed**, no skips, 1 warning, 38.63 seconds. Counts overlap and
must not be added. Warnings are JAX `ComplexWarning` messages from the existing
GW complex-to-real reverse pass; the corresponding finite-difference check
passes. The complete repository suite was not run.

Coverage includes:

- Independent NumPy dense-loop kernels versus matrix-free actions, diagonals,
  singlet/triplet roots and strengths; bounded dense versus Davidson results.
- Separate QP/screening spectra and optical/screening windows; no-screening
  HF/CIS limit versus PySCF, and the independent-particle limit.
- First derivatives through QP energies, screening, factors, isolated
  eigenvectors and dipoles, including a fixed-MO G0W0+BSE factor perturbation.
- Invalid QP coverage/convergence, gaps, negative modes, root nonconvergence
  and exact degeneracies; energy-only mode rejects incomplete property AD.
- Eager source freshness, failed-rerun invalidation, lazy dipole completion,
  capacity rejection before MO transformation, and oversized block requests.
- Shared direct multiple-RHS solves, nonsymmetric transpose response and
  failed-solve diagnostics.

## Executed QuAcK kernel oracle

The comparison script compiles unmodified QuAcK `phRLR_A.f90` and
`RGW_phBSE_static_kernel_A.f90` with GNU Fortran 15.1.0. Revision:
`2236bfcda24ff0971358f5107636b506dd201bb1`. Source SHA256 hashes, compiler flags,
backend and errors are recorded in
[the fixture metadata](../../../tests/bse/data/quack_static_seed83.json).

```sh
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  /opt/anaconda3/bin/python tests/comparisons/compare_quack_bse.py
```

The successful rerun took 2.70 seconds using cached, SHA256-verified upstream
files. Download attempts during this rerun failed; the cached sources came
from the earlier successful retrieval. Ordinary pytest uses the committed
fixture and does not require a network connection or Fortran compiler.

The synthetic reference has seed 83, 5 auxiliary functions, 5 spatial orbitals
and 2 occupied orbitals, with distinct QP and screening energies. Screening
uses an independent NumPy **full direct-RPA pole expansion**, eta=0 and
lambda=1. It does not use TDA screening. QuAcK supplies the bare response
matrix and static screening correction; NumPy supplies poles and transition
densities. This tests independent kernel algebra, not an external SCF/GW chain.

| Maximum absolute error (Hartree) | Singlet | Triplet |
| --- | ---: | ---: |
| TDA matrix | 2.22e-16 | 2.22e-16 |
| Excitation energies | 6.66e-16 | 3.55e-15 |
| Screening correction | 1.54e-16 | 1.60e-16 |

The fixture tests use 2e-13 Hartree for matrices and 2e-12 Hartree for roots.
Attribution and primary source
links are in [REFERENCES.md](REFERENCES.md). MOLGW and VOTCA were consulted
for planning but were not executed as numerical references.

## Native water example

```sh
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  /opt/anaconda3/bin/python examples/bse/water_tda.py
```

Geometry in Angstrom: O (0,0,0), H (0,-0.757,0.587), H (0,0.757,0.587).
STO-3G, native GradSCF RHF (`xc="hf"`), SCF tolerance 1e-12, G0W0 frequency
grid `nw=100`, default GW eta=1e-3 Hartree. All seven QP levels and the full
screening space are retained; BSE uses three roots and tolerance 1e-10 Hartree.

| Root | Singlet energy (eV) | Singlet oscillator strength | Triplet energy (eV) |
| --- | ---: | ---: | ---: |
| 1 | 12.8063271021 | 0.00343541344 | 10.7065034901 |
| 2 | 15.0149160453 | 1.01e-28 | 13.4388398840 |
| 3 | 16.7514947481 | 0.0795011079 | 13.6898877491 |

All QP levels converged. Maximum reported TDA residual was 4.25e-15 Hartree;
triplet oscillator strengths were zero. Example wall time was not recorded.

At `nw=40`, one valence QP root retained a Dyson residual of 3.04e-4 Hartree
and the BSE adapter correctly rejected it. At `nw=100`, the maximum QP
residual was 2.39e-10 Hartree. A separate `nw=160` check converged all QP
levels and changed the first three singlet/triplet excitations by a few
1e-8 Hartree (comparison from printed eight-decimal Hartree values).
This is a small-basis integration example, not an accuracy benchmark against
experiment or a complete external molecular GW+BSE package.

## Original P0/P1 validation boundaries

At the original P0/P1 checkpoint, no full-BSE X/Y property response,
open-shell/periodic systems, GPU performance,
degenerate-cluster observables, higher derivatives, nuclear/basis gradients,
or complete evGW/qsGW fixed-point response is claimed. The G0W0+BSE AD test
perturbs factors in a fixed orbital frame; it does not differentiate through
an SCF orbital optimization. Resource caps bound selected array dimensions,
not measured peak memory.

## Full-BSE and GW-provenance increment (2026-09-22)

This increment adds the bounded stable full-BSE reference and first-order X/Y
response described in [FULL_BSE_PLAN.md](FULL_BSE_PLAN.md). The backend remains
macOS arm64, CPU, JAX 0.8.1, float64. The full regression command in the first
section now ran **129 tests: all passed**, no skips, 8 existing GW complex-cast
warnings, 208.84 seconds. This preceded a final reduced-residual diagnostic
refinement and two additional solver regressions; see the final affected-area
verification below. Test counts from separate runs overlap.

Executed new checks cover the coupling block, full-BSE energies and oscillator
strengths against independent doubled matrices, positive-definiteness and
metric normalization, B=0 and independent-particle limits, native H2 G0W0/evGW
screening provenance, the water HF+W=v limit against PySCF TDHF for both spins,
and joint first-order QP/screening/factor/dipole response. TDHF comparison uses
water STO-3G, the example geometry, HF tolerance 1e-13, TDHF tolerance 1e-11,
and absolute comparison tolerance 2e-9 Hartree (energy) / 2e-9 (strength).

Review and diagnostic regressions address two cases:

- A request mixing isolated and degenerate roots now invalidates the whole
  requested derivative set, consistently for JVP and VJP. A smaller isolated
  prefix recovers finite response. Forward stable roots remain available.
- The reduced squared-frequency problem and reconstructed physical problem
  must both pass their solver residual checks. Large energy scales can give
  a small physical residual but a failed absolute reduced residual; this no
  longer reports valid response while the inner solver rejects its derivative.

### Additional executed QuAcK B-block oracle

The same opt-in comparison script now also compiles and executes unmodified
`phRLR_B.f90` and `RGW_phBSE_static_kernel_B.f90` at the same pinned revision.
All six reference sources are checked by SHA256. The new files downloaded
successfully; `phRLR_B.f90` required a retry after a timeout. The existing
A-block source cache was reused. The four actual Fortran A/B kernels run with
the same independent NumPy full direct-RPA screening poles as before.

Metadata and arrays: `tests/bse/data/quack_full_static_seed83.json` and `.npz`.
Command: the `compare_quack_bse.py` invocation above. Elapsed time: 4.60 seconds.
Full-BSE oracle eigenvalues are obtained by NumPy diagonalization of the doubled
matrix made from the Fortran blocks; no QuAcK eigensolver or complete external
molecular GW+BSE driver was run.

| Maximum absolute error (Hartree) | Singlet | Triplet |
| --- | ---: | ---: |
| Coupling B matrix | 6.94e-18 | 6.94e-18 |
| Full-BSE excitation energies | 2.66e-15 | 5.33e-15 |

Normal pytest now checks both the original TDA fixture and the new full-BSE
fixture offline, at 2e-13 Hartree for B and 2e-12 Hartree for energies.

### Native water full-BSE example

```sh
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  /opt/anaconda3/bin/python examples/bse/water_full.py
```

Same RHF/STO-3G geometry and nw=100 G0W0 setup as the original TDA example.
All QP levels converged. `tda=False, solver="dense"`, three roots, physical
residual tolerance 1e-10 Hartree. Example wall time was not recorded.

| Root | Singlet energy (eV) | Singlet strength | Triplet energy (eV) |
| --- | ---: | ---: | ---: |
| 1 | 12.75085069 | 0.00311599129 | 10.66783729 |
| 2 | 14.99788371 | 3.82e-29 | 13.25957339 |
| 3 | 16.56916724 | 0.0681077152 | 13.67628826 |

Triplet strengths are zero. Maximum full-BSE physical residual: 1.72e-14
Hartree. Minima of (A-B,A+B), Hartree: singlet (0.42686559,0.51438209), triplet
(0.42686559,0.36004790). These certify the specified static BSE matrices only.
The small-basis values are integration checks, not experimental benchmarks.

Boundaries at that checkpoint: full BSE was dense and bounded, requiring stable real
closed-shell input, and supports first-order isolated-state response only.
Matrix-free full-BSE response, degenerate cluster properties, open-shell and
periodic kernels, higher/nuclear derivatives, and outer evGW/qsGW AD remain
unverified or unimplemented. There is no new scGW spectral continuation.

### Final affected-area verification

After the reduced-residual validity fix and additional solver regressions:

```sh
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  /opt/anaconda3/bin/python -m pytest -q tests/bse tests/solvers --tb=short
```

**95 passed**, no skips, one existing GW complex-cast warning, 133.84 seconds.
This overlaps the 129-test run above; counts are not additive. The full repository
suite and GPU backends were not run. Independent code review found no remaining
important issue within this bounded real stable full-BSE scope after the
mixed-degeneracy validity fix.

## Matrix-free full-BSE increment (2026-09-23)

The implementation described in [MATRIX_FREE_BSE.md](MATRIX_FREE_BSE.md)
continues from `210cfee`. Full BSE now accepts `solver="davidson"` and calls the
shared `solve_rpa` action API, including first-order isolated X/Y response.
The inherited numerical RPA projection now preserves H/J rather than applying
ordinary Galerkin projection to the nonsymmetric R matrix. Lower residual signs,
paired reorthogonalization and subspace-capacity handling are also checked.

### Numerical checks

- Dense versus matrix-free molecular BSE energies and strengths for both spins;
  the existing saved QuAcK A/B fixture is checked by both solver paths. This
  stage reuses the previously executed Fortran fixture; it does not claim a new
  external molecular calculation.
- JIT, JVP and VJP optical-response checks against dense response and reconverged
  finite differences, including explicit QP/screening/factor/dipole perturbations.
- General positive-definite M=A-B/P=A+B matrices with indefinite B, n=10,
  seeds 0–3, 8-dimensional subspaces, 200-cycle limit, 1e-9 residual tolerance.
  All four cases converge after the metric-preserving projection correction.
  Independent review observed energy differences below 7e-16 Hartree and metric
  normalization errors below 3e-16 against the dense reference.
- Degenerate requested roots, failed iterations, unstable matrices and both
  invalid JVP/VJP directions. The invalid-state mask must not turn a failed
  derivative into a finite zero.
- Recursive forward/reverse JAXPR inspection for a 300-dimensional action:
  no (300,300) or (600,600) physical arrays. Callback tests also bound operator
  application width; the dense reference capacity can be set below ntrans
  without affecting the matrix-free calculation.

### Reproducible synthetic response run

```sh
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  /opt/anaconda3/bin/python tests/comparisons/compare_matrix_free_rpa.py
```

CPU, macOS 26.6.2 arm64, JAX 0.8.1, float64. Seed 41, n=300,
`A=diag(linspace(1,4))+U U.T`, U rank 2 with normal scale .002, `B=.05 I`,
2 requested roots, `max_subspace=8`, physical residual tolerance 1e-9 Hartree.
The differentiated scalar t multiplies U; the objective is the sum of the two
frequencies plus `sum(X[:3,:]**2)`. Both energies and X contribute to its AD.

Measured first JIT compilation plus value/gradient execution: **3.17 seconds**.
Process peak RSS through that call: **468,025,344 bytes** (macOS `ru_maxrss`),
including Python/JAX/compiler overhead. It is not isolated solver-array memory,
not a measured scaling law, and not a GPU or molecular benchmark.

| Quantity | Value |
| --- | ---: |
| First two frequencies (Hartree) | 0.9987555663712485, 1.0087961077630752 |
| Largest requested residual (Hartree) | 7.65e-10 |
| Objective | 4.008791515787621 |
| AD derivative at t=1 | 1.4273521769220886e-5 |
| Central difference, step 1e-4 | 1.4273520143603946e-5 |
| Absolute derivative difference | 1.63e-12 |

Both requested roots and derivative prerequisites passed. As intended for
iterative stability screening, `stability_certified=False`.

### Native integration

`examples/bse/water_full.py` now runs TDA, dense full BSE and matrix-free full
BSE using one native RHF/G0W0 calculation with the same STO-3G water geometry
and nw=100 grid as above. Both full-BSE methods print matching energies to
8 decimal eV places. Matrix-free singlet strengths are
`[0.00311599129, 3.53e-29, 0.0681077152]`; triplet strengths are zero. Maximum
matrix-free physical residual is 2.33e-15 Hartree. Dense certification and
iterative stability screening are printed separately.

No large molecular spectrum, GPU performance, global iterative stability proof,
complete degenerate-subspace response, higher derivatives, or full outer GW
self-consistency derivative is established by these checks. Native basis and
nuclear derivative contracts are unchanged.

### Matrix-free regression gate

```sh
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  /opt/anaconda3/bin/python -m pytest -q tests/bse tests/solvers \
  tests/test_tddft_eigensolvers.py tests/test_pyscf_style_excited_state_api.py --tb=short
```

**140 passed**, no skips, one existing GW complex-cast warning, 251.52 seconds.
After broadening the general-SPD regression to all four independently reviewed
seeds and allowing NumPy integer seeds, the final affected-area run was:

```sh
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  /opt/anaconda3/bin/python -m pytest -q tests/solvers/test_matrix_free_rpa.py \
  tests/bse/test_matrix_free.py tests/bse/test_quack_fixture.py --tb=short
```

**21 passed**, no warnings/skips, 76.87 seconds. These counts overlap and are
not additive. All 21 tests in the existing TDDFT eigensolver file were included
in the broader run. The complete repository suite and GPU backends were not run.
The synthetic run metadata is retained in
`tests/bse/data/matrix_free_rpa_300_cpu.json`; it is a measurement record, not
an external physical reference fixture.


## Screening and kernel memory update (2026-09-24)

A/B actions now screen contracted trial vectors or bounded factor slabs instead
of retaining full screened pair factors. The optional GMRES screening mode uses
an auxiliary operator and shared sequential multiple-RHS implicit solves. Direct
screening remains the default. Existing bare MO factors remain resident.

New checks cover independent A/B actions and diagonals, vector and block linear
transposes, TDA/full-BSE optical JVP/VJP through dense and Davidson roots, empty
screening/auxiliary spaces, explicit GMRES failure and invalid optical access,
and recursive forward/reverse JAXPR storage inspection. Narrow and wide trial
blocks must have no auxiliary-square matrix or full screened virtual-pair array.

A 384-auxiliary regression exposed a stopping-metric issue with left diagonal
preconditioning: a B-kernel RHS had a true residual 1.052 times the configured
limit. Without that optional preconditioner its ratio is 0.787. Screening now
uses unpreconditioned GMRES, retains the original strict physical-residual check,
and rejects failed columns. The shared scalar solver was not modified to relax
its convergence criteria.

### Synthetic action and gradient measurements

Command:

```sh
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  /opt/anaconda3/bin/python tests/comparisons/compare_bse_screening_memory.py
```

macOS 26.6.2 arm64 CPU, Python 3.12.2, JAX 0.8.1, float64, seed 701,
6 occupied orbitals, orbital block size 8. Factors are symmetric normal random
arrays with scale .03. The objective is `||(A+B)x||^2` for a fixed random probe;
the differentiated input is the full symmetric MO factor tensor. This is a
synthetic kernel/screening measurement, not SCF/GW or a molecular spectrum.

Baseline is release commit `13610c4`, using the same measurement function.
Each row runs in a fresh process. XLA temporary storage is from
`compiled.memory_analysis()`; peak RSS includes Python, compilation and runtime.
All MB below are decimal. Unrelated host CPU activity was present, so timings
are indicative rather than a controlled throughput benchmark.

| naux / nvir | Implementation | XLA temporary MB | Peak RSS MB | Steady value+grad seconds |
| --- | --- | ---: | ---: | ---: |
| 192 / 40 | baseline direct | 83.759 | 596.984 | 0.01427 |
| 192 / 40 | streamed direct | 12.689 | 377.635 | 0.00399 |
| 192 / 40 | streamed GMRES | 13.213 | 679.903 | 0.26838 |
| 384 / 64 | baseline direct | 734.992 | 1242.169 | 0.17096 |
| 384 / 64 | streamed direct | 59.139 | 588.349 | 0.04042 |
| 384 / 64 | streamed GMRES | 67.706 | 1008.976 | 1.58818 |

The larger streamed direct case reduces compiled temporary storage by 92.0%
and measured process peak RSS by 52.6%. Its objective equals the baseline at
printed precision; the maximum componentwise factor-gradient error is
8.53e-14. For GMRES, the larger objective error is 1.40e-10, maximum gradient
component error 3.60e-10, and relative gradient L2 error 8.80e-13. The objective
and its factor-scale directional derivative carry squared-Hartree units; they
are not excitation-energy error estimates.

GMRES removes the auxiliary-square array, but is slower and has higher total
RSS than streamed direct in both measured cases. It stays opt-in. These data do
not establish GPU performance, out-of-core scaling, or a reduction in the
memory of the preceding GW calculation. Raw settings, measurements and exact
baseline commands are in `tests/bse/data/screening_memory_cpu.json`.

### Native integration

The extended `examples/bse/water_full.py` runs one native RHF/STO-3G G0W0
calculation (`nw=100`) followed by TDA and full BSE, including Davidson with
GMRES screening. Geometry, SCF and BSE tolerances match the prior water example.
QP levels converge. Direct and GMRES screening print identical full-BSE roots
to 8 decimal eV places for singlets and triplets; singlet strengths are
`[0.00311599129, ~4e-29, 0.0681077152]` and triplet strengths are zero.
The largest GMRES-screened physical root residual is 9.71e-16 Hartree.
Iterative stability screening still reports `stability_certified=False`.

### Regression result

After the stopping-metric correction and final source changes:

```sh
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  /opt/anaconda3/bin/python -m pytest -q tests/bse \
  tests/solvers/test_block_linear.py tests/solvers/test_shared.py --tb=short
```

**77 passed**, no skips, one existing GW complex-to-real reverse-pass warning,
353.69 seconds. After extending storage inspection to both narrow and wide
trial blocks, the focused command
`python -m pytest -q tests/bse/test_screening_memory.py -k auxiliary_and_screened_pair_storage`
with the same environment gave **2 passed, 11 deselected**, 11.95 seconds.
These counts overlap. Independent read-only code review found no blocking
issues in the contractions, transposes, failure propagation or memory claims.
The full repository suite and GPU execution were not run.


## Scoped release integration (2026-09-30)

The GW/BSE changes from `32b279e` were ported onto release base `ff1a07e`.
The only change outside GW/BSE source, examples and dedicated tests is the
required multiple-RHS extension in `solvers/linear/implicit.py`, together with
its existing block-solve regression. Current scalar GMRES precision handling
and the simplified public namespaces are preserved. No SCF, DFT, CI, CC,
training, OFDFT or periodic-method source files are changed.

The 2026-09-24 performance table above is historical; its timing and memory
measurements were not repeated for this integration. Default screening stays
dense/direct; GMRES is explicitly selected. Full bare MO factors remain resident.

The native water example was rerun with CPU float64, JAX 0.8.1, the same
RHF/STO-3G geometry and `nw=100`. Dense/direct, Davidson/direct and
Davidson/GMRES full-BSE energies agree at the printed 8 decimal eV places:
singlets `[12.75085069, 14.99788371, 16.56916724]` and triplets
`[10.66783729, 13.25957339, 13.67628826]`. The largest GMRES-screened
root residual is `9.71e-16 Ha`; iterative stability remains uncertified.

```sh
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 python examples/bse/water_full.py
```

Fresh integration checks (CPU float64, Python 3.12.2, JAX 0.8.1):

```sh
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 python -m pytest -q \
  tests/solvers/test_block_linear.py tests/bse/test_screening_memory.py
# 15 passed, 375.53 seconds

PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 python -m pytest -q \
  tests/bse --ignore=tests/bse/test_screening_memory.py \
  tests/solvers/test_shared.py tests/solvers/test_gmres_accuracy.py
# 65 passed, 148.49 seconds; one existing GW complex-to-real AD warning
```

These two sets are disjoint: 80 tests passed. They include isolated-root optical
JVP/VJP, finite differences, shared nonsymmetric block transposes, failed-solve
propagation, true per-column residuals and forward/reverse storage checks.
The broader repository suite and GPU backends were not run.


## Molecular MolGW comparison and controls (2026-09-30)

This is an executed external molecular HF -> GW -> BSE comparison, distinct
from the earlier synthetic QuAcK A/B oracle. MolGW 3.4 was built from unmodified
tracked sources at `b831818d7a845c36f295d036dc8ceef59daf9991`. The comparison
uses MolGW's analytic spectral self-energy and graphical QP roots against
GradSCF's contour-deformation implementation. Source and executable hashes,
complete inputs, external values, deviations and per-case times are recorded in
[the committed JSON](../../../tests/bse/data/molgw_molecular.json).

Seven cases use HF/STO-3G: H2 G0W0/evGW0/evGW; water G0W0 with all states,
with its core excluded only from W, with that core excluded from both G/W,
and with one virtual removed only from W using matched Weigend RI. Each case
compares singlet TDA, singlet full BSE and triplet full BSE (21 optical runs).
H2 is at 0.74 Angstrom. Water has O=(0,0,0), H=(0,+/-0.757,0.587) Angstrom.
The optical space matches MolGW's screening window for this comparison; the
GradSCF API also permits independent optical windows.

Both codes use Cartesian orbitals, SCF tolerance 1e-12, and no implicit
frozen-core rule. The first six cases use full ERIs; the RI case uses the same
Weigend auxiliary basis. GradSCF uses nw=200 and eta=1e-5 Ha for GW; MolGW's
self-energy grid has 40001 points at step 5e-5 Ha and the same eta. evGW/evGW0
use 40 MolGW steps and a GradSCF residual target 1e-10 Ha. Optical eta is .01 Ha.
MolGW's documented 27.21138505 eV/Ha is used to parse its output, separately
from GradSCF's physical constants. All energies below are compared in Ha.

| Maximum absolute difference, all applicable cases | Value |
|---|---:|
| HF total energy (Ha; MolGW YAML is rounded) | 2.97e-8 |
| Computed QP energy (Ha) | 7.79e-9 |
| Local QP weight Z (dimensionless; G0W0 cases) | 1.29e-5 |
| BSE excitation energy (Ha) | 1.76e-8 |
| Oscillator strength | 1.34e-8 |
| Static polarizability tensor (a0^3) | 2.87e-8 |
| Sampled average absorption cross section (a0^2) | 1.77e-6 |

Measured case times sum to 37.07 s on macOS arm64 CPU, Python 3.12.2,
JAX 0.8.1, float64. Times include GradSCF compilation/calculation, MolGW
execution and I/O, not source compilation; they are not throughput benchmarks.
The finite STO-3G orbital spaces are implementation checks, not spectroscopy
accuracy claims. GPU, large molecules and unrestricted BSE were not benchmarked.

### Numerical settings discovered during comparison

- MolGW estimates graphical Z from adjacent self-energy grid points. At .001 Ha
  spacing the water error was about 2.45e-4; decreasing spacing to 5e-5 Ha brought
  it below the preselected 5e-5 tolerance. GradSCF uses the local AD derivative.
  We refined the independent reference rather than loosening the tolerance.
- MolGW's no-RI route also truncates its W product representation with its
  virtual cutoff. Tests with that route did not reproduce the intended
  independent W-transition-only model (observed QP differences up to .021 Ha).
  The independent virtual-window reference therefore uses RI. That no-RI
  cutoff case is not certified by these results or silently matched by
  changing GradSCF's window semantics. The remaining no-RI cutoff discrepancy
  was not resolved in this increment; independent virtual-window agreement
  is established by the RI case.
- MolGW discards Coulomb-metric modes below 1e-6. GradSCF's existing Cholesky
  whitening only uses its eigenvalue cutoff when Cholesky fails. The tested
  cc-pVDZ-RI metric had a 6.52e-7 mode, producing a 4.25e-5 Ha HF discrepancy.
  Weigend's smallest mode is 1.30e-5, so neither code drops a mode and the
  discrepancy vanishes. No integral-module behavior was changed here.

### Reproduction and offline tests

The standalone comparison requires a prebuilt executable and its exact source
checkout; it checks the pinned revision and that tracked sources are unmodified.
Each case directory must be empty to avoid stale RESTART/SCREENED_COULOMB files.
It writes no reference values unless MolGW actually executes successfully.

```sh
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
python tests/comparisons/compare_molgw_gw_bse.py /path/to/molgw \
  --source /path/to/molgw-source --workdir /tmp/fresh-molgw-cases \
  --output /tmp/molgw-results.json

PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
python -m pytest -q tests/bse/test_molgw_fixture.py
```

The offline fixture test reruns GradSCF for the seven recorded cases and checks
external values, not the saved GradSCF predictions: **7 passed in 34.21 s**.
It needs neither MolGW nor a Fortran compiler.

For this macOS reference build, GNU Fortran 15.1, Accelerate LAPACK/BLAS,
Homebrew LibXC and libcint were used. libcint was built at
`3d36c4f4e24ca5aaf91be7299f93dd541db0f50b`, with `WITH_CINT2_INTERFACE=ON`
and `WITH_FORTRAN=OFF`; PySCF's bundled library did not export MolGW's legacy
CINT2 symbols. `FCFLAGS=-cpp -O1 -ffree-line-length-none -fallow-argument-mismatch`.
Inherited Conda `LIBTOOL` and `LDFLAGS` were cleared in `my_machine.arch` to
avoid conflicting Fortran runtimes. The temporary libcint directory was supplied
through `DYLD_LIBRARY_PATH`. No numerical MolGW source was patched, and no
system-wide library or GradSCF integral backend was modified.


### Final affected-area verification

On the same CPU/float64 environment, the molecular-control, optical-response,
QP, evGW, G0W0 AD, BSE API/provenance/QuAcK, UGW and two complex Gamma/k-point
compatibility checks gave **63 passed in 90.73 s**, with 10 complex-to-real AD
warnings. Together with the disjoint seven-case offline MolGW fixture test,
**70 tests passed**. Older run counts above overlap and must not be added.
The new native `examples/bse/molecular_spectrum.py` was executed separately.

```sh
PYTHONPATH=src:tests/gw:tests/bse JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
python -m pytest -q --import-mode=importlib \
  tests/gw/test_molecular_controls.py tests/bse/test_optical_spectrum.py \
  tests/gw/test_gw_qp.py tests/gw/test_gw_evgw.py tests/gw/test_gw_differentiability.py \
  tests/bse/test_api.py tests/bse/test_gw_reference.py tests/bse/test_quack_fixture.py \
  tests/gw/test_gw_cd_ugw.py \
  tests/gw/pbc/test_kpoint_invariants.py::test_gamma_matches_single_k_with_complex_orbitals \
  tests/gw/pbc/test_kpoint_invariants.py::test_gamma_unrestricted_preserves_independent_spin_phases
```

Read-only review identified and resolved mutable-window signature aliasing and
stale MolGW W-file reuse. Regression tests cover immutable BSE snapshots and
in-place edits to a facade's assigned window list. A direct preflight check
confirmed a directory containing only stale SCREENED_COULOMB is rejected
before any reference execution. No SCF, integral or shared solver sources
were changed in this feature increment.


## Expanded molecular MolGW comparison (2026-09-30)

The additional executable reference results are recorded separately in
`tests/bse/data/molgw_expanded.json`. The pinned MolGW/libcint revisions,
CPU float64 environment and numerical tolerances are unchanged. Eight cases
all passed, each with TDA singlet, full-BSE singlet and full-BSE triplet:
H2/6-31G, water/6-31G, NH3/STO-3G, CH4/STO-3G, N2/STO-3G,
ethylene/STO-3G, and water/STO-3G evGW0 and evGW. The remaining cases use G0W0.
The summed per-case elapsed time was 101.68 s (macOS arm64, Python 3.12.2,
JAX 0.8.1 CPU). This is an implementation comparison in small bases.

| Observable | Maximum absolute error | Tolerance |
| --- | ---: | ---: |
| RHF energy / Ha | 4.71e-10 | 1e-7 |
| QP energy / Ha | 1.94e-8 | 2e-6 |
| QP weight Z | 4.45e-5 | 5e-5 |
| BSE energy / Ha | 2.88e-8 | 3e-6 |
| Oscillator strength | 3.05e-7 | 2e-5 |
| Static polarizability / a0^3 | 1.37e-7 | 3e-4 |
| Sampled cross section / a0^2 | 6.57e-6 | 2e-4 |

The extension exposed four distinct issues:

- Molecular evGW evaluated self-energies exactly at their Green-function
  poles. A strict residue mask omitted the boundary contribution, producing
  water QP errors of 0.226/0.229 Ha (evGW0/evGW). Analytic static-W subtraction
  now includes the half-residue limit. Both water errors fell below 4e-9 Ha.
  Independent single-RPA-pole tests check occupied/virtual values, frequency
  slopes, explicit empty/full-channel occupation signs and joint parameter
  JVP/VJP. Periodic q0 paths are unchanged. The subsequent matrix qsGW
  update also applies static-W subtraction; the diagonal comparison above
  does not independently certify that matrix path.
- Dense BSE inherited Davidson's default 40-vector subspace limit. The dense
  path now ignores that irrelevant limit, while retaining `max_dense`.
  Ethylene's 48-state reference and separate dense/Davidson cap tests pass.
- N2's hcore guess converged to a higher RHF stationary branch. Its negative
  BSE modes were physical instability diagnostics, not solver failures.
  The catalog explicitly uses `multistart(amplitudes=(.15,), seed=20260923)`;
  the selected energy is -107.49650051179786 Ha and its internal RHF curvature
  is positive. No SCF defaults or stability rejection rules were changed.
- Water/6-31G requires a finer MolGW graphical self-energy grid for Z:
  400001 points at 5e-6 Ha spacing, retaining the same +/-1 Ha interval.
  The Z discrepancy decreased from 4.02e-4 to 4.45e-5. This refines the
  independent finite-difference reference; the tolerance was not relaxed.

Run the executable suite in a fresh directory:

```sh
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
python tests/comparisons/compare_molgw_gw_bse.py /path/to/molgw \
  --source /path/to/molgw-source --suite extended --keep-going \
  --workdir /tmp/fresh-molgw-expanded --output /tmp/molgw-expanded.json
```

The combined offline fixtures (original seven plus eight new cases) passed
15 tests in 119.40 s. The final combined regression below passed **78 tests
in 210.82 s**, with 16 existing complex-to-real AD warnings and no skips.
The separate restricted PySCF GW comparison passed 2 tests in 35.87 s.
These focused checks do not certify the complete repository or GPU behavior.

```sh
PYTHONPATH=src:tests/gw:tests/bse JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
python -m pytest -q --import-mode=importlib \
  tests/gw/test_contour_pole_limit.py tests/gw/test_gw_evgw.py \
  tests/gw/test_gw_differentiability.py tests/gw/test_gw_qp.py \
  tests/gw/test_molecular_controls.py tests/gw/test_qp_methods.py \
  tests/bse/test_molgw_fixture.py tests/bse/test_dense_capacity.py \
  tests/gw/test_gw_cd_ugw.py \
  tests/gw/pbc/test_kpoint_invariants.py::test_gamma_matches_single_k_with_complex_orbitals \
  tests/gw/pbc/test_kpoint_invariants.py::test_gamma_unrestricted_preserves_independent_spin_phases
```


## Diffuse-basis roundoff and methane spectra (2026-09-30)

The functional BSE entry point now canonicalizes admissible roundoff in real
MO factors once, before constructing screening and both kernel operators.
The factor antisymmetry is checked first using a contraction-dimension-scaled
machine-precision bound. Larger errors remain invalid. Standalone screening
validation and all shared eigensolver structure/stability thresholds are
unchanged. This replaces the provisional screening-only threshold adjustment.

A 61-MO synthetic regression failed before the repair and passes afterward;
it verifies forward roots, JIT reverse gradients, JVP, finite differences,
symmetric factor gradients and rejection of a 1e-5 asymmetric perturbation.
The focused full-BSE/roundoff tests passed **12 tests in 25.30 s**. The disjoint
API/optical/matrix-free/MolGW-fixture suite passed **34 tests in 183.67 s**, with
two existing complex-to-real AD warnings: **46 tests total**, no skips.

```sh
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 python -m pytest -q \
  tests/bse/test_screening_roundoff.py tests/bse/test_full.py
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 python -m pytest -q \
  tests/bse/test_api.py tests/bse/test_optical_spectrum.py \
  tests/bse/test_matrix_free.py tests/bse/test_molgw_fixture.py
```

Native methane/Cartesian aug-cc-pVDZ/Weigend-RI G0W0, evGW0 and evGW spectra
were executed with all G/W states but an explicitly restricted QP/optical
window. All 145 selected full singlet BSE roots converged for all three
methods; the largest physical residual was 1.802e-13 Ha. See the
[archived results and convergence limitations](../../../reproducibility/gw_bse/methane_aug_cc_pvdz/README.md).
These results do not certify full-space evGW, optical-window/basis convergence,
experimental agreement, or evGW outer-fixed-point derivatives.


## Unrestricted BSE and molecular GW outer response (2026-10-04)

Environment: Apple M4 Pro, macOS arm64, Python 3.12.2, JAX 0.8.1,
CPU float64. Native GradSCF CPU integrals were built locally with
`PYTHONPATH=src python -m gradscf.integrals._native.build`.
No GPU or complete repository test suite was run. Times below are pytest
wall times with other local validation work, not performance benchmarks.

Final affected regression:

```sh
PYTHONPATH=src JAX_PLATFORMS=cpu python -m pytest -q \
  tests/gw/test_gw_outer_response.py tests/gw/test_gw_evgw.py \
  tests/gw/test_gw_qsgw.py tests/gw/test_gw_cd_ugw.py \
  tests/gw/test_gw_scgw_response.py tests/bse/test_unrestricted.py \
  tests/bse/test_tda.py tests/bse/test_full.py tests/bse/test_gw_reference.py \
  tests/bse/test_screening_roundoff.py
```

**94 passed**, 27 real/complex cotangent projection warnings, 401.19 s.
Complex molecular evGW physical inputs are separately tested to raise before
any real cast; these warnings arise in differentiation of complex self-energies
with real external parameters. The tested derivatives agree with reconverged
finite differences.

An earlier focused run of `tests/bse/test_unrestricted.py`,
`tests/bse/test_api.py`, and `tests/bse/test_screening_memory.py`, with
`-k 'not large_auxiliary'`, completed **36 passed, 1 deselected**, 533.23 s.
The excluded case is the existing 384-auxiliary-function physical-residual
stress test. Counts overlap and must not be added. Later input/provenance
repairs passed a focused 8-case regression and are covered by the final run.

New numerical evidence includes:

- Independent NumPy scalar-loop unrestricted A/B and dielectric matrices,
  with direct and GMRES screening, checked to 2e-12 Ha/matrix units.
- Restricted singlet/triplet limits of unrestricted TDA and full BSE, including
  oscillator strengths and their spin normalization.
- Open-shell optical JVP/VJP including the beta screening response; empty
  explicit-reference spin blocks, paired windows and coverage/status guards.
- Restricted/unrestricted evGW and evGW0 outer JIT response, compared with
  separately reconverged central finite differences (step 1e-4).
- Restricted qsGW orbital/density and energy response, AO factor perturbations,
  non-Euclidean initial frames and pure frame rotations; explicit degeneracy
  rejection and nonconverged/singular response behavior.
- Unrestricted G0W0/evGW/evGW0 -> BSE excitation-plus-strength gradients, with
  distinct alpha/beta frames and nocc=(2,1), compared with reconverged finite
  differences (absolute tolerance 3e-8, relative tolerance 3e-5).

Executed native examples (no PySCF SCF or command-line parser):

```sh
PYTHONPATH=src JAX_PLATFORMS=cpu python examples/bse/oh_unrestricted.py
PYTHONPATH=src JAX_PLATFORMS=cpu python examples/gw/implicit_response.py
```

OH, 0.9697 Angstrom, STO-3G, UHF, evGW0, nw=48, QP tolerance 1e-8 Ha:
QP levels 3/4/5 are computed; optical occupied windows ((4,),(3,)) and virtual
windows ((5,),(4,5)) use full reference screening. TDA roots are
0.28222595 and 11.44647563 eV; strengths are 0.00000000 and 0.00022168.
The maximum QP residual is 6.4076e-9 Ha; BSE residuals are below 1.2e-16 Ha.

H2, 0.74 Angstrom, STO-3G, nw=48: derivative of HOMO energy with respect to
an AO Coulomb-factor scale at fixed starting HF arrays (Ha per unit scale):

| Method | Implicit AD | Reconverged central difference |
| --- | ---: | ---: |
| evGW | 1.2977251783 | 1.2977251714 |
| evGW0 | 1.2935202972 | 1.2935202908 |
| qsGW | 1.2977251783 | 1.2977251714 |

These are implementation/derivative checks, not experimental accuracy or a
nuclear-gradient validation. Actual outputs are also comments at the ends of
the example files. The first unconfigured example launch selected the local
experimental Metal plugin and failed during device initialization; the CPU
commands above completed successfully.

Scope: real collinear spin-conserving BSE, isolated optical roots; no spin-flip,
SOC or excited-state S^2 diagnostic. qsGW outer response currently requires all
orbital levels isolated; UGW requires occupied and virtual orbitals in both
spins. Unrestricted fixed-W0 arrays are recomputed per outer step. Periodic
outer response and Matsubara analytic continuation were not extended.
