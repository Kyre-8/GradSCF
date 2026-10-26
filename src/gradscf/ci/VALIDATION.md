# CI validation

## Compact direct actions, bounded Davidson and molecular gradients — 2026-10-20

All numerical work in this increment ran on `jy_gpu_1`, CPU float64, with
`CUDA_VISIBLE_DEVICES=''` and `JAX_PLATFORMS=cpu`; GPU6 was never used.
The environment was Intel Xeon Platinum 8457C, Python 3.11.15, JAX/JAXlib
0.10.0 and PySCF 2.13.0. CI action/solver checks used CPU affinity 32–47
and OMP/OpenBLAS/MKL environment limits of 8; some existing PySCF fixtures
further reduce their own thread count. The independent nuclear-gradient
check used affinity 48–63 with the same thread limits.

The production CI action now uses compact particle/hole strings and tiled
one-/two-body contractions. The retained coefficient space is unchanged.
Slater–Condon connections remain an independent small-space oracle. Static
input/output rank limits apply to the complete operator, `P_out H P_in`;
the transpose exchanges the two projectors. A selected CID test explicitly
checks a double coupling that would be lost by multiplying two independently
projected single-excitation operators.

Recorded incremental runs are listed separately because they contain repeated
checks and different snapshots; their counts are not a unique aggregate suite.

| Increment | Measured result | Covered behavior |
| --- | --- | --- |
| Initial direct-action integration | 20 passed, 353.73 s | R/U sigma and diagonal comparisons, ranks 1–3, CID/frozen spaces, integral VJPs, fermion phases, one-body response/transpose, workspace guards, native N2/STO-3G forward |
| Static rank limits and focused regressions | 12 passed, 316.67 s | Eight `P_out H P_in`/transpose/diagonal cases, two occupied/virtual-block projections, restricted CI coefficient response and UCIS/TDA |
| Bounded basis and exact reference guesses | 4 passed, 403.57 s | Exact diagonal CI with one Davidson cycle; no default determinant-square basis in forward/backward Jaxprs; invalid energy/coefficient derivatives remain NaN |
| UCIS after initialization update | 1 passed, 16.25 s | The no-reference-index guess path retains UHF/TDA excitation energies |
| Independent CISD nuclear gradient | 1 passed, 208.04 s | Native moving integrals, smooth implicit HF frame, CI energy gradient, PySCF analytic gradient and reconverged finite differences |
| Large-basis native CISD nuclear gradient | 1 passed, 754.92 s | N2/Cartesian def2-TZVP, 295,296 determinants, native HF and moving integrals; independent analytic gradient and two reconverged finite differences |

The broad `tests/ci/test_ci.py tests/ci/test_uci.py` attempt, together with
the two block-projection cases, reached its 600 s budget after 15 completed
cases and exited 124. No failing test was printed, but the batch is incomplete.
Its completed cases are not added to the table as a successful suite. The
later narrow runs above completed with exit status 0. No entire-project,
entire-CI or GPU validation is claimed.

The two focused final test selections were:

```bash
python -m pytest -q \
  tests/ci/test_direct_action.py::test_static_input_output_rank_projection_and_transpose \
  tests/ci/test_direct_action.py::test_one_body_block_diagonal_projection_and_response \
  tests/ci/test_ci.py::test_implicit_ci_coefficient_response \
  tests/ci/test_uci.py::test_ucis_against_uhf_tda

python -m pytest -q \
  tests/ci/test_direct_action.py::test_diagonal_ci_keeps_exact_reference_guess_with_one_cycle \
  tests/ci/test_direct_action.py::test_default_ci_davidson_has_bounded_basis_in_forward_and_backward \
  tests/ci/test_ci.py::test_unconverged_eigenvalue_derivative_is_not_silently_zero \
  tests/ci/test_ci.py::test_unconverged_coefficient_response_is_invalid
```

### N2/STO-3G: occupied/virtual degeneracy and a nuclear gradient

`tests/ci/test_geometry.py::test_nitrogen_cisd_geometry_matches_pyscf_gradient`
uses N atoms at `(0,0,±1.04)` bohr, Cartesian STO-3G, all 14 electrons,
10 MOs and 610 CISD determinants. The native RHF seed is `1.4*I`, selecting
the usual closed-shell branch; the default hcore seed can converge to a
different, higher stationary solution. RHF energy/density/orbital-gradient
tolerances are `1e-13`, `1e-11` and `1e-10`; CISD uses residual tolerance
`1e-11`, 150 cycles and a 32-vector bound.

The path is native moving integrals → smooth implicit HF orbital frame →
shared AO-to-MO transformation → direct CI/shared Hermitian solver. The
occupied and virtual pi degeneracies do not require canonical eigenvector
derivatives. The CI ground state remains isolated.

| Quantity | Measured value |
| --- | ---: |
| CISD total energy | -107.642011614691 Ha |
| Energy difference from PySCF | 2.842e-13 Ha |
| Bond-direction energy derivative | -0.2608582717 Ha/bohr |
| Maximum Cartesian-gradient difference from PySCF analytic CISD | 5.789e-10 Ha/bohr |
| Maximum bond-direction finite-difference error | 1.127e-7 Ha/bohr |

Centered, reconverged displacements are `3e-4` and `1e-4` bohr along
`[(0,0,-.5),(0,0,.5)]`. The test also checks translational invariance.
Recorded gradient lowering/compilation/execution times are 76.235/70.403/
0.233 s; forward lowering/compilation and four displaced executions take
9.990/45.483/0.519 s. Peak process RSS is 5,153,688 KiB. The archived
source-before/source-after comparison reports zero changed files.

### N2/Cartesian def2-TZVP: 295,296 determinants

`tests/ci/benchmark_direct_action.py` uses the same nuclear coordinates,
72 Cartesian MOs, 7 occupied spatial orbitals, all electrons and no frozen
orbitals. PySCF supplies one common RHF/MO Hamiltonian and independent MP2/
CISD references. HF and integral-transform timings therefore belong to the
external reference; the measured GradSCF phases are the direct sigma action
and CI solve. Source hashes, affinity, versions, phase status and peak RSS
are saved with each result. Each job had a 1200 s limit.

The sigma-only run completed with exit 0 in 40.51 s:

| Phase or check | Measured value |
| --- | ---: |
| Retained determinant count | 295,296 |
| Host topology enumeration | 0.200 s |
| Operator-build dispatch | 26.280 s |
| First sigma, including pending build work | 10.290 s |
| Three warm sigma evaluations | 0.1993 / 0.2007 / 0.1981 s |
| Warm sigma median | 0.1993 s |
| Peak process RSS | 3,301,820 KiB (3.15 GiB) |
| MP2 from all normalized HF-to-determinant couplings | -0.4173335464947632 Ha |
| Independent PySCF MP2 | -0.4173335464947643 Ha |
| Absolute MP2 difference | 1.110e-15 Ha |

The direct action stores neither determinant-pair connections nor a full FCI
product vector. The random-vector Hermiticity defect in the diagnostic full
run was `1.18e-16` for a unit-scale bilinear probe.

The first full solve exposed a separate initialization issue. With the shared
default mixed unit/random guesses, the 150-cycle, 40-vector solve remained
unconverged. A diagnostic repeat returned energy `-100.54689075871114` Ha,
true residual `0.2687591964` and reference weight `1.80e-10`; PySCF converged
to `-109.37378941311151` Ha. The failed reports retain `complete=false`.

CI now passes exact reference/low-diagonal unit guesses plus separate
full-support auxiliary random columns through the existing shared solver's
`initial_vectors` argument. The default basis is bounded by
`min(ndet, max(40, 4*(nroots+1)))`; `None` no longer requests an
unbounded determinant-square allocation in the CI orchestration. The
shared solver and physical CI equations are unchanged.

The corrected full run used the same 150-cycle/40-vector controls and
`1e-10` residual tolerance, with exit 0:

| Quantity | Measured value |
| --- | ---: |
| GradSCF CISD energy | -109.37378941311160 Ha |
| PySCF CISD energy | -109.37378941311135 Ha |
| Absolute energy difference | 2.416e-13 Ha |
| Independently recomputed true residual | 4.678e-11 |
| Reference determinant weight | 0.9092807810 |
| Coefficient norm | 1.0000000000000007 |
| GradSCF CI phase after sigma warmup | 78.125 s |
| PySCF CI phase | 1.192 s |
| Complete job wall time | 120.97 s |
| Peak process RSS | 6,276,332 KiB (5.99 GiB) |

Both programs converged. The energy threshold remained `2e-9` Ha, and the
GradSCF residual threshold remained `1e-10`; neither was relaxed to obtain
agreement. The phase timings include different compilation/initialization
work, so they are recorded with their exact scope. This run demonstrates
large-space forward feasibility and accuracy; its nuclear gradient was not
calculated. Source-before/source-after hashes match for every recorded
numerical source and benchmark file, and no swap was recorded.

Local evidence is under `artifacts/ci_mp_validation/ci/`, with the independent
geometry check under `artifacts/ci_mp_validation/ci_geometry/`. Earlier failed
full-solve records remain in `ci/full_first_attempt/` and
`ci/full_diagnostic_attempt/`.

### Native N2/Cartesian def2-TZVP nuclear gradient

`tests/ci/benchmark_geometry.py::test_nitrogen_def2_tzvp_cisd_geometry`
explicitly invokes the same helper as the STO-3G regression, with the same
coordinates and 295,296-determinant space as the large forward benchmark.
The filename excludes it from default `test_*.py` collection. This calculation
uses GradSCF's own native RHF and moving integrals throughout; PySCF supplies
an independently converged RHF/CISD energy and analytic gradient. Its orbitals
are not substituted into the GradSCF path. The native initial density is
`[nelectron / Tr(S)] I`, sized from the actual AO overlap; its scale is
`0.1676016641882398` for this 72-AO basis. Native and independent HF energies
differ by `2.842e-13` Ha, and the minimum internal MO energy separation is
`4.441e-16` Ha. The first nuclear derivative therefore includes the occupied/
virtual pi degeneracies of the native HF frame at this larger scale.

The shared helper consumes packed or full native AO integrals through the
existing AO-to-MO API. HF tolerances match the small-basis test; CISD uses
`1e-10`, 150 cycles and 40 Davidson vectors. The numerical thresholds remain
`3e-9` Ha for energy, `5e-7` Ha/bohr for the independent Cartesian analytic
gradient and `6e-7` Ha/bohr for the two finite differences; none was loosened.

```bash
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  taskset -c 48-63 python -m pytest -q -s tests/ci/benchmark_geometry.py
```

The isolated run completed with **1 passed in 754.92 s**, exit 0:

| Quantity | Measured value |
| --- | ---: |
| Native HF energy | -108.98780330137585 Ha |
| GradSCF CISD total energy | -109.37378941311154 Ha |
| PySCF CISD total energy | -109.37378941311160 Ha |
| Absolute CISD energy difference | 5.684e-14 Ha |
| Bond-direction energy derivative | 0.0450332759 Ha/bohr |
| Maximum Cartesian-gradient difference from PySCF analytic CISD | 5.821e-8 Ha/bohr |
| Centered finite-difference error, step 3e-4 bohr | 8.578e-8 Ha/bohr |
| Centered finite-difference error, step 1e-4 bohr | 1.155e-8 Ha/bohr |
| Gradient lowering / compilation / first execution | 78.594 / 99.539 / 98.135 s |
| Four displaced value-and-gradient executions | 468.820 s |
| Complete watchdog runner wall time | 769.314 s |
| Peak process RSS from `/usr/bin/time` | 12,135,652 KiB (11.57 GiB) |

The displaced calculations reconverge native HF and CISD and reuse the
already compiled value-and-gradient executable. Their timing includes the
unused displaced gradients, rather than a separately compiled forward-only
path. Translational invariance also passes. A process-group watchdog sampled
11.56 GiB peak RSS and enforced a 32 GiB / 1200 s budget; neither limit was
triggered, and no swap was recorded.

Evidence is under `artifacts/ci_mp_validation/ci_large_geometry/`, including
JUnit, phase timings, both finite differences, the resource runner, native
library provenance, a frozen source archive and matching before/after hashes
for all 1,047 `src`/test snapshot files. The archive SHA256 is
`7741515cf831c84a30d7a007684b0a4aa3f6684d87fa6aa559a558cff9de8b51`.
Its `ci/direct.py` SHA256 is
`6745e60c08ab2a73477e7ddda1cc8cfee2c8d54d491fc658a8af33c549469f4f`;
this geometry result precedes the separately validated normal-ordered MP
fluctuation-action optimization and does not claim a rerun of that later
snapshot.

True degeneracy of an individual CI eigenstate and coefficient/property second
response retain their documented restrictions. The large-basis check covers
the CISD total-energy first nuclear derivative at one geometry, rather than
excited-state/property gradients or second derivatives.
Frozen/active subspaces remain fixed: mixing them by an orbital rotation
changes the frozen-space approximation, including when their energies are
degenerate. No automatic expansion of a frozen or active selection is applied.

## CID and total-spin diagnostics — 2026-09-22

Branch: `feat/ci-cc-forward-parity`, based on `b8f8535` for this increment.
Environment: local arm64 CPU, JAX 0.8.1, PySCF 2.9.0, float64. PySCF supplies
independent FCI/spin-operator oracles, not production runtime calculations.

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  python -m pytest -q tests/ci tests/cc tests/solvers/test_nonlinear.py
```

**165 passed, 2 warnings, 406.04 seconds**, no skipped tests. The warnings are
PySCF OpenMP-availability notices. The 18 new CID/spin cases also passed as a
focused invocation (**18 passed, 15.25 seconds**), before this regression.

Subsequent inspection found that changing `excitation_ranks` on an already
constructed CIS/UCIS object bypassed its constructor rejection. A failing
regression was added to the existing scope test, and the same guard was added
at `kernel()` entry. The modified scope case plus restricted singlet/triplet
CIS and UHF CIS/TDA comparisons passed (**4 passed, 1 warning, 4.21 seconds**).
Those four cases are repeats of cases in the 165-test selection, not additional
distinct tests. This is targeted validation, not the full repository suite.

Coverage includes:

- Direct CID rank selection and retained-space budget accounting; invalid
  rank sets and unattainable ranks; frozen, empty and polarized spaces.
- Restricted H4 and unrestricted H3 CID energies against independently
  projected PySCF FCI Hamiltonians; Davidson, multiple roots and energy AD.
- Arbitrary real CI vectors (seed 77) in common and different orbital frames,
  including frozen electrons, against `pyscf.fci.spin_op`.
- Known S=0, 1 and 2 eigenstates, giving S^2=0, 2 and 6, respectively.
- JIT/AD in CI coefficients and cross-spin overlaps; converged-state spin
  response to fixed-MO integral perturbations with implicit eigenvectors.
- Explicit-overlap requirements, stale rank changes, invalid shapes/complex
  inputs/nonfinite states, and native UHF/ROHF facade integration.

CID energy comparisons use 2e-10 Hartree (2e-9 for default-tolerance dispatch),
spin diagnostics 2e-12 to 2e-11 in hbar^2 units, and derivative comparisons
2e-8 absolute tolerance with steps 1e-4 (integrals) or 1e-5 (vector/overlap).

Independent review performed two additional probes, not counted as pytest cases:
720 small-space rank/frozen/budget combinations for nmo=1..5 agreed with direct
determinant enumeration; two different three-dimensional MO subspaces in a
six-dimensional AO space gave spin diagnostics within 4.44e-16 of PySCF even
though the cross-frame overlap was not orthogonal. No blocking issue was found.

## Native example

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  python examples/ci/cid_spin.py
```

This passed using only GradSCF for RHF/UHF/ROHF and CI. STO-3G, SCF tolerance
1e-12, CI residual tolerance 1e-11, float64 CPU; H4 z coordinates are
0, 0.8, 1.9, 3.1 Angstrom, and H3 z coordinates are 0, 0.85, 1.9 Angstrom.

| System/reference | Method/root | Energy / Hartree | S^2 / hbar^2 | Effective multiplicity |
| --- | --- | ---: | ---: | ---: |
| H4/RHF | CID/0 | -2.160079573952509 | 0.0 | 1.0 |
| H4/RHF | CID/1 | -1.4261379503015075 | 2.22e-16 | 1.0000000000000004 |
| H4/RHF | CISD/0 | -2.160092360349407 | -4.44e-16 | 0.9999999999999991 |
| H4/RHF | CISD/1 | -1.9365436926554969 | 1.9999999999999998 | 3.0 |
| H3/UHF | CID/0 | -1.5732621174265453 | 0.7529224790377552 | 2.0029203469312056 |
| H3/ROHF | CID/0 | -1.5690262487385913 | 0.7633245057281185 | 2.013280413383211 |

Near-zero signed S^2 values are roundoff and are not clipped by the code.
The H3 results illustrate mixed-spin truncated CI states; a ROHF determinant
does not guarantee that this rank-selected, fixed-M_s CI space is spin adapted.
Example wall time was not separately measured; no performance benchmark is
claimed. Large CI spaces, GPU, automatic total-spin selection, complex/GHF
states and full nuclear/orbital response are outside this validation.

For the earlier CI/CC forward-property validation, see
[`cc/VALIDATION.md`](../cc/VALIDATION.md). Definitions and current limits are in
[CID_SPIN.md](CID_SPIN.md).
