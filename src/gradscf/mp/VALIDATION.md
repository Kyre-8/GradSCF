# Molecular MP validation

## Invariant Fock resolvent and nuclear response, 2026-10-20

This increment adds `orbital_basis="noncanonical"` for real stationary RHF/UHF
frames. The occupied and virtual Fock blocks remain inside the residual and
its implicit derivative; no derivative of an individual degenerate canonical
orbital is taken. The default canonical API and its streaming MP2 path remain
available. The series engine reuses the compact CI action, shares each
`W Psi_j`, and prunes only identically absent excitation ranks. It does not
discard small amplitudes or regularize physical zero denominators.

All new numerical checks ran remotely on `jy_gpu_1`, Intel Xeon Platinum
8457C, CPU float64, Python 3.11.15, JAX/JAXlib 0.10.0 and PySCF 2.13.0.
CUDA devices were hidden, including GPU6. Independent jobs used disjoint
16-CPU affinity sets with eight BLAS threads; no local numerical tests ran.
Resource limits, commands, JUnit results and source snapshots are retained
under the worktree's ignored `artifacts/ci_mp_validation/` directory.

Focused MP2 checks passed 25 synthetic/legacy cases, one complete N2 geometry
case, and a later five-case frozen-boundary selection. They cover R/U internal
occupied/virtual rotations, exact internal degeneracy, JVP/VJP and HVP against
independent finite differences, canonical agreement, empty channels, and
invalid physical denominators. The frozen-boundary increment rejects a
partially frozen degenerate eigenspace and uses the configured denominator
tolerance. The complete N2 MP2 run preceded that narrow guard increment; it is
not evidence for general moving frozen-core projector response.

The higher-order independent oracle uses PySCF Hamiltonian actions followed by
ordinary NumPy Rayleigh--Schrodinger recursion. It shares neither the Taylor
lifting nor the production CI action. At exact internal degeneracy, both
`series` and `tensor` passed checks of each E2--E6 JVP and of the **sum of
corrections' HVP**: two cases, 783.72 s pytest time, 812.90 s wall time,
8.21 GiB peak process RSS. This includes compilation, not warmed execution.
These checks do not establish each coefficient's HVP or a nuclear Hessian.
Earlier MP4 coefficient/JVP/HVP and capacity/dispatch selections also passed;
the final MP6 pair checks the rank-pruned implementation.

A separate 400-determinant, six-electron/six-spatial-orbital check explicitly
applies `P_4 H P_6` to the rank-six determinant and obtains a nonzero rank-four
result. Its action, transpose, physical-integral directional VJP and state VJP
match the old Slater--Condon oracle (one case, 299.82 s pytest, 316.14 s wall,
5.84 GiB peak RSS). Two additional cases verify that nonzero coordinate
diagonal gaps do not conceal an actually singular occupied/virtual spectrum
(69.70 s pytest). Both representations reject that noninvertible recurrence,
including JVP/VJP; this is not a claim that every zero-coupling degenerate
model lacks a smooth energy extension.

The complete nuclear-response case uses all-electron N2/STO-3G at R=2.08 bohr,
Cartesian AOs, native GradSCF integrals and independently converged HF in both
programs. The symmetry-preserving AO density seed is `1.4 I_10`; the hcore
seed instead selects a different, high-energy RHF branch. The occupied pi
orbitals are degenerate to roundoff. GradSCF differentiates native integrals,
implicit HF, the transported occupied/virtual frame, MO integrals and the MP
residual. The independent MP4 reference reconverges PySCF at R and R +/- h,
with h=3e-4 and 1e-4 bohr.

| Quantity | GradSCF | Independent reference |
| --- | ---: | ---: |
| MP4 total / Ha | -107.650984563774 | -107.65098456377372 |
| MP4 dE/dR / (Ha/bohr) | -0.2816141277 | -0.28161424223 (h=3e-4) |
| MP4 dE/dR / (Ha/bohr) | -0.2816141277 | -0.28161414086 (h=1e-4) |
| MP6 total / Ha | -107.653992031040 | -107.653992031040 (rounded) |
| MP6 dE/dR / (Ha/bohr) | -0.2877278495 | -0.287727961516 (h=3e-4) |
| MP6 dE/dR / (Ha/bohr) | -0.2877278495 | -0.287727862442 (h=1e-4) |

The MP4 case passed, including all four displaced energies, internal finite
differences, transverse zero forces and translation invariance. Pytest took
635.50 s, wall time 649.46 s and peak RSS 8.17 GiB. Gradient lowering,
compilation and first execution took 86.112, 318.947 and 0.422 s, respectively.
An additional forward compilation accounts for most of the remaining time;
the current test reuses the compiled value-and-gradient function for the
displaced energies. The earlier unpruned graph was stopped within its resource
budget before producing a numerical result; it is not a passed comparison.

The corresponding default-JIT N2 MP6 geometry attempt completed lowering in
228.244 s but did not complete XLA backend compilation. It was stopped after
1140.53 s within the 1200 s budget, with return code -15 and peak RSS 16.23 GiB.
The compiler reported repeated `algebraic_simplifier` passes; neither the
energy, gradient nor finite differences were executed. The successful small
MP6 coefficient-response checks above must not be read as this larger nuclear
test having passed. Compilation scale remains a practical limitation.

An isolated compiler diagnostic disabled only XLA's `algsimp` pass for that
same lowered MP6 gradient. It also exhausted its budget before execution:
1140.58 s wall, 23.92 GiB peak RSS, no energy/gradient/FD result. Its unchanged
source produced approximately 1.275 million unoptimized HLO instructions,
122.22 MiB of HLO text. This does not isolate a single pass as the cause of
slow compilation, and no compiler-disable setting is installed by GradSCF.

The final algebraic reduction directly constructs normal-ordered W and removes
the lambda-independent HF offset from the Taylor quotient. Its ten focused
R/U, frozen-space, rank-projection, transpose and physical-direction response
cases passed in 167.96 s pytest time (179.88 s wall, 4.09 GiB peak RSS).
They also check a nonzero occupied--virtual Fock perturbation and the absence
of same-block Fock kernels in the W construction. Two additional forward cases
passed in 105.32 s: legacy canonical H2 low-order agreement and each E2--E6
against independent RS for the rotated noncanonical model.

On that frozen reduced source, the physical-CI default builder, the selected
doubles coupling, exact-degenerate MP4/series JVP plus total HVP, and
MP6/tensor JVP plus total HVP passed together: four cases, 543.71 s pytest,
564.57 s wall, 7.94 GiB peak RSS. No source changed during this run.

The complete N2 MP6 nuclear test then passed with **default JIT**, no plugin
or compiler-options override: one case, 1008.48 s pytest, 1033.34 s wall,
14.81 GiB peak process RSS. Lowering/compilation/first execution took
201.800/792.461/1.539 s. Internal FD errors were 1.121e-7 and 1.254e-8
Ha/bohr for h=3e-4 and 1e-4 bohr; independent RS FD errors were 1.120e-7
and 1.289e-8 Ha/bohr. Displaced-energy errors were at most 1.705e-13 Ha.
Translations and transverse gradients passed their original tolerances.
The compiler still warned about repeated simplifier passes; compilation scale
is reduced enough for this bounded case, not eliminated as a general concern.
The 1.539 s measurement is the first execution of an already compiled gradient,
not a repeated warm benchmark. The earlier MP4 nuclear run preceded this final
W/Rayleigh reduction; the MP6 and four-case response runs use the reduced source.

Fifteen selected legacy MP cases also passed (336.89 s pytest time), covering
R/U tensor MP4/MP6, low-order amplitudes and spin components, MP5 wavefunctions,
capacity/invalid guards and the existing implicit HF-to-MP4 basis-parameter
chain. This is a selected regression, not a rerun of the full suite or MP12.
General ROHF, complex orbitals, moving frozen-core geometry, a closed physical
occupied--virtual gap and large-system high-order scaling remain outside this
validation. An isolated HF branch and nonsingular excitation resolvents are
still required. CCSD(T)'s separate first-order streamed-response checks are
recorded in [the CC validation](../cc/VALIDATION.md).

## Shared Taylor graph optimization, 2026-10-10

[GRAPH_REUSE.md](GRAPH_REUSE.md) records the exact linear-node transformation,
old/new paired CPU timings and independent R/U coefficients through MP12.
The complete MP regression before four new MP12 cases passed 97 tests in
155.72 s; those additional cases passed 4 tests in 17.82 s. Physical space,
solver tolerances and public APIs are unchanged. The comparisons measure
compiled forward corrections; they do not establish GPU or peak-memory gains.

## Residual/Taylor extension, 2026-10-10

The subsequent nine-system Cartesian 6-31G* extension is recorded in
[MOLECULAR_VALIDATION.md](../../../examples/mp/MOLECULAR_VALIDATION.md).
H2, H4, He, LiH, doublet/quartet H3, H4+, Li and BeH complete E2--E6 comparisons
with maximum disagreement 4.429e-13 Ha. Both programs independently converge
their HF references. The complete MP suite after sharing the example-only
PySCF reference passed **91 tests in 168.85 s** with no skips or warnings:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu /opt/anaconda3/bin/python -m pytest -q tests/mp
```

Local macOS arm64 CPU, Python 3.12.2, JAX 0.8.1, PySCF 2.9.0, float64.
The generic engine is described in [SERIES.md](SERIES.md). The working branch
uses the same native CPU library as the parent checkout; no integral-kernel or
shared-solver source changes are included.

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu /opt/anaconda3/bin/python -m pytest -q \
  tests/mp tests/ci/test_ci.py tests/ci/test_uci.py tests/test_integral_namespace.py -x
```

Result: **127 passed**, 201.39 s. No skips. The one warning is PySCF's
unavailable OpenMP build. This includes restricted individual E2--E6 and complete
state coefficients on H2, LiH and asymmetric H6/STO-3G; unrestricted E2--E5 on
Li/6-31G; frozen spaces; JIT outputs; integral gradients/HVPs; capacity preflight;
empty spaces; invalid response NaNs; and full implicit HF-to-MP4 contraction
parameter response against reconverged finite differences. The reference uses
PySCF full-space Hamiltonian actions and an independent RS recurrence, not
Taylor AD or GradSCF CI connections. It is a numerical oracle, not a production
FCI calculation used by the MP engine.

The subsequent water public-example regression passed separately:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu /opt/anaconda3/bin/python -m pytest -q \
  tests/mp/test_series.py::test_water_public_example_matches_independent_pyscf
```

Result: **1 passed**, 8.98 s. The example also ran directly with assertions:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu /opt/anaconda3/bin/python examples/mp/compare_water_pyscf.py
```

Water: O (0,0,0), H (0,0.75,0.58), H (0,-0.75,0.58) Angstrom, Cartesian
STO-3G, all electrons, real RHF with `conv_tol=1e-13`, `conv_tol_grad=1e-11`.
Both programs converge their own HF/integrals. GradSCF and PySCF RHF totals
are -74.961376619994 Ha. Both retained and reference spaces contain 441
determinants; this small molecule's finite excitation space is exhausted.

| Order | GradSCF correction / Ha | PySCF-action RS correction / Ha | Absolute difference / Ha |
| --- | ---: | ---: | ---: |
| E2 | -0.034747934183 | -0.034747934183 | 7.339e-14 |
| E3 | -0.009340345100 | -0.009340345100 | 4.046e-14 |
| E4 | -0.002813547195 | -0.002813547195 | 1.718e-14 |
| E5 | -0.000916760464 | -0.000916760464 | 7.240e-15 |
| E6 | -0.000318643445 | -0.000318643445 | 3.748e-15 |
| E7 | -0.000116080027 | -0.000116080027 | 1.902e-15 |
| E8 | -0.000043500906 | -0.000043500906 | 9.195e-16 |

MP8 partial total is -75.009673431315 Ha; PySCF FCI total is
-75.009700279727 Ha. The remaining 2.6848412e-5 Ha is a finite-order truncation
error in this basis, not disagreement between the MP coefficient engines.
PySCF 2.9.0 and the inspected [official MP interface](https://pyscf.org/user/mp.html)
provide native MP2; the higher-order reference is explicitly our independent
recurrence using PySCF's action, not a native PySCF MP8 API.

The same water geometry with Cartesian 6-31G*, frozen=1, gives native MP2
correlation energies -0.185314572684 Ha in both programs, differing by
4.241e-14 Ha. This larger-basis check is MP2 only. High-order full-space water
calculations with 6-31G* were not performed.

The first cold end-to-end STO-3G runs took 5.600 s (GradSCF RHF + Taylor) and
0.043 s (PySCF RHF + RS). These regions have different compilation/AD work and
are not a matched speed benchmark. Cold startup is included. No warmed timing,
GPU peak memory or scaling claim follows from them.

The native GradSCF H6 [example](../../../examples/mp/taylor_series.py) also ran:
HF -3.172676142364 Ha, E2 -0.051659165678, E3 -0.016557683628,
E4 -0.006267145387, E5 -0.002416510625, E6 -0.000954427910 Ha;
MP6 partial total -3.250531075592 Ha. It retains all 400 allowed determinants
through actual rank six and complete states through order three. Maximum
linear-response residual is 4.813e-17 Ha. Values are included as output comments.

No entire-project suite, GPU execution or high-order large-system benchmark
was performed. The generic path is in-core and grows combinatorially. Real
canonical RHF/UHF are supported; ROHF/noncanonical/complex variants and MP
density matrices remain outside this implementation. The parameter-response
test does not establish nuclear-coordinate derivative coverage or convergence
of an arbitrary MP series at lambda=1.

## Initial low-order implementation, 2026-10-09

Date: 2026-10-09. Local macOS arm64, CPU backend, JAX float64,
Python 3.12.2, JAX 0.8.1, PySCF 2.9.0. Native forward integrals reuse the
same built library as the parent checkout; no vendor or native-kernel change.

## Focused tests and shared-code regressions

```bash
PYTHONPATH=src /opt/anaconda3/bin/python -m pytest -q \
  tests/mp tests/cc/test_ground.py tests/cc/test_ucc.py tests/ci/test_ci.py \
  tests/test_pyscf_style_namespace.py tests/test_integral_namespace.py
```

Result: **120 passed**, 229.20 s. This includes 37 MP tests and existing CI/CC
regressions for the generalized selected-block MO transformation. No dependency
tests were skipped. One PySCF warning reported that OpenMP is unavailable in
the installed build; it did not affect the assertions.

- RHF H2O/Cartesian 6-31G*: all-electron, frozen core and frozen core/virtual
  MP2 energies, E2 spin components and amplitudes match PySCF. Energy tolerances
  are 1e-11 Ha (correlation) and 1e-10 Ha (total); amplitudes 2e-9.
- UHF H3/STO-3G, spin=1: MP2 with no freezing and different alpha/beta frozen
  spaces matches PySCF under the same tolerances. Reference SCF uses
  `conv_tol=1e-14`, `conv_tol_grad=1e-11`.
- GradSCF RHF water/STO-3G and UHF H3/STO-3G: selected packed/DF facade results
  match full-MO array reference results at 1e-12 Ha; monkeypatched full-MO
  transforms must not be called by MP2. RHF energy-only and stored-amplitude
  modes are both exercised.
- MP3 H2/3-21G and LiH/STO-3G: individual E2 and E3 match independent determinant
  perturbation coefficients at 1e-11 Ha, including frozen core/virtual cases.
- JIT energy derivatives use an interaction-scaling identity with a fixed Fock
  operator, preserving the canonical-input condition. MP2/UMP2 HVPs and MP3
  gradients are checked. The MP3 HVP assertion added to that same test was then
  rerun with the MP3 file, separately from the 120-test run: **11 passed**,
  25.19 s.
- Invalid denominators and noncanonical states are rejected; Pauli-forbidden
  same-spin entries are excluded from minimum-denominator diagnostics. Empty
  doubles spaces return valid zero correlation energy. Stale facade sources
  raise and clear previous successful result/status.

## Public examples

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu python examples/mp/molecular.py
```

H2O geometry in Angstrom: O (0,0,0), H (0,0.75,0.58), H (0,-0.75,0.58).
Cartesian 6-31G*, frozen=1; native RHF, `conv_tol=1e-12`,
`conv_tol_grad=1e-10`. DF uses the default `def2-universal-jkfit` auxiliary basis.

| Quantity | Hartree |
| --- | ---: |
| RHF total | -76.010721112637 |
| E2 correction | -0.185314572684 |
| E3 correction | -0.005996691699 |
| MP3 total | -76.202032377021 |
| DF-MP2 total | -76.196039459056 |

UHF H3/STO-3G at z=(0,0.85,1.9) Angstrom, spin=1, all-electron:
UMP2 total -1.562947444299 Ha.

`examples/mp/implicit_gradient.py` is exercised by the test suite. H2/STO-3G
at 0.9 Angstrom, dimensionless scaling of one primitive contraction coefficient
on the first atom; shared implicit orbital HF response; centered reconverged
finite difference step 1e-4, `atol=2e-7`, `rtol=1e-5`.

| Method | Total / Ha | AD derivative / Ha | FD derivative / Ha |
| --- | ---: | ---: | ---: |
| MP2 | -1.109271845527 | 0.025838166 | 0.025838166 |
| MP3 | -1.116349572980 | 0.025318695 | 0.025318695 |

The example computes native primitive integrals once and differentiates JAX
contraction, implicit HF orbitals, MO integrals and MP energy. It does not prove
nuclear-coordinate gradient coverage. Output comments in both examples are
these measured values, not external reference tables.

## Boundaries

No entire-project test run, GPU execution, GPU peak-memory measurement or
large-system scaling benchmark was performed. MP3 remains in-core; UMP3,
MP4/higher orders, noncanonical and ROHF variants, and density matrices are not
implemented in that initial version. The 2026-10-10 extension above adds UMP3
and generic higher orders. See README.md for result/amplitude semantics.
