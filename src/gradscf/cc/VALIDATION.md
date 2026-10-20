# Ground-state CC validation

## Streamed second-order response — 2026-10-20

This increment supersedes the second-AD rejection recorded in the earlier
streamed-response stages below. The explicit `algorithm="streamed"` path now
supports first and second JVPs/VJPs, HVPs, reverse-over-reverse directional
curvatures and `jax.hessian`. Third-order AD still raises explicitly.
The existing `algorithm="auto"` paths are unchanged. The derivation and
memory/time tradeoff are in [SEMICANONICAL.md](SEMICANONICAL.md).

All numerical checks ran on `jy_gpu_1` (host `c20`, Intel Xeon Platinum 8457C),
Python 3.11.15, JAX/jaxlib 0.10.0, PySCF 2.13.0 and CPU float64. Every job hid
all GPUs, used a disjoint 16-CPU affinity set within CPUs 0--63, and limited
OMP/OpenBLAS/MKL to eight threads. No local numerical or GPU test was run.

| Final-source selection | Result | Pytest time | CPU affinity |
| --- | --- | ---: | --- |
| 21 response cases plus 12 invalid-derivative cases | 33 passed | 96.60 s | 48--63 |
| Three molecular integration cases plus two UCC curvature cases | 5 passed | 113.30 s | 32--47 |
| Newly appended direct `jit(jax.hessian)` API case | 1 passed | 5.12 s | 0--15 |

All three invocations exited zero, with no failures, errors or skips. The
single Hessian-API case was appended after the 33-case run; the preceding
21 response cases were byte-identical. These are separate invocations, not
one full CC-suite run. The exact final commands, from their source stages, were:

```bash
export CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=src
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
taskset -c 48-63 timeout 600 python -m pytest -q \
  tests/cc/test_streamed_triples_response.py \
  tests/cc/test_streamed_triples_invalid.py
taskset -c 32-47 timeout 600 python -m pytest -q -s \
  tests/cc/test_streamed_triples_geometry.py \
  tests/cc/test_semicanonical.py::test_semicanonical_cc_response_at_exact_orbital_degeneracy
taskset -c 0-15 timeout 600 python -m pytest -q -s \
  tests/cc/test_streamed_triples_response.py::test_streamed_jit_hessian_api_matches_two_direction_six_index_oracle
```

### Independent algebra and workspace checks

The 4-occupied/4-virtual spin-orbital model has twofold occupied and virtual
Fock eigenvalues. Joint physical directions vary the full symmetric Fock
matrix, consistently antisymmetrized ERI slices, T1 and T2. Connected,
disconnected and total HVPs and curvatures agree with the retained six-index
tensor-resolvent oracle. Independent occupied/virtual rotations also produce
noncanonical Fock blocks while preserving the repeated spectra; the maximum
rotated HVP discrepancy is 4.34e-18 and the energy spread is 6.07e-18 Hartree.
The direct two-parameter `jax.hessian` result is nonzero and symmetric, with
maximum oracle error 1.74e-18 and symmetry error 2.17e-19. This is a complete
2-by-2 parameter Hessian, not a full molecular Cartesian Hessian.

Recursive Jaxpr checks cover forward, VJP, HVP and scalar grad-grad graphs.
They reject six-axis arrays and arrays larger than 256 elements for this
fixture, including flattened panel histories. These are structural checks;
they are not universal process-memory or large-system scaling bounds.

Independent review also reproduced a pre-existing failure-path defect:
singular denominators or invalid CC states could yield finite or zero
derivatives. Twelve regressions now require invalid active first/second
derivatives to be NaN. The private kernel reuses the shared convergence
derivative guard, while the public streamed energy mask propagates failure
through AD. Valid-state algebra and default algorithms are unchanged.

### Complete native nuclear HVP

The final integration group uses the N2/STO-3G real-RHF setup and converged
HF/CC thresholds documented below, including exact pi-orbital degeneracy.
The differentiated map includes native integral second response, implicit
HF, metric-preserving MO transport, MO transformations, implicit CCSD and
streamed `(T)`. Two dynamic directions are checked: a bond stretch and a
normalized mixed Cartesian displacement. At each displaced geometry, the
external PySCF reference independently reconverges RHF and CCSD, solves the
CCSD(T)-corrected Lambda equations, and evaluates its analytic gradient.

| Direction | Max HVP error at h=3e-4 bohr | Max HVP error at h=1e-4 bohr |
| --- | ---: | ---: |
| Bond stretch | 3.277e-7 | 3.653e-8 |
| Mixed Cartesian | 3.864e-7 | 4.634e-8 |

HVP errors are Hartree/bohr squared. The mixed Hessian symmetry discrepancy
is 3.064e-14 in the same units. The final first-order rerun has energy error
1.549e-12 Hartree and maximum analytic-gradient error 6.197e-12 Hartree/bohr.
The UCC cases separately include implicit amplitude response and a nonzero
triples correction at exact internal orbital degeneracy; both `auto` and
`streamed` curvatures agree with gradient differences at two steps.

The five-case runner took 116.26 s and reached 2,475,820 KiB maximum RSS
(2.36 GiB); this includes compilation and reference calculations, not just a
production HVP. Source/library manifests were unchanged during every run.
The final molecular stage checked 1,568 files and verified native-source and
reused-library hashes. The common streamed-kernel SHA256 was
`61be0d11ce5786777a0a6c982352ecf80e322c2e35f2ef621ddf72ee2965cc03`.
JUnit, resource records and source manifests are retained under the remote
root `/home/yjiao/gradscf_checks/ci-mp-response-20261020/`, in
`cc_second_guard/artifacts/regression`,
`cc_second_geometry/artifacts/final_integration` and
`cc_second_order/artifacts/hessian_api`.

The full repository/CC suites, GPU response, unrestricted nuclear Hessians,
a full N2 Cartesian Hessian and third-order response were not validated by
this increment. Third-order streamed response remains unsupported.

## Restricted QCISD/(T) — 2026-09-22

Branch: `feat/ci-cc-forward-parity`, increment based on `07a54e5`. Environment:
arm64 CPU, JAX 0.8.1, PySCF 2.9.0 and JAX float64. No new solver implementation
or PySCF runtime dependency was introduced.

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  python -m pytest -q tests/cc tests/ci tests/solvers/test_nonlinear.py
```

**147 passed, 2 warnings, 383.71 seconds**, with no skips. The warnings are
PySCF's OpenMP-availability notices. This is a targeted regression selection,
not the full repository suite. The 13 new QCI cases also passed independently
in **36.94 seconds** and are included in the 147-case run.

New coverage includes optimized PySCF random QCISD residuals/energies with and
without noncanonical Fock perturbations (seed 67); H4/STO-3G full and frozen
energies, amplitudes and triples; H2O/STO-3G energy/triples; quadratic residual
degree (seed 68); noninteracting H4-fragment additivity; implicit energy/amplitude
response and model-density finite differences; model/frozen-state, empty-space
and unsupported-scope guards; nonconvergence AD and native facade integration.

Independent review additionally checked LiH/STO-3G with nocc=2, nvir=4 and random
noncanonical Fock/amplitude inputs. Residual maximum errors against optimized
PySCF QCISD were 3.68e-16 and 3.33e-16; energy error was 3.82e-17 Hartree.
With state/canonical tolerances relaxed only in that formula probe, the QCI
triples expression including F_vo*T2 agreed with PySCF's slow formula to
8.67e-19 Hartree. This probe does not broaden the public canonical-(T) contract.
It is separate from the automated regression count.

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  python examples/cc/restricted_qcisd.py
```

This example passed using GradSCF native RHF, H2O coordinates
`O 0 0 0; H 0 -.757 .587; H 0 .757 .587` Angstrom, STO-3G,
SCF `conv_tol=1e-12`, QCI `conv_tol=1e-12`, `residual_tol=1e-11`:

| Quantity | Measured value |
| --- | ---: |
| RHF energy / Hartree | -74.96306312972915 |
| QCISD energy / Hartree | -75.0125474982174 |
| QCISD(T) correction / Hartree | -5.72851626487058e-5 |
| QCISD(T) energy / Hartree | -75.01260478338006 |
| QCI residual infinity norm | 2.95e-12 |
| QCI model 1-RDM trace | 10.0 |
| QCI adjoint converged | True |

Example wall time was not separately measured; this is a numerical smoke test,
not a performance benchmark. Unrestricted/ROHF QCI, semicanonical QCI triples,
QCI 2-RDMs, triples densities, complete nuclear gradients and GPU/large-system
performance are not covered. Definitions and boundaries are in [QCISD.md](QCISD.md).

## Semicanonical real-reference triples — 2026-09-22

Branch: `feat/ci-cc-forward-parity`, increment based on `748df39`. Same local
arm64 CPU, JAX 0.8.1, PySCF 2.9.0 and float64 environment as the preceding stage.

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  python -m pytest -q tests/cc tests/ci tests/solvers
```

**174 passed, 2 warnings, 437.32 seconds**, no skips. The warnings are PySCF's
OpenMP-availability warnings. After this invocation, review identified that an
empty tensor RHS bypassed validation of other nonempty factors. A failing
regression was added, the input check was moved before the empty-space return,
and `tests/solvers/test_tensor_sum.py` passed separately (**3 passed, 2.78 seconds**).
The final selection contains 175 distinct tests; all have been exercised across
these invocations. The second run repeats two existing tensor-sum cases and
adds the empty-input case, rather than constituting three additional tests.

The new CC checks cover:

- OH/STO-3G ROHF, R=0.97 Angstrom, no frozen orbitals, one frozen core per spin,
  and an alpha-only frozen core, compared with independently semicanonicalized
  PySCF UCCSD/(T). Energy tolerance: 2e-10 Hartree for the correction.
- Random independent alpha/beta occupied and virtual rotations, seed 51,
  with consistent integral/amplitude transformation: invariant (T) to 2e-12 Hartree.
- RHF H4/STO-3G canonical equivalence (2e-11 Hartree), and JIT total-energy
  response to noncanonical perturbations versus reconverged finite differences
  (step 1e-4; absolute derivative tolerance 2e-7).
- An interacting synthetic model, seed 52, with exact within-spin occupied and
  virtual degeneracies; JIT triples response versus finite differences to 2e-9.
- Empty triples and invalid-state/API/capacity rejection.

The generic tensor solver was checked against a dense Kronecker matrix, including
exact factor degeneracy and JVP/VJP agreement. Independent review also probed
simultaneously varying factors/RHS and found value/first/second derivative errors
of 8.33e-17, 0 and 1.73e-16 against a 12-by-12 dense solve. That extra probe does
not establish full CCSD(T) second derivatives.

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  python examples/cc/semicanonical_triples.py
```

This native GradSCF example passed with OH/STO-3G at 0.97 Angstrom, spin=1,
SCF `conv_tol=1e-12`, CC `conv_tol=1e-12`, residual tolerance 1e-10:

| Reference | CCSD / Hartree | Semicanonical (T) / Hartree | CCSD(T) / Hartree |
| --- | ---: | ---: | ---: |
| UHF | -74.3871842074733 | -2.3003890403e-7 | -74.3871844375122 |
| ROHF | -74.38718434215826 | -1.6519820636e-7 | -74.38718450735647 |

The input Fock off-diagonal maxima were 2.34e-10 and 0.02301698 Hartree.
The UHF semicanonical correction also matched the default streamed correction
within 1e-11 Hartree. No PySCF runtime is used by this example. Example time
was not separately measured; this is validation, not a performance benchmark.
Full-memory moments, the per-tensor capacity guard and the conservative full
Cartesian-spectrum singularity check are explained in [SEMICANONICAL.md](SEMICANONICAL.md).
GPU, large-basis scaling, spin-adapted ROCCSD and complete nuclear gradients
remain outside the validated contract.

## CI/CC forward properties — 2026-09-22

Branch: `feat/ci-cc-forward-parity`, based on `52d2d93`. Environment:
macOS 26.6.2 arm64 CPU, JAX 0.8.1, PySCF 2.9.0, float64.

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  python -m pytest -q tests/ci tests/cc tests/solvers/test_nonlinear.py
```

**126 passed, 2 warnings, 317.42 seconds**, with no skips. Both warnings report
that the installed PySCF lacks OpenMP; they do not indicate a failed calculation.
A subsequently added independent arbitrary-amplitude spin-density oracle test,
`tests/cc/test_forward_properties.py::test_spin_density_parts_arbitrary_amplitudes_against_pyscf`,
passed separately (**1 passed, 3.34 seconds**). Thus 127 distinct test cases were
verified across these two invocations, not in one combined invocation.

This includes 22 new cases: real CI 1/2-RDMs against FCI, arbitrary normalized
vectors, frozen electrons and virtuals, UCC Lambda and density comparisons,
R/U CCSD 2-RDM energy/particle-number identities, canonical UCCSD(T) against
PySCF for OH/6-31G with/without a frozen core, restricted-limit agreement,
JIT and fixed-MO response finite differences, empty spin/amplitude spaces,
invalid/stale-state rejection and arbitrary same-spin density contractions.

The initial baseline worktree lacked the native integral library (99 passed,
4 failed during SCF initialization). Reusing the same compiled CPU library from
the parent checkout resolved all four failures; those four tests passed before
the full regression above. No mathematical backend fallback was used.

The standalone `examples/cc/open_shell_properties.py` ran successfully with
GradSCF native CPU UHF, OH at 0.97 Angstrom, STO-3G, spin=1, SCF `conv_tol=1e-12`,
CI `conv_tol=1e-10`, CC `conv_tol=1e-12` and `residual_tol=1e-10`:

| Quantity | Measured value |
| --- | ---: |
| UHF energy / Hartree | -74.3626691947672 |
| UCISD energy / Hartree | -74.38718440414765 |
| UCCSD energy / Hartree | -74.3871842074733 |
| UCCSD(T) energy / Hartree | -74.3871844375122 |
| UCISD density energy reconstruction error / Hartree | 1.99e-13 |
| UCCSD density energy reconstruction error / Hartree | -7.11e-14 |
| Alpha/beta density traces | 5 / 4 |
| Lambda converged | True |

No PySCF runtime is used in this example. It is a numerical smoke test, not a
performance benchmark; example wall time was not separately measured. GPU,
large-basis performance, general ROHF triples, complete nuclear gradients and
CCSD(T) densities remain unvalidated/unimplemented as specified in
[FORWARD_PARITY.md](FORWARD_PARITY.md). No full repository test-suite claim is made.

## Restricted ground-state CC — 2026-09-21

Environment: arm64 CPU (`TFRT_CPU_0`), JAX 0.8.1, PySCF 2.9.0, float64.
Tests were run in the isolated `feat/cc-ground-state` worktree.

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=1 \
python -m pytest -q tests/cc tests/ci tests/solvers \
  tests/test_scf_diis_state.py tests/test_scf_diis_scaling.py \
  tests/test_tddft_eigensolvers.py tests/test_scf_higher_order.py
```

Initial comprehensive result: **129 passed**, two PySCF OpenMP-availability warnings, 185.64 seconds.
No tests in this selection were skipped. This is targeted validation, not the
complete repository test suite or a GPU benchmark.

The 34 CC tests in that run include:

- CCSD/CC2 random-amplitude residuals versus PySCF;
- independent determinant-space BCH checks for CCSD, CCD and CCS, and
  H + [H,T] checks for LCCD/LCCSD;
- energies, right amplitudes and spin-adapted Lambda amplitudes;
- conventional (T) on H4 and H2O, and the two-electron zero-(T) limit;
- frozen core/virtual spaces and the fully frozen reference limit;
- two-electron FCI and noninteracting-fragment additivity;
- implicit energy response for every exported model, amplitude response,
  CCSD(T) total response, and independence from converged DIIS trajectories;
- nonconvergence, invalid settings, unsupported methods and complex amplitudes.

A subsequent float32-restart/float64-integral regression exposed a JAX loop dtype
mismatch. Restart amplitudes now use the common integral/amplitude dtype before
packing. The new regression, namespace smoke test and nonconvergence derivative
test passed together: **3 passed**, 12.56 seconds (`tests/cc -k
'lower_precision or cc_public or failure_and_unsupported'`). The current CC test
file therefore contains 35 cases. The wheel was rebuilt after this code fix.

## Runnable H4 example

`examples/cc/restricted_ground.py` uses H4 at z = 0, 0.8, 1.9, 3.1 Angstrom,
STO-3G, real RHF and the native CPU integral path. Energies are Hartree.

| Method | Total energy |
| --- | ---: |
| RHF | -2.089978546810821 |
| CCS | -2.089978546810820 |
| CCD | -2.161519207807380 |
| LCCD | -2.165569628785267 |
| LCCSD | -2.165593627670407 |
| CC2 | -2.132771236216811 |
| CCSD | -2.161533025614623 |
| CCSD(T) | -2.161619299668401 |

These methods are not variational bounds; a lower number in this table does not
establish a more accurate method. CCSD converged in 12 iterations with residual
3.20e-11 Hartree; the Lambda linear residual was 1.71e-16. The (T) correction
was -8.627405377840567e-5 Hartree.

For a dimensionless coupling scale that preserves the canonical Fock matrix,
the complete CCSD(T) derivative was -2.5222909909633544 Hartree. Central finite
difference with step 1e-4 gave -2.5222909920175063 Hartree, absolute difference
1.054e-9 Hartree. This is an integral-parameter test, not a nuclear-force test.

## Packaging and runtime independence

- A no-index, no-dependency wheel build succeeded.
- The wheel includes the CC implementation, shared helpers, Apache license,
  source-code NOTICE and bibliographic files.
- A separate process blocked every PySCF import and successfully ran CCSD,
  Lambda and (T) on a two-orbital noninteracting Hamiltonian.
- `git diff --check` and local documentation-link checks passed.

See [README.md](README.md) for limits and [REFERENCES.md](REFERENCES.md) for
the distinction between theory, adapted code and independent numerical oracles.

## OpenMolcas-informed refinements

The same comprehensive command above, after adding the refinements in
`tests/cc/test_refinements.py`, completed with **142 passed**, two PySCF
OpenMP-availability warnings, no skips, in 233.95 seconds (CPU/float64).
The CC selection now includes 47 cases. The 12 new cases cover:

- iteration-level-shift invariance of energy and implicit response;
- explicit Urban and conventional triples variants and diagnostic components;
- failed/shape-inconsistent/mismatched states, including empty triples spaces;
- full/frozen-space 1-RDM comparisons with PySCF;
- density response including both right and left-state response;
- invalidation after changing the reference, method or frozen space;
- an independent determinant-space WT2 moment check and a finite-difference
  derivative of the Urban correction itself.

The updated H4 example gave Urban's correction -8.642093119764708e-5 Hartree,
the conventional correction -8.627405376957714e-5 Hartree, and the added singles
component +1.4687742806995303e-7 Hartree. The minimum triples denominator magnitude
was 2.0090627405364425 Hartree. The 1-RDM trace was 4.0 and its symmetry error
was zero. The complete conventional CCSD(T) response still agreed with central
finite difference to 1.0542e-9 Hartree along the documented canonical path.

The inspected OpenMolcas source revision and hashes are recorded in
[OPENMOLCAS.md](OPENMOLCAS.md). These checks do not claim that an OpenMolcas
executable was run or that its ROHF/Cholesky implementations were ported.
## Open-shell extension (2026-09-21)

The open-shell implementation is described in [OPEN_SHELL.md](OPEN_SHELL.md).
It uses the same common nonlinear/linear/eigenvalue solvers as the restricted
implementation. All tests here use CPU and float64; PySCF 2.9.0 is an independent
test dependency. No GPU, nuclear-gradient or spin-adapted ROCCSD claim is made.

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=1 \
  python -m pytest -q tests/cc tests/ci tests/solvers \
  tests/test_scf_uhf.py tests/test_scf_rohf_roks.py tests/test_tddft_eigensolvers.py
```

Result: **158 passed, 5 skipped**, 294.70 s on macOS arm64, JAX 0.8.1. The five
skips are existing semilocal ROKS comparisons requiring unavailable `jax_xc`
(three functionals on an identical grid, the PBE facade, and the closed-shell
PBE limit). Two PySCF warnings report unavailable OpenMP. Subsequent narrowly
targeted regressions include the UKS response-cache fix and additional CI
triples/empty-amplitude cases; see the focused command below.

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=1 \
  python -m pytest -q tests/ci/test_uci.py tests/cc/test_ucc.py \
  tests/cc/test_refinements.py::test_facade_rejects_stale_posthoc_state
```

Final focused result: **27 passed**, 55.67 s (24 open-shell cases plus three
existing restricted facade-freshness cases). Separately, the UHF response-cache
regression and `tests/test_reference_uks.py` give **6 passed**, 9.11 s.

Coverage includes:

- H3 doublet/STO-3G, z = 0, 0.85, 1.9 Angstrom: unrestricted CI Hamiltonian
  compared with PySCF UHF-FCI, UCISD energies with PySCF UCISD, UCIS excitations
  with UHF/TDA, UCCSD energies and all amplitude blocks with PySCF UCCSD.
- OH doublet/6-31G, R = 0.97 Angstrom: nonzero alpha-alpha and beta-beta
  doubles, UCCSD energies/amplitudes, random noncanonical residuals against
  PySCF GCCSD. Oracle CC uses up to 150 cycles with energy tolerance 1e-12 Ha
  and amplitude-update tolerance 1e-10; GradSCF checks the physical residual.
- A four-MO synthetic Hamiltonian with distinct spin orbital frames and three
  electrons: nonempty triple excitations, projected FCI matrices, variational
  lowering from doubles to triples, and the full-CI limit.
- Full/s4/DF AO transforms; separate frozen alpha/beta occupied and virtual
  orbitals; alpha-beta exchange of labels; level shift; float32 restarts into
  float64 calculations; single empty spin channels and all-frozen amplitudes.
- JIT energy/coefficient/amplitude response versus central finite differences
  (step 1e-4) for one- and two-electron MO integral perturbations. Tolerances are
  2e-8 to 3e-8 for derivative comparisons, 2e-9 Ha for molecular energy oracles,
  3e-8 for amplitudes, and 3e-12 for random residuals. Unconverged UCC response
  must be nonfinite rather than silently zero.
- GradSCF UHF/ROHF facade dispatch and stale-SCF rejection, including the existing
  UHF response adapter after adding the shared input-freshness bookkeeping.

### Reproducible public-API example

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  python examples/cc/open_shell_ground.py
```

The example prints its hardware/backend, geometry, convergence tolerances,
amplitude shapes, elapsed time and derivative comparison. Measured on
2026-09-21, macOS 26.6.2 arm64, JAX 0.8.1, TFRT CPU, float64, in 16.15 s:

| Reference | HF (Ha) | UCCSD (Ha) | CC residual norm | Fixed-MO derivative absolute error |
| --- | ---: | ---: | ---: | ---: |
| UHF | -1.5506267047570963 | -1.5741669980551043 | 2.44e-12 | 1.54e-12 |
| ROHF | -1.5415902126151726 | -1.5741669980551203 | 2.67e-13 | 8.27e-13 |

UCISD and full CI also give -1.57416699805511 Ha for this small system. Here the
orbital/electron counts allow at most two spin-conserving excitations; this
equality must not be generalized to arbitrary three-electron systems. The
derivative perturbs only h-alpha by `x*diag(0.2,-0.1,0.07)` at fixed MO orbitals,
with finite-difference step 1e-4. It is not a nuclear force.

## Streamed full-Fock triples response — 2026-10-20

The opt-in `algorithm="streamed"` path was tested remotely on `jy_gpu_1`,
CPU backend/float64, affinity restricted to logical CPUs 48–63, with all GPUs
hidden. OMP, OpenBLAS and MKL thread limits were each 8. No local numerical
tests or GPU tests were run for this change.

The initial focused oracle/degeneracy group passed 5 tests in 12.59 s. The
new response module plus the existing semicanonical module then passed
17 tests in 136.52 s. After strengthening the workspace assertion and adding
an explicit streamed UCC parameter, the two affected cases passed in 20.00 s.
These are test-suite elapsed times, not production timing/scaling benchmarks.

```bash
env CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONPATH=src \
  taskset -c 48-63 timeout 600 python -m pytest -q \
  tests/cc/test_streamed_triples_response.py tests/cc/test_semicanonical.py
```

The new synthetic 4-occupied/4-virtual spin-orbital case compares the connected
term, disconnected term and their sum, plus all input VJPs, with the retained
six-index tensor-solve oracle. It includes a nonzero `F_vo*T2` contribution.
Exactly repeated occupied and virtual eigenvalues are perturbed off-diagonally;
both JVPs and VJPs agree with the oracle and central differences (step 1e-4).
Forward/reverse Jaxprs are checked for both array rank and element count so
that flattening a retained triples-panel history cannot evade the memory test.
This structural check is not a measured peak-process-memory benchmark.

A restricted four-spatial-orbital interacting model compares complete CCSD
amplitude-response derivatives with the tensor oracle and finite differences.
The fixed-amplitude derivative is explicitly different. That test rejects any
call to the full spin-integral builder from the restricted streamed path.
The existing three-electron UCC exact-degeneracy model also passes with the
streamed option. Workspace/tensor-capacity separation, physical denominator
singularity rejection and explicit second-order-AD rejection are covered.

JUnit evidence is in the remote validation stage
`/home/yjiao/gradscf_checks/ci-mp-response-20261020/cc/`:
`streamed_triples_junit.xml` and `streamed_triples_delta_junit.xml`.
The old default algorithm remains unchanged. These checks establish bounded
moment representation and first-order parameter response; large-molecule
performance, fully spin-adapted storage and complete nuclear forces are not
established by this test group.

### Complete native molecular geometry chain

The same validation date added `CorrelatedEnergy(mf, "ccsd(t)")` by extending
the existing molecular adapter. The unsupported-method regression failed
before the implementation; the constructor/configuration check and complete
N2 geometry calculation then passed. The final two-test run took 24.86 s
(CPU48–63, float64, OMP/OpenBLAS/MKL limits 8, all GPUs hidden). Its source and
test hashes were unchanged across the run. No local numerical test was used.

The molecule is N2 at `(0,0,-1.04)` and `(0,0,1.04)` bohr, all electrons,
Cartesian STO-3G, real RHF. The symmetry-preserving AO density seed is `1.4 I`;
HF convergence tolerances are 1e-13 Hartree, density 1e-11 and orbital gradient
1e-10. The occupied/virtual pi pairs are exactly degenerate to 1e-9 Hartree.
The adapter differentiates native AO integrals, implicit HF, the transported
MO frame, MO integral transformations, implicit CCSD and the streamed `(T)`
correction inside a single total-energy map.

| Quantity | Measured value |
| --- | ---: |
| Total CCSD(T) energy | -107.652291980331 Hartree |
| Bond derivative dE/dR | -0.2832401921 Hartree/bohr |
| Energy absolute error versus PySCF | 1.862e-12 Hartree |
| Maximum component error versus analytic PySCF CCSD(T) gradient | 6.602e-12 Hartree/bohr |
| Maximum bond-gradient error versus independently reconverged PySCF FD | 1.134e-7 Hartree/bohr |
| Maximum bond-gradient error versus GradSCF FD | 1.133e-7 Hartree/bohr |

The independent analytic reference explicitly solves
`pyscf.cc.ccsd_t_lambda.kernel(tol=1e-11)` and supplies those amplitudes to
`pyscf.grad.ccsd_t.Gradients`; ordinary CCSD Lambda amplitudes are not used.
Each external displaced sample independently reconverges RHF, CCSD and `(T)`.
Central bond-length steps are 3e-4 and 1e-4 bohr. Transverse force components
and the translational gradient sum pass at 2e-8 Hartree/bohr.

```bash
env CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONPATH=src \
  taskset -c 48-63 timeout 1200 python -m pytest -q \
  tests/cc/test_streamed_triples_geometry.py
```

Final evidence is in
`/home/yjiao/gradscf_checks/ci-mp-response-20261020/ccgeom/`:
`ccgeom_final_junit.xml`, `ccgeom_final_resources.txt`, and matching
`source_final_before.json`/`source_final_after.json`. This establishes the
complete first-order geometry chain for this restricted example; it does not
establish second-order `(T)` response, a general large-molecule benchmark or
an unrestricted nuclear-gradient comparison.

### Mixed input precision

Independent review found that float32 integrals with float64 restart
amplitudes produced inconsistent loop-carry dtypes in the new streamed kernel.
Two focused regressions reproduced this failure before the repair. The kernel
now promotes its working inputs to their common dtype before entering the
custom AD rule; the casts' transposes retain each original input's cotangent
dtype. No global precision setting or default algorithm was changed.

The two new cases, the three existing float64 connected/disconnected/total
value-and-VJP oracles, and explicit second-AD rejection passed together:
six cases, 6.62 s pytest, 7.15 s wall time. A subsequent restricted spin-bridge
check and the two mixed cases passed in 5.67 s pytest. The bridge already
preserved the common amplitude dtype and required no implementation change.
These overlapping groups are not nine unique cases. Both groups used remote
CPU32--47, float64 enabled, eight BLAS threads and all CUDA devices hidden.
Evidence is under `artifacts/ci_mp_validation/cc_dtype/` in the worktree.
The complete nuclear-gradient group above preceded this common-dtype repair;
its float64 algebra is unchanged, but it was not rerun for the mixed-input
increment.
