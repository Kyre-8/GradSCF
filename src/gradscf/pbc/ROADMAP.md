# Periodic development acceptance ledger

Goal started from release/v1.0.0 at `01de656` on 2026-10-13. The objective remains
incomplete until all five stages meet their implementation and validation gates.
A prototype, a passed unit test, and a converged material benchmark are distinct
states. An unsupported combination must be documented and resolved before its
parent capability is marked complete.

## Status review: 2026-10-15

Reviewed against local `release/v1.0.0` at `2a916c2`, after the stage-1 merge
and example reorganization. This review checks source/API restrictions and
the recorded evidence below; it does not rerun numerical calculations. Local
uncommitted NNAO/SCF work is not counted as delivered periodic functionality.
This document is the canonical five-stage checklist; detailed theory and
benchmark records remain in their owning modules.

| Stage | Current state | Open acceptance items |
| --- | --- | ---: |
| 1. Ground-state foundations | Complete within the declared scalar 3D Gaussian/GTH scope | 0 |
| 2. Forces, stress and reference consistency | Derivative infrastructure implemented; material and branch-continuation gates open | 3 |
| 3. DFPT and phonons | HF/KS primitive response, OFDFT supercell reference and Fourier IFC implemented; extensions and material validation open | 6 |
| 4. Correlation and spectra | q=0 TD and partial periodic GW implemented; method/spin/AD/optical coverage incomplete | 7 |
| 5. Spinors and macroscopic fields | Foundational implementations and validation still required | 5 |

There are **21 open acceptance items**. They are umbrella tasks, not equal-sized
units or a percentage estimate. An open item can contain working infrastructure;
its missing implementation and validation are specified below. Historical
"Stage 2" labels inside `gw/pbc` refer to that subproject's own development
sequence, not stage 2 of this roadmap.

## Execution constraints

- Develop in isolated worktrees; preserve unrelated main-worktree changes.
- Run every test, numerical probe, example and benchmark on `jy_gpu_1` using
  `jax_scf`. Local work is editing, source review and static checks only.
- Short tests default to CPU. Material jobs may use one currently idle GPU,
  after a real JAX/CUDA probe, with a timeout of at most two hours per job.
  Multi-GPU or longer jobs require separate user approval. Do not displace an
  existing workload or assume that a past free-device observation is current.
- Reuse the integral, XC derivative, SCF, linear and eigenvalue infrastructure.
- Record source revision/snapshot, hardware/backend, dtype, physical input,
  convergence settings, reference method, errors and elapsed time for gates.
- Merge/push completed milestones when requested; do not publish unfinished work
  or include unrelated local changes. Keep the goal active across milestones.

## Current priority

Stage 1 is complete as of 2026-10-15 within the declared scalar, neutral 3D
Gaussian/GTH scope. Periodic snapshots use the shared Sample/Trainer in all
three modes. Spatial maps now drive irreducible target J/K and XC projection;
the final state and implicit residual retain the full mesh. External functionals
can explicitly define their finite-q energy, whose AD Hessian supplies the
response kernel. The final boundary and derivative checks are recorded below.

The next priority is stage 2: independent force/stress comparisons at relaxed
material geometries, basis/FFT/k convergence and explicit SCF branch tracking.
Stages 2 through 5 remain incomplete; stage-1 closure does not close the goal.

Work order: establish branch-consistent displaced/strained references (P2.3),
complete matched force/stress and discretization records (P2.1/P2.2), then
finish a converged nonpolar phonon benchmark (P3.1). Primitive OFDFT response
(P3.2) is a separate implementation task. Dielectric/Born-charge work (P3.3)
needs the electric-field/polarization infrastructure in P5.4/P5.5 before the
LO-TO gate (P3.4) can close; the stage numbers do not remove this dependency.

The silicon pilot's memory and timing work is parked. Its diagnostic resource
limits are not project acceptance criteria and must not block unrelated
foundation capabilities. Resume material performance work when its stage-2/3
acceptance gate is being addressed explicitly.

## 1. Periodic ground-state foundations

- [x] Offline GTH bases/pseudopotentials with provenance and coverage queries.
- [x] Full and shifted k meshes, quadrature weights and Fermi occupations.
- [x] First thermal density response and time-reversal diagonalization reduction.
- [x] External energy functional integration and three array-level training modes.
- [x] Scalar Cartesian space-group operations, atom/basis mapping and AO/Bloch rotations.
- [x] Irreducible integration/J/K with numerical equivalence to the full mesh.
- [x] Periodic Trainer energy/density samples in fixed-density, explicit and implicit modes.
- [x] Explicit external-functional finite-q contract and corresponding response tests.

Acceptance: full/reduced forward agreement; geometry and parameter derivatives
against independent differences; symmetry-breaking perturbations retained by
the full response residual; physical model restrictions enforced.

Scope: reduction acts on target contractions and occupation/energy quadrature,
not on source-k sums or full AO/one-body input storage. Only symmetries of the
finite reciprocal box, real FFT grid, unwrapped k mesh and weights are used;
automatic discovery can yield identity only. External finite-q response requires
an explicit energy extension of the same functional and a closed, equally
weighted full mesh. No finite-q model is inferred from a diagonal-k callback.

## 2. Forces, stress and reference-state consistency

- [x] HF/KS and grid/Gaussian OFDFT forces and stress with moving basis/Pulay terms.
- [x] Finite-temperature free-energy derivatives and stale-reference rejection.
- [ ] **P2.1 — Relaxed-material force/stress validation.** Existing
  [geometry derivatives](grad.py) cover HF/KS and both periodic OFDFT
  representations. Add relaxed material references and independent force/stress
  comparisons with matched geometry, basis, PP, XC, occupations and grids.
  Report Eh/Bohr forces and Eh/Bohr^3 tensile stress, including finite-temperature
  free-energy derivatives where applicable. Small-cell derivative tests alone
  do not close this material gate.
- [ ] **P2.2 — Discretization convergence.** Record basis, FFT and k-mesh
  sensitivity separately at fixed physical settings. Declare energy, force and
  stress tolerances before comparing the runs. Existing Si FFT/basis checks are
  partial evidence; full k convergence and relaxed references remain open.
- [ ] **P2.3 — Explicit SCF branch continuation.** The current geometry facade
  reconverges from its default seed and rejects a density inconsistent with the
  stored reference; it does not follow an arbitrary metastable branch. Add
  continuation from the selected state and observable diagnostics for unintended
  branch changes. Verify displaced and strained finite differences stay on the
  intended branch, while stale-reference and unconverged-state checks remain.

Acceptance: state/energy/force consistency, translation and strain diagnostics,
external numerical references, and no hidden projection of raw force errors.

## 3. DFPT and phonons

- [x] Static molecular/periodic KS and OFDFT stationary response.
- [x] Primitive finite-q HF/KS displacement sources and full geometry curvature.
- [x] Restricted/unrestricted, zero/finite-T checks and shifted/even-mesh gates.
- [x] Independent commensurate-supercell HF/KS/OFDFT reference path.
- [x] Fourier force constants, shortest-image weighting, dispersion and DOS.
- [ ] **P3.1 — Converged nonpolar phonons and scaling.** Finish an independent
  material dispersion comparison after P2.1/P2.2, with raw force constants,
  ASR and imaginary modes retained. Measure complete calculations, cold/warm
  timings and peak RSS over meaningful basis/k/q sizes. The Si pilot, one large-k
  response column and compile-only buffer estimates do not close this gate.
- [ ] **P3.2 — Primitive finite-q OFDFT.** Extend the stationary/KKT response to
  grid and Gaussian OFDFT with atomic sources and geometry curvature. Reuse the
  OFDFT functional derivatives and shared response solvers; validate number
  constraints and q/-q symmetry against the independent supercell path.
  [Phonons](phonon.py) currently rejects primitive finite-q OFDFT explicitly.
- [ ] **P3.3 — Dielectric tensors and Born effective charges.** Build the required
  periodic field/polarization interface together with P5.4/P5.5. Validate volume,
  charge, unit and gauge conventions, tensor symmetries and independent physical
  references. A response to an arbitrary AO potential is not this capability.
- [ ] **P3.4 — Polar IFCs and LO-TO splitting.** Once P3.3 is validated, implement
  consistent long-range subtraction before interpolation and restoration at the
  target q, including the direction-dependent nonanalytic limit. Compare a polar
  material to an independent reference and check against double counting.
- [ ] **P3.5 — Higher periodic response.** Extend Taylor propagation to all
  momentum sectors generated by the perturbation (including 0, q and 2q), with
  the necessary primitive/geometry derivative rules. Verify coefficients against
  independent differences or small reference problems. A one-sector linear
  finite-q solve does not establish all-order or thermal Taylor support.
- [ ] **P3.6 — Degenerate phonon observables.** Reuse shared spectral-subspace
  response for projectors, cluster traces and other basis-invariant observables.
  Test exact/near degeneracy and a separated cluster boundary; individual sorted
  frequencies or mode vectors need not have unique derivatives at a crossing.

Acceptance: q-grid round trips, Hermiticity and q/-q conjugacy, unit/gauge/mass
checks, independent models and first-principles comparisons, raw ASR diagnostics,
converged material dispersions and performance records. Imaginary modes remain
visible. A projected Gamma zero is not evidence of mesh convergence.

Completed pilot implementation: **Fourier force constants and silicon**. Interpolation
passes analytic-lattice and small-cell tests; the full silicon 2-by-2-by-2 q grid
and DOS run on CPU. The fixed-geometry SZV pilot has imaginary branches and is
not a converged nonpolar material benchmark. Its remaining convergence and
performance gates stay open.

The negative L curvature is independently reproduced by PySCF. FFT31³ to 41³
and SZV to DZVP sensitivity checks are complete, but neither removes the pair.
Finite-q actions now use runtime arrays, independent reference snapshots,
batched k operations and streamed bilinear rows. DZVP/k2³ completes within the
monitored resource envelope. Phase profiling is complete: reference-force
construction finishes, but the original full-column action requests about
39.90 GiB of temporary buffers. Streaming the two shared response contractions
reduces that estimate to 23.35 GiB. The earlier 32-GiB diagnostic threshold was not met; a
separate 48-GiB diagnostic completes one k4³ column and its hot repeat, taking
249.39 s (cold) and 178.93 s (hot). The full matrix and k-convergence gate remain
open. Future performance work should measure the remaining AO/geometry and
projection costs before selecting a larger k block. Do not infer
material stability from ASR projection.

The 32/48-GiB values above describe historical diagnostics, not a required
production memory budget. The outstanding scientific gate is a complete,
converged material calculation with measured resource use.

## 4. Periodic correlation and spectra

- [x] Existing q=0 periodic TDHF/TDDFT and partial periodic GW infrastructure.
- [ ] **P4.1 — Periodic static BSE.** Add periodic reference/screening and
  momentum-resolved electron-hole operators while reusing the molecular BSE
  and shared solver infrastructure where mathematically applicable. Validate
  excitations and optical observables against independent periodic results;
  the existing [BSE facade](../bse/api.py) is molecular.
- [ ] **P4.2 — Unrestricted k-point GW.** Extend the existing Gamma-point
  [KUGW](../gw/pbc/kugw.py) to full k sampling and implement its missing q-to-zero
  head/wing correction. Check spin factors, closed-shell limits and corrected
  versus uncorrected finite-size behavior independently.
- [ ] **P4.3 — Periodic GW method and AD coverage.** Start from the existing
  restricted Gamma/k-point G0W0 and Gamma evGW paths. Record a method-by-method
  matrix for G0W0, evGW0, evGW, qsGW and scGW: spin, sampling, occupation,
  correction, forward and backward coverage. Complete the required endpoints
  with independent validation. In particular, [k-point G0W0](../gw/pbc/ksigma.py)
  currently rejects JAX transformations; molecular implicit differentiation does
  not establish a working periodic backward path.
- [ ] **P4.4 — Metallic excited-state response.** Supply fractional-occupation
  TD/GW/BSE kernels and their supported gradients, including intraband and
  zero-temperature limiting checks. Thermal SCF and static first response are
  already present, but their existence does not establish metallic spectra.
- [ ] **P4.5 — General periodic optics and dynamics.** Extend the current
  restricted-Gamma velocity-gauge [optical strengths](optics.py) to the required
  k/spin sectors and finite-q dynamical response. Keep per-cell oscillator
  strengths distinct from macroscopic dielectric/absorption quantities and
  establish gauge, normalization and nonlocal-PP checks.
- [ ] **P4.6 — Periodic MP/CC.** Establish momentum-conserving excitation spaces,
  integral actions and method endpoints before adding response. Reuse shared
  Taylor/nonlinear/linear solvers; validate Gamma/supercell limits and matched
  periodic energies. Molecular methods alone do not satisfy this gate.
- [ ] **P4.7 — Electronic/phonon coupling integration.** Connect validated
  periodic displacement responses, mode conventions and k/q indexing to GW/BSE
  spectral consumers. The existing prescribed-model periodic Fan example is
  not a first-principles periodic phonon-to-GW/BSE workflow. Validate phases,
  masses, units, momentum conservation and supported derivatives end to end.

Acceptance: method/occupation/spin coverage recorded explicitly; matched reference
inputs for external comparisons; forward and supported backward checks; zero
temperature insulating limits; stable memory and iteration diagnostics.

## 5. Spinors and macroscopic fields

- [ ] **P5.1 — Spinor/noncollinear ground states.** Implement spinor densities
  and compatible XC energy/potential/kernel contracts; establish collinear and
  scalar limits before claiming noncollinear response.
- [ ] **P5.2 — SOC and relativistic data.** Add traceable relativistic PP assets
  and SOC operators, with explicit conventions and independent SOC-off and
  finite-SOC numerical checks.
- [ ] **P5.3 — Spinor response and symmetry.** Extend spatial/time-reversal
  transformations, response operators and invariant degenerate observables to
  spinors. Complex scalar AO arrays are not a spinor implementation.
- [ ] **P5.4 — Berry-phase polarization.** Implement gauge-invariant k-link
  operations and polarization branches, with mesh and gauge tests. Reuse this
  field interface for P3.3 rather than inventing a molecular position operator
  for a periodic crystal.
- [ ] **P5.5 — Macroscopic fields and derivatives.** Add compatible field,
  force and mixed-response interfaces, validating the connection to P3.3 Born
  charges and dielectric tensors at consistent conventions and normalization.

Acceptance: scalar/SOC-off and collinear limits, gauge tests, physical reference
systems and external numerical comparisons, and tested first derivatives of
the supported invariant observables. Complex arrays alone do not meet this gate.

## Ownership and reuse

This checklist is stored in `pbc`, but does not move every periodic algorithm
into this package or introduce module-local copies of shared solvers.

| Work | Existing implementation owners | Starting regression/reference files |
| --- | --- | --- |
| P2 geometry and branches | [pbc.grad](grad.py), [pbc.scf](scf.py), periodic integrals, OFDFT | [force tests](../../../tests/pbc/test_forces.py), [gradient tests](../../../tests/pbc/test_gradients.py), [example](../../../examples/pbc/periodic_forces.py) |
| P3 electronic/atomic response | [dfpt](../dfpt/README.md), [pbc.phonon](phonon.py), [pbc.ifc](ifc.py), OFDFT and shared solvers | [primitive Hessian tests](../../../tests/dfpt/test_primitive_hessian.py), [phonon tests](../../../tests/pbc/test_phonon.py), [silicon workflow](../../../examples/dfpt/README.md) |
| P4 correlation/spectra | [gw/pbc](../gw/pbc/), [bse](../bse/), [pbc.tdscf](tdscf.py), [optics](optics.py), MP/CC | [periodic GW tests](../../../tests/gw/pbc/), [TD tests](../../../tests/pbc/test_tdscf.py), [optical tests](../../../tests/pbc/test_optics.py) |
| P5 spin/field foundations | periodic cell/integrals/SCF/XC, [symmetry](symmetry.py), DFPT | Extend [symmetry tests](../../../tests/pbc/test_symmetry.py) and add independent spinor/field references |

## Coverage boundaries and verification debt

These boundaries qualify the existing capabilities; they are not a reason to
reopen the already accepted stage-1 scope or silently expand it.

- The current native cell path is neutral, three-dimensional, scalar and
  Cartesian Gaussian/GTH. Lower-dimensional electrostatics, charged-cell
  conventions and a general all-electron periodic backend are outside the
  accepted first-stage scope.
- Ground-state built-in XC covers HF/LDA/GGA/global hybrids, not a complete
  molecular XC catalogue. [Off-mesh band queries](bands.py) currently support
  pure LDA/GGA; hybrid and external-model bands need additional operators.
- External `q_energy_fn` supplies **electronic potential response**. The native
  displacement and primitive-force adapters still reject external models;
  atomic/moving-basis and supercell extensions must be defined and checked
  before claiming arbitrary-model phonons. External finite-q hybrids also
  require an explicit adapter. Track these as response coverage within P3/P4.
- Periodic [Trainer](training.py) supports energy and grid-density labels in
  all three modes. Force/stress/response losses are not yet supplied by this
  periodic sample adapter; array-level derivatives do not automatically add
  those loss types. Extend the existing trainer if these targets are required.
- The accepted irreducible path reduces target contractions and iteration
  diagonalization, while retaining full AO/source-k data and full implicit
  response. Larger-scale source/storage reduction is a separate measured
  optimization, not an unimplemented part of its existing mathematical contract.
- The stage-1 closure suite skipped a PySCF-2.9.0-pinned basis-source oracle
  and deselected three larger basis/material cases. Retain these qualifications
  when reporting validation; rerun them in the appropriate reference environment
  or material campaign rather than counting them as passes.
- The recorded JAX 0.10.0 silicon CUDA compiler failure is an environment/backend
  validation gap. A successful small CUDA probe does not establish a working
  material calculation. CPU results and GPU validation must be reported separately.
- Existing experimental material timings, negative modes and memory diagnostics
  remain in [IFC.md](IFC.md). Resume that work for P2/P3 scientific acceptance;
  do not let an arbitrary memory cap replace feature or accuracy requirements.

## Evidence

- 2026-10-15, graphite-driven extension: `Phonons.project` reuses the existing
  force-response action and checks the generalized mass metric, full-space
  leakage and basis-scale-independent Hermiticity. Local frozen-mode fit/direct
  jet utilities and relative one-phonon loss/gain spectra are implemented with
  explicit scope. Seventy distinct focused/regression cases pass on `jy_gpu_1`
  CPU float64. The [graphite pilot](../../../examples/thermally_excited_graphite/RESULTS.md)
  verifies directional AD/DFPT curvature against relaxed displaced energies and
  demonstrates electronic FFT convergence against PySCF. It does **not** close
  P3.1 or P3.5: the default material sampling is unconverged, and neither a local
  Gamma expansion nor instrumental broadening is a bulk anharmonic self-energy.
- Baseline `01de656`: see [ATOMIC_RESPONSE.md](../dfpt/ATOMIC_RESPONSE.md) for native phonon and regression
  evidence. The merge gate verified 811 committed source/test/data files and
  passed 42 tests (one PySCF-version-pinned source oracle skipped) on `jy_gpu_1`.
- All new evidence below must identify the actual tested snapshot and backend.
- IFC snapshot `ebafa62a3dc0927fd6c3cca1c8c99056b105a7e151e809fdea9021e1f390f84f`:
  `tests/pbc/test_ifc.py tests/pbc/test_phonon.py -k 'not finite_q'` gave
  **13 passed, 4 deselected**, 82.05 s, JAX 0.10.0 CPU float64 on `jy_gpu_1`.
- Si PBE/GTH-SZV/GTH-PBE, a=5.43 Angstrom, k/q=2³, FFT=31³: eight primitive
  q solves completed in 847.39 s including compilation; maximum response
  residual 4.08e-12, raw Gamma ASR norm 5.26e-5 Eh/Bohr². DOS integrates to six.
  These are cold per-point timings, not a warm-call scaling measurement.
- FFT 21³ to 31³ changes the raw Gamma force-constant matrix by up to
  1.47e-4 Eh/Bohr², despite only a 3.28e-7 Eh/cell total-energy difference.
  The pilot therefore does not satisfy the material convergence gate.
- Independent re-converged PySCF energy differences pass at both Gamma and
  q=(0,0,1/2), using a 1x1x2 supercell for the latter. At displacement 0.001
  Bohr, errors are 3.09e-6 and 4.01e-6 Eh/Bohr², respectively (gate 2e-5).
  Both also pass at 0.002 Bohr. See example 07 for exact inputs and output.
- CUDA small-array probes passed, but silicon SCF with installed JAX 0.10.0
  failed an XLA layout-normalization assertion. Disabling cuBLASLt for one
  diagnostic job did not fix it. No permanent compiler flags were installed;
  this environment-specific GPU gate remains open.
- 2026-10-14: negative-mode PySCF curvature -0.005424312 versus native
  -0.005430210 Eh/Bohr² at a 0.001-Bohr step; all six Gamma/boundary directional
  differences pass the 2e-5 Eh/Bohr² gate in example 07.
- Runtime-array regression initially failed with 45.69 MB of captured constants;
  snapshot-mutation and k-trace-growth regressions also reproduced their failures
  before repair. Final Bloch tests: 3 passed, 10.19 s. Batching integration gate:
  23 passed, 741.74 s. Final streamed action gates: 3 AD/snapshot tests passed in
  508.20 s and 3 physical tests passed in 246.28 s. Snapshot archive hashes are
  `47125553c1ed17d567f27abe608eeebc3ad9f59aca9445ace12993bdd8183621`
  (batched) and `725e3e4bf4e1eae985f3db8f1167c1f4281c8a2b9df7b9fd63d66935596ae839`
  (final streamed source/test overlay); CPU float64, JAX 0.10.0 throughout.
- Matched SZV full-IFC difference after streaming: 9.38e-15 Eh/Bohr². Completed
  SZV/DZVP single-L-point cold calculations took 185.63/313.82 s and peaked at
  15.14/30.97 GiB RSS. This is not a warm-call speedup or a scaling fit. The
  final k4³ attempt exited 143 after the memory monitor fired; no k4³ spectrum
  or force-constant matrix was obtained. See IFC.md for the detailed boundaries.
- Profiling `8da6036`: SCF 31.49 s; reference forces 24.00 s, peak process RSS
  15.72 GiB at that stage. XLA finished compiling the first action before the
  0.2-second RSS monitor stopped it. Separate compile-only probes estimate
  39.90 GiB of full-column temporary storage and 30.00 GiB for a dynamic
  single-row action; neither probe executes the large compiled action. Exact
  orbital translation reuse and a two-JIT-stage trial offered no useful
  reduction and were reverted. See IFC.md for logs and measured boundaries.
- Shared k-streamed contractions: eight focused value/AD/dtype/memory checks
  pass in 12.77 s; eight HF/PBE kernel/displacement/geometry checks pass in
  154.84 s. Final full-column XLA temporary estimate: 25,068,718,408 bytes
  (23.35 GiB). The 48-GiB material diagnostic completes a column with residual
  4.21e-13 and repeats it within 1e-12. This is not a completed k4³ IFC matrix
  or evidence of a speedup against an executed original k4³ calculation.
- Complete-action snapshot/JVP/VJP checks: three passed in 507.19 s (19 passing
  checks total for this contraction change). The material process peaked at
  32.71 GiB and timed out after 600 s while computing the second column; its
  measured hot repeat took 178.93 s. No completed k4³ spectrum is claimed.
- 2026-10-15, periodic Trainer adapter on baseline `b21ef41`: ten new regressions
  first failed on the baseline, then passed in 101.39 s. Three additional Gamma,
  unrestricted and deliberate convergence-opt-out checks passed in 72.52 s;
  26 molecular training regressions passed in 58.42 s. The latter reported one
  existing JAX deprecation warning from the invalid-learning-rate list test.
  All 39 checks ran on `jy_gpu_1`, `jax_scf`, CPU float64, JAX 0.10.0, with
  OMP/OpenBLAS threads set to two. No local numerical tests were run.
  The source/test overlay SHA256 is
  `e51f95bad239afc2a528989ee5c4f97e1eb670fae0e537eaff801384206e0e06`;
  added coverage/example overlay is
  `eaba01e08844f5e40fffd2a1b793ddd8b15c47abedd827817e11682a9fa894df`.
  Logs are `red.log`, `green.log`, `coverage.log`, `molecular.log`, `example.log`
  under `/home/yjiao/gradscf_checks/pbc-training-b21ef41`.
- The shared-Trainer four-layer periodic XC example completes three Adam steps
  in each mode: MSE falls from approximately 8.04e-5 to 4.14e-5 Eh²; the separate
  q=0 DFPT electron-number derivative is -2.78e-17. Its labels now use internal
  energy per cell, explicitly replacing the earlier hand-written free-energy
  training loop. This is a synthetic teacher/API example, not a material model.
- 2026-10-15, spatial symmetry on baseline `bb24e5d`: 16 initial tests failed
  before the module existed. Four finite-geometry regressions reproduced the
  NaN-center acceptance hole and nonspecific rejection of other NaN geometry;
  these now fail early with explicit finite-real checks. Final symmetry suite:
  **23 passed in 8.81 s**, including non-symmorphic phases/composition, atoms
  outside the cell, Cartesian s through g polynomials, contracted-f Fourier
  covariance, nonstandard lattice bases, PySCF overlap/kinetic covariance,
  stale metadata and JAX derivatives. Seven existing k-point/time-reversal and
  optional-dependency checks passed in 111.36 s. No numerical tests ran locally.
  Environment: `jy_gpu_1`, `jax_scf`, CPU float64, OMP/OpenBLAS threads two,
  JAX 0.10.0, spglib 2.7.0, PySCF 2.13.0. Six upstream spglib deprecation
  warnings were reported; no tests were skipped. Automatic discovery is an
  optional dependency; explicit operations remain independent of it.
  Final source/test overlay SHA256:
  `ca84d172677baac030322c4c0b90e601190b81df55fdcb0adb91431a6e3b8f01`.
  Logs: `/home/yjiao/gradscf_checks/pbc-symmetry-bb24e5d/{red,boundary-red,final,regression,example}.log`.
  The example discovers 48 operations for its hexagonal two-He cell and
  preserves `Tr(P F)` to 1.21e-13. It is an AO/gauge example, not an SCF result.
  Finite-FFT covariance, irreducible J/K and symmetry-breaking response remain
  the next integration gates; no space-group speedup is claimed here.
- 2026-10-15, stage-1 completion on `e8b888c`: the combined irreducible/finite-q
  gate passed **35 tests in 475.11 s**; existing HF/KS/thermal/time-reversal
  regressions passed **28 in 356.03 s**. Basis/sampling/occupation/external-XC/
  Trainer regressions passed **34 in 353.55 s**, with one version-pinned basis
  source oracle skipped (requires PySCF 2.9.0; installed 2.13.0) and three
  larger basis/material cases deselected. These are 97 passes, not a full-suite
  or material convergence result. The combined source/test overlay SHA256 is
  `d815ad5acde535e1b2666e757bbf1ce03662a4e40554f733e8e85eeaecbd046b`.
- Two final boundary checks were added: one-spin J-only output shape and a
  symmetry-tied Gaussian-exponent derivative against full-grid differences.
  The former failed before the one-line spin-axis fix. Final irreducible/combined
  boundary files passed **16 tests in 208.55 s**; this repeats 14 earlier cases
  and adds two, for **99 distinct passing cases** across this completion gate.
  Final source/test overlay SHA256:
  `82408fbd7069e71ebf70acf21b0bf72577762a548a764159f8b5468b809c4cb2`.
  No other numerical source changed after the combined gate. Five upstream
  spglib deprecation warnings were reported in this final run.
- All completion tests ran on `jy_gpu_1`, Intel Xeon Platinum 8457C, `jax_scf`,
  JAX 0.10.0 CPU float64, OMP/OpenBLAS threads two, with a 900-second timeout
  per batch. Logs are under `/home/yjiao/gradscf_checks/pbc-ibz-e8b888c`:
  `combined-new.log`, `ground-regression.log`, `api-regression.log` and
  `final-boundaries.log`. Commands, from that directory, with
  `CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=2
  OPENBLAS_NUM_THREADS=2 PYTHONPATH=src`:

  ```sh
  python -m pytest -q tests/pbc/test_reduction.py tests/pbc/test_irreducible_scf.py tests/pbc/test_ibz_boundaries.py tests/dfpt/test_external_finite_q.py
  python -m pytest -q tests/pbc/test_scf.py tests/pbc/test_dft.py tests/pbc/test_thermal_scf.py tests/pbc/test_time_reversal_scf.py
  python -m pytest -q tests/pbc/test_basis_library.py tests/pbc/test_foundation.py tests/pbc/test_weighted_kpoints.py tests/pbc/test_occupations.py tests/pbc/test_external_functional.py tests/pbc/test_training.py -k 'not silicon_molopt and not molopt_hydrogen'
  python -m pytest -q tests/pbc/test_irreducible_scf.py tests/pbc/test_ibz_boundaries.py
  ```

  The first command now collects two additional cases on the final source.
  Tested runtime/test files were checked by SHA256 against the local worktree;
  no numerical tests ran locally. Example formatting/comments changed only
  after execution, with exact Python AST equivalence checked locally.
- Irreducible tests cover HF with/without Ewald, handwritten PBE, thermal and
  integer occupations, unrestricted spin, nonunitary Cartesian s/p/d frames,
  explicit/implicit geometry and model-parameter derivatives, symmetry-breaking
  implicit response, stabilizers and finite-grid incompatibilities. The H2
  example uses a 6-Bohr cubic cell, H at +/-0.7 Bohr, the default GTH-SZV/GTH-PADE
  inputs, FFT 9 cubed, k=(3,1,1), kBT=0.2 Eh and no Ewald correction. It has
  3 full/2 representative k points, 9 full/6 representative target-source exchange
  pairs, identical free energy -1.1169946341008274 Eh/cell and maximum density
  difference 1.82e-13. This is a coarse API check, not a timing/scaling benchmark.
- External finite-q tests include independent full-matrix supercell Hessians
  for a nonlinear nonlocal energy at two and three k points, q/-q conjugacy,
  complex linearity, self-inverse q and parameter/reference AD. A nonlinear
  GGA-type SCF-to-screened-response calculation agrees with its independent
  supercell AO response to 2.68e-10; its relaxed parameter derivative is
  -0.0306164785926 versus finite difference -0.0306164786079. Freezing the
  reference gives -0.0263801801854, confirming that relaxation matters.
  The nonlocal model is validated at Hessian level, not as a separate complete
  nonlocal SCF-loss benchmark. See [FINITE_Q.md](../dfpt/FINITE_Q.md) for inputs,
  normalizations and independent-oracle construction.
- Both `examples/pbc/periodic_irreducible.py` and
  `examples/dfpt/03_external_finite_q.py` exited zero on the final runtime
  snapshot. Their logs are `irreducible-example.log` and `external-example.log`
  in the same remote directory; measured output is included as code comments.
  The finite-q example reports free energy -0.6104181508891107 Eh/cell and
  density-response norm 0.05878070762111205.
