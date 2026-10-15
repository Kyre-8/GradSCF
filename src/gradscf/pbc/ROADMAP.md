# Periodic development acceptance ledger

Goal started from release/v1.0.0 at `01de656` on 2026-10-13. The objective remains
incomplete until all five stages meet their implementation and validation gates.
A prototype, a passed unit test, and a converged material benchmark are distinct
states. An unsupported combination must be documented and resolved before its
parent capability is marked complete.

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
- [ ] Materials with relaxed geometries and independent force/stress comparisons.
- [ ] Basis/FFT/k-mesh convergence records at fixed physical settings.
- [ ] Explicit branch-following support and rejection of unintended SCF branch changes.

Acceptance: state/energy/force consistency, translation and strain diagnostics,
external numerical references, and no hidden projection of raw force errors.

## 3. DFPT and phonons

- [x] Static molecular/periodic KS and OFDFT stationary response.
- [x] Primitive finite-q HF/KS displacement sources and full geometry curvature.
- [x] Restricted/unrestricted, zero/finite-T checks and shifted/even-mesh gates.
- [x] Independent commensurate-supercell HF/KS/OFDFT reference path.
- [x] Fourier force constants, shortest-image weighting, dispersion and DOS.
- [ ] Nonpolar material benchmark, external comparison and measured scaling.
- [ ] Primitive finite-q OFDFT response and geometry, in both representations.
- [ ] Dielectric tensors and Born effective charges with normalization checks.
- [ ] Long-range subtraction/restoration and nonanalytic LO-TO correction.
- [ ] Higher-response/Taylor support with all required momentum sectors.
- [ ] Degenerate-subspace invariant phonon response observables.

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
reduces that estimate to 23.35 GiB. The 32-GiB RSS gate is still not met; a
separate 48-GiB diagnostic completes one k4³ column and its hot repeat, taking
249.39 s (cold) and 178.93 s (hot). The full matrix and k-convergence gate remain
open. Future performance work should measure the remaining AO/geometry and
projection costs before selecting a larger k block. Do not infer
material stability from ASR projection.

## 4. Periodic correlation and spectra

- [x] Existing q=0 periodic TDHF/TDDFT and partial periodic GW infrastructure.
- [ ] Periodic static BSE with external excitation/optical comparisons.
- [ ] Full k-point unrestricted GW and its q-to-zero head/wing corrections.
- [ ] Audit and complete periodic support of the required GW self-consistency paths.
- [ ] Metallic/fractional-occupation TD/GW/BSE response and associated gradients.
- [ ] General periodic optical matrix elements and finite-q dynamical response.
- [ ] Momentum-conserving periodic MP/CC foundations with validated method endpoints.
- [ ] Consistent periodic electronic/phonon coupling interfaces for spectral consumers.

Acceptance: method/occupation/spin coverage recorded explicitly; matched reference
inputs for external comparisons; forward and supported backward checks; zero
temperature insulating limits; stable memory and iteration diagnostics.

## 5. Spinors and macroscopic fields

- [ ] Spinor/noncollinear SCF and compatible XC density/potential/kernel interfaces.
- [ ] SOC operators with traceable relativistic pseudopotential data.
- [ ] Spinor response, symmetry and degeneracy-invariant observables.
- [ ] Berry-phase polarization and gauge-invariant k-link operations.
- [ ] Macroscopic electric-field response and compatible forces/Born charges.

Acceptance: scalar/SOC-off and collinear limits, gauge tests, physical reference
systems and external numerical comparisons, and tested first derivatives of
the supported invariant observables. Complex arrays alone do not meet this gate.

## Evidence

- Baseline `01de656`: see `ATOMIC_RESPONSE.md` for native phonon and regression
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
- Both `examples/periodic_irreducible.py` and
  `examples/dfpt/03_external_finite_q.py` exited zero on the final runtime
  snapshot. Their logs are `irreducible-example.log` and `external-example.log`
  in the same remote directory; measured output is included as code comments.
  The finite-q example reports free energy -0.6104181508891107 Eh/cell and
  density-response norm 0.05878070762111205.
