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

## 1. Periodic ground-state foundations

- [x] Offline GTH bases/pseudopotentials with provenance and coverage queries.
- [x] Full and shifted k meshes, quadrature weights and Fermi occupations.
- [x] First thermal density response and time-reversal diagonalization reduction.
- [x] External energy functional integration and three array-level training modes.
- [ ] Physical space-group operations, atom/basis mapping and AO/Bloch rotations.
- [ ] Irreducible integration/J/K with numerical equivalence to the full mesh.
- [ ] Periodic Trainer samples, preserving fixed-density, explicit and implicit modes.
- [ ] Explicit external-functional finite-q contract and corresponding response tests.

Acceptance: full/reduced forward agreement; geometry and parameter derivatives
against independent differences; symmetry-breaking perturbations retained by
the full response residual; physical model restrictions enforced.

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

Current milestone: **Fourier force constants and a silicon pilot**. Interpolation
passes analytic-lattice and small-cell tests; the full silicon 2-by-2-by-2 q grid
and DOS run on CPU. The fixed-geometry SZV pilot has imaginary branches and is
not a converged nonpolar material benchmark. Its remaining convergence and
performance gates stay open.

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
