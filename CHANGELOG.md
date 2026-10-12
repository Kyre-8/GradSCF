# Changelog

## Unreleased — GradSCF

### Model-owned training and workflows

- Move XC training and workflows into `gradscf.model.neural_xc`; remove the
  former top-level namespaces and update their callers.
- Separate NNAO integration, initialization, training and CLI workflows.
  Keep the unchanged MACE snapshot inside `model/nnao/vendor/mace-jax`.
- Document model ownership and the standard development/validation process
  in the root README and model guides. Keep local editor settings and root
  working documentation out of repository synchronization.

### Native differentiable basis integrals

- Add CPU native contraction-coefficient JVP/VJP and Hessian products, and
  first-order Gaussian-exponent JVP/VJP for one-electron and fixed-auxiliary
  three-center integrals. Keep normalization and fixed-metric whitening in JAX.
- Organize basis definitions/resources, molecular contractions, native FFI/AD
  and SCF input assembly by ownership; preserve basis data and vendor bytes.
- Share the physical-input implicit DF-RHF solver across basis updates and add
  joint MACE-coefficient/exponent Adam training with immutable-parameter masks,
  parent-energy/gradient checks and accepted-state checkpoints.
- Preserve failed-candidate diagnostics without applying invalid gradients or
  Adam moments; add numerical GMRES solve margin while retaining true-residual
  acceptance thresholds.
- Validate CPU integral/SCF/training boundaries and wheel resource ownership.
  Exponent Hessians, mixed basis/geometry rules and GPU integral FFI remain
  outside the new native basis-derivative coverage.

### Shared MP Taylor graphs

- Reuse each linear Hamiltonian coefficient across residual lifting and the
  Rayleigh energy graph, preserving the same complete states and outer AD.
- Add moving-integral/gap gradient and HVP regressions and MP12 R/U references.
- Record paired MP4--MP12 timings on H4, H3 and Li/Cartesian 6-31G*: observed
  CPU forward speedups of 2.32--4.86 on Apple M4 Pro, with unchanged state spaces.

### Residual-driven MP series

- Add `MP(mf, order=p)` and a generic restricted/unrestricted canonical MP
  engine with order-limited determinant spaces, normalized Taylor lifting of
  one eigen residual, and shared checked linear forward/backward solves.
- Use complete wavefunctions through floor(p/2) and a Rayleigh quotient to
  obtain all energy coefficients through p; allow higher state orders, with
  explicit capacity checks and invalid-response guards.
- Retain optimized low-order kernels, automatically route UHF MP3 to the series
  engine, and expose per-order energies and optional wavefunction coefficients.
- Add native H6 MP2–MP6 and water MP2–MP8 demonstrations, independent full-space
  RS comparisons, and a larger-basis water native-PySCF MP2 comparison.
- Extend Cartesian 6-31G* comparisons to nine RHF/UHF systems through MP6,
  including quadruple excitation ranks, frozen cores and a polarized quartet;
  share the example-only PySCF RS reference with the water comparison.

### Molecular Moller--Plesset perturbation theory

- Add canonical real RHF MP2/MP3 and collinear UHF MP2 with short eager
  interfaces and a JIT/AD array entry, frozen cores/virtuals, spin-resolved E2
  and explicit physical-denominator/canonical-state validity diagnostics.
- Stream DF-MP2 occupied-index slices without full MO ERIs, optionally omit
  stored doubles, and share selected dense/packed/DF orbital-block transforms.
- Obtain MP3 from the existing connected CC doubles interaction action,
  validate it against independent determinant-space perturbation coefficients,
  and demonstrate complete implicit-HF parameter response. The specialized MP3
  path remains in-core; the generic engine above extends its initial order scope.

### Exciton–vibration optical response

- Add `bse.ep_coupling` with restricted TDA electron-minus-hole projection,
  optional electronic-kernel derivative actions, fixed-bath bosonic Fan/DW,
  causal matrix spectra and resonant/antiresonant optical response.
- Support explicit excitation Hamiltonian matrices without eigenvector AD,
  including degenerate spaces; retain static BSE amplitude-response boundaries.
- Reuse shared checked linear solves through `solve_complex`, and add compact
  diagonal-mode quadratic storage to the common phonon/DW data contract.
- Add native H2/3-21G calculation and AD checks. State the low-population,
  one-phonon, frozen-GW/kernel and strong-coupling limits; reject non-passive
  absorption rather than repairing signs.
- Add differentiable finite-space vibronic Hamiltonian assembly, with proper
  quadratic cutoff boundaries, and a native three-mode water example comparing
  Fan/DW with multiphonon spectra and an independent Franck-Condon oracle.

### GW/BSE response and electron–phonon coupling

- Add real collinear unrestricted spin-conserving molecular BSE in TDA and
  full form, sharing screening and isolated-root optical response machinery.
- Add opt-in outer implicit response for molecular restricted/unrestricted
  evGW and evGW0, and restricted nondegenerate qsGW; retain explicit validity
  and convergence boundaries.
- Add fixed-phonon molecular Fan and Debye–Waller self-energies to the shared
  Matsubara scGW step, plus causal real-axis spectra, linewidths and periodic
  q-weighted kernels. Phonon feedback and a periodic Dyson loop are not included.
- Validate real-axis kernels against pinned ElectronPhonon.jl functions.
  Keep literature-derived Cu(111)/adamantane illustrations separate from the
  new native GradSCF adamantane calculation and its experimental comparison.

### Native coordinate Hessians and molecular vibrations

- Add analytic second native geometry JVP/VJP contractions for overlap,
  kinetic, nuclear attraction, dipole and full ERI, in Cartesian/spherical
  bases, including independent nuclear/center/origin mixed derivatives.
- Compose the integral Hessian with shared implicit RHF response for molecular
  Hessian-vector products and maximum-overlap transported Fock derivatives.
  Avoid differentiating arbitrary canonical orbitals in degenerate subspaces.
- Add verified s/p signed-permutation symmetry reduction and a checkpointed
  adamantane RHF/STO-3G → G0W0 → Fan/DW spectrum example. Record its minimal-basis,
  frozen-GW, harmonic and numerical-resolution limits explicitly.

### Restricted Hartree-Fock entry point

- Add `scf.RHF(mol).run()` with fixed HF settings, sharing the existing RKS
  solver, integral backends and reference pipeline. Keep `RKS(xc="hf")` valid.
- Use the RHF entry in tracked HF examples and public documentation.

### Public API ownership

- Keep domain namespaces at the root; migrate 144 flat exports to their owners.
- Replace long SCF reference-builder entry points with `mf.to_reference()`;
  let `training.Sample` and `Trainer.predict` accept converged RKS/UKS objects.
- Reuse solved SCF data and reject stale configuration or modified orbitals.
- Remove four DFT forwarding modules and six duplicate workflow wrappers;
  consolidate two preset names and publish the complete import migration table.

### Functional and model ownership

- Keep `model.neural_xc` as the concrete built-in neural XC model. Move the
  generic callback adapter to `dft.Functional` and shared training to
  `gradscf.model.neural_xc.training` (`Sample` / `Trainer`).
- Move XC derivatives to `dft.derivatives` and shared physical HFX/PT2 inputs
  to `dft.hfx` / `dft.pt2`. SCF, data loading, and external-model training no
  longer import the built-in neural model to obtain physical primitives.
- Remove old forwarding paths and the ambiguous root `Functional` export.
  Move user-defined-network examples to `examples/training` and verify the
  existing three-mode H2/FCI training histories remain unchanged.

### Full configuration interaction

- Add an independent real common-orbital FCI module with alpha/beta string
  contractions, shared Hermitian solve/response, density and transition density
  matrices, and total-spin diagnostics. Support fixed-Ms open and closed shells.
- Expose an integral-driven FCISolver protocol and frozen-core active-space
  facade. Fold cores in AO space and transform active MO integrals only.
- Preserve complete-subspace energy/projector response at degeneracy; reject
  incomplete density derivatives in energy-only mode. Reuse one fermionic phase
  implementation across CI and FCI.

- Remove the `gradscf.traditional_xc` compatibility namespace (use
  `gradscf.dft` / `gradscf.dft.xc` directly), and move the XC backends from
  `gradscf.xc_backend` to `gradscf.dft.libxc_jax`.

- Move the neural-network model code into the new `gradscf.model` subpackage:
  `gradscf.neural_xc` -> `gradscf.model.neural_xc`, `gradscf.neural_d` ->
  `gradscf.model.neural_d`,
  and the top-level `nnao` package -> `gradscf.model.nnao`.  The vendored
  mace-jax project remains a separate top-level package (sources unchanged).
  At that stage, top-level convenience symbols were unchanged; the subsequent
  public API cleanup above moves them to their owning modules.  No compatibility aliases are
  kept for the old import paths.

- Rename the distribution and Python namespace from `td-graddft` / `td_graddft`
  to `gradscf`; rename `td_graddft_tools` to `gradscf_tools`.
- Update active imports, dynamic module paths, package-data declarations,
  examples, command-line tools, and contributor documentation.
- Retain scientific method names, upstream GradDFT names, historical
  reproducibility artifacts, and the existing NPZ target-bundle format marker.
- Retire the old import namespaces without a compatibility alias.

### Closed-shell EOM-CCSD

- Repair roundoff-degenerate real Ritz bases with complete cluster duals before
  root truncation, fixing CO's rank-deficient left states while retaining the
  invalid isolated-root AD policy and true residual checks.
- Add explicit restricted SCF multistart with native orbital-rotation guesses,
  all-attempt summaries and lowest-converged-candidate selection; ordinary
  single-run defaults and the source object are preserved.

- Add a shared restarted non-Hermitian Davidson solver for all three sectors,
  with right/left/guard residual checks, orbital-difference preconditioning and
  first-order energy AD without a physical dense EOM matrix. Expose subspace
  dimensions, restarts and incomplete spectral certification explicitly.
- Validate water/6-31G (EE dimension 860) against the full PySCF action spectrum;
  distinguish the oracle's complete spectrum from roots found by its default
  two-root iterative guess.

- Add singlet EE and doublet IP/EA sectors, with frozen-space support,
  biorthogonal left/right states and first-order isolated-energy JVP/VJP
  including the converged CC amplitude response.
- Add a bounded dense real non-Hermitian reference to `gradscf.solvers`;
  retain complex/gap/conditioning diagnostics and reject invalid derivatives.
  Vector/cluster response and transition properties remain outside this implementation.
- Add native GradSCF spectrum/response examples, independent PySCF action and
  energy comparisons, and H2 particle-number FCI checks.

### Shared Hermitian eigensolver

- Unify isolated-state and spectral-subspace differentiation behind
  `solve_hermitian` and `EigenResponseConfig`, sharing one numerical Ritz solve,
  guard/boundary diagnostics and constrained Sylvester response core.
- Migrate CI, restricted/unrestricted TDA and TDA-BSE. Preserve TDA's explicit
  positive-root window and distinguish physical RPA frequency gaps from squared
  reduced eigenvalue gaps. Expand guesses to explore hidden invariant sectors.
- Remove `solve_spectral_projector`, `SpectralProjectorResult` and standalone
  implicit Hermitian Davidson AD wrappers. Use the unified subspace target for
  projector actions and cluster energy sums; raw Ritz arrays are diagnostics.

- Fix native unrestricted HF TDA/TDHF routing by representing its semilocal
  response as exactly zero while retaining nonlocal exact exchange.

### Molecular GW and BSE

- Add real closed-shell static singlet/triplet TDA-BSE with factorized Davidson
  actions and a bounded dense oracle. Reuse shared eigensolvers and screened
  linear solves; include isolated-root energy and optical-property response.
- Add bounded stable full BSE (`tda=False, solver="dense"`) with coupling
  blocks, metric-normalized X/Y and first-order amplitude response through a
  shared Cholesky-Hermitian RPA solver. Report stability margins and reject
  invalid derivatives. Add matrix-free full-BSE Davidson with a projected H/J
  pencil and a checked preconditioned metric adjoint. Distinguish iterative
  stability estimates from dense certification.
- Record QP-computation coverage and the actual screening spectrum in CD GW
  results. Add checked G0W0/evGW result snapshots for fixed-frame BSE inputs,
  without claiming evGW outer fixed-point differentiation.
- Add native water TDA/full-BSE examples, pinned QuAcK Fortran A/B kernel
  fixtures, TDHF limit comparisons and first-order response regressions.


### SCF and native integrals

- Add an explicit restricted spin-breaking stability channel with full-reference
  stationarity checks, sharing existing UKS energies and the curvature solver.
- Share bounded signed negative-mode trials between facade and integral APIs;
  only finite converged energy-lowering candidates are accepted, with complete
  attempt histories. RKS->UKS promotion requires the explicit spin channel.
- Reuse fresh solved sources without rerunning SCF and validate public energy
  aliases before following a mode, protecting energy acceptance from stale data.

- Extend the existing restricted multistart to multiple explicit seeds with one
  baseline and optional internal-stability filtering; retain all attempts.
- Share real RKS/UKS curvature analysis through the public Hermitian solver,
  reusing occupation rotations and each model's existing energy evaluator.
- Add fresh stored-state SCF stationarity and SCF/CC/EOM residual reports;
  these expose upstream precision failures without changing solver tolerances.

- Add RHF/UHF, ROHF/ROKS, and GHF/GKS workflows and opt-in UHF/UKS
  internal-stability analysis with bounded lower-energy restarts.
- Consolidate convergence, DIIS, and energy/Fock assembly; reuse converged
  RKS results when preparing response references.
- Unify implicit/unrolled SCF differentiation and force-training helpers;
  retain only the current XC binding protocols.
- Consolidate integral code, full basis assets, and vendored native sources
  under `gradscf.integrals`; remove production PySCF/GPU4PySCF calls.
- Add native coordinate JVP/VJP and mixed force/parameter differentiation.
  Native basis derivatives and pure coordinate Hessians remain unsupported.

### Periodic calculations

- Add neutral 3D GTH Gaussian HF/DFT at Gamma and uniform k meshes, with
  FFT density fitting and q=0 TDA/TDDFT. Complex k-point full TDDFT uses
  a bounded dense solver; metals, finite-q response, and low-dimensional
  electrostatics are not supported.
- Reject cached periodic response results after cell or FFT mesh changes
  until the SCF reference is recomputed.
- Add fixed-density LDA/GGA band queries and restricted Gamma velocity-gauge
  oscillator strengths including the nonlocal GTH commutator.
- Add reproducible silicon/diamond band and silicon Gamma TDDFT spectrum
  comparisons with PySCF. Gamma strengths are not macroscopic absorption
  coefficients; basis and k-mesh convergence are not claimed.

## 1.0.0 - 2026-08-24

### Features

- Provide differentiable restricted and unrestricted SCF with explicit and
  implicit gradient modes.
- Provide restricted and unrestricted TDA/Casida response solvers, Davidson
  eigensolvers, and differentiable Neural XC response paths.
- Support semilocal channels plus fixed-cache `nograd` local-HF and optional
  PT2/CIS(D)-style Neural XC channels. Ground-state HFX/PT2-SCF recomputation is
  not part of v1.0.0.
- Support CPU/libcint, JAX, density-fitting, and GPU4PySCF integral backends.
- Provide the manuscript training and evaluation entry points for H2+, H2, N2,
  QM9 ground-state, and QM9GWBSE S1/TDA tasks.

### Reproducibility

- Add selected manuscript checkpoints, final inference CSV/JSON files, compact
  reference tables, final figures, and a SHA-256 manifest under
  `reproducibility/v1.0.0/`.
- Validate the published QM9 ground and QM9GWBSE S1 checkpoints with the v1.0.0
  inference API.
- Keep only manuscript-facing examples and tools, and document conventional
  DFT/TDDFT plus Neural XC training, checkpoint, and inference workflows.
- Normalize public QM9 naming and remove machine-specific paths from released
  reproducibility metadata.

### Fixes

- Align the standalone JAX molecular grid with PySCF 2.13 for levels 0-9:
  full Lebedev tables through 1454 points, Treutler-Ahlrichs radial constants,
  NWChem pruning, Becke partitioning, and the PySCF Angstrom-to-Bohr constant.
  Keep the grid JIT- and geometry-gradient-safe without caching traced arrays.
- Solve implicit-SCF adjoint systems with JAX GMRES and remove the previous
  regularized fixed-point residual bias.
- Align the matrix-free full-TDDFT Davidson solver with PySCF-style dual
  subspaces, residual-only convergence, and differentiable symplectic Rayleigh
  reconstruction.
- Extend restricted and unrestricted TDA/TDDFT validation to end-to-end JAX
  SCF references, including excitation energies and degenerate-cluster
  oscillator strengths.
- Use canonical `excitation_gap_*` metric names in closed-shell checkpoint
  evaluation summaries.
- Keep restricted two-AO references out of unrestricted spin-axis dispatch and
  keep occupation conversion JIT-safe.
- Remove obsolete training loss/RSH modules and legacy research drivers from
  the public release tree.
