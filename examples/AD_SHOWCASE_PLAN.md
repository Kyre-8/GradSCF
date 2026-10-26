# Automatic differentiation examples: complete implementation plan

Date: 2026-10-15. Working branch: `examples/ad-showcase`.

This document covers the entire proposed example collection. The first ten
examples are an implementation milestone, not a reduction of the scope.
Entries below are planned deliverables, not claims of available or validated
derivative support. Existing examples provide starting points; each complete
observable must pass its own numerical checks.

## Purpose and organization

Demonstrate how derivatives of an electronic-structure calculation become
measurable molecular and periodic properties. Every example identifies the
independent variables, the self-consistent response being differentiated, the
physical approximation, and an independent validation.

Use readable, ordinary Python scripts with explicit inputs and no CLI.
Keep GradSCF calculations independent of PySCF reference scripts. Reuse public
interfaces and shared solvers; do not put new SCF, response, or eigenvalue
solvers in examples. Extract a helper only when several examples actually need
it. Keep method-specific calculations in their current directories and link
them from the property-oriented index rather than copying implementations.

Suggested locations:

- `examples/geometry/`: forces, optimization, Hessians and potential surfaces.
- `examples/response/`: electric properties and their derivatives.
- `examples/vibrations/`: normal modes, IR, Raman and vibrational spectra.
- `examples/tdscf/`: CIS/TDA/TDDFT excited-state properties where no suitable
  existing example exists.
- Existing `ci/`, `cc/`, `fci/`, `mp/`, `gw/`, `bse/`, `dfpt/`, `pbc/`,
  `ofdft/`, `basis/`, `localization/` and `training/`: retain method ownership.
- `reproducibility/ad_showcase/<example_id>/`: selected data, figures and a
  short explanation of results. Use the existing spelling `reproducibility`.

## Derivative conventions

Use atomic units in calculations: coordinates in Bohr, energy in Hartree,
forces in Hartree/Bohr and Cartesian Hessians in Hartree/Bohr squared.
Convert displayed frequencies to inverse centimetres and wavelengths to nm
with recorded constants. Record isotope masses and normal-coordinate units.

For a static electric field F, use

    E(R,F) = E(R,0) - mu(R).F - (1/2) F.alpha(R).F + ...
    force = -dE/dR
    Hessian = d2E/dR2
    mu = -dE/dF
    alpha = -d2E/dF2
    IR amplitude = dmu/dQ
    Raman tensor = dalpha/dQ

Include the nuclear field interaction in the total energy. A complete geometry
derivative includes moving basis functions, overlap response and, for DFT,
the chosen quadrature's coordinate dependence. State precisely whether the
electronic response uses explicit AD, implicit differentiation, or a stationary
energy identity. A frozen-density energy derivative is not a general response
method and must not be reused to infer a relaxed Hessian or polarizability.

Static Raman uses a mixed third derivative, d3E/(dR dF dF); it does not require
three nuclear-coordinate derivatives of AO integrals. Higher-order capability
must be checked through the entire SCF chain, not inferred from integral AD.

For excited state I, distinguish d(omega_I)/dR from
d(E_ground + omega_I)/dR. Track state identity when geometry changes. At exact
degeneracy, test invariant subspace observables; derivatives of arbitrary
eigenvector components or a sorted crossing root need not exist uniquely.

## Complete example catalog

Status of each new entry starts as **planned**. Reuse links identify a starting
point only. Implementation and measured validation will be tracked separately.

### A. Geometry and nuclear response

| ID | Example and initial systems | Methods / AD target | Validation | Visualization |
| --- | --- | --- | --- | --- |
| G01 | Water nuclear gradients | RHF, then PBE; dE/dR | Central differences at several steps; independent analytic gradients; net force and torque | Force arrows and component parity |
| G02 | Forces across electronic-structure levels; water/NH3 | HF, PBE, PBE0, B3LYP, MP2, CCSD; small-basis FCI reference | Same geometry/basis conventions; relaxed correlated response | Per-atom method differences |
| G03 | Open-shell forces; OH/NO | UHF, UKS; UMP2/UCCSD extension | Spin state, spin contamination, convergence and directional finite differences | Force errors and bond scans |
| G04 | Water geometry optimization | RHF/PBE energy and AD force | Optimizer and SCF convergence; final gradient and Hessian | Energy/force histories, structure trajectory |
| G05 | Hessian and normal modes; water/CO2/CH4 | HF/DFT d2E/dR2 | Finite differences of gradients; symmetry; raw translational/rotational residuals | Hessian heatmap, mode animations |
| G06 | Hessian-vector products; ethylene | JVP of relaxed gradient | HVP versus dense small-system Hessian and directional differences | Error/time/memory versus system size |
| G07 | Smooth surfaces and branch changes; H2 stretch/NH3 inversion | E, gradient and curvature | SCF stability and branch continuity | Energy/derivative curves with branch labels |
| G08 | Selected cubic/quartic force constants; water | Directional higher AD or Taylor propagation after support audit | Step convergence and symmetry identities | Anharmonic potential versus harmonic model |

### B. Electric response and vibrational spectroscopy

| ID | Example and initial systems | Methods / AD target | Validation | Visualization |
| --- | --- | --- | --- | --- |
| R01 | Permanent dipole; water/HF/CO | HF/DFT -dE/dF | Total electronic+nuclear dipole and finite fields | Dipole vectors and method comparison |
| R02 | Static polarizability; water/CO2/ethylene | HF/DFT -d2E/dF2 | DFPT response, finite fields, tensor symmetry | Tensor heatmap and principal axes |
| R03 | Hyperpolarizability; asymmetric small molecule | Selected third/fourth field derivatives | Field-step convergence; permutation symmetry | Field-dependent dipole and nonlinear fit |
| V01 | Harmonic water IR | HF/PBE, then PBE0/B3LYP | Equilibrium Hessian, dmu/dQ finite differences, reference intensities | Sticks, broadened spectrum, mode animations |
| V02 | Nonresonant water Raman | HF/PBE, then PBE0/B3LYP; dalpha/dQ | Polarizability differences along modes; rotational invariants | Raman activities, depolarization and spectrum |
| V03 | IR/Raman selection rules; CO2/N2/CH4 | HF/DFT | Symmetry-forbidden strengths, degenerate-mode sums | Paired IR/Raman plots with mode assignment |
| V04 | Isotope shifts; H2O/D2O and carbon isotopes | Fixed electronic force constants, changed masses | Mode normalization and mass scaling | Overlaid isotope spectra |
| V05 | Correlated IR/Raman; water/CO | MP2, CCSD; CCSD(T)/FCI only after derivative gates | Relaxed density and response; same-level geometry; reference data | Frequency/intensity changes by level |
| V06 | Dynamic polarizability and frequency-dependent Raman | TDHF/TDDFT and correlated response when available | Static limit, frequency grid, damping dependence | Polarizability and Raman versus laser frequency |
| V07 | Resonance Raman | State-resolved excited response and vibronic model | Explicit resonance approximation; damping and state convergence | Excitation profiles and mode enhancement |
| V08 | Anharmonic fundamentals, overtones and combinations | VPT2 or bounded small-mode vibrational diagonalization | Harmonic limit, force-constant convergence, resonance handling | Fundamental/overtone spectrum |

For V01/V02, report both fixed-geometry comparisons and same-method optimized
geometries explicitly; do not attribute geometry differences solely to an
electronic response model. Molecular r2SCAN is an extension only after the
current molecular meta-GGA response path is verified. Periodic r2SCAN support
does not establish molecular support.

Raman activity is not an experimental intensity. If converting activities,
state laser frequency, temperature, scattering geometry and polarization.
An imposed Gaussian/Lorentzian width is a plotting choice, not a calculated
lifetime. Experimental comparisons need a traceable source, phase, temperature,
isotope and uncertainty; harmonic frequencies are not experimental fundamentals.

### C. Excited states and their derivatives

| ID | Example and initial systems | Methods / AD target | Validation | Visualization |
| --- | --- | --- | --- | --- |
| X01 | Vertical absorption; formaldehyde/water/ethylene | CIS, TDA, TDDFT; EOM-EE-CCSD and GW-BSE extensions | Energies, transition dipoles, oscillator strengths; independent reference | Assigned stick and broadened spectra |
| X02 | Excitation and total excited-state gradients; formaldehyde | CIS/TDA first, then TDDFT/EOM/BSE as supported | State-tracked finite differences of omega and E0+omega separately | Ground/excited force arrows and derivative parity |
| X03 | Excited-state geometry optimization; formaldehyde | TDA/TDDFT, later EOM | State overlaps, stationary total excited energy and final curvature | Optimization trace and structure change |
| X04 | Excited-state dipoles and difference densities | Relaxed response for chosen method | Field derivatives; density trace and electron count | Dipole changes and density isosurfaces |
| X05 | Transition densities, NTOs and attachment/detachment | CIS/TDA/TDDFT/EOM/BSE supported definitions | Normalization, phase invariance, reconstruction | NTO pairs and charge-transfer diagnostics |
| X06 | Derivatives of excitation coefficients/reference orbitals | Extend existing water CIS probes | Gauge-invariant probes; directional finite differences | Sensitivity heatmaps; reuse ci/cis_h2o_gradients.py |
| X07 | Oscillator-strength geometry derivatives | Bright/dark formaldehyde and ethylene states | Tracked-state finite differences and symmetry | Intensity modulation along normal modes |
| X08 | Excited-state polarizability and Stark shifts | Field response of total excited-state energy | Finite fields and state identity | Stark curves and tensor changes |
| X09 | Excited-state absorption | State-to-state transitions with appropriate response theory | Independent transition moments, selection rules | Ground/excited-state absorption comparison |
| X10 | Excited-state Hessians, emission and adiabatic energies | Same-level ground/excited minima and curvatures | Hessian symmetry, mode analysis, vertical versus adiabatic bookkeeping | Absorption/emission energy diagram and mode shifts |
| X11 | Franck-Condon progression and Duschinsky mixing | Ground/excited modes and displacements | Identity/displaced-oscillator limits, mode/sign alignment | Vibronic spectrum and Duschinsky heatmap |
| X12 | Herzberg-Teller vibronic intensities | Transition-dipole mode derivatives | Finite differences; constant-dipole limit | FC versus FC+HT spectra |
| X13 | Nonadiabatic coupling and near-degeneracy | Small CI/FCI controlled models before production TDDFT | Gauge convention, state tracking, invariant subspaces | Coupling/energy-gap scans with singular points marked |
| X14 | Ionization/attachment and spectral derivatives | EOM-IP/EA, GW; appropriate Dyson weights | Energies/weights and their geometry dependence | Photoelectron spectrum and sensitivity |
| X15 | Vibrational broadening of electronic spectra | Molecular GW/BSE electron-vibration modules | Zero-coupling limit, temperature dependence, sum/normalization checks | Temperature-dependent spectra; reuse existing water BSE example |

Do not identify a relaxed excited-state difference density with an unrelaxed
attachment/detachment construction. Non-Hermitian EOM properties require the
appropriate left/right states and normalization. BSE derivatives must state
which QP and screening responses are included; no silent frozen-screening
approximation. Existing forward methods do not imply all these derivatives.

### D. Periodic and orbital-free extensions

| ID | Example | AD target / scope | Validation | Visualization |
| --- | --- | --- | --- | --- |
| P01 | Periodic atomic forces | Gaussian DFT and applicable OFDFT models | Displacements, net force, grid and k convergence | Forces and convergence plots |
| P02 | Stress and lattice optimization | Cell derivatives with declared coordinates/occupations | Finite strains and unit conventions | Equation of state and stress convergence |
| P03 | Harmonic phonon dispersion | Finite-q DFPT or validated supercell force constants | Conjugacy before ASR; supercell comparisons; interpolation convergence | Phonon bands, DOS and selected animations |
| P04 | Dielectric tensor and Born effective charges | Proper periodic polarization/electric-field treatment | Acoustic charge sum, finite perturbations, insulating gap | Charge tensors and dielectric response |
| P05 | Crystal IR | Modes, Born charges and long-range corrections | LO-TO limits, q direction and oscillator sums | Polarized IR and LO-TO splitting |
| P06 | Crystal Raman | Mode derivatives of periodic dielectric response | Tensor symmetry, finite differences, normalization by volume | Polarized spectra and Raman tensors |
| P07 | Phonon sensitivities to strain/temperature/XC parameters | Derivatives of validated dynamical matrices | Isolate electronic temperature, expansion and anharmonicity | Mode shifts and sensitivity maps |
| P08 | OFDFT parameter and density response | External kinetic functional; fixed-electron stationary response | Independent TF/vW cases and finite differences | Density/energy/response maps |

Molecular dipole integrals cannot be transplanted to periodic electric response.
For finite-temperature examples specify fixed N versus fixed chemical potential
and energy versus free-energy derivatives. ASR or symmetry projection must not
hide a raw derivative inconsistency. Reuse existing DFPT/PBC/OFDFT demonstrations
and retain their validated physical boundaries.

### E. AD-specific demonstrations and cross-links

| ID | Example | Purpose | Validation / outputs |
| --- | --- | --- | --- |
| A01 | Explicit versus implicit response | Same converged observable and independent variables | Derivative parity; forward/backward and memory comparison |
| A02 | JVP/VJP/HVP | Avoid dense response tensors when only contractions are needed | Adjoint identity and small dense references |
| A03 | Degenerate eigenspaces | Projector/subspace energy versus sorted-root derivative | Reuse solvers/degenerate_subspace.py; crossing plots |
| A04 | Taylor MPn | Higher perturbation coefficients and their convergence | Reuse MP examples and known convergent/divergent series |
| A05 | Basis parameter derivatives | Exponent/contraction optimization | Existing basis examples, independent finite differences |
| A06 | Orbital localization | Boys loss gradient and HVP at fixed orbital subspace | Existing localization examples; density/orthogonality invariance |
| A07 | External functional sensitivities | Fixed-density, explicit and implicit training modes | Existing training examples; physical-response targets and derivative checks |

These entries primarily index or extend existing examples. Do not rebuild their
solvers or introduce neural-model training as a dependency of spectroscopy.

## First milestone: ten examples in dependency order

1. **G01:** water RHF geometry gradient; validate the native-integral and
   converged-SCF chain, then add PBE with quadrature response.
2. **G04:** water geometry optimization using the verified energy/gradient.
3. **G05:** water Hessian, mass weighting and normal-mode analysis.
4. **R01/R02:** water dipole and polarizability in one small introductory
   electric-response example, with independent field checks.
5. **V01:** water harmonic IR at HF and PBE levels.
6. **V02:** water static Raman at HF and PBE levels.
7. **V03:** CO2 complementary IR/Raman selection rules.
8. **X01:** formaldehyde TDA/TDDFT absorption and state assignment.
9. **X02:** formaldehyde excitation and total excited-state gradients.
10. **X03:** formaldehyde excited-state optimization with state tracking.

Use water/6-31G* for the first molecular derivative demonstration; use smaller
bases only for explicit smoke tests. Spectral production settings must be
converged for the stated target, especially diffuse/Rydberg excitations.

Subsequent milestones:

- **M2:** G02/G03/G06/G07, V04/V05, hybrid-DFT spectra, R03, A01/A02.
- **M3:** X04-X08, X10-X12 and cross-method X01/X02 comparisons.
- **M4:** V06-V08, G08, X09/X13-X15 after their response prerequisites.
- **M5:** P01-P08, extending existing periodic and OFDFT infrastructure.
- **GW/BSE molecular track:** execute the dedicated sequence below across
  M3/M4. Its forward baseline does not depend on completion of the periodic
  M5 work or of resonance-Raman/anharmonic extensions.
- A03-A07: cross-link existing validated examples throughout; extend only where
  their AD or visualization explanation is missing.

## GW and BSE molecular track

GW/BSE is a required part of this collection. The original catalog mentioned
it in X01/X02/X05/X14/X15; this section makes its execution and acceptance
explicit. The five work packages below refine those existing entries and the
relevant optical-response extensions. They do not add five entries to the
49-item catalog. A validated CIS/TDDFT component does not complete the GW/BSE
part of the same entry.

Start with finite real closed-shell molecules and singlet static TDA/full BSE.
Use water/Cartesian 6-31G* for the small nuclear-response chain and the archived
methane/aug-cc-pVDZ setup for a diffuse-basis absorption comparison. Match its
geometry, auxiliary basis, G/W/QP/optical windows and frequency conventions
before comparing with the previous run or MolGW. Small smoke-test bases must
remain separate from the spectra used for physical interpretation.

| Order | Catalog links | Required calculation and deliverable | Acceptance gate |
| --- | --- | --- | --- |
| 1. Current-source forward baseline | X01, X14 | Recompute G0W0, evGW0 and evGW from the same RHF reference, then TDA and full BSE. Save QP energies/residues, excitations, transition moments, oscillator strengths, polarizability and absorption. | Check SCF/QP/outer-loop/BSE residuals, stability, window and quadrature convergence. Compare matched methods with MolGW and small dense versus matrix-free paths. Historical output alone is not a new pass. |
| 2. QP and screening derivatives | X14; prerequisite for X02 | Connect moving native integrals or RI factors, relaxed SCF orbitals, self-energy and screening to QP nuclear-coordinate derivatives. Start with G0W0, then differentiate the evGW0/evGW outer fixed point. | Check AD self-energy frequency slopes and QP geometry derivatives against several independently reconverged finite differences; record pole/root identity, QP residues and actual backward residuals. A scalar QP root derivative alone does not validate a self-consistent GW derivative. |
| 3. BSE energy and optical derivatives | X02, X05, X07, V06 | Propagate the verified QP/screening response into BSE excitation-energy, transition-moment and oscillator-strength derivatives. Extend to dynamic polarizability and its normal-mode derivative after the relevant higher-response gate. | Track states or supported invariant subspaces, include the AO/MO transition-operator response, and compare with fully rebuilt displaced GW+BSE calculations. Check optical normalization, static limits, damping and frequency resolution. TDA and full-BSE response receive separate gates. |
| 4. GW photoelectron spectrum and sensitivity | X14 | Refresh a small molecular photoelectron example before extending the existing adamantane workflow. Report QP positions and residues or the explicitly defined spectral-function weights, then their supported geometry/coupling sensitivities. | Check QP branch identity, spectral normalization over the stated window, missing tails and numerical broadening. Distinguish GW residues from EOM Dyson observables; do not silently substitute equal peak weights. Experimental comparison needs compatible, traceable data. |
| 5. BSE electron-vibration absorption | X15; groundwork for V07 | Refresh the existing water BSE electron-vibration example on the current source, then add supported finite-temperature and response extensions. Retain the frozen-QP/kernel model as a separately labelled baseline; compare with relaxed vertices only after packages 2/3 pass. | Check zero-coupling and zero-temperature limits, vibrational/electronic cutoffs, spectral weights, and AD versus recomputed differences. Save temperature-dependent spectra and distinguish vibronic redistribution from imposed smoothing or a calculated lifetime. |

The intended complete nuclear chain is

```text
R -> AO overlap/integrals/grid or RI factors -> stationary SCF
  -> Sigma(omega), W, QP roots -> BSE A/B -> excitations and optical observables.
```

Its method boundaries must remain explicit:

- In evGW0, W0 is fixed during QP self-consistency, but depends on the initial
  reference geometry. Rebuilding W0 at each displaced geometry and including
  that dependence in AD is distinct from freezing W0 with respect to R.
- Frozen orbitals, QP shifts, screening, BSE kernels and transition operators
  are separately declared approximations. Preserve useful diagnostic frozen
  calculations alongside the complete calculation, without labelling them as
  full nuclear response. Derivatives of RI factors/metric are included when
  that representation is used.
- Begin X02 with dOmega/dR. A proposed total excited-state energy gradient
  requires an explicit ground-energy definition and approximation; a BSE
  excitation alone does not establish a variational excited-state surface.
- Degenerate TDA clusters use invariant observables. Full-BSE and GW pole
  responses obey their own stability, gap and solver contracts; the Hermitian
  higher-response tests do not establish RPA/non-Hermitian higher response.
- RHF is the controlled baseline. Add PBE/PBE0/B3LYP GW starting points only
  with matched XC subtraction and references. Original omegaB97X-D requires
  a validated full-range/long-range GW reference adapter and XC subtraction;
  its supported SCF/TDA/TDDFT paths do not authorize removing the GW guard.
  qsGW/scGW, open-shell and periodic extensions need their own method gates.
- A BSE absorption resolvent is not a resonance-Raman scattering theory.
  V07 requires a specified scattering approximation and intermediate-state
  response before a resonance-Raman result is claimed.

Reuse `src/gradscf/gw`, `src/gradscf/bse`, their `ep_coupling` submodules and
the shared linear/nonlinear/eigenvalue solvers. Add source functionality only
where a demonstrated missing response blocks a work package. Keep readable
calculation scripts in `examples/gw` or `examples/bse`, with independent
reference scripts alongside them; do not introduce method-local solvers or a
CLI. Store new compact results under the corresponding showcase IDs with
GW/BSE-specific suffixes, preserving the historical `reproducibility/gw_bse`
records and their original source/environment provenance.

Each package has its own forward, derivative and independent-reference status.
Use normalized geometric directions and multiple displacement steps (initially
1e-3, 3e-4 and 1e-4 bohr), with tighter electronic solves as needed to resolve
the differences. Declare observable-specific units and acceptance tolerances
before production runs. A derivative gate that fails records its actual
residual, branch change or missing primitive and the smallest enabling test.
It does not become a completed result by replacing AD with differences.

All numerical work and plotting follows the existing remote-only contract:
`jy_gpu_1`, float64, frozen source/native-library manifests, and bounded jobs
of at most two hours. Separate compile, warm forward and warm backward timings
and peak-memory measurements. This planning update launches no calculations;
the GW/BSE statuses remain pending until new logs and results establish them.

### Functional coverage and implementation gates

The complete collection was authorized on 2026-10-15, with original **omegaB97X-D**
added to the requested functionals alongside HF, PBE, PBE0 and B3LYP. It must not
be silently replaced by omegaB97X-D3, omegaB97X-D4, or its semilocal LibXC term.
The full electronic model requires separate full-range and long-range exchange
contractions; the original geometry-dependent dispersion must also enter forces
and force constants. Record electronic and dispersion energies separately.
Functional name recognition alone does not establish these capabilities.

Before widening the molecular examples, validate these small dependencies:

1. Complete PBE geometry response through integrals, atom-centered grid, Becke
   weights, AO values and the relaxed SCF state, including grid convergence.
2. One normal-coordinate derivative of the relaxed static polarizability.
   Rebuild the array-valued response problem inside the differentiated map;
   a response snapshot constructed outside it freezes the reference response.
3. Cross-geometry state identity before X02 finite differences and X03
   optimization, including the changing AO metric and degenerate subspaces.

The [implementation status](AD_SHOWCASE_STATUS.md) separates existing reusable
work, implemented examples, saved validation, and missing response prerequisites.
Blocking one physical observable does not stop independent examples. Numerical
finite differences remain validation oracles and never replace promised AD.

Milestones are dependency ordering, not promises that every high-order
derivative currently works. A blocked item records the exact missing API,
derivative rule or physical approximation, plus the narrow test needed to
enable it. Never replace missing AD silently with finite differences.

## Verification and reproducibility contract

For each completed example:

1. Save system, charge/spin, geometry, basis, XC/grid, SCF/response tolerances,
   method approximations, units, dtype, backend and seeds where relevant.
2. Establish forward convergence and state identity before testing derivatives.
3. Compare AD against several finite-difference step sizes and an independent
   implementation where available. Set tolerances before accepting results.
   Initial force target: max absolute error <= 1e-6 Hartree/Bohr on the small
   water benchmark, with a visible finite-difference convergence window.
4. Check applicable invariants: electron count, tensor symmetry, translations,
   rotations, state normalization and degeneracy-invariant sums.
5. Separate compilation time, warm forward time and warm derivative time.
   Record machine, software versions, source revision/patch and command.
6. Run numerical validation and plotting on `jy_gpu_1`; do not run local
   numerical tests. Use bounded jobs and an isolated source snapshot.
7. Preserve compact JSON/CSV, PNG plus PDF, and a results README under
   `reproducibility/ad_showcase/<example_id>/`. Keep caches, checkpoints and
   bulk arrays outside tracked deliverables. Figures must come from saved data.
8. Include measured representative terminal output as trailing code comments.
   Never fill these comments or plots with invented numbers.

Plots use labelled units, legible fonts, restrained consistent method colors,
and disclosed broadening parameters. Save raw sticks alongside smooth spectra.
Include mode animations when they explain assignments. Experimental overlays
require verified primary sources and compatible physical conditions; otherwise
label them as theoretical or software comparisons.

## Progress log

- 2026-10-15: complete catalog written; isolated worktree created from `8f8c93d`.
  Existing CI, DFPT, MP and localization examples identified for reuse.
- 2026-10-15: G01 RHF component implemented and validated on `jy_gpu_1` CPU:
  10 native geometry JVP/VJP tests passed; water/Cartesian 6-31G* gradient
  agrees with independent PySCF to 4.52e-12 Hartree/Bohr and with central
  differences (1e-4 Bohr) to 1.75e-9 Hartree/Bohr. Force/torque checks pass.
  JSON and PNG/PDF deliverables are under `reproducibility/ad_showcase/`.
  PBE for G01 and the other new catalog entries remain planned.
- 2026-10-15: full collection authorized; original omegaB97X-D added as an
  explicit implementation target. PBE geometry response and RHF optimization/
  Hessian examples are in progress. Existing method examples are being indexed
  with their actual physical and differentiation boundaries.
- 2026-10-15: G01 PBE completed: moving-grid AD versus matched PySCF gradient
  max error 4.833e-12 Eh/bohr; the separate level-3 to level-5 quadrature change
  is 1.652e-6 Eh/bohr. G04/G05 RHF and H2O/D2O V04 completed with full implicit
  Hessian and independent PySCF comparison (max error 2.699e-8 Eh/bohr squared).
- 2026-10-15: water RHF R01/R02 and V01/V02 completed with independent finite
  fields/mode displacements, normal-coordinate units and saved plots. Mixed
  Raman AD agrees with independent PySCF mode differences below 6.83e-8 in
  atomic normal-coordinate units. PBE higher-order response remains gated:
  a zero-coordinate AO-polynomial derivative issue is under investigation.
- 2026-10-15: original omegaB97X-D prerequisite components have passed separate
  LR-ERI/geometry-AD and local-XC/CHG tests. Complete SCF/response integration
  is in progress; prerequisite passes are not full-functional certification.
- 2026-10-15: the first ten examples have passed their primary numerical gates.
  Water HF/PBE geometry, electric response and IR/Raman include moving-grid
  response and independent comparisons. CO2 selection rules and carbon isotope
  shifts pass, with complete native and independent mode differences. X01 now
  compares 60 transitions across HF/PBE/PBE0/B3LYP/original omegaB97X-D.
  X02/X03 separately validate CIS total/excitation gradients and a tracked local
  minimum. Scripts, plots and validation records are being consolidated in the
  isolated worktree; no whole-project completion or merge is implied.
- 2026-10-15: M2 components include matched water DFT/MP2/CCSD forces and a
  separate small-basis FCI panel, H2/water/ethylene HVP cost measurements, H2
  branch/stability scans, and third/fourth electric-field response. Hybrid water
  vibrational workflows and open-shell forces are in progress. The shared
  Hermitian higher response and bounded CIS response policy also enabled X10:
  independently validated paired RHF/CIS Cartesian Hessians, mode shifts and
  vertical/adiabatic/harmonic 0-0 energy bookkeeping. Other M3-M5 gates remain
  separate and are tracked in AD_SHOWCASE_STATUS.md.
- 2026-10-16: water PBE0/B3LYP/original omegaB97X-D optimization, Hessian,
  H2O/D2O modes and static IR/Raman components passed their complete-response
  checks. MP2/CCSD water now includes the same workflow; independent Hessian
  errors are 5.24e-9/5.27e-9 Eh/bohr squared. Each calculation retains its
  executed source snapshot, finite-grid or finite-field error and timed phases.
  Shared nonlinear and Hermitian higher-response fixes passed focused current
  regressions; this is not a whole-suite or whole-project completion claim.
  X04-X09 provide relaxed CIS density, NTO, physical wavefunction probes,
  transition/Stark response and a finite-space FCI ESA benchmark. X11 supplies
  a declared common-Eckart harmonic Franck-Condon model. Native third/fourth
  nuclear derivatives, further dynamic spectroscopy, the molecular GW/BSE
  track and periodic electric-response prerequisites remain active work.
- 2026-10-16: G03 now contains all eight OH/NO UHF/UKS-PBE/UMP2/UCCSD
  force comparisons, with explicit independent UCPHF gauge treatment and
  original failed-reference evidence. X12 adds a linear Herzberg-Teller
  model with raw spectral-weight closure and independent quadrature. V06
  adds complete-space molecular dynamic polarizability and harmonic Placzek
  mode response, validated against PySCF and frequency/damping refinement.
  Native third/fourth geometry integrals and P08 remain under development.
  H2 complete GW/BSE nuclear response passes; water revealed multiple-QP-root
  selection as a separate forward boundary, now recorded and tested by
  conditional branch continuation without changing the physical windows.
- 2026-10-16: selected native Cartesian nuclear derivatives now reach fourth
  order (G08); P08 supplies independently checked fixed-N external TF/vW
  parameter/density response. Water G0W0 main-peak seeds and all-70-state
  TDA/full-BSE forward calculations are recorded. Their six-root complete
  geometry JVP/FD checks pass, including all QP residues after step refinement.
  Independent MolGW optical gates pass; its QP residue graphical-grid gate
  remains open and failed coarse/fine controls are preserved. Periodic SCF
  implicit-forward tape removal is separately validated before material
  convergence work. None of these component gates closes the full catalog.

## Requested extensions: large-system accuracy and degenerate backward response

### Execution scope: functional demonstrations

The user's 2026-10-16 clarification makes this a collection of readable
GradSCF capability examples. A working finite-model calculation, a meaningful
derivative or observable, one relevant independent check and a clear figure
are the delivery target. Preserve measured numerical limitations in the
result notes. Material convergence studies and exhaustive combinations of
systems, methods and response orders are optional follow-up benchmarks.

Reuse completed calculations. Do not rerun them solely to obtain another
validation count. The remaining compact examples are an additional OFDFT
size, a matched PBE/PBE0 band demonstration, selected directly computed
periodic GW points and a small static periodic BSE spectrum. Larger cases
remain available through ordinary Python parameters when they fit the
declared resource limits.

Added on 2026-10-16. The user explicitly places the Davidson demonstration's
main emphasis on **implicit backward propagation through a complete degenerate
subspace**, rather than on forward diagonalization speed. These extend A03
and P08 and do not create new catalog IDs. Remote submission was reauthorized
after the server restart; first inspect checkpoints before resuming jobs.

### A03: nontrivial subspace VJP at exact and near degeneracy

- Start with a reproducible high-dimensional real symmetric operator with a
  known complete degenerate cluster and genuine cluster-to-complement coupling.
  Initial dimensions are 256 and 1024; expand only after accuracy/resource
  gates. Do not merely embed the old 4x4 example in an inactive large space.
- Differentiate a nontrivial projector-probe loss, such as
  `L = <W, B.T @ P @ B>`, with a full parameter VJP. The subspace energy sum is
  supplementary: its first stationary derivative does not exercise the
  response solve and cannot be the main backward demonstration.
- At fixed external separation, sweep the internal splitting down to exact
  zero. Verify finite VJPs, independent analytic/projector-FD agreement,
  JVP/VJP duality and invariance under rotations within the degenerate frame.
  Separately shrink the external gap and retain conditioning loss or rejected
  responses. No regularization may be presented as resolving an undefined
  individual eigenvector or a sorted crossing-root derivative.
- Compare dense forward plus the **same** shared implicit backward with
  Davidson forward plus that backward. Dense is a valid numerical reference.
  A separate ordinary `eigh -> projector -> loss` AD control uses the same
  mathematically differentiable observable. Any NaN there is an implementation
  robustness problem, unlike the genuinely undefined individual-vector or
  cut-cluster case. Do not use failed gradients in performance speed ratios.
- The principal performance records are VJP preparation, retained-pullback
  execution and memory, at equal residual tolerances and fixed block/restart
  capacities. Forward timing is secondary. Never materialize large `eye(n)`,
  full projectors or parameter Jacobians merely for the benchmark.
- Actual loss-dependent adjoint residuals must be distinguished from Ritz
  residuals and `response_valid`. Reuse/optionally expose the existing shared
  checked linear solve's diagnostics; do not implement another solver or
  infer runtime counts from Python tracing. Separate diagnostic instrumentation
  from timing runs.

The physical example is regular-hexagon benzene RHF/CIS (HF/TDA), initially
Cartesian 6-31G*, no frozen orbitals: 102 AOs, 21 occupied and 81 virtual,
giving 1701 single excitations. Identify the complete multiplet and its
external gaps numerically before fixing the selected rank. The invariant
observable is the cluster transition tensor `T = D.T @ P_cluster @ D`.
Use an inversion-even symmetry-breaking perturbation so the main VJP is
nonzero; a uniform electric-field derivative can vanish by symmetry.
Fixed-reference response and fully relaxed geometry response are separate
gates. The latter can reuse existing Cholesky metric transport, stationary
orbital-angle response and full noncanonical oo/vv Fock blocks, but must
include moving dipoles/integrals and pass fresh molecular checks. Current
isolated-root CIS/TDA facades must not silently accept an unsupported subspace
mode: connect their existing operator to the shared solver without copying it.

### P08: large-system OFDFT accuracy and scaling

- Use Al/WGC/PZ-LDA/OEPP as the primary material family, with Mg as a later
  cross-check. Compare identical discrete models against DFTpy plus the
  existing independent NumPy WGC adapter; do not call that adapter native
  DFTpy WGC support. Keep KEDF-versus-KS physical error in a separate table.
- Keep real-space resolution fixed when expanding the cell. Candidate Al
  repetitions `s=1,3,5,7` give `1,27,125,343` atoms and `(17*s)^3` mesh points.
  Pilot only the first two sizes before scheduling larger ones. These are
  proposed sizes, not a promise that every case fits the resource budget.
- Keep perfect replicated-crystal checks separate from nontrivial local or
  long-wavelength perturbations. At least small/medium sizes need independently
  reconverged reference densities; a tiled primitive reference is not an
  independent large-system physical validation.
- Record energy per atom and components, weighted density errors, electron
  number, raw stationarity residual and normalized Euler RMS `||r||/(2*N)`
  in Eh. Use comparable physical convergence across system sizes; fixed raw
  tolerance alone can impose size-dependent accuracy requirements.
- Measure input assembly, compilation/first execution, warm forward, density
  JVP, nonconstant density-observable VJP and selected energy HVP separately.
  Use fixed-count normalized directions and multistep differences. All output
  pytrees must be synchronized. Report actual process/device memory, hardware,
  seeds and forward iteration counts; unavailable adjoint iteration counts
  remain unavailable until proper shared diagnostics exist.
- The current atlas RECPOT builder is NumPy/eager and cannot supply nuclear
  AD by wrapping it in grad. Coordinate-dependent phase/Ewald assembly needs
  an explicit JAX path before force claims. Its grid-by-atom structure array
  and the current atom-pair/image Ewald array are possible quadratic costs;
  include them in end-to-end accounting. Do not assert near-linear scaling
  from a solver-only timer or from the small dense P08 KKT oracle.

All new numerical work stays on `jy_gpu_1`. Start with bounded CPU probes;
each job remains below two hours. A single idle GPU may be used after an
actual backend probe and with separate CPU/GPU records. No performance or
large-system accuracy claim is established by this local plan update.

## Added material bands, periodic GW/BSE and reduced-dimensional cells

The user added Si, monolayer graphene, GaAs and GaN on 2026-10-16, with
different DFT methods, GW bands and BSE results, and requested one- and
two-dimensional supercell support. This extends the project; the earlier
unfinished phonon, Raman and OFDFT work remains active.

Initial structural conventions are diamond Si, zincblende GaAs and wurtzite
GaN, at explicitly fixed example geometries. Compare LDA, PBE, r2SCAN and
PBE0 using the same geometry, basis and declared pseudopotential within each
controlled comparison. Preserve the actual core/valence partition, including
Ga semicore states. Scalar calculations do not include spin-orbit splitting.

1. **Native band workflow.** Reuse periodic SCF and `get_bands`; keep ASE
   limited to structure/path construction and PySCF in an independent script.
   Record actual fractional paths, raw eigenvalues, energy alignment and
   sampled/path gaps separately. Verify on-mesh consistency and independent
   off-mesh values before interpreting material trends. The present band API
   supports LDA/GGA/meta-GGA and validated 3D global-hybrid off-mesh bands with
   matched spherical exchange. The compact Si PBE/PBE0 example now has an
   independent numerical comparison; other material hybrid runs are follow-up
   examples. TD/DFPT consumers reject unsupported exchange conventions.
2. **Dimensional boundary infrastructure.** Explicit periodic axes, vacuum
   directions and allowed k/q meshes must be shared with the Coulomb model.
   Repeated 3D slab/wire pilots are labeled as such, never as isolated lower-
   dimensional results. Add slab and wire interaction treatments, consistent
   electron-ion/ion-ion terms and exchange conventions; validate neutral
   electrostatics, vacuum convergence and derivatives. Do not merely remove
   `dimension != 3` guards. Use a small gapped H2 chain as the first 1D test.
3. **Periodic GW.** Reuse the existing contour-deformation/QP kernels. Remove
   the impractical dense full-FFT dielectric storage before material runs;
   reuse shared screening/linear solves and transition-space representations.
   Three-dimensional head/wing prefactors must not be applied to slabs/wires.
   Start with G0W0 and distinguish directly computed QP points from validated
   interpolation. Existing non-Gamma GW is eager-only; do not imply AD support
   from a forward result. Gapless graphene requires occupation/intraband and
   long-wavelength treatment, not an artificially imposed insulating gap.
4. **Periodic BSE.** Build complex momentum-conserving electron-hole actions
   on the shared GW screening, with existing eigensolver ownership. Validate
   static singlet TDA first, then full BSE where supported. Compare independent-
   particle and electron-hole spectra, optical onset, weights and exciton
   binding with matched QP gaps. For 2D/1D, report sheet/line polarizability
   or conductivity with explicit normalization, not a vacuum-dependent bulk
   dielectric curve labeled as an intrinsic material spectrum.

2026-10-17 GW band update: [the public native example](pbc/band_structures/gw_bands.py)
now evaluates complete compact G0W0@PBE paths for all four materials, with
the [selected arrays and figures](../reproducibility/ad_showcase/material_gw_bands/README.md)
kept separate from the earlier DZVP DFT gallery. Accepted/requested QP counts
are Si 113/114, GaAs 114/114, GaN 132/132 and graphene 89/91. Failed candidates
remain masked. The query treats complete degenerate KS subspaces, explicit
neutral T=0 band touching and analytic static-pole subtraction, reusing
shared matrix, linear and scalar solvers. Focused remote gates record 67
query passes and 17 separate momentum passes. These coarse body-only paths
do not complete material convergence, metallic/intraband or long-wavelength
screening, periodic GW AD, or material BSE validation.

The subsequent [61-point direct paths](../reproducibility/ad_showcase/material_gw_bands/dense61/README.md)
finish all four systems with the user-approved maximum of 96 logical CPUs.
Accepted/requested counts are 359/366, 363/366, 366/366 and 413/427 for
Si/GaAs/GaN/graphene. Denser query sampling leaves source/FFT settings fixed
and preserves failed masks. Accepted-root correction jumps remain visible;
continuous physical QP-branch validation is still open.

The cutoff implementation should be checked against Ismail-Beigi,
[Phys. Rev. B 73, 233103 (2006)](https://doi.org/10.1103/PhysRevB.73.233103),
and independent slab/wire calculations. The [BerkeleyGW cutoff interface](https://manual.berkeleygw.org/4.0/epsilon-keywords/)
and [Yambo cutoff documentation](https://wiki.yambo-code.eu/wiki/index.php/Coulomb_2D)
are implementation references, not substitutes for matched numerical gates.
All calculations remain on jy_gpu_1 in isolated snapshots, at most one idle
GPU per task and two hours per job. No new solver copies or hidden finite-
difference replacement of an advertised derivative are permitted.
