# Automatic differentiation showcase: implementation inventory

Project started: **2026-10-15**. Baseline revision: **`8f8c93d`**.
This living inventory covers all 49 entries in the
[implementation plan](AD_SHOWCASE_PLAN.md). Recorded numerical checks are
linked explicitly as implementation proceeds. Existing test files identify
validation targets; their presence alone is not a fresh passing result.

**Remote work resumed with user authorization on 2026-10-16.** The server and
saved checkpoints have been inspected after the restart. The historical
[pause checkpoint](../artifacts/ad_showcase/development/PAUSED_SERVER_RESTART_20261016.md)
records interrupted jobs; the
[resumption log](../artifacts/ad_showcase/development/RESUMED_20261016.md) records fresh
submissions and outcomes. Older component STATE files can contain stale
running/paused descriptions and must be checked against these dated records.
Of the 49 catalog entries, 44 have validation records for their named bounded
components, four remain partial (P03/P05/P06/P07), and A04 retains reusable
historical plus partial current checks. These counts describe coverage, not
full-scope completion. The checkpoint passed the selected integration checks
listed below. Remote publication is a separate action.

**Current delivery scope (2026-10-16): capability examples.** Reuse completed
finite-model checks and figures. Material/basis/grid convergence and exhaustive
method combinations are follow-up benchmarks, rather than prerequisites for
every example. The matched Si PBE/PBE0 band pair and selected Si GW points
are now present. Al125 native AD and its independent
reference are now packaged with the three-size summary plot. Small 3D, wire
and layer BSE spectra and direct H2/Si GW segments are now packaged; full
q-cell corrections and periodic BSE AD remain open. Scientific limitations
stay visible in each record.

**Workspace organization:** runnable examples, selected results and local
development evidence now have separate locations. Executed-source copies and
raw job records are under `artifacts/ad_showcase/provenance/`; earlier local
diagnostics are under `artifacts/ad_showcase/development/`. The relocation map
preserves original paths and hashes. The earlier periodic BSE drafts remain archived. The current SRC body
module now passes the corrected full-channel static-screening checks and has
a runnable H2 spectrum; source and raw execution records have separate locations.

All numerical tests, calculations, benchmarks and plotting for this project
run on **`jy_gpu_1`**, using bounded jobs and an isolated source snapshot.
Reuse the current public interfaces and shared SCF, linear, nonlinear and
eigenvalue solvers. Do not implement replacement solvers inside examples.
Keep method examples in their existing directories and cross-link them.

Status vocabulary:

- **planned**: the requested observable or prerequisite is not yet implemented
  as an example, even if a related forward method exists.
- **partial**: useful components or narrower examples exist; the requested
  physical response or validation chain remains incomplete.
- **reusable**: an existing example already demonstrates a relevant observable
  within its stated scope. Reuse it without treating the whole catalog entry
  as completed.
- **validated-component**: recorded numerical evidence supports the named
  component only. Broader methods, settings and new project gates remain open.

No status below means that a complete catalog entry has passed every item of
the showcase verification and reproducibility contract. In particular,
fixed-MO integral response, basis-parameter response and a differentiable
forward solver do not establish a complete nuclear-coordinate response.

## Geometry and nuclear response

| ID | Existing entry | Implemented / validated boundary | Limits and follow-up |
| --- | --- | --- | --- |
| G01 | [Water RHF gradient](geometry/water_rhf_gradient.py); [PBE gradient](geometry/water_pbe_gradient.py); [PBE results](../reproducibility/ad_showcase/G01_water_pbe_gradient/README.md) | **validated-component**: water/Cartesian 6-31G* RHF and PBE complete geometry gradients pass multi-step finite differences and independent PySCF comparisons. PBE includes moving grid/AO/weights. | Preserve the reported grid error separately from AD error; extend functional coverage only after its full response is validated. |
| G02 | [Water correlated forces](geometry/water_correlated_forces.py); [DFT forces](geometry/water_dft_forces.py); [results](../reproducibility/ad_showcase/G02_water_correlated_forces/README.md) | **validated-component**: matched HF/PBE/PBE0/B3LYP/MP2/CCSD forces in Cartesian 6-31G*, plus a separate STO-3G FCI panel; complete response, independent analytic gradients and multistep directional differences pass. | Extend to NH3 and full omegaB97X-D; retain distinct FCI basis and DFT finite-grid boundaries. |
| G03 | [OH/NO forces](geometry/open_shell_forces.py); [independent reference](geometry/compare_open_shell_forces_pyscf.py); [results](../reproducibility/ad_showcase/G03_open_shell_forces/README.md) | **validated-component**: all eight OH/NO UHF/UKS-PBE/UMP2/UCCSD nuclear-gradient cases pass multistep differences and independent references. Correlated reference uses PySCF assembly with a SciPy augmented physical-pi-gauge oracle, checked full residuals and NO/MP2 force gauge invariance; SCF/reference spin and complete internal spectra retained. Two native source families are explicit. | Establish independent PBE grid convergence and broader open-shell systems/response orders. Stock iterative UCPHF did not meet the true-residual gate; this does not define arbitrary degenerate orbital response or correlated-state spin/stability. |
| G04 | [Water HF/PBE optimization](geometry/water_pbe_optimization.py); [hybrid workflow](vibrations/water_hybrid_spectra.py) | **validated-component**: HF/PBE/PBE0/B3LYP/original omegaB97X-D water Cartesian 6-31G* minima, evaluation histories, final gradients and independently checked positive internal curvatures. | Extend molecular and basis/grid coverage; preserve raw finite-grid rotational diagnostics. |
| G05 | [HF/PBE Hessians](geometry/water_pbe_hessian.py); [hybrid workflow](vibrations/water_hybrid_spectra.py); [hybrid results index](vibrations/README.md) | **validated-component**: relaxed HF/PBE/PBE0/B3LYP/original omegaB97X-D water Hessians, multistep HVP checks, isotope modes and animations. DFT includes moving-grid response and independent PySCF full-grid gradient differences. | Extend beyond water/CO2; retain raw rotational residuals, including the matched omegaB97X-D finite-grid residual, before internal-mode projection. |
| G06 | [Hessian vectors](geometry/hf_hessian_vectors.py); [results](../reproducibility/ad_showcase/G06_hessian_vectors/README.md) | **validated-component**: H2/water/ethylene HVPs versus all-column Hessians and multistep gradient differences, with cold/warm time and fresh-process high-water RSS. | Three small systems do not establish an asymptotic exponent; full AO ERIs and cumulative runtime memory remain explicit. |
| G07 | [H2 response surface](geometry/h2_response_surface.py); [results](../reproducibility/ad_showcase/G07_h2_surface/README.md) | **validated-component**: RHF energy/gradient/curvature, occupied-overlap continuity and multistep checks; shared spin stability/following identifies lower unrestricted branches. Independent PySCF gradients, Hessians and selected-branch energies pass. | The AD target is the constrained RHF branch; discrete branch selection and the selected UHF derivatives are not inferred from it. |
| G08 | [Cubic/quartic molecular response](geometry/molecular_cubic_quartic.py); [native and PySCF validation](../reproducibility/ad_showcase/G08_molecular_cubic_quartic/README.md) | **validated-component**: Cartesian fixed-basis S/T/nuclear/full-Coulomb derivatives through fourth order, with relaxed RHF H2/6-31G and water/STO-3G/6-31G* directional E3/E4. Independent nuclei/AO slots, permutations, adjoints and low-order regression gates passed. | These are local directional derivatives, not full anharmonic spectra. Water bend quartic is negative. Higher spherical/dipole/LR/RI/ECP, mixed basis response, fifth order and jet remain outside scope. |

## Electric response and vibrational spectroscopy

| ID | Existing entry | Implemented / validated boundary | Limits and follow-up |
| --- | --- | --- | --- |
| R01 | [Shared HF/PBE water electric response](response/water_electric_response.py); [RHF results](../reproducibility/ad_showcase/R01_R02_water_electric/README.md); [PBE results](../reproducibility/ad_showcase/R01_R02_water_pbe_electric/README.md) | **validated-component**: total RHF/PBE water dipoles from E(R,F), including a consistent fixed origin and nuclear field energy; independent PySCF agreement at each method's own optimized geometry. | Extend molecular/functional coverage and establish basis/grid convergence. |
| R02 | [Shared HF/PBE water electric response](response/water_electric_response.py); [RHF results](../reproducibility/ad_showcase/R01_R02_water_electric/README.md); [PBE results](../reproducibility/ad_showcase/R01_R02_water_pbe_electric/README.md) | **validated-component**: full RHF/PBE water tensors, DFPT agreement, multistep GradSCF/PySCF finite fields and symmetry checks at matched geometries within each comparison. | Extend to hybrids, CO2 and ethylene; establish diffuse-basis and grid convergence for physical prediction. |
| R03 | [HF-molecule nonlinear response](response/hf_hyperpolarizability.py); [results](../reproducibility/ad_showcase/R03_hf_hyperpolarizability/README.md) | **validated-component**: relaxed third/fourth field derivatives, third-tensor permutation symmetry, multistep lower-response differences and independent PySCF dipole fits. | This verifies electric-field orders only; preserve field-fit noise and fixed-geometry/diffuse-basis boundaries. |
| V01 | [Water IR/Raman](vibrations/water_ir_raman.py); [hybrid workflow](vibrations/water_hybrid_spectra.py); [results index](vibrations/README.md) | **validated-component**: HF/PBE/PBE0/B3LYP/original omegaB97X-D water IR, relaxed dipole JVPs, independent mode differences, SI intensity conversion, sticks and plotted spectra. Original omegaB97X-D includes its separate CHG geometry correction. | Physical basis/grid convergence and molecular coverage remain; plotting widths are not lifetimes. |
| V02 | [Water IR/Raman](vibrations/water_ir_raman.py); [hybrid workflow](vibrations/water_hybrid_spectra.py); [results index](vibrations/README.md) | **validated-component**: HF/PBE/PBE0/B3LYP/original omegaB97X-D mixed RFF derivatives, static Raman activities/depolarization, native and independent mode differences. | Retain independent finite-difference noise and raw finite-grid rotational residuals. Activities are not experimental intensities; dynamic/resonant scattering is a separate extension. |
| V03 | [CO2 selection rules](vibrations/co2_selection_rules.py); [results](../reproducibility/ad_showcase/V03_co2_selection_rules/README.md) | **validated-component**: linear RHF minimum, independent Hessian/frequencies and intensity checks, complete two-step native mode differences, AD mutual exclusion and degenerate-bend sum invariance. | Extend beyond CO2 after the same response gates; initial rejected SCF references and the incomplete direct-energy timing run remain explicitly excluded. |
| V04 | [Water isotope modes](geometry/water_rhf_hessian.py); [hybrid/correlated workflows](vibrations/README.md); [carbon isotope](vibrations/co2_carbon_isotope.py) | **validated-component**: H2O/D2O at HF/PBE/PBE0/B3LYP/original omegaB97X-D/MP2/CCSD and 12CO2/13CO2 share their respective electronic Hessians; mass normalization and the independent CO2 isotope-scaling identity pass. | Preserve fixed electronic force constants and harmonic approximation; no isotope-dependent electronic optimization is implied. |
| V05 | [Water correlated spectra](vibrations/water_correlated_spectra.py); [independent reference](vibrations/compare_water_correlated_pyscf.py); [MP2 results](../reproducibility/ad_showcase/V05_water_mp2_spectra/README.md); [CCSD results](../reproducibility/ad_showcase/V05_water_ccsd_spectra/README.md) | **validated-component**: water/Cartesian 6-31G* MP2 and CCSD optimization, complete AD Hessians, H2O/D2O modes, IR and static Raman pass native differences and independent total-response checks. HF orbitals and correlated energies/amplitudes respond. | Canonical MP2 MO roots are isolated at sampled points. CO, triples, FCI vibrations and basis convergence remain separate extensions. |
| V06 | [TDHF example/results](../reproducibility/ad_showcase/V06_water_dynamic_raman/README.md); [full-BSE example](vibrations/water_bse_dynamic_raman.py); [BSE results](../reproducibility/ad_showcase/V06_water_bse_dynamic_raman/README.md) | **validated-component**: water/6-31G* complete TDHF and all-70-state G0W0/full static BSE polarizabilities and three mode JVPs. TDHF has independent PySCF alpha/FD checks; BSE has an independent 140-dimensional resolvent, full spectral geometry FD and frequency/activity-grid checks. | Harmonic frequency-dependent Placzek activities with imposed width, not vibronic KHD or experimental Raman intensity. TDHF and BSE run at the same saved RHF minimum, with separately recorded source/solver boundaries; no zero-damping resonance-height or basis-convergence claim. |
| V07 | [HCHO single-state KHD Raman](vibrations/formaldehyde_khd_raman.py); [theory, independent resolvent checks and figures](../reproducibility/ad_showcase/V07_formaldehyde_khd/README.md) | **validated-component**: frozen X11/X12 harmonic/linear-HT model, six ground fundamentals, both scattering time orderings and complete complex Cartesian tensors. N=24 retains 593775 intermediate states; tensor increment/tail bounds are 2.04e-7/1.34e-7 relative to the global peak. Three Gamma values pass frequency-grid gates; 30 moment/KHD/FC/HT tests and an independent two-mode resolvent comparison pass. | One electronic intermediate state and one local well at 0 K; phenomenological damping. AD electronic inputs are reused from X07, while the vibronic sum is host-side. Raw amplitudes/strengths only: no absolute cross section, other electronic states, anharmonicity, global tunneling/parity completion or experimental-spectrum claim. |
| V08 | [H₂ one-coordinate model](../reproducibility/ad_showcase/V08_h2_quartic/README.md); [water two-stretch model](../reproducibility/ad_showcase/V08_water_two_stretch/README.md); [bounded matrix helper](../src/gradscf_tools/anharmonic.py) | **validated-component**: optimized H₂ and fixed-mode water use native relaxed cubic/quartic AD, positive/unique polynomial certificates, padded oscillator matrices and the shared eigensolver. Independent PySCF derivatives, DVR grid/domain and harmonic-limit checks passed; water includes physical quadratic-dipole overtone/combination strengths and raw state mixing. | Finite local-polynomial models: sampled Taylor errors and wavefunction tails limit molecular accuracy. Water bend/rotor are frozen; no full VPT2/Watson/Coriolis, experimental-spectrum accuracy or end-to-end optimization/mode AD claim. |

## Excited states and their derivatives

| ID | Existing entry | Implemented / validated boundary | Limits and follow-up |
| --- | --- | --- | --- |
| X01 | [Formaldehyde spectra](tdscf/formaldehyde_absorption.py); [independent comparison](tdscf/compare_formaldehyde_pyscf.py); [record](../reproducibility/ad_showcase/X01_formaldehyde_absorption/README.md) | **validated-component**: 60 HF/PBE/PBE0/B3LYP/original omegaB97X-D TDA/full-TD transitions checked against independent PySCF matrices and SciPy dense roots. Broader initial seeding fixes the observed root-coverage failure; no general completeness claim. | Converge basis/grid and extend assigned spectra to EOM/BSE. RSH TD remains restricted singlet/full ERI, and its input-response tests do not imply a full SCF geometry chain. |
| X02 | [CIS geometry gradients](tdscf/formaldehyde_cis_gradients.py); [tracking helper](../src/gradscf_tools/excited.py); [record](../reproducibility/ad_showcase/X02_formaldehyde_cis_gradients/README.md) | **validated-component**: RHF/CIS `domega/dR` and `d(E0+omega)/dR` separately pass tracked finite differences and independent analytic gradients. Matching uses transition-density overlaps. | Extend to DFT TDA/TDDFT and correlated geometry response; validate tracking near crossings separately. |
| X03 | [Tracked CIS optimization](tdscf/formaldehyde_cis_optimization.py); [independent comparison](tdscf/compare_formaldehyde_cis_optimization.py); [record](../reproducibility/ad_showcase/X03_formaldehyde_cis_optimization/README.md) | **validated-component**: RHF/CIS S1 local minimum passes full Cartesian stationarity, tracked continuation and positive chart curvature from finite differences of native/PySCF gradients. This is not an AD Hessian. | Extend DFT/correlated response, sample other basins and basis convergence. The separate X10 workflow now validates the full RHF/CIS AD Hessian. |
| X04 | [Relaxed CIS density/dipole](tdscf/formaldehyde_cis_density.py); [independent comparison](tdscf/compare_formaldehyde_cis_density.py); [record](../reproducibility/ad_showcase/X04_formaldehyde_cis_density/README.md) | **validated-component**: RHF/CIS total-energy AO-potential and field derivatives pass electron-count, six controlled AO-direction and independent finite-field gates. Unrelaxed wavefunction 1-RDM and normalized attachment/detachment are separate; relaxed response occupations need not lie in [0,2]. | Extend beyond isolated real RHF/CIS singlets. External relaxed-density validation is directional, not componentwise; TDDFT/EOM, basis and geometry convergence remain. |
| X05 | [CIS NTOs and transition densities](tdscf/formaldehyde_cis_ntos.py); [independent comparison](tdscf/compare_formaldehyde_cis_ntos.py); [record](../reproducibility/ad_showcase/X05_formaldehyde_cis_ntos/README.md) | **validated-component**: real singlet CIS transition normalization, one-electron attachment/detachment, NTO reconstruction, gauge invariants and independent PySCF weights/projectors/moments pass. Figures show S1 and the brightest of six retained roots. | Extend nonzero-Y TDDFT and EOM definitions separately; individual degenerate SVD frames have no derivative contract. Finite-basis centroid separation is not a complete charge-transfer assignment. |
| X06 | [Water CIS wavefunction probes](ci/cis_h2o_gradients.py); [independent comparison](ci/compare_cis_h2o_probes.py); [record](../reproducibility/ad_showcase/X06_water_cis_probes/README.md) | **validated-component**: physical squared transition-moment probe passes tracked geometry, normalized-amplitude and fixed-X reference-rotation directional checks, phase/block-gauge tests and sensitivity heatmaps. Reference rotations do not re-solve off-HF CIS. | The three parameter maps are distinct, not additive pieces of nuclear response. Extend higher derivatives, near-crossing states and other electronic methods only after separate gates. |
| X07 | [Tracked CIS oscillator response](tdscf/formaldehyde_oscillator_response.py); [independent comparison](tdscf/compare_formaldehyde_oscillator_response.py); [record](../reproducibility/ad_showcase/X07_formaldehyde_oscillator_response/README.md) | **validated-component**: formaldehyde RHF/CIS bright/dark moment and strength derivatives pass six internal-mode differences, state/phase tracking and rigid covariance. Dark quadratic intensity uses first moment derivatives; finite-amplitude curves are independently checked. | Add the planned ethylene and other-method components. Fixed harmonic directions and electronic intensity modulation do not constitute a vibronic spectrum; basis and crossing coverage remain. |
| X08 | [CIS polarizability/Stark response](tdscf/formaldehyde_cis_stark.py); [independent comparison](tdscf/compare_formaldehyde_cis_stark.py); [record](../reproducibility/ad_showcase/X08_formaldehyde_cis_stark/README.md) | **validated-component**: clamped-nuclei total S1 energy field Hessian, ground/excited tensor differences and tracked Stark curves pass independent first/second finite fields. Nuclear field energy is retained; no excited-alpha positivity premise. | Extend beyond isolated RHF/CIS, one geometry and basis. Nuclear relaxation, dynamic-frequency response, TDDFT/EOM and near-crossing states need separate gates. |
| X09 | [H2 full-FCI ESA](fci/h2_excited_absorption.py); [independent reference](fci/compare_h2_excited_absorption_pyscf.py); [record](../reproducibility/ad_showcase/X09_h2_fci_esa/README.md) | **validated-component**: complete H2/6-31G states, physical state-to-state RDMs/weights, separated upward ESA and downward stimulated emission, full moving-metric bond AD, multistep PySCF differences and finite-space commutator closure pass. | Isolated roots and one local bond coordinate in a finite basis; closure does not imply TRK=Ne. Basis convergence, crossings and production TDDFT/EOM ESA remain open. |
| X10 | [Paired RHF/CIS minima and spectra](../reproducibility/ad_showcase/X10_formaldehyde_cis_emission/README.md) | **validated-component**: independently checked complete Cartesian AD Hessians, paired local minima, absorption/emission/adiabatic/harmonic 0-0 bookkeeping and matched mode shifts. Bounded shared direct CIS response is explicit. | Extend to DFT/TDDFT and other correlated methods only after their higher-response gates; basis, basin and anharmonic convergence remain. EOM still has its separate first-order contract. |
| X11 | [Formaldehyde FC model](tdscf/formaldehyde_franck_condon.py); [model, figure and gates](../reproducibility/ad_showcase/X11_formaldehyde_franck_condon/README.md); [FC helper](../src/gradscf_tools/franck_condon.py) | **validated-component**: common-ground-Eckart-slice double-harmonic model from X10; 8 identity/Poisson/Gaussian/quadrature/Fock/AD tests and molecular rigid/mode-gauge covariance gates passed. N=30 raw probability sum 0.999999999938; fixed-shift P00 AD/FD error 1.86e-9. | Explicit approximation: raw J nonorthogonality 0.03243, final frequency change up to −54.87 cm^-1 and model 0–0 4.398663 eV versus original X10 4.402847 eV. Unit-Condon probabilities only; no absolute dark S1 absorption, HT, full curvilinear or through-mode spectral AD claim. |
| X12 | [Formaldehyde linear HT](tdscf/formaldehyde_herzberg_teller.py); [spectra, closure and scope](../reproducibility/ad_showcase/X12_formaldehyde_herzberg_teller/README.md) | **validated-component**: frozen X11 common-slice model plus X07 full electronic dipole derivatives; 16 FC/HT tests, independent 2D quadrature, combined electronic/mode/rigid-frame gauges. N=30 raw dipole-strength closure remainder 1.81e-9; total oscillator strength 3.43056e-4, FC-only 2.84667e-31. | Single-local-well, zero-temperature double-harmonic/linear-dipole model; same X11 frequency/0–0 approximation. Three Cartesian components and signed interference retained without component normalization. Near-dark FC is a numerical symmetry residual; no anharmonic/tunneling, experimental or through-mode full-spectrum AD claim. |
| X13 | [H3 full-FCI NAC](fci/h3_nonadiabatic_coupling.py); [independent reference](fci/compare_h3_nac_pyscf.py); [record](../reproducibility/ad_showcase/X13_h3_fci_nac/README.md) | **validated-component**: real nine-determinant H3/STO-3G, fixed Ms=+1/2; complete CI/frame/AO-overlap NAC passes multistep PySCF differences, orbital-gauge and rigid-generator checks. At exact degeneracy only the complete rank-two spatial-state pair is differentiated; individual NAC is undefined. | Cross-frame overlap currently requires at most three occupied orbitals per spin. Larger bases/spaces, complex states, ETF/ERF dynamics and production TDDFT/EOM NAC remain open. |
| X14 | [Water GW spectrum](gw/water_photoelectron_response.py); [GW results](../reproducibility/ad_showcase/X14_water_qp_removal/README.md); [H2 EOM Dyson example](cc/h2_dyson_response.py); [EOM results](../reproducibility/ad_showcase/X14_h2_eom_dyson/README.md) | **validated-component**: water occupied-QP trace uses actual 2Z weights and complete geometry response. Restricted H2/STO-3G CCSD/IP/EA Dyson residues include native AO/SCF/T/Lambda/both-vector response, independent full FCI and multistep spectrum checks; final spectral FD error 2.79e-4/(eV bohr) passes the fixed 1e-3 gate. | GW residues and biorthogonal EOM Dyson products are different observables. QP-only water omits satellites; neither example supplies photoionization matrix elements or a lifetime. Exact H2 EA/FCI agreement is specific to two spatial orbitals; larger systems/bases remain separate extensions. |
| X15 | [Refreshed water baseline](../reproducibility/ad_showcase/X15_water_bse_baseline/README.md); [thermal response](../reproducibility/ad_showcase/X15_water_bse_temperature/README.md); [matched vertices](../reproducibility/ad_showcase/X15_water_matched_vertices/README.md); [example index](bse/README.md) | **validated-component**: frozen-kernel RHF/3-21G baseline with 0 K multiphonon cutoffs 28/40; separate one-phonon Fan/Fan+DW spectra at 0/300/600/1200 K pass temperature AD/FD, grid, independent resolvent and raw-weight/tail checks. Matched water/6-31G* frozen/relaxed diagonal vertices pass complete geometry FD and zero-temperature FC cutoff/weight checks. | The thermal model changes only the Bose bath; it is not thermal multiphonon FC. The matched comparison holds modes, electronic origins and dipoles fixed, omitting off-diagonal coupling in both models. Full nonadiabatic matrices, relaxed quadratic vertices and global excited-state surfaces remain outside these results. |

## Periodic and orbital-free extensions

| ID | Existing entry | Implemented / validated boundary | Limits and follow-up |
| --- | --- | --- | --- |
| P01 | [LiH force/stress](pbc/lih_forces_stress.py); [independent PySCF](pbc/compare_lih_forces_stress.py); [results, figures and source](../reproducibility/ad_showcase/P01_P02_lih_periodic/README.md) | **validated finite-model capsule**: 12 native cases / 9 independent property comparisons; FFT41→51 and k6→8 increments pass declared gates. Original PBE multi-step FD and repaired-source HF full Voigt FD pass. Implicit-memory fix: 6 new + 21 existing tests; same-case eager RSS 75.7→5.6 GiB. | Independent k checks stop at k4; k5/6/8 are native convergence only. SR-DZVP changes the force sign; no basis-complete, jointly FFT/k-converged or mechanical-stability claim. Original PBE full FD was not repeated after the memory-only repair. |
| P02 | [Cubic LiH optimization](pbc/lih_lattice_optimization.py); [independent energy minimum](pbc/compare_lih_lattice_pyscf.py); [EOS and conventions](../reproducibility/ad_showcase/P01_P02_lih_periodic/README.md) | **validated scalar cubic model**: SZV/k2³/FFT31³ AD-stress minimum a=7.7800322767 bohr; independent PySCF a=7.7800454231 bohr. Energy/stress curvatures are positive and agree; independent energy, slope and curvature gates pass. Four figure pairs and source-checked tables archived with P01. | Fixed fractional coordinates/kpoints. The minimum is specific to this finite basis/k/FFT model; fixed-a convergence does not establish optimized-a convergence. No full-shape, internal-coordinate or phonon-stability claim. |
| P03 | [LiH q-grid](pbc/lih_qmesh_phonons.py); [bands/DOS](pbc/lih_ifc_bands.py); [complex-q reference](pbc/compare_lih_complex_q.py); [results](../reproducibility/ad_showcase/P03_lih_qmesh/README.md) | **Finite-model dispersion demonstration available**: all 12 k223/FFT41 response matrices, bands/DOS and animation are archived. Independent complex q displacement differences pass at both steps (4.67e-9 and 2.87e-8 Eh/bohr²); independent Phonopy interpolation agrees. Earlier Gamma/boundary gates remain archived. | Material convergence is a follow-up. The additional k226 set is partial and is not mixed into the accepted k223 interpolation; smaller FD steps show numerical noise. |
| P04 | [Berry polarization](../src/gradscf/pbc/polarization.py); [LiH Born record](../reproducibility/ad_showcase/P04_lih_born/README.md); [LiH electric response](../reproducibility/ad_showcase/P04_lih_dielectric/README.md); [small mixed-response gates](../reproducibility/ad_showcase/P04_electric_mixed/README.md) | **finite-model gates passed**: full-mesh Berry/gauge/BZ/AD foundation; FFT31 Born independently checked with 25 PySCF endpoints (max error 2.76e-8 e). Matched PBE/SZV/k2³/FFT71³ full Born/chi now pass all six geometry directions, native displacement pairs and 12 finite-field endpoints; raw charge sum 5.41e-9 e, chi FD error 2.53e-8. No charge/tensor projection. Dimer/H2 field oracles, full-root enthalpy Hessian, Maxwell and one-direction mixed response pass separately. | Basis/k/string-discretization convergence and general material mixed response remain open; nonzero raw epsilon off-diagonals are retained. Downstream spectra must use matching FFT/model data; small-system nested AD does not establish general Taylor-jet support. |
| P05 | [Gamma IR](pbc/lih_infrared.py); [polar bands](pbc/lih_polar_bands.py); [material results](../reproducibility/ad_showcase/P05_lih_polar_bands/README.md); [independent Phonopy](pbc/compare_lih_polar_phonopy.py) | **Finite-model polar-spectrum demonstration available**: matched Born/dielectric and coarse-q IFC data give IR, LO–TO limits and polar bands; analytic and independent Phonopy checks pass. No ASR or charge projection. | Coarse-grid polar decomposition remains width-sensitive, so the figure illustrates the finite model rather than a converged material prediction. Further parameter/q-grid scans are optional follow-up work. |
| P06 | [Si Raman tensor](pbc/silicon_raman_tensor.py); [response checks](pbc/check_silicon_raman.py); [results](../reproducibility/ad_showcase/P06_silicon_raman/README.md) | **Finite-model mixed-response demonstration available**: Gamma/k2 full tensors, gauge/zero/translation checks and two independent cold-SCF differences are archived. k4/FFT25 single-column response also passes its true residual gates. | Raw forbidden components remain visible; k4/FFT25 zy/xy is about 6.73%. This does not establish a converged Raman selection rule. Finer-grid/FD attempts reached the 32 GiB budget and are not repeated for the showcase. |
| P07 | [Strain example/results](../reproducibility/ad_showcase/P07_periodic_strain/README.md); [positive-temperature example](pbc/h2_temperature_response.py); [thermal results](../reproducibility/ad_showcase/P07_periodic_temperature/README.md); [graphite workflow](thermally_excited_graphite/README.md) | **partial; local strain, thermal and XC-parameter gates passed**: H2 Gamma/FFT9 energy/free-energy Hessians agree with projector DFPT; both log-volume and positive-kBT curvature/mode derivatives pass three-step rebuilt differences. The thermal path checks fixed N and F_theta=-S/kB. No new core solver was needed. | The coarse/unoptimized model has three negative raw curvatures. The [external XC parameter example](../reproducibility/ad_showcase/P07_periodic_xc/README.md) now passes independent FFT41 energy/Hessian/mixed-response gates, with failed FFT9/31 retained. Converged material sensitivities remain; volume and electronic temperature are separate paths, with no thermal-expansion or anharmonic claim. Use complete subspace invariants at degeneracy. |
| P08 | [Parameter response](ofdft/parameter_density_response.py); [large-system response](ofdft/large_system_response.py); [large-system results](../reproducibility/ad_showcase/P08_ofdft_parameter_response/large_system_scaling/README.md) | **validated-component**: the original manufactured TF/vW fixed-N response and independent KKT/Taylor gates pass. Material Al1/Al27/Al125 WGC forward/JVP/VJP/HVP and DFTpy-optimizer + NumPy-WGC reference checks are archived on a common GPU6 protocol, with three-size figures. | Three sizes do not establish asymptotic scaling; the KS comparison does not isolate a KEDF-only physical error. Al343 is optional. |

## AD demonstrations and cross-links

| ID | Existing entry | Implemented / validated boundary | Limits and follow-up |
| --- | --- | --- | --- |
| A01 | [Explicit/implicit molecular response](response/explicit_implicit_response.py); [results and timings](../reproducibility/ad_showcase/A01_explicit_implicit/README.md) | **validated-component**: H2/6-31G HF and PBE energy/dipole, eight fresh processes, same seed/controls and nine nonzero iterations each; mode derivative parity ≤1.26e-16 and independent multistep FD ≤8.10e-10 at the smallest step. Forward, VJP preparation and retained pullback are timed separately. | Frozen v7 CPU source; RSS is process high-water, not backward-only allocation. Small shared-node timings do not establish universal speedups or later-core performance. |
| A02 | [Molecular JVP/VJP/HVP](response/molecular_ad_products.py); [scaled matrices and independent checks](../reproducibility/ad_showcase/A02_molecular_products/README.md) | **validated-component**: water RHF/Cartesian 6-31G*, energy/dipole in 12 scaled coordinate/field variables; dense/adjoint errors ≤1.22e-14, native multistep FD and independent PySCF Jacobian/four-energy HVP passed. | H is d²(E/Eh)/dtheta², not an unscaled Cartesian Hessian; fixed nonminimum geometry, no vibrational-frequency or third-nuclear-derivative claim. |
| A03 | [Degenerate projector crossing](solvers/degenerate_subspace.py); [figure and diagnostics](../reproducibility/ad_showcase/A03_degenerate_subspace/README.md) | **validated-component**: the original crossing and [high-dimensional/benzene backward capsule](../reproducibility/ad_showcase/A03_degenerate_vjp/README.md) pass. n=256/1024/4096 complete-subspace VJPs agree with analytic references. Ordinary eigh AD on the same smooth projector loss gives NaN at exact degeneracy; shared implicit remains finite. Native benzene/6-31G*, 1701 singles, fixed-reference bright-doublet VJP passes independent PySCF FD. The new [CH4 geometry example](solvers/ch4_cis_cluster.py) adds complete HF/AO/integral/dipole relaxation, a 15-coordinate VJP and two independent geometry differences; its [frame check and figure](../reproducibility/ad_showcase/A03_ch4_cluster_geometry/README.md) pass. | Actual loss-dependent adjoint residuals/counts are not exposed; no automatic Davidson speed/memory advantage is claimed. CH4's random MO frame requires full 90-vector forward capacity at the unchanged Ritz gate; capacities 32/64 are retained failures. Cut clusters and sorted crossing roots remain undefined. |
| A04 | [Taylor MP](mp/taylor_series.py); [recorded convergence report](mp/CONVERGENCE.md); [current and historical evidence](../reproducibility/ad_showcase/A04_taylor_mp/README.md) | **reusable**: historical native He2 MP2–30 and external atomic curves retained. Current H6 MP2–6 and independent water MP2–8 gates passed (max coefficient error 7.08e-14 Eh). | Current full He2 MP30 attempt exited 134 with LLVM allocation failure; incomplete rows are not a pass. Historical figures/source provenance and the failure log remain separate; no shared cache change. |
| A05 | [Bounded DF-HF exponent optimization](basis/optimize_exponents.py); [independent results](../reproducibility/ad_showcase/A05_basis_exponents/README.md); [existing contractions](basis/optimize_bsse_contractions.py) | **validated-component**: H2 fixed-topology/fixed-auxiliary DF-HF converged in 15 updates; positive bounded exponent scales, normalization and relaxed gradients checked; independent FD error ≤2.58e-9 Eh. | DF approximation explicitly separated from exact four-center energy. First main-basis exponent derivatives only; no auxiliary, geometry, higher/mixed exponent AD or global basis-optimum claim. |
| A06 | [Existing Boys example](localization/ethylene_boys.py); [archive wrapper](localization/archive_ethylene_boys.py); [figure/results](../reproducibility/ad_showcase/A06_boys_localization/README.md) | **validated-component**: current ethylene fixed-occupied-space run passed gradient/HVP FD, analytic local gradient, density/orthogonality invariance and independent PySCF multistart score comparison (6.40e-14 bohr²). | Eight occupied orbitals including cores, 28 rotation variables; no SCF/nuclear derivative, unique gauge or global optimum claim. Existing animation scripts remain linked. |
| A07 | [External functional](training/external_functional.py); [DFPT response target](training/dfpt_response.py); [evidence and mode boundaries](../reproducibility/ad_showcase/A07_external_functional/README.md) | **validated-component**: current three-mode H2 energy and explicit/implicit response-target runs passed; 8 focused parameter/response FD tests passed. Historical 100-step FCI histories remain separately identified. | Synthetic teacher labels do not establish chemical accuracy; fixed-density and relaxed objectives differ. Historical single-geometry FCI energy error remains 0.29041 Eh; force example not rerun in this group. |

## GW/BSE method-specific execution status

The [dedicated molecular track](AD_SHOWCASE_PLAN.md#gw-and-bse-molecular-track)
is required work within the existing catalog. The phase rows below are not
additional catalog entries. In particular, X01/X02/X05/X07 may have validated
CIS/TDDFT components while their GW/BSE extensions remain pending.

| Track phase | Existing catalog coverage | Current GW/BSE evidence | Next required gate |
| --- | --- | --- | --- |
| Current-source forward baseline | X01, X14 | **validated-component**: pinned MolGW/H2, three-method methane and all-19-QP/70-state water comparisons pass their stated gates. The [fixed-reference water G0W0 checks](../reproducibility/ad_showcase/GW_BSE_forward_gates/README.md) additionally cover four imaginary-axis quadratures, nine partial windows and eight solver/screening representations. All numerical solves converge; smaller windows fail the requested approximation accuracy. | Broader bases/physical regimes and integrated-source validation remain. G0W0 window/quadrature results do not establish evGW convergence. The [lazy structured-RPA optimization](../reproducibility/ad_showcase/GW_BSE_lazy_response/README.md) now passes 67 related regressions and complete water geometry response with direct/GMRES screening. Same-input whole-result warm forward is 4.56 s versus the historical 54.84 s; backward costs remain explicit. |
| QP and screening nuclear response | X14; prerequisite for X02 | **validated-component**: H2 three-method QP/Z/W JVP/VJP and facade parity; water all-19 G0W0, evGW0 and evGW JVP/FD on explicitly continued branches. Shared optional DIIS/outer seeds passed 35 affected regressions. [Diagnostics](../reproducibility/ad_showcase/GW_BSE_response_gates/README.md) retain convergence and branch failures separately. | Final evGW uses a known native center; MF-initialization/BLAS failures remain unresolved. Host-only eta=0 seeds and local continuation do not define a globally differentiable peak selector. |
| BSE excitation and optical response | X02, X05, X07, V06 | **validated-component**: H2 and water G0W0/evGW0/evGW -> TDA/full six-root complete directional response. [G0W0/full BSE dynamic response](../reproducibility/ad_showcase/V06_water_bse_dynamic_raman/README.md) includes all 70 states, three relaxed mode derivatives, three widths and 1761 frequencies; independent resolvent error 4.98e-12 bohr^3 and all spectral FD/grid gates pass. | Higher response, basis convergence and vibronic extensions remain. The dynamic model is harmonic frequency-dependent Placzek activity at an RHF minimum, not KHD or GW forces. Individual state branches and finite-convergence limits remain explicit. |
| GW photoelectron sensitivity | X14 | **validated-component**: water occupied-QP trace with actual 2Z weights and complete stored geometry response; final spectral FD discrepancy 7.66e-7 /(eV bohr), area/area-response integration errors 4.02e-10/2.22e-11. [Record](../reproducibility/ad_showcase/X14_water_qp_removal/README.md). | QP-only, imposed width, one geometry/direction; no satellite, photoionization-matrix-element or EOM Dyson claim. The larger-system and electron-vibration extensions remain. |
| BSE electron-vibration absorption | X15 | **validated-component**: the frozen water model has a fresh bounded rerun; its separate finite-T one-phonon extension passes raw spectral normalization and temperature JVP checks. The controlled same-reference diagonal frozen/relaxed vertex comparison also passes geometry FD and FC weight/cutoff checks. [Examples and results](bse/README.md). | Full nonadiabatic matrices, quadratic relaxed vertices and finite-temperature multiphonon spectra remain separate extensions. Historical and new outputs, and the three physical approximations, remain distinct. |

Original omegaB97X-D SCF and singlet TD support does not establish a GW
starting-point adapter. Retain unsupported-reference guards until the matching
full/long-range exchange and XC-subtraction tests pass. Periodic and open-shell
GW/BSE remain separate extensions. No GW/BSE phase is promoted by a
shared-solver test alone; the table records actual molecular calculation gates.

The [nested-JIT BSE regression](../reproducibility/ad_showcase/GW_BSE_response_gates/validation/nested_jit/README.md)
adds 15 new and 9 existing numerical passes plus the original molecular probe.
All 19 QP and three TDA mode slopes match the previously accepted raw-JVP
array exactly. The change keeps static indices and validity gates traceable
inside BSE operator closures; it does not modify the shared eigensolver or
establish general higher-order GW response.

The separate lazy structured-RPA change passed 67 related tests in 1205.52 s
and complete water geometry-response gates with direct and GMRES screening.
The same frozen water full-Davidson/GMRES warm forward is now 4.5633 s versus
the earlier whole-result 54.8380 s, with identical energies/strengths. The
actual GMRES-screened geometry JVP still costs 54.36 s warm: the response solve
is deferred to AD, not removed. Exact source/input hashes, old timings and
new derivative checks are preserved in the
[acceptance capsule](../reproducibility/ad_showcase/GW_BSE_lazy_response/README.md).
No higher-order RPA or whole-worktree validation is inferred.

## Scope boundaries and follow-up benchmarks

- **Higher nuclear derivatives:** G08 now validates selected Cartesian
  fixed-basis fourth-order RHF geometry response. V08 now validates declared
  H2 and water two-stretch quartic Hamiltonians with separate domain and
  spectral-convergence checks; this is not a globally accurate anharmonic
  molecular potential. Finite differences remain independent oracles.
- **Correlated relaxation:** G02 now validates the restricted water chain;
  V05 validates relaxed MP2/CCSD water Hessians and static IR/Raman. G03
  validates eight open-shell first-gradient cases with its explicit reference
  gauge and source boundaries. MO densities with Lambda response alone do
  not supply orbital response or establish higher-order correlated response.
- **EOM properties:** opt-in shared non-Hermitian left/right response passed
  38 solver regressions; restricted IP/EA Dyson endpoints and their fixed-MO
  response passed a separate 10-test group. H2 complete molecular geometry and
  independent FCI/spectral response now pass. The default vector contract remains
  stopped. Final integration is pending; these focused gates do not establish
  general EE transition properties or larger-basis EA exactness.
- **Periodic electric response:** Berry polarization and macroscopic electric
  enthalpy now have focused tests and finite-model LiH Born/chi checks. Crystal
  spectroscopy still needs matching converged phonons/electric tensors,
  long-range corrections and validated mixed dielectric-mode derivatives.
- **Frozen vibronic models:** existing X15 outputs are useful deliverables
  within their declared model. Preserve frozen QP/kernel, temperature, Condon
  and coupling-scale-derivative qualifications in any index, plot or results file.

## Completion worktree: delivered examples and open directions

| Item | Implementation status | Remaining delivery |
| --- | --- | --- |
| Periodic static BSE absorption | Validated 3D static TDA body kernel and [H2 spectrum](bse/periodic_hydrogen.py); 18 core and 4 optical tests passed. | Direct q0 head/wing corrections, larger material examples and periodic BSE AD remain open. |
| One-/two-dimensional GW and BSE | [H2 wire/layer examples](bse/reduced_hydrogen.py) now compute G0W0 and static TDA-BSE with matching Coulomb boundaries, checked FFT metrics and explicit bare finite heads. [Line/sheet polarizability and independent signed-v checks](../reproducibility/ad_showcase/reduced_bse_body/README.md) are packaged. | No missing compact insulating example. Singular q-cell integration, material convergence and periodic BSE AD remain method extensions. Gapless graphene additionally needs occupations/intraband treatment. |
| Off-mesh GW points and short paths | [Arbitrary-k API and H2 path](gw/periodic_bands.py) now have independent full-response NumPy and analytic contour checks. [Si's three-point generic segment](../reproducibility/ad_showcase/periodic_gw_query/silicon/README.md) computes all six requested levels successfully. | No missing compact path example. Full high-symmetry paths, degenerate subspace self-energies, q-cell corrections and interpolation remain separate extensions; the old four-point attempt remains preserved with its one failed root. |
| Fully relaxed degenerate molecular response | [CH4 complete-cluster geometry response](../reproducibility/ad_showcase/A03_ch4_cluster_geometry/README.md) now passes two independent PySCF directional differences and a random occupied/virtual frame check; nine focused tests pass. | No missing compact geometry example. Fifteen-coordinate VJP is computed, with directional FD validation; broader systems and componentwise FD remain separate extensions. |
| Larger OFDFT scaling | Al1/Al27/Al125 native forward/JVP/VJP/HVP, independent DFTpy/NumPy-WGC references and three-size plots are packaged. | No missing compact scaling example; Al343 and asymptotic/material-accuracy studies remain optional. |
| Expanded material/method matrix | LDA/PBE/r2SCAN workflows and Si PBE0 are present. | Additional hybrid material runs and some reference/result packaging. [GaN compact r2SCAN](../reproducibility/ad_showcase/PBC_material_bands/gan_r2scan_compact/README.md) fails its independent comparison. A fixed-DM diagnosis locates an AO/integral representation difference before SCF; identical-input XC checks pass. The failed model remains unvalidated and is not plotted as accepted. |

The [saved band gallery](../reproducibility/ad_showcase/PBC_material_bands/saved_bands/README.md)
plots all 17 complete material records (Si 5, GaAs 3, GaN 3, graphene 6),
plus the two separate Si GW data sets. Thirteen material comparisons pass,
three have no saved independent reference and one fails; these states remain
visible in the figures. Three incomplete records have no arrays to plot.
The gallery reused the saved numerical inputs and ran no new SCF/GW.

2026-10-19: additional validation found that the
[archived FFT21 GaAs demonstration](../reproducibility/ad_showcase/gaas_four_gw_20261019/README.md)
is underresolved for the tight Ga 3d Gaussian primitives. Its reciprocal-image
error is distinct from scGW continuation sensitivity. The
[FFT61 correction](../reproducibility/ad_showcase/gaas_four_gw_fft61_20261019/README.md)
records the causal grid diagnosis, independent PySCF PBE agreement and new
common-model G0W0/evGW/qsGW/scGW runs. The public workflow now checks
reciprocal-image bands before GW and retains a continuous physical path.
Both fine-grid scGW frequency calculations converged. A separate symmetry
test identified time-reversal breaking in the unconstrained continuation;
the model-aware measure now restores the validated source symmetry and
passes the saved-query replay. Quasiparticle-root coverage is recorded
separately from the retarded spectral map and DOS, with unavailable gaps
left explicit. Numerical-rank/window sensitivity remains measurable.
The subsequent frequency-cluster precision correction removes a false Gamma
rejection without changing scalar root tolerances: the full path/DOS query
now has 2604/2604 certified selected QPs. All eight source input files, retarded
A/DOS and previously certified roots remain byte-identical. The final source
passed 26 core tests and 52 caller regressions (three shared tests), including
genuine-crossing rejection and preservation of negative-Z diagnostics.
Spectral maps and selected-root diagnostics remain distinct; this is not
a bulk-gap, periodic-AD or continuation-stability completion claim. Separate
eight-source-point CD checks and the controlled BerkeleyGW reference remain
linked from the archived result, with their original finite-model settings.

2026-10-19 integration checkpoint: further periodic GW/BSE work is deferred.
The [external-reference sampling record](../reproducibility/ad_showcase/gaas_reference_sampling_20261019/README.md)
retains accepted BerkeleyGW G0W0 and ecalj mode-A qsGW sampling checks,
the unconverged external evGW state, and an exact-Hamiltonian GREEN/native
single-map comparison (92 common Matsubara nodes, maximum full-matrix Sigma
difference 7.18e-12 Ha). Full external GREEN scGW was not submitted. These
checks do not establish bulk/source-mesh convergence or resolve the native
off-node L-star anisotropy. Large checkpoints remain separate as described
in [result packaging](../reproducibility/ad_showcase/PACKAGING.md).

Al343, Mg, additional basis/k/grid/vacuum scans and the remaining phonon/Raman
convergence studies are optional benchmarks. They are not required to establish
the demonstrated finite-model capability. A02's historical result bundle was
recovered from its accidental nesting under A03; that relocation was not a new
numerical run.

## Checkpoint integration verification

The branch-only source snapshot passed **175 selected tests** on `jy_gpu_1`:
49 shared-response tests (264.16 s), 63 periodic/GW-screening/OFDFT tests
(412.81 s), and 63 native-integral/SCF/FCI/EOM/correlated-response tests
(1389.48 s). Each group used a separate 16-CPU affinity pool, CPU float64,
a 1800-second timeout and a 32-GiB RSS budget; peak RSS was 2.75, 6.48 and
9.63 GiB. No selected test was skipped.

This validates the committed showcase source, not the full repository suite
or the unrelated uncommitted main-workspace changes. All staged source/test
bytes match the verified source manifest. The existing native library matched
65 private native-source/export files before use. Code and result links were
checked separately; there are no executable code files in the result tree.
Raw commands, logs, source manifests and resource records remain in the local
ignored development archive.

The new `feat/ad-showcase-completion` worktree adds separate remote batches of
18 periodic-BSE core tests, four optical tests and nine CIS-cluster tests.
It also packages the H2 body spectrum, the fully relaxed CH4 cluster and the
Al125 independent reference. Those focused checks do not repeat or extend
the earlier 175-test integration claim. Current source identities and measured
responses are recorded in each result capsule. GaN's fixed-density diagnostic
locates the retained coarse-model discrepancy; it is not a passing SCF parity
test. Integration and publication are tracked separately by the release Git
history; these records describe the bounded numerical checks above.

The later periodic extensions have their own bounded checks: 16 full-Lehmann
and mixed-Fourier tests plus 22 BSE tests passed together; three existing
regressions passed separately after their old toy fixture received its missing
boundary/lattice metadata. The query API has 18 distinct 3D checks, followed
by 25 low-dimensional and related scalar/domain checks (four tests deselected).
Reduced interactions have 13 focused passes; BSE low-dimensional callers have
23 passes and a 10-case exact-zero-head follow-up that overlaps that batch.
These counts must not be added into a new whole-suite claim. The wire and
layer each have four converged QP roots, independent signed-v BSE matrices
and complex line/sheet spectral checks.
