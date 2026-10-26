# Material bands and periodic excitation benchmarks

This extension covers diamond Si, zincblende GaAs, wurtzite GaN and monolayer
graphene. Structures and reciprocal paths are explicit in [systems.py](systems.py).
The fixed geometries are example inputs; each functional does not receive a
separately optimized lattice. ASE constructs structures/paths, GradSCF performs
the primary electronic calculation, and PySCF is an independent comparison.

The [saved-data gallery](../../../reproducibility/ad_showcase/PBC_material_bands/saved_bands/README.md)
now plots all 17 completed material cases and both Si GW data sets. It contains
a semilocal overview and separate Si/GaAs/GaN/graphene PNG/PDF figures.
Different numerical protocols are separated into panels. Missing references
and the failed GaN r2SCAN comparison remain labelled explicitly; no new
electronic calculation was needed for the gallery.

The intended method ladder is LDA/PBE/r2SCAN/PBE0, periodic G0W0 and static BSE.
The native script supports the three pure DFT methods and PBE0. Hybrid
off-mesh bands use `exxdiv='vcut_sph'` consistently in SCF and the independent
reference. This is a declared finite-cell exchange quadrature; it is distinct
from the Ewald exchange used in the pure-functional pilots and current GW.

```python
from examples.pbc.band_structures.native import run

run('silicon', 'pbe', case='silicon_pbe_k2_fft31')

# A small, matched hybrid comparison; no material convergence claim.
options = {'kmesh': (2, 1, 1), 'mesh': (21, 21, 21), 'path_points': 61}
run('silicon', 'pbe', case='silicon_pbe_compact', overrides=options)
run('silicon', 'pbe0', case='silicon_pbe0_compact', overrides=options)
```

Each bounded remote job runs one case. Use ordinary Python overrides for
separate convergence studies, for example `overrides={'kmesh': (3, 3, 3)}`.
Change one numerical control at a time and keep earlier case names/results.
[compare_pyscf.py](compare_pyscf.py) constructs its own SCF and off-mesh bands
using the same numerical basis/pseudopotential definitions and actual k path.

The common basis is `gth-dzvp-molopt-sr`, with GTH-PBE potentials held fixed
across the first functional comparison. Ga uses its q13 semicore potential;
C/Si use q4 and As/N q5. This isolates the functional change within a fixed
pseudopotential model and is not an all-electron or spin-orbit calculation.

For bulk semiconductors, figures use the sampled valence maximum as their
energy zero. Graphene uses the SCF chemical potential at a declared electronic
temperature. Raw energies, both SCF-mesh and path gaps, fractional k points,
source hashes and runtime are retained. A gap sampled along a plotting path
is not automatically the Brillouin-zone fundamental gap.

Graphene can use either a labeled 3D repeated slab or a 2D Coulomb-truncated
cell with `overrides={'dimension': 2}`. Slab/wire electrostatics and a small
gapped H2 chain have separate numerical checks. The example records the
actual electrostatic dimension; finite vacuum thickness is still an explicit
numerical input. Reduced-dimensional GW long-wave corrections remain open.
Gapless graphene GW/BSE also requires occupation/intraband validation; an
artificial insulating gap will not be used to bypass that requirement.

The [small periodic H2 BSE example](../../bse/periodic_hydrogen.py) now uses
matched QP energies and screening, a complex momentum-conserving electron-hole
kernel and shared numerical solvers. It demonstrates a three-dimensional
static singlet TDA body spectrum with per-cell oscillator strengths. Direct
q0 head/wing corrections, periodic BSE AD and material absorption coefficients
remain outside that example. Bulk dielectric spectra and sheet/line
polarizabilities require their respective normalizations.

See the [project plan](../../AD_SHOWCASE_PLAN.md#added-material-bands-periodic-gwbse-and-reduced-dimensional-cells)
for the dependencies and acceptance boundaries. All numerical work stays on
`jy_gpu_1`; no local tests or material calculations are performed.

The compact Si PBE/PBE0 comparison has passed one independent numerical
check: frontier-band discrepancies are 3.06e-9 and 3.69e-9 Eh. Its sampled
path gaps are 0.30049 and 1.93646 eV. See the
[band comparison figure](../../../reproducibility/ad_showcase/PBC_material_bands/silicon_hybrid.png)
and [result inventory](../../../reproducibility/ad_showcase/PBC_material_bands/).

The [direct GW query example](../../gw/periodic_bands.py) now also supplies a
[three-point Si generic segment](../../../reproducibility/ad_showcase/periodic_gw_query/silicon/README.md).
All six requested QP values converge, with full occupied/virtual screening
and shifted internal bands. This deliberately small GTH-SZV, two-q, body-only
model is a functional demonstration; it is not a complete high-symmetry
path or a bulk Si prediction. That record used nondegenerate output bands;
the current query also supports complete degenerate subspaces.

## Compact native GW material paths

The [2026-10-18 BerkeleyGW review](../../../src/gradscf/gw/pbc/BERKELEYGW_REVIEW.md)
and [two-point root diagnostic](../../../reproducibility/ad_showcase/material_gw_bands/multiroot_audit/README.md)
confirm that different seeds can converge to different QP roots in the
graphene path. These saved curves are diagnostic finite-model outputs;
the convergence masks certify equation residuals, not physical branch
selection. Long-wave screening and material convergence remain open.

[gw_bands.py](gw_bands.py) adds a separate PBE/GTH-SZV finite-model workflow
for the same Si, GaAs and GaN geometries. The bulk source k mesh is `(2,1,1)`;
FFT grids are `21^3` for Si/GaAs and `(21,21,33)` for GaN. These inputs differ
from the DZVP material gallery and must be plotted in separate panels.
Output windows contain three occupied and three virtual frontier bands:
Si bands 1--6, GaAs bands 6--11 and GaN
bands 15--20 (zero-based). The saved full-path KS preflight verified these
windows as numerically closed for this declared finite model. The initial
eight-band Si pilot retained two failed roots for its deepest valence band 0;
that separate record remains archived. Band 0 is outside the displayed frontier
window, but stays in internal G and screening W. A new path or
numerical model requires its own preflight before an explicit `orbs` override
is chosen. All internal G bands and all OV+VO screening
transitions remain in the calculation. Any partially selected degenerate
cluster is rejected instead of silently enlarging the output window.

```python
from examples.pbc.band_structures.gw_bands import run

# Inspect native SCF and Gamma/generic query spectra before the GW pilot.
run('silicon', case='silicon_szv_reference', calculate_gw=False,
    checkpoint='reproducibility/ad_showcase/PBC_silicon_gw/scf_reference.npz')

# A later fresh case starts from that converged density via public kernel(dm0).
run('silicon', case='silicon_szv_qp_pilot',
    checkpoint='reproducibility/ad_showcase/material_gw_bands/'
               'silicon_szv_reference/scf_reference.npz')

# Denser, directly computed bands with the same SCF/screening model.
run('silicon', case='silicon_szv_dense61', path_points=61,
    checkpoint='reproducibility/ad_showcase/material_gw_bands/'
               'silicon/scf_reference.npz')
```

The bulk two-point pilots contain Gamma and one generic point; graphene uses
Gamma and K so that its pilot checks the Dirac-contact treatment. `path_points=19`
requests a small high-symmetry path through ASE; every requested QP is then
evaluated at an actual path point, one query per call. No spline or scissor
correction supplies missing QPs. The saved computed/converged masks control
which QPs can appear in a figure; connecting accepted adjacent samples is a
visual guide, not an interpolation calculation. Partial or rejected cases
retain their reference spectra, completed point indices and diagnostics.

The denser paths retain the declared source k mesh, FFT, basis and frequency
grid. The stored 19-point and 61-point records use different query source
hashes; the latter includes the bounded hybrid retry, so their comparison
does not isolate path sampling alone. Additional samples reduce polygonal
plotting artifacts; failed
roots and changes between independently selected nonlinear roots can still
produce gaps or jumps. The figures preserve those numerical results.

The [completed 61-point GW paths](../../../reproducibility/ad_showcase/material_gw_bands/dense61/README.md)
contain the denser gallery, raw arrays and adjacent-correction diagnostics.
Si/GaAs/GaN/graphene have 359/366, 363/366, 366/366 and 413/427 converged QPs,
respectively. The denser samples still expose accepted-root jumps, so these
curves do not establish continuous physical QP branches.

This work uses the SRC projected Dyson treatment for complete KS degenerate
subspaces. It is eager forward only and retains the finite-model body
interaction; no GW geometry AD, long-wave correction or converged material
gap is implied. In particular, the bulk `(2,1,1)` source mesh is anisotropic;
its density is not converged with respect to crystal symmetry or k sampling.
Pilot calculations establish actual time, RSS and QP convergence before
increasing the path length. Earlier coarse-contour results are retained as
baseline records: Si and GaAs full paths contain failed roots, and the first
GaN pilot converged all 12 requested values. These records are distinct from
the stabilized protocol below.

The current example explicitly enables `pole_subtraction=True`. It evaluates
the same static screened matrix elements at zero frequency, subtracts them
from the numerical imaginary integral and real-pole residues, and restores
their analytic contribution. For an internal-state occupation `f`, that
contribution is `(1/2-f)*Wc(0)`. This resolves the cancellation near a Green
pole without adding a separate root or screening solver. The imaginary-axis
Green denominator uses zero eta in this protocol; signed eta is retained in
the real-frequency screened action. It agrees with the earlier formulation
as eta tends to zero, but it is a declared change of finite-eta convention.
The ordinary query API keeps its previous protocol by default; a band-touching
call always uses the subtraction. Independent low-frequency RPA pole and
complex-frame checks are part of the 63-test remote validation.

Stable failed scalar secants receive one bounded retry through the shared
AD-hybrid root solver, starting again from the original energy guess. This
does not create a second self-energy model or a private solver. Matrix
branches and the default non-subtracted protocol retain their stated solver
behavior; a failed retry remains masked. The final focused query gate has
67 passes, including four regressions for this scalar recovery.

Long eager paths release JIT executables between points. `resume_from=...`
can reuse a partial path in a new case after matching the model, physical
source hashes, k points and reconverged KS energies. Its timings cover only
the new process and must be reported alongside the original run. Results
from the earlier contour convention cannot be resumed into the new one.

Graphene uses its unchanged physical structure with a newly converged
zero-temperature SCF, source k mesh `(2,2,1)`, a two-dimensional finite-part
boundary and `exxdiv=None`. Its 12 Angstrom cell height and FFT `(25,25,97)`
are compact demonstration inputs; they differ from the older 20 Angstrom
DZVP gallery. No existing warm-density state is converted to integer
occupations. The `band_touching=True` query option permits a neutral numerical
Dirac contact at a stated `gap_tol=1e-5 Ha`. The saved K-point KS gap is
7.2674325298e-6 Ha (about 0.198 meV) on this finite grid, rather than exactly
zero. The source `(2,2,1)` mesh has a sampled gap of about 0.16643 Ha.
This numerical classification does not modify KS energies, introduce a gap
or remove complete response channels. Earlier frozen reference-only records
retain their originally declared `1e-6 Ha` tolerance; the new GW records
capture the updated caller tolerance and source hash.
The output includes bands 0--6: the extra sigma state completes the K-point
doublet cut by a six-band frontier window. This is an explicit input choice,
not hidden automatic closure. The sampled product-Gram validity checks still apply. The `GMKG`
path uses 13 directly calculated points. This is not finite-temperature
metallic/intraband GW, a q0-converged calculation or a vacuum-converged
isolated-layer prediction.

The [selected material GW results](../../../reproducibility/ad_showcase/material_gw_bands/README.md)
provide raw arrays, convergence masks, numerical controls, separate full-path
and retry resources, individual figures and a four-material gallery. These
figures compare G0W0 with the matching SZV PBE bands. Their energy axes cover
all requested bands; path distances use the actual continuous reciprocal
path rather than overlapping sparse ASE ticks.

These earlier GaAs FFT21 records predate the Gaussian Fourier-resolution
diagnosis below. They remain historical finite-resolution diagnostics; use
the FFT61 reference for the new four-method GaAs example. Other materials
need their own reciprocal-image and FFT convergence checks.

## GaAs self-consistent GW and independent reference

[`gaas_self_consistent_gw.py`](gaas_self_consistent_gw.py) exposes `reference`
and `calculate` for diagonal G0W0, eigenvalue-only GW, full mode-A qsGW and
full matrix Matsubara scGW. They share a finite PBE/GTH-SZV/GTH-PBE source
Hamiltonian, all 14 orbitals and 18 electrons. The example uses `fc=False`,
`exxdiv=None` and no spin-orbit coupling; dielectric cutoff and the full-FFT
bare interaction are independent. These settings define a finite-model
demonstration, not a converged bulk GaAs prediction.

[`gaas_gw_spectra.py`](gaas_gw_spectra.py) consumes four converged saved states.
It checks source geometry, orbitals and computed/converged QP masks, then
interpolates source operators onto a dense path and an independent uniform
DOS mesh. scGW uses the shared causal Hermitian realization, held-out
Matsubara samples, its first self-energy moment and a separate full-spectral
electron-count check. Its bands solve the continued matrix Dyson equation;
they are not static Fock eigenvalues. `frequency_cutoff` is an explicit Ha
control requiring a separate forward-frequency study; excluded-frequency
errors remain in the report. Rank, range and broadening stability also need
independent checks. Static QP DOS counts unit-weight bands, whereas scGW DOS
is the interacting spectral trace including satellites.

The GaAs continuation passes its source model to preserve spinless time
reversal through complex MO sewing matrices, after validating the original
full-frequency state. A valid retarded spectral function does not require a
unique ReSigma quasiparticle root for each orbital. Uncertified roots remain
NaN with their candidates and masks saved; each reported gap requires complete
roots for its own sample, otherwise that gap and its change are unavailable.
The recorded FFT61 case now has 2604/2604 certified selected roots after
separating eigenvalue resolution from root stopping tolerances. The change
leaves the interacting spectrum and all previously certified roots unchanged;
the result record preserves the earlier Gamma rejection as a diagnostic.

The [archived FFT21 demonstration](../../../reproducibility/ad_showcase/gaas_four_gw_20261019/README.md)
contains 121-point paths and independent `4^3` DOS samples, but subsequent
validation found that its Gaussian AO Fourier grid is underresolved. Its
equivalent reciprocal representatives give different PBE bands. These
figures remain diagnostics and must not be used as resolved material results.
The [FFT61 correction and independent PBE validation](../../../reproducibility/ad_showcase/gaas_four_gw_fft61_20261019/README.md)
retain the `(2,1,1)` / 16 Ry finite model and report the replacement runs.
The example rejects reciprocal-image errors above `1e-6 Ha` before GW and
keeps the physical plotting path continuous. `precision` does not select the
explicit FFT mesh. scGW uses 1024 positive frequencies and Gauss time
integration; its main panel displays the continued spectral function.
Selected-root curves remain a separate diagnostic because their real-axis
stability needs independent checks. The completed `(2,2,2)` CD checks are
recorded separately, without mixing their mesh with this source model.

[`berkeleygw_gaas.py`](berkeleygw_gaas.py) exports the actual native orbitals
for an independent BerkeleyGW Epsilon/Sigma comparison. The executed
[matched-frequency evidence](../../../reproducibility/ad_showcase/gaas_berkeleygw_matched_eta_20261018/README.md)
records the finite-body head adapter, remaining model differences, original
multiple-root flags and scope-specific errors. It must not be relabelled as
standard semiconductor-head output or a full-Dyson scGW comparison.
