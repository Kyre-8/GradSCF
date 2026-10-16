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
path or a bulk Si prediction. Nondegenerate selected bands are required.
