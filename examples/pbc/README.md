# Periodic structure, response and vibrations

The showcase separates electronic/nuclear response from spectral
postprocessing. Read the result records before combining calculations:

| Entry | Examples and evidence |
| --- | --- |
| Forces, stress and cubic optimization (P01/P02) | The LiH workflow below; [results](../../reproducibility/ad_showcase/P01_P02_lih_periodic/README.md). |
| Raw Gamma phonons (P03) | [lih_gamma_phonons.py](lih_gamma_phonons.py), [independent gradient FD](compare_lih_gamma_phonons.py), [grid gates](compare_lih_gamma_grids.py) and [results](../../reproducibility/ad_showcase/P03_lih_gamma_phonons/README.md). |
| Self-inverse zone boundary (P03) | [lih_zone_boundary.py](lih_zone_boundary.py), [independent two-cell check](compare_lih_zone_boundary.py) and [phase, folding and status](../../reproducibility/ad_showcase/P03_lih_zone_boundary/README.md). Probe gates pass; full curvature validation remains in progress. |
| Born charges and dielectric response (P04) | [lih_born_charges.py](lih_born_charges.py), [lih_dielectric.py](lih_dielectric.py), [finite-field checks](check_lih_finite_field.py); separate [Born](../../reproducibility/ad_showcase/P04_lih_born/README.md) and [field](../../reproducibility/ad_showcase/P04_lih_dielectric/README.md) records. |
| Gamma infrared and LO-TO limit (P05) | [lih_infrared.py](lih_infrared.py) consumes matching-grid inputs; [plotting](plot_lih_infrared.py), [array-kernel validation](../../reproducibility/ad_showcase/P05_infrared_kernels/README.md) and [theory](../../src/gradscf/pbc/INFRARED.md). Material checks remain distinct from the analytic tests. |
| Polar IFC decomposition (P05) | [polar_interpolation.py](polar_interpolation.py), [independent Phonopy kernel](compare_polar_ifc_phonopy.py), [plots](plot_polar_interpolation.py) and [results](../../reproducibility/ad_showcase/P05_polar_interpolation/README.md). Manufactured two-atom model; reciprocal/q-mesh and q-response gates pass. |
| Local curvature versus volume (P07) | [h2_strain_response.py](h2_strain_response.py), [plotting](plot_h2_strain_response.py) and [results](../../reproducibility/ad_showcase/P07_periodic_strain/README.md). Complete periodic energy/SCF third response in a deliberately small model; raw unstable curvatures and nonstationary geometry remain explicit. |
| Local curvature versus electronic temperature (P07) | [h2_temperature_response.py](h2_temperature_response.py), [plotting](plot_h2_temperature_response.py) and [results](../../reproducibility/ad_showcase/P07_periodic_temperature/README.md). Positive-kBT, fixed-N Mermin response at fixed geometry/cell; no expansion or anharmonic shift. |
| Local curvature versus external XC parameter (P07) | [h2_xc_response.py](h2_xc_response.py), [independent PySCF](compare_h2_xc_response.py), [figure/results](../../reproducibility/ad_showcase/P07_periodic_xc/README.md). Complete relaxed mixed third response; FFT41 passes independent energy/Hessian/parameter gates; failed coarse grids retained. |
| Finite-q phonon dispersion (P03/P05) | [lih_qmesh_phonons.py](lih_qmesh_phonons.py), [lih_ifc_bands.py](lih_ifc_bands.py) and [lih_polar_bands.py](lih_polar_bands.py) reuse saved response matrices to show bands and DOS. |
| Material bands | [band_structures](band_structures/README.md): Si, GaAs, GaN and graphene; separate native and independent scripts. |
| One- and two-dimensional cells | [reduced_dimensions.py](reduced_dimensions.py): neutral H2 wire/slab SCF and off-mesh bands using consistent truncated electrostatics. |

The Gamma correction does not supply polar finite-q IFC interpolation. Keep
raw charge sums, acoustic defects and grid failures visible; selecting a
stable optical subspace is not permission to erase those diagnostics.

## LiH forces and cubic equation of state

The LiH showcase uses a two-atom rocksalt primitive cell, Cartesian Gaussian
orbitals, GTH-Pade pseudopotentials and explicit FFT/k sampling. Read the
[results and limitations](../../reproducibility/ad_showcase/P01_P02_lih_periodic/README.md)
for the accepted settings, independent comparisons, rejected basis case,
resource limits and source hashes.

The archived capsule contains 12 native displaced-cell cases, nine independent
property comparisons and paired native/PySCF cubic minima. Four PNG/PDF figure
pairs show forces, separate numerical sensitivities, final increments and the
equation of state. The fixed-a FFT/k checks and two-basis sensitivity must not
be interpreted as a converged prediction of the optimized lattice parameter.

The essential public-API calculation is:

```python
import jax
import numpy as np
jax.config.update("jax_enable_x64", True)

from gradscf.pbc import dft, gto
from gradscf.scf import SCFDifferentiationConfig

a = 7.6  # bohr; conventional cubic lattice parameter
lattice = .5*a*np.array([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]])
positions = np.array([[0., 0., 0.], [.5, .5, .5]]) @ lattice
positions[1] += [.025, -.015, .010]  # nonzero restoring force
cell = gto.M(atom=list(zip(["Li", "H"], positions)), a=lattice, unit="Bohr",
             basis="gth-szv", pseudo="gth-pade", cart=True, mesh=(31,)*3)
mf = dft.KRKS(cell, xc="pbe", kpts=cell.make_kpts((2, 2, 2)),
              temperature=None, exxdiv="ewald", max_cycle=150, damping=.1,
              conv_tol=1e-12, conv_tol_density=1e-11, conv_tol_grad=1e-10,
              differentiation=SCFDifferentiationConfig(
                  mode="implicit", tolerance=1e-11, max_iter=100, restart=40)).run()
assert mf.converged
gradient = mf.nuc_grad_method()
forces = -gradient.kernel()       # Eh / bohr
stress = gradient.stress()        # tensile-positive, Eh / bohr^3
pressure = -np.trace(stress)/3
```

The potential supplies Li q3/H q1, hence **four explicit electrons**. The
scripts additionally enforce a positive sampled gap and an AO metric condition
gate. Fixed fractional atoms and fractional k points define the strain
derivative. HF uses the same declared Ewald exchange convention as its PySCF
reference. The example does not establish a converged material prediction.

| Script | Role |
| --- | --- |
| [lih_forces_stress.py](lih_forces_stress.py) | Public SCF/geometry derivative flow, diagnostics and one explicit settings dictionary. |
| [compare_lih_forces_stress.py](compare_lih_forces_stress.py) | Independent PySCF E/F/stress and multi-step reconverged displacement/strain FD; also retains the initial volume pilot. |
| [lih_lattice_optimization.py](lih_lattice_optimization.py) | Fresh-cell, fresh-process scalar cubic optimization from a measured bracket, using AD stress. |
| [compare_lih_lattice_pyscf.py](compare_lih_lattice_pyscf.py) | Independent bounded energy minimization and curvature/FD checks. |
| [plot_lih_forces_stress.py](plot_lih_forces_stress.py) | Source-checked saved-data figures and convergence tables. |

There are no command-line options. Change the ordinary Python settings or use
the small functions directly, for example:

```python
from examples.pbc.lih_forces_stress import SETTINGS, run_case
from examples.pbc.compare_lih_forces_stress import compare_case

run_case("my_lih_case", dict(SETTINGS, mesh=[41]*3))
assert compare_case("my_lih_case", finite_differences=True)["passed"]
```

Run from the repository root with `src` and the root on `PYTHONPATH`. Keep
numerical calculations and plotting on the declared validation host when
reproducing this capsule. Each archived job records its bounded command and
environment. Large k-mesh runs are separate from the small default example.

The archived `pbe_szv_k2_fft31_repaired` case is the repeated primary baseline
after the implicit-forward memory repair. The unsuffixed original result and
its finite differences remain separately preserved; the suffix is an output
label, not a different physical method. The
[memory diagnostic](../../reproducibility/ad_showcase/P01_P02_lih_periodic/MEMORY_DIAGNOSTIC.md)
keeps scientific checks, resource measurements and infrastructure failures
separate.

Other periodic capabilities and their current boundaries are listed in the
[PBC acceptance ledger](../../src/gradscf/pbc/ROADMAP.md). The pre-existing
[periodic_forces.py](periodic_forces.py) remains a smaller thermal/OFDFT example.
