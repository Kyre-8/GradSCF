# Thermally excited graphite

This example develops a GradSCF calculation around the observations in
[McCauley, Koch and Haas, arXiv:2610.18561v1](https://arxiv.org/html/2610.18561v1).
It separates quantities that can otherwise be confused: electronic-temperature
changes of harmonic frequencies, Bose populations, a local anharmonic potential,
and instrumental broadening. It does not fit an empirical redshift or lifetime
to the experimental points.

## Run order

Run from the repository root with `PYTHONPATH=src`, in an environment with JAX
float64, the usual GradSCF DFT dependencies, NumPy, SciPy and Matplotlib. PySCF is
needed only for the independent comparison. Inputs are edited directly in
`settings.py`; the scripts have no CLI. Generated files go to `graphite_results/`
under the working directory.

```sh
python examples/thermally_excited_graphite/00_reference.py
python examples/thermally_excited_graphite/01_gamma_temperature.py
python examples/thermally_excited_graphite/02_frozen_mode.py
python examples/thermally_excited_graphite/03_compare_pyscf.py
python examples/thermally_excited_graphite/04_plot_spectra.py
```

Use `JAX_PLATFORMS=cpu` for the recorded CPU path. The separate, more expensive
`05_dispersion.py` samples the in-plane Gamma--K--M--Gamma dispersion with a
6x6x1 k mesh. Sixths are needed to represent both K and M exactly. Its connected
plot is a set of direct samples, not a dense Fourier interpolation.
`06_grid_convergence.py` independently compares both electronic solvers on
three FFT meshes, records any bounded PySCF self-restart, and plots the energy
differences. It does not require a preceding displacement scan.
After 05 completes, `07_momentum_spectrum.py` uses its direct q samples to plot
relative momentum-resolved loss/gain maps in the high-energy window. It holds
the electronic modes fixed and varies Bose occupations only.

`08_gamma_convergence.py` separately refines the Gamma phonon FFT grid, retaining
the complete in-plane eigensystem and response diagnostics. Its cases and
temperatures are editable at the top of the file; each setting writes to its
own `graphite_results/gamma_precision/` subdirectory. This does not overwrite
the coarse pilot or certify convergence in basis, k points or lattice geometry.
`09_compare_paper.py` compares unresolved synthetic peaks to the paper's
reported experimental peaks. It reads the pilot and any completed refinement
cases, keeping their settings separate; temperature shifts require both
endpoints from the same case. Experimental inputs and their source locations
are recorded in `experiment_reference.json`.

`10_r2scan.py` changes only XC relative to the SZV, FFT=(33,33,63), k=(3,3,1)
PBE Gamma reference and writes a separate r2SCAN case for 09. The GTH-PBE
pseudopotential is retained and explicitly identified; it is not an
r2SCAN-generated pseudopotential. This calculation tests the new tau-dependent
SCF/DFPT path without conflating XC with a basis or lattice change. Its two
temperatures describe electronic occupations, not a bulk anharmonic model.
The alternative FFT39 case at the top of 10 reproduces the separate 300 K grid
diagnostic. `11_translation_check.py` checks the FFT33 case by rigidly translating
all atoms and independently reconverging SCF at two displacement step sizes.
It tests the raw discretized energy without enforcing an acoustic sum rule.
When 09 detects multiple local spectral maxima, it saves multipeak plots and
diagnostics and excludes that entire case from single-Voigt experimental bias
and temperature-shift comparisons.

| File | Main calculation | Reusable implementation |
| --- | --- | --- |
| [settings.py](settings.py) | Four-carbon AB geometry, basis, grids and SCF controls | Existing `pbc.gto` and `pbc.dft` |
| [00_reference.py](00_reference.py) | Finite-electron-temperature PBE reference and occupations | Existing periodic SCF |
| [01_gamma_temperature.py](01_gamma_temperature.py) | In-plane force response; select the four highest optical modes | `Phonons.project` |
| [02_frozen_mode.py](02_frozen_mode.py) | Re-converged displaced free energies; compare orders and windows | `pbc.anharmonic.fit_frozen_mode` |
| [03_compare_pyscf.py](03_compare_pyscf.py) | Independent SCF and central-difference curvature | PySCF FFTDF/PBE |
| [04_plot_spectra.py](04_plot_spectra.py) | Relative loss/gain spectra and comparison panels | `pbc.vibrational` |
| [05_dispersion.py](05_dispersion.py) | Direct finite-q in-plane phonon samples | Existing DFPT plus `Phonons.project` |
| [06_grid_convergence.py](06_grid_convergence.py) | Matched k vectors and independent FFT energy convergence | GradSCF and PySCF SCF |
| [07_momentum_spectrum.py](07_momentum_spectrum.py) | High-energy momentum-resolved gain/loss maps | `pbc.vibrational`, saved direct-q modes |
| [08_gamma_convergence.py](08_gamma_convergence.py) | Gamma phonon FFT sensitivity, with raw signed modes | `Phonons.project` |
| [09_compare_paper.py](09_compare_paper.py) | Unresolved spectral peaks, widths and experimental endpoint comparison | `pbc.vibrational`, SciPy peak fitting |
| [10_r2scan.py](10_r2scan.py) | Matched-setting r2SCAN Gamma optical modes | Periodic meta-GGA SCF and DFPT |
| [11_translation_check.py](11_translation_check.py) | Reconverged rigid-translation free-energy curvature | Existing periodic input builder and SCF |

## What the calculation means

The starting geometry is fixed AB graphite with a=2.46 Angstrom and
c=6.708 Angstrom. These are representative input values, not an optimized PBE
geometry or a reconstruction of the experimental specimen. The pilot uses
GTH-SZV/GTH-PBE, FFT=(15,15,31), k=(3,3,1), and electronic temperatures 300,
800 and 1377 K. Neither basis, FFT nor out-of-plane k convergence is established.
Electron temperature enters SCF as kBT in Hartree; phonon Bose temperature is
passed separately in kelvin to the spectrum routine.

The projected response uses the complete eight-dimensional in-plane displacement
space. Four intralayer-opposite trial vectors alone need not span an invariant
subspace at a finite discretization. The full Cartesian dynamical residual is
therefore checked, with an explicit relative threshold of 1e-3. Signed energies
of all in-plane modes are saved; only the positive highest four are used for the
optical illustration. No ASR projection or absolute-value replacement hides
unstable modes. Tight SCF tolerances are needed to satisfy DFPT reference checks.

The frozen-mode coordinate is u=s*d, where d is a stated unit Euclidean direction
and s is in Bohr. Every displacement reconverges the same electronic free energy
at 300 K. The fit exposes a_n in F(s)=sum(a_n*s^n) and derivatives n!*a_n, in
Eh/Bohr^n. Both polynomial order and displacement window are compared. This is
a **finite-displacement fit**, not automatic differentiation through SCF to
fourth order. `directional_energy_taylor` is separately available for energy
graphs with supported JAX jet rules; unsupported SCF/primitive rules are not
silently replaced by finite differences.

The 300 K finite k mesh has a small local expansion radius: the default scan
uses |s|<=0.0012 Bohr, not a thermal displacement distribution. Wider fits can
have small-looking energy residuals yet severely bias high-order derivatives.
The comparison checks the fitted quadratic coefficient against AD/DFPT and
reports the independent finite-step curvature separately. Native Fourier AO
integrals and PySCF analytic AO integrals converge differently with FFT size;
the same mesh dimensions do not make their finite discretizations identical.

The spectral illustration uses Q equal to a nonzero reciprocal vector, equal
unit atomic form factors, and a fixed Gaussian FWHM of 11 meV. It includes
coherent mode/atom weights and the 1/(2*phonon_energy) zero-point factor. The
plot contrasts Te=Tph with fixed 300 K electronic modes and variable Tph, so
population changes can be distinguished from changes of the electronic
reference. These assumptions yield relative kinematic intensities, not an
absolute EELS cross section, a lifetime prediction or the experimental detector
acceptance. Data files carry settings and are checked before being combined.

## Experimental targets and remaining physics

Section III.A of the paper reports an unresolved Gamma optical peak at
193.7+/-0.2 meV at room temperature and 185.2+/-0.3 meV at 1377+/-11 K.
The plot includes these as external markers; it does not assign them to an
individual calculated mode or calibrate the model to them. The stated 11 meV
instrumental resolution is used as a Gaussian illustration of broadening.
The room-temperature marker is placed at 300 K for display.

The more precise comparison distinguishes individual eigenvalues, the maximum
and FWHM of the broadened envelope, and the center/width of a single Voigt fit.
Its instrumental profile is an assumed 11 meV FWHM Gaussian; the paper reports
an approximately 11 meV zero-loss-peak width but does not supply the response
function used here. A fitted Lorentzian component in an unresolved synthetic
mixture is not evidence of a computed lifetime. The paper's approximate 7 meV
width increase belongs to its time-resolved dataset (Fig. 4f); it is not used
as a static room-temperature-to-1377 K linewidth target. No unreported endpoint
widths, errors or intermediate experimental temperatures are invented.

A quantitative temperature-dependent reproduction still requires converged
electronic/phonon sampling, lattice expansion, mixed-mode cubic/quartic vertices,
Brillouin-zone phonon self-energies, and the relevant electron-phonon corrections.
A single frozen Gamma direction cannot supply these. Experimental peak width
also contains unresolved-mode and momentum-window effects. Predicting the
time trace further needs an input or predicted temperature history and gated
detector averaging. No temperature-dependent lifetime or laser-heating model
is implemented by these scripts.

For the nonharmonic theory, see
[Bonini et al., 2007](https://arxiv.org/abs/0708.4259),
[Giura et al., 2012](https://arxiv.org/abs/1207.6364), and
[Paulatto et al., 2013](https://arxiv.org/abs/1304.2626).
Implementation conventions are in
[pbc/VIBRATIONS.md](../../src/gradscf/pbc/VIBRATIONS.md).

Measured output, verification records and the limits of the pilot are collected
in [RESULTS.md](RESULTS.md), with selected reference tables and figures under
`results/`. Those figures are generated by the scripts, not fitted to the paper.
