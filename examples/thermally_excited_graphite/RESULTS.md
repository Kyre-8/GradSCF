# Graphite pilot: measured results and limits

Measured on 2026-10-15 using `jy_gpu_1` (`jax_scf`), Intel Xeon Platinum 8457C,
JAX 0.10.0, PySCF 2.13.0 and CPU float64. OMP/OpenBLAS thread settings were two;
this does not fix the number of XLA compiler threads. Jobs sometimes overlapped,
so the recorded elapsed times are execution records, not isolated speed tests.
No numerical tests or plotting ran on the local workstation.

**This is a verified implementation pilot, not a quantitative reproduction of
the paper's thermal redshift or linewidth.** The default basis and sampling are
not converged. Electronic occupation effects alone do not provide the required
bulk anharmonic self-energy. A fixed Gaussian width is instrumental broadening,
not a computed phonon lifetime.

## Source and regression checks

The work starts from GradSCF commit `9181981`; the reusable numerical additions
are committed as `ea3edf9`. The final numerical source/test
hashes are recorded in `results/source_manifest.json`. Tests were executed on
the corresponding remote snapshot in
`/home/yjiao/gradscf_checks/graphite-9181981`.

| Tests | Passed |
| --- | ---: |
| Projected modes, mass metric, full-space residuals, finite q and invalid/stale states | 16 |
| Frozen-mode fits and direct jet, factorials, gradients, invalid/overflow data | 24 |
| Loss/gain spectra, detailed balance, phases, units, JIT/AD and degeneracy | 24 |
| Existing Gamma and supercell HF/OFDFT phonon regressions | 6 |
| Total distinct tests | 70 |

The final combined selection passed 62 tests in 175.36 s; the four complementary
supercell cases passed in 236.00 s. Logs are `final-tests.log` and
`supercell-regression.log`. The two selections together cover all cases in the
four original test selections, without skipped cases. Independent review added
four projected-mode regressions; the final complete projection file passed
**16 tests in 198.20 s**, recorded in `projection-final.log`. Unit-direction
solves and mass-whitened Hermiticity checks prevent direction scaling from
weakening the numerical gate. Finite-q integer directions are promoted to
floating point before JVP. Missing APIs and the subsequent numerical guard
failures were observed before their repairs.

## Default fixed-lattice calculation

AB graphite, four carbon atoms, a=2.46 Angstrom, c=6.708 Angstrom,
Cartesian GTH-SZV/GTH-PBE, PBE, k=(3,3,1), FFT=(15,15,31), no Ewald exchange
correction. The lattice is fixed rather than optimized. The SCF thresholds are
1e-12 Eh in free-energy change, 1e-11 density change and 1e-10 residual.

At 300 K electronic temperature:

- Internal energy: **-22.472858173664125 Eh/cell**.
- Electronic free energy: **-22.473441428470178 Eh/cell**.
- The four highest in-plane modes are listed below. The complete in-plane
  subspace passes the full-space relative residual check at roughly 1e-13.

| Electron temperature (K) | Four optical energies (meV) |
| ---: | --- |
| 300 | 175.086390, 175.453281, 178.559947, 178.922688 |
| 800 | 176.227509, 176.590976, 178.762269, 179.125516 |
| 1377 | 178.064789, 178.424223, 180.415972, 180.776184 |

These adiabatic electronic-temperature shifts are upward in this pilot. They
do not reproduce the experimental downward shift. No mode has been manually
shifted to match an experimental peak. The initial four-vector optical trial
space had relative leakage about 0.012 and was rejected; the full eight-vector
in-plane space is used instead. All signed in-plane energies remain in the raw
output, including unstable/poorly converged low-frequency modes.

## Local derivatives and the independent reference

For the unit Euclidean frozen direction in example 02, with scalar displacement
in Bohr and |s|<=0.0012:

| Quantity | Value |
| --- | ---: |
| AD/DFPT directional second derivative (Eh/Bohr^2) | 0.942508323908003 |
| Sixth-order fit second derivative (Eh/Bohr^2) | 0.942511917668752 |
| Fourth-order fit second derivative (Eh/Bohr^2) | 0.942300915569165 |
| Fourth-order fit fourth derivative (Eh/Bohr^4) | -185839.841162 |
| Sixth-order fit fourth derivative (Eh/Bohr^4) | -190678.468446 |
| Fourth-order fit RMS energy residual (Eh) | 6.45349e-12 |

The quadratic fit/DFPT agreement validates the electronic relaxation and units
in this directional check. The fourth derivative still varies by a few percent
with fit order. A small energy residual alone is not a high-order accuracy
certificate. Much wider windows produced strongly different coefficients and
were not used for the displayed local derivative.

At a matched central-difference step of 0.0004 Bohr, GradSCF gives
0.939971855907 Eh/Bohr^2 and PySCF gives 0.907564223773 Eh/Bohr^2. Finite-step
bias and the different finite AO/FFT discretizations must be distinguished:
neither finite-step value is silently substituted for the AD derivative.
This coarse-grid curvature comparison does not meet a tight cross-code
derivative agreement criterion. The higher-order coefficients are local
Gamma diagnostics, not Brillouin-zone-integrated thermal corrections.

## Electronic grid convergence

Independent calculations at identical absolute k vectors give:

| FFT mesh | GradSCF free energy | PySCF free energy | Difference (Eh/cell) |
| --- | ---: | ---: | ---: |
| 15x15x31 | -22.473441428470178 | -22.478401245399947 | 4.95981693e-3 |
| 21x21x41 | -22.475620909761393 | -22.475758925741165 | 1.38015980e-4 |
| 27x27x51 | -22.475556882778235 | -22.475557665773163 | 7.82994928e-7 |

All these references ultimately converged. In the independent diagnostic run,
the largest-grid PySCF reference required one bounded restart from its own
density after its first 180 cycles. No native density initialized PySCF.
Native finite-G AO/S/T construction and PySCF real-space AO/analytic S/T
construction approach one another with grid refinement; the nuclear energies
agree to about 7e-15 Eh in that diagnostic. This supports a discretization
explanation, rather than an unexplained constant energy offset. It does not
establish convergence of forces, higher derivatives, k sampling or phonons.

The public `06_grid_convergence.py` was subsequently executed end to end and
reproduced these differences (last-digit roundoff aside), including the bounded
self-restart. Its record is `public-grid.log` and
[grid_convergence.json](results/grid_convergence.json). The
[grid plot](results/grid_convergence.png) shows why equal FFT dimensions must
not be mistaken for identical finite AO discretization.

## Reading the figures

[The pilot figure](results/graphite_pilot.png) separates the electronic-mode
comparison, Te=Tph spectra, fixed-electronic-reference Bose spectra and the local
potential. [PDF](results/graphite_pilot.pdf) is available for inspection/export.
The experimental Gamma markers are taken from Sec. III.A of
[arXiv:2610.18561v1](https://arxiv.org/html/2610.18561v1), not digitized or fitted.
They describe an unresolved measured peak, not four individually assigned modes.
The supplied 11 meV Gaussian and unit atomic form factors are explicit display
assumptions. No lifetime, multiple-scattering intensity or time trace is inferred.

The finite-q script uses a different, 6x6x1 electronic mesh to include K and M.
Its results must not be concatenated with the 3x3x1 Gamma scan as if they shared
one reference. In particular, the k6 Gamma optical energies around 220–225 meV
already differ strongly from the k3 results above. Raw unstable modes and this
sampling sensitivity must remain visible in any follow-up material study.

All six direct q samples completed successfully in `dispersion.log`; their
per-q times, including compilation, total **3496.63 s**. This excludes the
preceding SCF and does not measure warm-call scaling. The selected
[signed-energy table](results/dispersion_summary.json) retains every in-plane
mode. The [dispersion figure](results/dispersion_samples.png) connects only those
samples. In addition to Gamma, q=(1/6,0,0) has an unstable low mode in this pilot.
No absolute value or ASR projection removes it.

The [momentum-resolved map](results/momentum_spectrum.png) is generated by
example 07 from the completed modes, using a common logarithmic intensity scale
for 300 and 1377 K Bose populations and fixed 300 K electronic frequencies.
It explicitly limits the displayed mode set to energies above 100 meV; the
low-frequency instabilities in the separate dispersion prevent interpreting
this as a stable bulk thermodynamic prediction. No thermal redshift, intrinsic
linewidth or detector angular averaging is introduced by this visualization.

Examples 00–07 were executed on the remote CPU path. Minor comment/progress-text
updates followed those runs; numerical inputs and computations are retained.
Selected small JSON reference tables and PNG/PDF figures are versioned here;
the full arrays and logs remain in the remote validation directory.
