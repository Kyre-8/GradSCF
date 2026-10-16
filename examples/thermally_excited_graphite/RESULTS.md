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
The experimental Gamma markers are taken from Sec. III.1 of
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

## PBE baseline: unresolved peaks and FFT refinement

Examples 08–09 add a Gamma-only numerical refinement and a comparison of the
**synthesized unresolved optical peak** with the measured peak. All calculations
in this section retain PBE/GTH-PBE, GTH-SZV, k=(3,3,1), and the same fixed lattice.
These are PBE baselines, not r2SCAN calculations. No extra q points, fitted
electronic-frequency corrections, or experimental spectral curves were added.

The source values and their evidence status are stored in
[experiment_reference.json](experiment_reference.json). The paper reports
193.7 +/- 0.2 meV at room temperature and 185.2 +/- 0.3 meV at 1377 +/- 11 K
(III.1, Fig. 3). Mapping room temperature to 300 K is a comparison assumption.
Their difference is -8.5 meV; **0.361 meV is only the formal propagated error
assuming independent endpoint fit errors**. The paper does not specify their
covariance or confidence coverage. Its approximately 7 meV width increase
refers to the time-resolved data in Fig. 4f (IV Discussion), not the static
300-to-1377 K endpoints. No precise width endpoints or error bars were extracted.
The dispersions quoted around 197 meV at Gamma are secondary theory from the
paper's reference 28, not additional measured peak targets.

Example 09 uses `pbc.vibrational` with full Q equal to the first reciprocal
lattice vector, unit atomic form factors, and a Gaussian instrumental FWHM of
11 meV. The paper reports an 11 meV ZLP width; its assignment to a Gaussian
profile is our model assumption. The measured momentum interval of 0.26
Angstrom^-1 is not integrated. The saved table distinguishes raw optical
eigenvalues, the envelope maximum/FWHM, and a single-Voigt fitted center/FWHM.
The fit has free area, center, Gaussian sigma and Lorentzian gamma, uses uniform
residual weights and zero background, and supplies no model uncertainty.
Its Lorentzian parameter is a shape descriptor, not an intrinsic lifetime.

| FFT mesh | Fitted peak at 300 K (meV) | Fitted peak at 1377 K (meV) | Peak shift (meV) | Envelope FWHM, 300 / 1377 K (meV) |
| --- | ---: | ---: | ---: | ---: |
| 15x15x31, pilot | 177.675920 | 179.689772 | +2.013851 | 11.645485 / 11.330125 |
| 21x21x41 | 179.355316 | not calculated | not compared | 11.502559 / not calculated |
| 27x27x51 | 177.539785 | 179.771248 | +2.231463 | 11.438451 / 11.135579 |
| 33x33x63 | 177.459208 | 179.691779 | +2.232571 | 11.438723 / 11.135533 |

Between FFT27 and FFT33, the fitted centers change by -0.080577 meV at
300 K and -0.079469 meV at 1377 K; the temperature shift changes by
0.001107 meV. This bounds the observed sensitivity of these particular
quantities to this FFT refinement only. It does not establish basis, k-point,
or complete material convergence. The FFT33 model shifts upward by 2.232571
meV, opposite to the experimental -8.5 meV, leaving a shift mismatch of
10.732571 meV. Its absolute center errors are -16.240792 and -5.508221 meV.
Keeping the FFT33 electronic modes fixed at 300 K and changing only Bose
occupation shifts the fitted center by -0.004279 meV.

The FFT33 raw optical modes form two in-plane pairs at 175.134191 and
178.190563 meV at 300 K, and 178.438921 and 180.117403 meV at 1377 K.
The low in-plane signed energies still contain negative pairs at approximately
-23.441639 and -9.319860 meV, respectively. These remain in the JSON and are
not broadened, clipped, or removed by ASR. Consequently the positive-optical
spectrum is not evidence of a stable bulk reference. Thermal expansion,
bulk three-/four-phonon self-energies, lifetimes, and the experimental scattering
geometry remain absent.

A separate GTH-DZVP forward probe did **not** converge within 200 SCF cycles;
its largest overlap condition number was 6.83448e8 and smallest overlap
eigenvalue 1.02396e-8. No DZVP phonon comparison was run. This probe is an
unresolved numerical limitation, not evidence of basis convergence.

The final comparison ran on `jy_gpu_1`, CPU float64 with OMP/OpenBLAS thread
settings of two, in 13.33 s with exit code 0. It checked a single-Gaussian limit
(center/FWHM tolerances 1e-5 meV; fitted FWHM 1e-4 meV), compared 0.01 and
0.02 meV energy sampling within 1e-5 meV, and rejected significant tail
truncation or fits near the interval boundary. These are postprocessing checks,
not uncertainty estimates for the electronic calculation. Missing 1377 K data
for FFT21 is explicit and excluded from the temperature-shift comparison.

Selected results are the [comparison JSON](results/paper_comparison/comparison.json),
[CSV table](results/paper_comparison/comparison.csv),
[settings comparison](results/paper_comparison/settings_comparison.png)
([PDF](results/paper_comparison/settings_comparison.pdf)), and
[FFT33 detailed comparison](results/paper_comparison/szv_fft33_k3_comparison.png)
([PDF](results/paper_comparison/szv_fft33_k3_comparison.pdf)). Equivalent detailed
figures for the pilot, FFT21 and FFT27 are in the same directory. The JSON
records all input hashes, source hashes, settings, case completeness, and
fit diagnostics. Full normalized-shape CSVs and execution logs remain in the
ignored archive, with remote originals in
`/home/yjiao/gradscf_checks/graphite-paper-comparison-8509c60`.

## r2SCAN implementation and grid diagnostics

The periodic named-functional route now includes orbital kinetic density in
the SCF potential, finite-q kernel, displacement source and mixed geometry
curvature. The same SCF/response solvers are reused. The mathematical convention
and API boundaries are documented in [META_GGA.md](../../src/gradscf/pbc/META_GGA.md).

Example 10 uses the same SZV basis, k=(3,3,1), fixed lattice and **GTH-PBE**
pseudopotential as the refined PBE reference. The pseudopotential has not been
regenerated for r2SCAN. The following are raw optical mode energies, each
approximately doubly degenerate, not fitted experimental peak positions:

| XC and FFT mesh | Electronic temperature (K) | Optical pair energies (meV) | Total elapsed (s) |
| --- | ---: | --- | ---: |
| PBE, 33x33x63 | 300 | 175.134191, 178.190563 | 283.36 |
| r2SCAN, 33x33x63 | 300 | 174.795605, 207.074948 | 712.37 |
| r2SCAN, 33x33x63 | 1377 | 177.093742, 208.897770 | 638.75 |
| r2SCAN, 39x39x75 | 300 | 176.464612, 194.607590 | 1068.06 |

Times include SCF and compilation on the same remote CPU environment; they
are not isolated warm-call benchmarks. FFT39 was executed as a separately
archived settings-only variant of 10; the public example exposes that case at
its top. Changing FFT33 to FFT39 shifts the upper pair by -12.467358 meV,
although the free energies differ by only -5.14620e-6 Eh/cell. The apparent
proximity of the FFT39 upper pair to the measured 193.7 meV peak is therefore
**not evidence of convergence or agreement with experiment**.

Both r2SCAN meshes produce resolved multiple peaks with the assumed 11 meV
Gaussian broadening. Example 09 saves their [raw multipeak spectra](results/r2scan_comparison/r2scan_szv_fft33_k3_multipeak.png)
and [FFT39 spectrum](results/r2scan_comparison/r2scan_szv_fft39_k3_multipeak.png),
local peak positions and integrated mode weights. It does not force a single
Voigt fit: single-peak center, FWHM, temperature shift and experimental bias
are explicitly null for the excluded cases. Mode-resolved weights within an
exactly degenerate pair depend on its basis; summed pair weights and spectra
are invariant. No mode is selected just because its energy is near experiment.

### Rigid translation exposes the remaining numerical error

The continuum energy is invariant under translating every atom. A unit
Euclidean rigid-x displacement instead gives these raw curvature diagnostics:

| 300 K calculation | Curvature (Eh/Bohr^2) | Signed Rayleigh energy (meV) |
| --- | ---: | ---: |
| PBE FFT33 | 0.00120078522 | 6.372557 |
| r2SCAN FFT33 | 0.21540947254 | 85.351947 |
| r2SCAN FFT39 | 0.13553338998 | 67.702418 |

These energies are diagnostics of broken translation invariance, not physical
acoustic gaps. The saved mass-metric overlaps show that rigid translation mixes
with more than one low-frequency eigenspace. FFT33 also retains negative low
modes. Positive low eigenvalues on FFT39 do not resolve its translation error.

Example 11 independently reconverged the FFT33 r2SCAN free energy after rigid
translation. Central differences with steps 0.0004 and 0.0002 Bohr give
0.21196993050 and 0.21474626521 Eh/Bohr^2, respectively, approaching the DFPT
value 0.21540947254. The smaller-step difference remains about 0.3%; this is
not claimed as a tight derivative benchmark. Crucially, the independently
evaluated discretized energy itself has a nonzero translation cost.
This supports a grid-discretization problem rather than attributing the
entire effect to a spurious DFPT derivative. No acoustic sum rule or fitted
correction was used. See the [translation diagnostic figure](results/r2scan_comparison/rigid_translation_diagnostics.png)
and [independent displacement data](results/r2scan_comparison/r2scan_szv_fft33_k3_translation_check.json).

The final r2SCAN postprocessing completed in 14.27 s, exit code 0. Single/double/weak
secondary-peak checks, mode-area sums, stored-Phi rediagonalization and
mass-metric normalization passed. Figures were inspected. The
[complete comparison JSON](results/r2scan_comparison/comparison.json) records
the exclusions, inputs and hashes. This is an implementation and numerical
diagnostic, not a quantitative reproduction of the thermal EELS experiment.
The next material-level prerequisite is converged translation-invariant force
constants, followed by basis/k sampling and the missing anharmonic physics.

### Implementation verification

The r2SCAN validation uses Python 3.11.15, JAX/jaxlib 0.10.0, jax-xc 0.0.9,
PySCF 2.13.0, NumPy 2.4.4 and SciPy 1.17.1 on `jy_gpu_1` (hostname `c20`).
All numerical checks ran remotely on CPU in float64.

| Check | Observed result |
| --- | --- |
| Point XC/LibXC, potential/HVP, variational projection, Gamma and k SCF/bands, existing interface regressions | 32 passed, 232.27 s |
| Finite-q displacement and independently reconverged supercell density response | 2 passed, 165.51 s |
| Primitive mixed geometry curvature versus independent supercell AD | 1 passed, 369.42 s |
| Whole-loss JIT gradient through relaxed reference and finite-q response | 1 passed, 517.05 s |

There are **36 distinct repository tests** in these completed selections.
Repeated intermediate Gamma/basic checks are not counted again. The initial
three-test run failed before the implementation, and the unrestricted MGGA
feature-bundle guard also had an observed failing test before its repair.
The finite-q Fock-source maximum difference is 2.63e-13 Eh; the density-response
maximum difference is 9.18e-10 in the AO density matrix.

An additional diagnostic ran the relaxed-reference response-loss gradient with
the same implicit SCF and linear-response solvers, omitting only the outermost
`jax.jit` around the loss gradient. It passed in 1127.95 s:

    AD:                 2.9926929804712583
    central difference: 2.992692980541278

The finite-difference parameter step is 1e-5; the comparison uses atol=1e-7
and rtol=2e-5. This tests a derivative through the XC Hessian and therefore
involves local third XC derivatives. Internal JIT/staging remains active, and
the SCF derivative still uses the converged implicit residual.

The initial whole-loss JIT run reached its 1800 s compilation limit with exit
code 124 before returning a derivative. That timeout is retained separately
from numerical failures and from the successful diagnostic above. Persistent
CPU-cache diagnostic runs also emitted XLA AOT target-feature warnings; logs
are retained rather than represented as warning-free execution.

The final implementation precomputes each grid point's AD-generated 10-by-10
XC Hessian and reuses it in finite-q matrix applications. It preserves the
Hessian's reference-density dependence, the implicit solvers, convergence
tolerances and original test. The identical whole-loss JIT test then passed:

    optimized JIT AD:    2.9926929804712596
    central difference: 2.992692980496869

Its pytest duration was 517.05 s and cold-process wall time 530.29 s; one
logged gradient compilation took 162.43 s. Peak process RSS was approximately
20.93 GiB. These include reference/response solves and finite differences and
are not warm-gradient timings or an exclusive-host speed benchmark. The
final runs disabled persistent compilation caching. Obsolete original CPU/GPU
compilation comparisons were cancelled after this identical CPU test passed;
no complete GPU r2SCAN regression is claimed. A separate small-grid original
JIT diagnostic reached its 1200 s limit without a derivative.

For isolation, a scalar third-XC contraction on the actual 729-by-10 reference
fields also agreed with finite differences within 6.02e-15. Its lowering took
13.47 s, compilation 375.29 s, first execution 39 ms and one subsequent
execution 14 ms. Those timings concern only this contraction, not full DFPT.

Material calculations above predate the local-Hessian reuse optimization;
their exact executed scripts and numerical source snapshots are retained in
the local archive. The optimization changes finite-q kernel application and
shares the existing scalar XC formula; the final implementation was separately
verified by the 36 tests above. Reproducibility metadata are in
[implementation_validation.json](results/r2scan_comparison/implementation_validation.json).
