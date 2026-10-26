# Methane: G0W0, evGW0 and evGW followed by full BSE

Executed on 2026-09-30 using native GradSCF throughout, Apple M4 Pro,
macOS arm64, Python 3.12.2, JAX 0.8.1 CPU, float64. These are windowed
calculations in one basis. No experimental spectrum is fitted or reproduced.

## Reproduction

From the repository root with the native integral library built:

```sh
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
python examples/bse/methane_gw_spectrum.py
python examples/bse/plot_methane_gw_spectrum.py
```

The plain Python examples have no CLI. Generated working data, including
fixed-MO NPZ snapshots for optical-window checks, go to `outputs/methane_gw_bse`.
The selected JSON/CSV/PDF/PNG outputs are archived in this directory.

## Settings and scope

- Tetrahedral geometry: C at (0,0,0), H at (.629,.629,.629),
  (.629,-.629,-.629), (-.629,.629,-.629), (-.629,-.629,.629) Angstrom.
- Cartesian aug-cc-pVDZ, Weigend RI, 61 MOs and 5 doubly occupied MOs.
- Same native RHF reference: E = -40.199597689343 Ha, tolerance 1e-12.
- QP window: all 5 occupied plus lowest 29 virtual MOs (zero-based indices
  0 through 33), ending at a complete degenerate shell. G and W sums retain
  all 61 MOs. Higher virtual energies remain RHF values during evGW updates.
- G0W0 and evGW0 BSE use W0; evGW BSE uses its recorded updated W.
- Imaginary quadrature: 200 points; GW real-axis eta = 1e-5 Ha.
  evGW maximum 80 iterations, damping 0.3, Dyson tolerance 1e-8 Ha.
- Static full singlet BSE (not TDA), all 145 roots of the selected optical
  space; dense shared solver, tolerance 1e-9 Ha. No BSE root truncation
  within this window. All three calculations are stable and converged.
- Photon range 8-24 eV, spacing 0.005 eV. Retarded polarizability with
  resonant and antiresonant poles; spectral HWHM 0.15 eV. Cross sections
  are orientational averages in Mb (1e-18 cm^2), with no normalization.
- Stick plots sum oscillator strengths within 1e-5 eV degenerate groups.

## Measured results

| Method | First bright band / eV | Band summed f | Strongest bright band in 8-24 eV / eV | Max QP residual / Ha | Max BSE residual / Ha |
| --- | ---: | ---: | ---: | ---: | ---: |
| G0W0+BSE | 10.857565 | 0.34629912 | 16.503242 | 6.624e-11 | 1.113e-13 |
| evGW0+BSE | 10.833348 | 0.34571233 | 16.473598 | 5.165e-9 | 1.698e-13 |
| evGW+BSE | 10.808672 | 0.34283750 | 16.445142 | 3.228e-9 | 1.802e-13 |

Band energies are unbroadened eigenvalues, not sampled maxima of overlapping
broadened lines. First bright and strongest bands here are threefold degenerate.
The minimum selected QP Z is 0.886308 / 0.842398 / 0.835719, respectively.
The first bright band redshifts by 0.024217 eV (evGW0) or 0.048893 eV (evGW)
relative to G0W0 under these common settings.

Total oscillator strengths over the selected 145 roots are 6.293224,
6.280403 and 6.272524. Corresponding isotropic static polarizabilities,
computed as sum_s f_s/Omega_s^2 with Omega in Ha, are 14.974079,
14.998272 and 15.004495 a0^3. These are truncated-space diagnostics, not
claims of complete oscillator-strength sums or basis convergence.

RHF took 2.60 s; GW+BSE per-method times were 19.96, 41.88 and 46.33 s.
Tests overlapped part of this run, so these wall times are not an isolated
performance comparison.

## Convergence checks and limitations

The original all-61-MO evGW0 probe failed after 40 iterations (damping 0.5,
100 quadrature points), with final maximum Dyson residual 1.522 Ha. It is
not plotted. The successful results above must not be described as full-space
evGW. High-energy full-space G0W0 roots also showed low Z and branch-sensitive
splitting; the present work does not implement new global QP root selection.

G0W0's selected QP energies changed by at most 1.78e-15 Ha between 100 and
200 quadrature points. Separate evGW grid convergence was not measured.

A smaller optical window of 20 virtual MOs, with the SAME saved QP and W
spectra, gave first bright bands of 10.870636, 10.846433 and 10.821543 eV.
The largest change among the lowest 39 sorted roots (all below 18 eV in
the 29-virtual reference) was about 0.151 eV. Absolute excitation energies
are therefore not certified converged with respect to the optical window.
Method-relative shifts changed by at most 0.000169 eV (evGW0 minus G0W0)
and 0.002550 eV (evGW minus G0W0) in this check. This tests optical truncation,
not convergence of the QP update window, AO basis or auxiliary basis.

To repeat the optical check, load each generated `*_reference.npz` into
`bse.BSEReference`, supply nocc=5 and the full screening windows, then run
`bse.BSE(ref, occupied=range(5), virtual=range(5,25), tda=False,
solver='dense', nroots=100, max_dense=320, conv_tol=1e-9)`. Keep the saved
QP coverage masks; never mark uncomputed high virtual QP levels as converged.

## Numerical repair used by this run

BSE's functional entry point canonicalizes only contraction-sized asymmetry
in real MO factors, once, before static screening and both kernel blocks.
Larger asymmetry is left for rejection. The previous standalone screening
threshold experiment was withdrawn. Shared eigensolver thresholds and
stability checks were not relaxed; no solver implementation was duplicated.
The changed path has JIT forward, VJP/JVP, finite-difference and invalid-input
checks in `tests/bse/test_screening_roundoff.py`.

## Comparison with experiment

Run `python examples/bse/compare_methane_experiment.py` to reproduce
`methane_bse_experiment.png` / `.pdf` and `experimental_comparison.json`.
The script reads the archived spectra and checksum-verified experimental
files locally; it performs no electronic-structure calculation or download.
Raw experimental tables are not committed. First download the exact source
files and verify their SHA256 checksums as described in
[experimental sources](experiment/README.md), which also records primary
publications, archive provenance, temperatures and unit conversion.

The main reference is the Kameta et al. (2002) 298 K total absorption data,
obtained from the MPI-Mainz Atlas. It contains 3077 points over approximately
9.948-23.818 eV. Independent Samson et al. data agree with Kameta by a mean
absolute 0.48 Mb at 42 overlapping samples from 13.1 to 23.8 eV. These datasets
are plotted separately. The lower-energy Lee/Chiang curve is figure-digitized
and is not used in the quantitative comparison metrics.

**The present calculation does not reproduce the measured spectral shape.**
No energy shift, intensity scaling, broadening fit or experimental smoothing
was applied. The theoretical HWHM remains 0.15 eV. In the common interval,
the experimental maximum is 52.74 Mb at 13.499 eV, whereas the calculated
maximum is 527-529 Mb at 16.445-16.505 eV. These are envelope maxima, not
an assignment of the same individual excited state.

| Photon energy / eV | Experiment / Mb | G0W0+BSE / Mb | evGW0+BSE / Mb | evGW+BSE / Mb |
| --- | ---: | ---: | ---: | ---: |
| 10.2 | 18.95 | 4.18 | 4.47 | 4.76 |
| 13.5 | 52.71 | 16.24 | 17.99 | 20.24 |
| 16.5 | 44.94 | 528.89 | 512.76 | 466.35 |
| 21.2 | 31.22 | 7.13 | 8.10 | 9.41 |

Values are linearly interpolated within each sampled range. Cross-section
heights are broadening-sensitive; they are not oscillator strengths.
On a uniform energy grid from 10 to 23.8 eV, RMSE values (uniform integration over energy) are
65.87, 65.65 and 65.45 Mb. The small difference between these errors is not
evidence that one GW variant is decisively superior.

Integrated cross sections over 10-23.8 eV are 508.47 Mb eV experimentally
and 474.65, 474.45, 474.17 Mb eV theoretically (6.7% lower). In contrast,
over 10-12.5 eV the values are 64.95 versus 40.39, 40.36, 40.10 Mb eV,
about 38% lower. The near agreement in the larger-window integral hides
substantial redistribution into a few narrow calculated peaks.

The calculation uses one fixed tetrahedral geometry and finite Gaussian
virtual states, with an artificial common linewidth. It does not explicitly
include vibronic/nuclear-ensemble broadening, dissociation dynamics, or a
converged ionization continuum. Experimental ionization begins near 12.61 eV;
absolutely comparing narrow finite-basis lines above it to a smooth total
absorption continuum requires particular care. These missing ingredients
and the already measured orbital-window sensitivity are plausible contributors,
not a quantitative attribution of the discrepancy. The comparison alone does
not establish that changing linewidth, including nuclear motion, or selecting
a different GW iteration would resolve the mismatch.

Before claiming experimental accuracy, converge the optical/QP and diffuse
basis spaces, validate the fixed-geometry oscillator-strength distribution
against an independent calculation at the same settings, and then assess
nuclear motion and continuum representation. Do not tune a single linewidth
or rescale the curve to conceal this discrepancy.

## Independent MolGW calculation (2026-10-01)

All three methane GW/BSE chains were executed independently in MolGW.
With screening/optical inputs and RI retention aligned, the maximum excitation
energy difference is 6.36e-7 eV over all 145 roots. MolGW reproduces the same
strong narrow peak near 16.5 eV. Its default RI cutoff gives a small but real
~0.005 eV difference; that numerical setting is documented separately.
See [full comparison, external input adapter and provenance](molgw/README.md),
[plot](methane_molgw_comparison.png) and [numerical arrays](molgw_comparison.json).
The agreement validates this bounded implementation comparison, not agreement
with experiment or convergence of the physical model.
