# Cu(111) electron–phonon photoemission broadening

This is a literature-input numerical demonstration using GradSCF's
`fan_linewidth` and `spectral_function`. It is not a new Cu slab GW, DFPT,
Wannier interpolation, or fully self-consistent GW+phonon calculation.
No linewidth, coupling constant, or baseline is fitted to experiment.

## Reference and extraction

A. Eiguren et al., “Role of Bulk and Surface Phonons in the Decay of Metal
Surface States”, Physical Review Letters 88, 066805 (2002),
[DOI](https://doi.org/10.1103/PhysRevLett.88.066805),
[arXiv v1 PDF](https://arxiv.org/pdf/cond-mat/0111029v1).

- Fig. 1(b), PDF page 5: solid Eliashberg alpha2F curve, not the dashed
  Rayleigh-only contribution. Extracted from original vector paths.
- Fig. 3, PDF page 7: seven Cu(111) intrinsic Lorentzian FWHM markers;
  only its single displayed 125 K error bar is retained.
- Fig. 3 inset: 85 experimental EDC circle markers at each of 55,160,285 K.
  Fitted/connecting curves are not used as experimental points.
- Table I: the calculated electronic linewidth baseline is 14 meV, stated
  in the source to be a lower bound. Its tabulated phonon width is 6.6 meV
  and lambda is 0.16.
- The experiment used He I, 21.23 eV, with approximately 3 meV energy and
  0.3-degree angular resolution (PDF page 3). Published FWHM points already
  represent fitted intrinsic widths, not raw peak widths.

The source PDF SHA256 and exact vector mappings are in `sources.json` and
`tests/comparisons/extract_cu111_reference.py`. CSVs are graph-derived
values, not original instrument files; their many stored digits do not
imply corresponding experimental precision. Absent error bars are unknown,
not zero. The rendered source figures were visually inspected before extraction.

## Computation

We discretize the digitized alpha2F in 480 phonon-energy midpoint bins over
0–30 meV and a locally flat electronic continuum with 801 points over
[-0.52,-0.36] eV, for an external hole at -0.44 eV relative to EF.
The rectangular zero-point-normalized vertices satisfy

```
g[l,m]^2 = alpha2F(Omega_l) * dOmega_l * dE_m
```

They are converted from eV to Ha before calling GradSCF. The on-shell Fan
integral uses a 1 meV Gaussian numerical delta. This numerical integration
width is distinct from the 3 meV instrumental Gaussian applied afterward
only to plotted spectra. The 960-mode/1601-electron check changes widths by
at most 0.000694 meV; this checks quadrature, not the accuracy of the
physical alpha2F input.

The local quasiparticle approximation uses Sigma=-i Gamma/2 and
Gamma=14 meV+Gamma_Fan(T). `spectral_function` supplies the Lorentzian
resolvent. A 3 meV FWHM Gaussian represents the reported energy resolution;
full angular convolution, experimental background, and thermal peak shifts
are not modeled. The EDC comparison centers each measured trace on its own
maximum and normalizes peak heights; this does not fit its width.

The extracted theoretical curve gives lambda=0.162906 and a T→0 phonon
width of 6.886550 meV, close but not identical to the printed table values
0.16 and 6.6 meV. We retain the curve-derived result without rescaling it.
This reconstruction difference should not be interpreted as a new physics result.

## Numerical comparison

| Temperature (K) | Experiment (meV) | Calculation (meV) | Calculation − experiment (meV) |
| ---: | ---: | ---: | ---: |
| 55.0 | 22.80 | 22.03 | -0.77 |
| 85.0 | 24.60 | 23.78 | -0.82 |
| 125.0 | 27.20 | 26.67 | -0.53 |
| 160.0 | 28.10 | 29.42 | +1.32 |
| 200.0 | 30.80 | 32.70 | +1.90 |
| 240.1 | 33.50 | 36.06 | +2.56 |
| 285.1 | 34.70 | 39.89 | +5.19 |

The seven-point RMSE is 2.403 meV. Low-temperature agreement is close;
the calculation overestimates high-temperature broadening. No fitted offset
or coupling rescaling was introduced to remove that difference. The original
paper also notes a discrepancy at its highest Cu temperature.

## Reproduction

```
python tests/comparisons/extract_cu111_reference.py /path/to/paper.pdf reproducibility/gw_bse/cu111_photoemission
PYTHONPATH=src JAX_PLATFORMS=cpu python examples/gw/cu111_photoemission.py
```

The extractor requires the recorded PDF hash and PyMuPDF 1.27.2.3 was used.
The numerical run used Python 3.12.2, JAX 0.8.1, CPU float64 on
Apple M4 Pro. A writable MPLCONFIGDIR can be supplied if the default cache
is inaccessible. PNG and vector PDF figures were inspected after rendering.
The example has no CLI and requires no PySCF calculation or external solver.
