# Isolated adamantane: published photoemission benchmark

This directory is a literature illustration, **not a GradSCF calculation**.
All experimental and theoretical curves are vector-redrawn from the adamantane
panel of Figure 1(b), PDF page 3, in:

A. Gali et al., *Electron-vibration coupling induced renormalization in the
photoemission spectrum of diamondoids*, Nature Communications **7**, 11327
(2016), [DOI](https://doi.org/10.1038/ncomms11327),
[original open PDF](https://www.nature.com/articles/ncomms11327.pdf).
The source article is CC BY 4.0. It attributes this experimental PES to
W. Schmidt, Tetrahedron **29**, 2129–2134 (1973). Attribution is retained;
changes here are vector extraction, common normalization and new layout.

## What the curves mean

- Black: published experimental molecular PES, not original instrument data.
- Blue dashed: published single-shot GW QP levels convolved with static
  Huang–Rhys (HR) broadening. It **already contains vibrational effects**.
- Red: published dynamical electron–vibration spectral functions convolved
  with HR broadening. It includes more physics than a Fan-only linewidth.
- Gray ticks: published QP marker positions from panel (b). Tick heights
  are arbitrary and must not be interpreted as spectral weights.

The original common vertical scale is preserved and divided by the maximum
experimental intensity. No per-curve rescaling, new energy shift, parameter
fit, or additional convolution is applied. Original Figure 1(b) had already
used the authors' normalization and zero-point energy alignment; these are
inherited, not an independent prediction of absolute ionization energies.

## Why this matters for ep_coupling

The isolated-molecule spectrum is a multimode vibronic envelope with weight
redistribution and shoulders, not a single Lorentzian lifetime width as in
the Cu(111) demonstration. The source connects important HOMO coupling to
C–C modes near 124–162 meV, including a T2 mode at 159.8 meV, and discusses
Jahn–Teller effects and quasiparticle breakdown. It also uses HR broadening
for multiple vibrational excitations. Consequently, reproducing this figure
requires more than a uniform Gaussian width on the GW levels.

The supplementary tables list spectral peak contributions (`residual`), not
the complete complex electron–vibration coupling matrices. These must not
be substituted for g_lmn. No sufficient full matrix input was obtained here
to claim a GradSCF reproduction. A future calculation requires independently
constructed modes/vertices, the corresponding GW reference, and a separately
defined treatment of multiphonon/Franck–Condon contributions.

The main article describes zero-temperature electronic spectral functions;
do not assign an unverified temperature to the adamantane experimental curve
based on the separate room-temperature statement for diamantane.

## Reproduction and verification

```
python tests/comparisons/extract_adamantane_reference.py /path/to/ncomms11327.pdf reproducibility/gw_bse/adamantane_photoemission
MPLCONFIGDIR=/tmp/mpl-config python examples/gw/adamantane_literature_spectrum.py
```

`sources.json` records the PDF SHA256, page/panel, curve roles and CSV hashes.
The extractor requires the exact source hash. It follows PDF line segments
and samples original cubic Bezier paths, rather than fitting guessed data.
The source page and the rendered output PDF were visually inspected.
No new electronic-structure calculations or numerical solver tests are claimed.
