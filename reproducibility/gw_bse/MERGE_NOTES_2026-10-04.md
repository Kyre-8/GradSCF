# GW/BSE response, native Hessians and molecular electron–vibration spectra

Target branch: `release/v1.0.0`. Base: `bdd4415`.
This update integrates the completed work from
`feat/unrestricted-bse-gw-response`. It does not change the package version or
create a release tag. Unrelated local Boys/NNAO changes are excluded.

## Changes

- **Unrestricted BSE:** real collinear spin-conserving TDA/full molecular BSE,
  common screening, spin-resolved amplitudes, optical quantities and isolated
  root response. Restricted singlet/triplet paths remain available.
- **GW outer response:** opt-in implicit differentiation of restricted and
  unrestricted evGW/evGW0, and restricted qsGW in its documented nondegenerate,
  complete-orbital window. QP and source-state validity checks remain explicit.
- **Fixed-phonon coupling:** molecular Fan/Debye–Waller self-energies share the
  same scGW update in forward and implicit response. Real-axis spectral
  functions, linewidth kernels and periodic q-weighted contractions are
  available independently. Phonon frequencies/vertices are external inputs;
  no self-consistent phonon feedback is claimed. With EP present, a total
  electron-plus-phonon energy is not reported as an electronic-only energy.
- **Native second derivatives:** shell-wise analytic coordinate Hessian JVP/VJP
  for overlap, kinetic, nuclear attraction, dipole and full ERI, Cartesian and
  spherical. Independent nuclei, basis centers and dipole-origin mixed terms
  are covered. Upstream libcint `hess.c` is pinned and byte-verified.
- **Molecular calculation workflow:** reuse the shared implicit RHF solver to
  form energy HVPs and transported Fock derivatives without differentiating
  arbitrary degenerate canonical orbitals. Verified s/p signed-permutation
  symmetry reduces the adamantane calculation from 78 to six direct Cartesian
  response directions, reconstructing and checking the full result.
- **Examples and data:** add native adamantane calculation/plot examples,
  fixed-phonon scGW, periodic Fan and GW implicit-response examples. Keep the
  literature-derived Cu(111) illustration and digitized adamantane curves
  separately labelled and attributed.

## Adamantane calculation

C10H16, native RHF/STO-3G, 66 MOs, CPU float64, 300 K. The optimized energy is
-383.515360723451 Ha with maximum gradient 1.97e-7 Ha/Bohr. All 72 internal
frequencies are positive (349.42–3724.70 cm^-1). Hessian asymmetry and the
translation residual are 2.73e-12 and 4.28e-12 Ha/Bohr².

All 66 G0W0 roots converge with maximum residual 1.73e-11 Ha. Both the Fan
internal propagator and external QP Hamiltonian use those energies; the nuclear
derivatives use screened HF vertices with a frozen GW correction. Linear
vertices and molecular Hessians are analytic. Quadratic vertices use central
finite displacements, with a separate half-step check.

The HOMO QP energy corresponds to 9.04255 eV ionization. At the stated plotting
resolution, the projected main peak moves from 9.045 eV to 8.535 eV with
Fan+DW; the digitized experimental first-band maximum is 9.455 eV. The model
therefore **does not quantitatively reproduce experiment**. The HOMO central
80% spectral-weight span increases from 0.06165 to 1.03842 eV at eta=10 meV;
this is a percentile interval over 7–19 eV, not an intrinsic lifetime/FWHM.
Its 5/20 meV resolution checks give 1.0234/1.0733 eV.

Figures, unrounded numbers and method boundaries are in
[`reproducibility/gw_bse/adamantane_native`](adamantane_native/README.md).
The experimental curve is attributed to Gali et al., *Nature Communications*
7, 11327 (2016), https://doi.org/10.1038/ncomms11327, Fig. 1b. No energy shift,
coupling fit, experimental linewidth fit or Huang–Rhys convolution is used.

## Validation and review

- Native CPU source build succeeds; vendor checksums and symbol/export tests
  are covered by the focused integral suite.
- Native integral and implicit SCF Hessian checks previously passed 112 tests.
  H2/3-21G curvature agrees with the PySCF analytic RHF Hessian to 2.18e-9
  Ha/Bohr².
- Molecular response/symmetry and EP spectral regressions pass 35 tests,
  including displaced self-consistent references, degenerate-frame covariance,
  response-failure handling and unsupported-input rejection.
- Pinned ElectronPhonon.jl original numerical functions were evaluated by a
  small standalone harness; this is not a full Julia-package/materials test.
- Independent pre-merge reviews found no blocking numerical/API issues. Two
  stale GW scope statements were corrected. Independent NumPy recomputation
  of six archived spectral points agrees within 2.91e-12 eV^-1.
- Final combined pre-merge regression: **296 passed, 35 warnings in 573.39 s**
  on macOS arm64, JAX 0.8.1, CPU float64. The warnings concern JAX's projection
  of complex cotangents back to real inputs; no tests failed or were skipped.
  This run covers the BSE API/TDA/full/matrix-free/unrestricted paths, molecular
  G0W0/evGW/qsGW/scGW and outer response, EP/Julia fixtures, native
  values/layouts/geometry/Hessians/packed paths, and molecular response/symmetry.

Linux native builds, GPU execution and a full repository test run are not
claimed. The native compiled library is not committed; rebuild after pulling:

```sh
PYTHONPATH=src python -m gradscf.integrals._native.build
```

Third coordinate derivatives, native exponent/contraction AD, packed-ERI/RI/
direct-JK geometry derivatives, spin-flip BSE, periodic BSE, a periodic EP
Dyson closure and self-consistent phonon feedback remain outside this update.
The adamantane spectrum is a minimal-basis harmonic frozen-GW model with
unit-weight QP poles and equal photoemission matrix-element weights.
