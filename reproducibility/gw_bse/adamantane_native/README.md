# Native GradSCF adamantane electron–vibration calculation

This directory receives numerical results from
`examples/gw/adamantane_photoemission.py` and figures from
`examples/gw/adamantane_photoemission_plot.py`. Files are generated only after
the corresponding calculations and checks finish. The separate
`../adamantane_photoemission/` directory contains digitized literature curves,
not GradSCF calculations.

## Calculated results (2026-10-04)

| Quantity | Result |
| --- | --- |
| Optimized RHF energy | -383.515360723451 Ha |
| Maximum Cartesian gradient | 1.97e-7 Ha/Bohr |
| Positive internal modes | 72; 349.42–3724.70 cm^-1 |
| Hessian asymmetry / translation residual | 2.73e-12 / 4.28e-12 Ha/Bohr² |
| Converged QP levels | 66; maximum residual 1.73e-11 Ha |
| HOMO HF / G0W0 ionization energy | 9.76493 / 9.04255 eV |
| Displayed HOMO main peak, QP / Fan / Fan+DW | 9.045 / 8.220 / 8.535 eV |
| Experimental first-band maximum on the sampled grid | 9.455 eV |
| HOMO central 80% spectral-weight span, QP / Fan+DW | 0.06165 / 1.03842 eV |

The calculated HOMO main peak including Fan+DW is about **0.92 eV below**
the experimental first-band maximum. Thus this calculation demonstrates
spectral redistribution and satellites, but does **not** quantitatively
reproduce the experimental PES. The width quoted above is a percentile span
over 7–19 eV at eta=10 meV, not a lifetime or FWHM. Changing eta from 5 to
20 meV changes this span from 1.0234 to 1.0733 eV; the broad spectral-weight
distribution therefore persists as the individual poles become sharper.

The quadratic-vertex step check uses mode 45 (1650.75 cm^-1), selected by
the largest individual HOMO coupling norm. Reducing the Cartesian displacement
from 0.002 to 0.001 Bohr changes the maximum valence-block vertex element by
4.60e-6 eV and the HOMO block by 1.51e-6 eV. The complete DW matrix's largest
point-group covariance error is 4.46e-6 eV. These are numerical consistency
checks, not estimates of the method's error against experiment.

`results.json`, `comparison_metrics.json` and `dw_step_check.json` contain the
unrounded numbers. `invocation_wall_seconds` measures the last checkpointed
invocation only, not the full calculation. Per-direction response timings
are recorded separately; six native response directions were required.

## Model and units

The isolated neutral C10H16 molecule has 76 electrons. A restricted HF/STO-3G
calculation uses 66 real molecular orbitals. Its geometry is optimized within
the five totally symmetric cage coordinates, with the full Cartesian force
checked afterwards. All integrals, SCF calculations and responses use GradSCF
on the native CPU backend in float64. Coordinates are in Bohr and energies in
Hartree internally.

The analytic molecular Hessian includes both native second integral
derivatives and the response of the converged HF density. The response reuses
the shared implicit SCF residual and linear solver. It does not differentiate
canonical eigenvectors inside degenerate occupied or virtual spaces. Six
rigid translations/rotations are projected out of the mass-weighted Hessian;
all remaining 72 squared frequencies must be positive. Atomic masses are
12 u for carbon and 1.00782503223 u for hydrogen.

The optimized cage retains 24 verified Td signed-permutation operations.
Only six non-equivalent Cartesian directions are evaluated directly:
0, 12, 13, 30, 42 and 43 (zero-based flattened atom/axis indices). Coordinate
and AO/MO representations reconstruct the other columns. Geometry, basis
parameters, overlap, Fock, density and MO orthogonality are checked before
using a symmetry. The full Hessian must satisfy symmetry and translational
sum rules after reconstruction. This is an exact symmetry reduction; no
geometry is projected onto an approximate point group.

For mass-weighted eigenvectors e and frequency omega, the linear vertices are

    g[nu] = sum_Aalpha (d F_transport / d R_Aalpha)
                  * e[Aalpha,nu] / sqrt(2 M_A omega[nu]).

F_transport is the self-consistent HF Fock matrix transported by the polar
factor of the cross-geometry orbital overlap into the reference MO frame.
The local derivative of that polar factor at the reference is skew(dM),
which avoids differentiating the completely degenerate singular values of
M=I. These are HF-screened vertices, not derivatives of a GW self-energy.

Quadratic mode vertices use reconverged displacements of the same transported
Hamiltonian. The central-difference step is chosen so the largest Cartesian
motion is 0.002 Bohr. With the dimensionless phonon coordinate b+b†,
Lambda[nu] = (d²F/dQ_nu²)/(2 omega_nu) and
Sigma_DW = sum_nu Lambda[nu] (2 n_B + 1)/2.

The G0W0 calculation starts from the optimized RHF reference. It uses native
ERI spectral factors (eigenvalue cutoff 1e-9), contour deformation with 64
imaginary-frequency quadrature points, and the resolvent implementation.
QP equations are solved for all 66 molecular orbitals, including virtual and
core states. All electronic states remain in screening and self-energy sums.
The displayed photoemission trace contains the 28 valence orbitals, indices
10 through 37; it excludes the deep carbon-core photoemission bands.

The real-axis spectrum uses a unit-weight diagonal valence QP Hamiltonian and full
matrix Fan self-energy, with or without the static DW term:

    A(w) = -(G(w) - G(w)†)/(2 pi i)
    G(w) = [(w + i eta) I - H_QP - Sigma_Fan(w) - Sigma_DW]^-1.

Both Fan internal poles and external QP levels use the same complete G0W0
energy spectrum, in the reference frame of the HF vertices. This defines a
frozen-GW-correction model: H_eff(R) = F_HF,transport(R) + Delta_GW(R0), so
the response of Delta_GW itself is omitted. It is not a fully self-consistent
GW-plus-vibration Dyson solution. Temperature
is 300 K, and eta=0.01 eV is a common numerical resolution for all curves.
The CSV records the per-spin spatial-orbital trace per eV and the trace projected onto the whole
three-dimensional HOMO subspace. Traces assume equal photoemission
matrix-element weights. No lifetime is inferred from eta.

## Interpretation and experimental comparison

The spectral changes include shifts and satellite weight from the finite
molecule's discrete electron–vibration poles. A 50 meV Gaussian standard
deviation is applied equally to all calculated curves only for display;
raw values remain in spectra.csv. In the experiment comparison, curves are
area-normalized over 8.0–15.8 eV. No energy shift or linewidth fit is applied.

The experimental curve is the vector-extracted black trace in Fig. 1b of
Gali et al., *Nature Communications* **7**, 11327 (2016),
https://doi.org/10.1038/ncomms11327. That paper attributes the adamantane
measurement to W. Schmidt, *Tetrahedron* **29**, 2129 (1973).
The extraction procedure, original PDF hash and curve checksums are in
`../adamantane_photoemission/sources.json`. The source paper's colored spectra
also include Huang–Rhys broadening and a zero-point energy correction; this
GradSCF calculation does not.

STO-3G, HF-screened vertices, harmonic modes, fixed phonons, and the absence
of multimode Franck–Condon/Huang–Rhys contributions limit quantitative
comparison. Agreement between the analytic response and displaced SCF tests
validates the implementation, not this approximation's experimental accuracy.

## Reproduce

From the worktree containing the native second derivative implementation:

```sh
PYTHONPATH=src python -m gradscf.integrals._native.build
PYTHONPATH=src JAX_PLATFORMS=cpu python -u examples/gw/adamantane_photoemission.py
PYTHONPATH=src MPLCONFIGDIR=/private/tmp/gradscf-mpl-cache python examples/gw/adamantane_photoemission_plot.py
```

The calculation has no CLI. Set `GRADSCF_ADAMANTANE_CACHE` to choose a checkpoint
directory; the default is `/private/tmp/gradscf-adamantane-native/calculation`.
Intermediate full ERIs and compilation caches are not repository artifacts.
