# Graphene two-point QP root diagnostic

Completed on 2026-10-18 on `jy_gpu_1`, Intel Xeon Platinum 8457C, JAX CPU
0.10.0 with float64. The wrapper returned exit code 0 after 167.472 s,
using CPU IDs 0--15, with peak monitored RSS 6,249,308 KiB. The imposed
limits were 600 s and 16 GiB. This was a two-point diagnostic, not a rerun
of the complete path or a BerkeleyGW calculation.

[result.json](result.json) records roots and fixed-G/W AD frequency slopes
for zero-based dense61 points 6 and 7, band 0. [budget.json](budget.json)
records the command, resource measurement and exit state.

The source/native SHA256 fingerprints were checked before computation.
The numerical configuration was PBE/GTH-SZV/GTH-PBE, graphene source mesh
`(2,2,1)`, FFT `(25,25,97)`, cell height 12 Angstrom, T=0, dimension 2,
`exxdiv=None`, `fc=False`, `nw=16`, `eta=0.001 Ha`,
`pole_subtraction=True`, `band_touching=True`, `gap_tol=1e-5 Ha`.
All available internal G states and OV+VO response channels were retained.
The linear solve used the shared direct solver (`rtol=1e-11`, `atol=1e-13`).
The original QP solve used secant (`ftol=xtol=1e-9`, maxiter=80), with the
driver's existing bounded hybrid recovery. Alternative seeds used the
shared hybrid solver at the same tolerances, maxiter=120.

The diagnostic invoked `gradscf.gw.pbc.g0w0_bands` and temporarily intercepted
the actual scalar equation inside its process. It reused that equation and
correlation evaluator immediately, before the driver advanced its closure.
The original solver result was returned unchanged. It tested the driver
seed, the neighbouring saved QP, the same saved QP and the KS energy;
`jax.jvp` evaluated the equation's frequency slope. Twenty-five frequency
samples are also retained. No self-energy formula was reconstructed in the
workflow, and no original path energies were replaced.

The two public energies reproduce the dense61 stored values within 5e-15 Ha.
Different seeds expose at least three distinct converged roots at each
point. The formal diagnostic `Z=1/dyson_slope_AD` assumes the implemented
real-frequency equation; it is not a certified finite-eta retarded spectral
residue. Stored `sigma_c_imag_hartree` must not be interpreted directly as a
linewidth. The tested roots are not a complete root enumeration and no
physical QP branch has been certified.

The ignored workflow is archived at
`artifacts/ad_showcase/material_gw_bands/berkeleygw_reference/graphene_pair_20261018/`.
Its remote job directory was
`jobs/berkeleygw_review/graphene_pair_20261018/` under the existing
`ad-showcase-query-lowdim-20261016` validation checkout. No executable code
is stored in this results directory.

See the [BerkeleyGW review](../../../../src/gradscf/gw/pbc/BERKELEYGW_REVIEW.md)
for the source comparison, assumptions and correction priorities.
