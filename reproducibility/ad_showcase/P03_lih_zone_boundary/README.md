# P03 — LiH self-inverse zone-boundary gate

**Native matrix and selected independent curvature gates passed.** The native FFT31 H x
response passed its residual and raw imaginary-component checks. Independent
two-cell energy and density folding also passed. The full native matrix passes
its raw checks and has four negative curvatures, retained explicitly.
Independent two-step H x gradient differences also pass. The separate,
predeclared transverse-direction check independently confirms a negative
curvature of this same finite model at both displacement steps.
This follows the completed
[Gamma gate](../P03_lih_gamma_phonons/README.md). It does not establish phonon
dispersion, interpolation convergence or general complex q/−q conjugacy.

The fixed finite model is PBE / GTH-Pade / Cartesian GTH-SZV, four explicit
electrons, a full k2³ mesh, and the P02 cubic lattice parameter
`a = 7.780032276700287 bohr`. The initial resource probe uses FFT31³. The
already converged Gamma FFT71 result is not assumed to bound finite-q errors
or memory. The existing primitive response and shared GMRES are reused;
no solver or production force formula is changed.

## Phase and normalization

The driver uses fractional reciprocal `q=(0,0,1/2)` and the convention

`Phi_ab(q) = sum_l C_ab(l) exp(2 pi i q.l)`.

The phase contains the integer lattice translation `l`, without an extra atomic
position phase. The public phonon driver wraps this q to `(0,0,-1/2)`.
Both are the same self-inverse point. Calling the wrapped q and −q twice
would not constitute an independent complex-conjugacy check. At this point
the raw force constants should be real and Hermitian; both defects are kept.

For an independent PySCF supercell, use `S=diag(1,1,2)`, row lattice `S A`,
four atoms, eight electrons and k2×2×1. Its FFT is `(m,m,2m)`, which preserves
the primitive real-space grid spacing and quadrature weight. Primitive k
points map to supercell fractional coordinates `k S.T`, modulo integers.
Each supercell k groups two primitive k points. The density transformation
has blocks `B_lk = exp(i R_l.k) I_AO / sqrt(2)` and
`D_super = B diag(D_primitive) B†`.

For the real alternating displacement `u_l=(-1)^l u`, write `g_l=dE_super/dR_l`
for the analytic nuclear gradient. Then

`Phi(q) u = (g'_0 - g'_1)/2`,

and `d²E_super/dt² /2 = u.T Phi(q) u`. There is no extra factor of two from
combining sine and cosine: the sine pattern vanishes at this self-inverse q.
The gradient-FD comparison uses this unweighted Cartesian force constant in
Eh/bohr². It does not compare a mass-weighted matrix to an unweighted one.

The doubled FFT z dimension is even (62 for the initial probe), while the
primitive dimensions are odd. Their Nyquist/Fourier support need not coincide
exactly when a finite G window is folded. If an energy/density folding gate
fails, first inspect actual Fourier support and AO phases; do not relax the
gate, symmetrize the result or apply ASR to hide the discrepancy.

## Predeclared checks and resource order

The initial native probe computes the full response column for a unit H x
displacement. The finite-q density chart has 288 real variables, compared
with 144 at Gamma, so it uses the existing GMRES (`rtol=1e-10`, `atol=1e-12`,
`restart=40`, `maxiter=100`). Explicit workspace limits are 30000 grid points
and 20000000 pair elements. The independent probe verifies the undisplaced
two-cell SCF, gradient and folding against the already completed independent
FFT31 primitive reference. No completed primitive endpoint is rerun.

| Quantity | Gate |
| --- | ---: |
| Maximum true linear residual | 1e-9 |
| Largest raw imaginary component | 1e-8 Eh/bohr² |
| Largest raw Hermiticity component, full matrix only | 1e-8 Eh/bohr² |
| Independent primitive vs supercell energy/2 | 1e-8 Eh/cell |
| Independent folded density relative difference | 1e-7 |
| Native vs independent primitive energy | 2e-6 Eh/cell |
| Native action vs independent gradient FD, each step | 5e-6 Eh/bohr² |
| Two-step independent FD change | 5e-6 Eh/bohr² |

The finite-difference displacements are `3e-4` and `1e-4 bohr`. Reference SCF
gates retain the Gamma requirements: a sampled gap above `1e-3 Eh`, orbital
gradient norm below `2e-8`, overlap minimum above
`1e-8 * max(1, overlap maximum)`, and electron count error below `1e-9`.
The original one-column probe could not establish full-matrix Hermiticity;
the completed six-column result below now supplies that check. No ASR
projection is used.

Each probe is a fresh CPU process on `jy_gpu_1` with its own 1800 s timeout;
native RSS is bounded by 32 GiB and PySCF by 8 GiB. Node memory is checked
before launch. The process watchdog retains actual exit status and peak RSS.
Only after these resource and folding probes will a full matrix or FFT71
calculation be scheduled. Times will include SCF and compilation; OpenMP/BLAS
thread settings do not limit all JAX threads.

- [Native response](../../../examples/pbc/lih_zone_boundary.py).
- [Independent folding and gradient FD](../../../examples/pbc/compare_lih_zone_boundary.py).

The frozen production source is the Gamma `0de7c6c920f84563529fe04238cd8c92b625c3e05665bfe2fcccc278e0077470`
family. Original Gamma snapshots and data are kept unchanged. A separate
finite-q snapshot records its own executed example hashes. Neither this probe
nor the preceding Gamma gate validates the current entire worktree.

## Measured probe status

The native H x column completed with exit 0 in `130.917 s`, at peak RSS
`8302656 KiB` (7.918 GiB). Its maximum true linear residual is `6.91348e-13`
and largest raw imaginary component `6.55269e-10 Eh/bohr²`, both below the
predeclared gates. The exact raw column and reciprocal-shift table are in
`pbe_szv_k2_fft31_qhalf_probe.json/.npz`; `native_probe31_resources.json`
records the watchdog result. A one-column pass does not establish full-matrix
Hermiticity or independent agreement.

The independent folding probe completed with exit 0 in `689.877 s`, at peak RSS
`874420 KiB` (0.834 GiB). Its energy error per primitive cell is
`1.77636e-15 Eh`, density folding relative error `1.84606e-11`, and native versus
independent primitive energy difference `5.37989e-8 Eh`. The supercell has
eight electrons, gap `0.1427264 Eh` and orbital-gradient norm `1.75432e-11`.
The actual FFT31 → FFT(31,31,62) folding passes at this finite model; this
does not make arbitrary finite-G supports exactly equivalent.

The native six-column job also exited 0, in `231.160 s` at peak RSS
`8656552 KiB` (8.256 GiB). Its maximum raw Hermiticity component is
`3.6430e-14 Eh/bohr²`, maximum imaginary component `6.5527e-10 Eh/bohr²`,
and maximum true linear residual `6.9139e-13`. Its H x column exactly matches
the earlier single-column probe. All original full-matrix checks pass.

Existing numerical mode postprocessing, without ASR, gives signed frequencies
approximately `[-438.550, -438.487, -103.768, -103.737, 596.123, 1313.085] cm⁻¹`.
These four negative curvatures are retained in `raw_mode_diagnostic.json`.
The scalar cubic minimum is not a proof of full structural stability, and the
small SZV basis has not been converged. These results do not establish a
physical instability of LiH. Do not change the lattice or project these modes
merely to eliminate the negative values.

Independently, four reconverged endpoints at the two declared steps checked
**only H x** under 8 GiB / 5400 s. This job exited 0 in `1470.297 s`, with
`872856 KiB` peak RSS (0.832 GiB). Maximum native/reference column errors are
`2.04604e-7` and `2.06883e-7 Eh/bohr²` at the larger and smaller steps. The
two FD columns differ by `2.27891e-9 Eh/bohr²`. All original `5e-6` gates pass.
The error remains near a code-to-code floor rather than decreasing as h²;
the independent FD step stability and native/reference discrepancy are distinct.
The completed independent primitive reference and supercell density were reused.

The [raw matrix and H x comparison figure](lih_zone_boundary.png), also in PDF,
shows the signed native modes and clearly separates full native validation
from the one independently checked column. The data table is
`zone_boundary_hx.csv`; `plot_validation.json` fingerprints the inputs.

The additional direction was fixed in advance: `u_Li=(0,0,0)` and
`u_H=(1,-1,0)/sqrt(2)`, with unit Cartesian norm. Its native Rayleigh value is
`u.T Phi(q) u = -0.007335242225150866 Eh/bohr²`. This is a specified real
transverse displacement, not a selected eigenvector from a nearly degenerate
negative pair. After the H x gates passed, four independent endpoints at the
same two step sizes were launched under 8 GiB / 5400 s. The same `5e-6 Eh/bohr²`
limit applies to both the action and scalar Rayleigh comparison. That run is
now complete and passed. The independent values are
`-0.007335447401768588` and `-0.007335449472465499 Eh/bohr²` at
`h=3e-4` and `1e-4 bohr`, respectively. The largest scalar discrepancy is
`2.07247e-7 Eh/bohr²`, and the maximum action-component discrepancy is
`1.46547e-7 Eh/bohr²`; both satisfy the unchanged `5e-6` gates. The two scalar
FD estimates differ by `2.07070e-9 Eh/bohr²`.

The resumed invocation completed in `865.672 s` with `876124 KiB` peak RSS,
under the existing 8 GiB / 5400 s bound. It reused the saved larger-step
positive endpoint and computed the three remaining endpoints; this runtime is
not the cost of recomputing all four endpoints or the prior folding reference.
The [completed independent result](pyscf_pbe_szv_k2_fft31_qhalf_full_rayleigh_fd.json),
its endpoint checkpoints and [resumed-job records](../../../artifacts/ad_showcase/provenance/P03_lih_zone_boundary/verification/jobs/pyscf-negative-rayleigh-resume)
retain the source and input hashes. The [signed-curvature figure](lih_negative_direction.png)
and PDF show the fixed direction and the two independent steps. This establishes
one negative direction of the specified SZV/k2³/FFT31³ model; basis, k and FFT
convergence of the material instability remains open.

The reference script marks the FD stage incomplete before starting an endpoint,
even when continuing a successful folding probe. The original folding JSON
remains unchanged; a distinct `_fd.json` records its parent fingerprint.
Every completed ± endpoint is saved atomically with its geometry, source and
density-state hashes. Four synthetic I/O regressions passed remotely: an
interrupted FD cannot retain completed status, the original probe is unchanged,
an already completed positive endpoint is reused, and an altered density cache
is rejected. These are state-machine checks, not fabricated physical results.
`source_manifest_fd.json` and `artifacts/ad_showcase/provenance/P03_lih_zone_boundary/verification/source_fd` retain the version with
these I/O changes; the original source manifest and probe versions remain
separate. No physical formula or acceptance threshold changed.
