# Fourier interpolation of force constants

`Phonons.mesh(qmesh, method='primitive')` collects raw force-constant matrices
on a complete Gamma-centered rectangular mesh and returns a `ForceConstants`
object. Existing data can instead be passed to
`ForceConstants.from_qmesh(cell, qpoints, matrices, masses=masses)`.
The sampling must close on the reference electronic k mesh. The independent
supercell method retains its existing odd-replication restrictions; primitive
finite-q OFDFT is not implemented.

## Convention and transform

Lattice vectors are rows of A, positions are in Bohr, and q is in fractional
reciprocal coordinates. The native displacement convention is

\[
 u_{Rb}=u_b\exp(2\pi i q\cdot R),\qquad
 \Phi_{ab}(q)=\sum_R C_{ab}(R)\exp(2\pi i q\cdot R).
\]

Each atom pair has a 3-by-3 Cartesian block. Input matrices are **unweighted**
force constants in Eh/Bohr², rather than mass-weighted dynamical matrices.
For a mesh containing N points, the inverse is

\[
 \bar C_{ab}(r)=\frac{1}{N}\sum_q\Phi_{ab}(q)
                         \exp(-2\pi i q\cdot r).
\]

Consequently the implementation uses `fftn / N`. The finite mesh only fixes
the sum of coefficients at translations congruent modulo the mesh. For each
atom pair we select all shortest images of
`r @ A + tau[b] - tau[a]` in the mesh superlattice and split the coefficient
equally between tied images. This choice retains both directions at boundaries
of even meshes. A singular-value bound limits the search to a finite integer
box, including for oblique cells; severely ill-conditioned searches are rejected.

The atomic positions enter image selection, **not** the Fourier phase. Moving
an atom to another lattice representative therefore changes matrix phases;
the phonon eigenvalues remain invariant under the corresponding basis gauge.
The convention is unitarily equivalent to the position-phase dynamical matrix
used by many phonon packages, but matrices cannot be compared without applying
that gauge transformation.

Construction checks complete sampling, Hermiticity and raw q/-q conjugacy.
Raw invalid matrices cannot be repaired silently by the optional ASR projection.
On the input grid, the reconstructed matrices equal their inputs up to floating
point error, except for the explicitly requested Gamma projection.

## Acoustic sum rule and modes

By default no projection is applied. `asr=True` replaces only the sampled
Gamma matrix by P Phi(0) P, where

\[
 T_{a\alpha,\beta}=\delta_{\alpha\beta}/\sqrt{N_{\mathrm{atom}}},
 \qquad P=I-TT^\mathsf{T}.
\]

This is a Cartesian unweighted translation projection. The object records
`raw_asr_residual = ||Phi(0) (sqrt(Natom) T)||_F` and
`asr_correction_norm = ||P Phi(0) P - Phi(0)||_F`, both in Eh/Bohr².
Projection also affects interpolation away from the sampled points. It is not
a proof of FFT, k-mesh, geometry, or force convergence, and it does not impose
rotational sum rules.

`bands(qpoints)` reuses the existing shared Hermitian mode solver after dividing
each Cartesian index by the square root of its mass. Masses are supplied in amu
and converted to electron masses. Frequencies are signed hbar omega in Eh:
negative values represent imaginary modes, and are never clipped to zero.
Mode vectors are normalized in mass-weighted coordinates.

`dos(qpoints, energies, width=..., weights=...)` uses normalized Gaussian
broadening, with sigma `width` in Eh. It returns states/Eh/cell and integrates
to 3 Natom on a sufficiently wide and resolved energy grid. Supplied q weights
must be nonnegative and sum to one. Negative signed frequencies are retained.
The Gaussian sum is processed in chunks to avoid allocating one full
energy-grid-by-all-modes array.

## Differentiation and scope

`matrix(q)` supports batched queries, JIT and derivatives with respect to q.
Mesh reconstruction and shortest-image selection are eager preprocessing.
`bands` and `dos` are numerical postprocessing: they do not establish
differentiable eigenvectors within degenerate phonon subspaces, or a geometry
derivative through changes of the discrete shortest-image topology.

This is short-range interpolation. For polar crystals, long-range dipolar
subtraction/restoration, dielectric tensors, Born charges and the direction-
dependent nonanalytic correction require a separate implementation. Do not use
the uncorrected Gamma interpolation to claim LO-TO splitting.

## Validation and example

`tests/pbc/test_ifc.py` covers harmonic chains, diatomic gauge changes, oblique
images, even meshes, shuffled reciprocal representatives, invalid conjugacy,
ASR diagnostics, mass scaling, DOS normalization and JIT q derivatives. A
small first-principles HF test checks `Phonons.mesh` against a direct q solve.
The remote CPU regression at snapshot
`ebafa62a3dc0927fd6c3cca1c8c99056b105a7e151e809fdea9021e1f390f84f`
passed 13 tests with 4 deselected in 82.05 s (JAX 0.10.0, float64, `jy_gpu_1`,
`jax_scf`; `tests/pbc/test_ifc.py tests/pbc/test_phonon.py -k 'not finite_q'`).

`examples/dfpt/06_silicon_phonons.py` demonstrates per-q checkpoints, raw ASR
diagnostics, a band path and DOS for Si at a fixed geometry. Its GTH-SZV basis
and 2-by-2-by-2 k/q sampling are pilot settings, not a converged material
benchmark. The acceptance ledger in [ROADMAP.md](ROADMAP.md) tracks the remaining
material, long-range and response work separately.

The CPU pilot completed all eight q points with maximum response residual
4.08e-12. The broadened DOS integrates to six modes per primitive cell. Its
raw Gamma ASR residual is 5.26e-5 Eh/Bohr²; the projected Gamma optical modes
are 671.5532 cm^-1, and the L transverse pair has signed frequency -71.4779
cm^-1. These values describe the specified coarse calculation, not experimental
or converged silicon. Refining FFT 21³ to 31³ changes the raw Gamma matrix by
as much as 1.47e-4 Eh/Bohr². `07_silicon_pyscf_reference.py` independently checks
directional curvatures with re-converged PySCF displaced energies, using the
primitive cell at Gamma and a doubled cell at q=(0,0,1/2).

| Step (Bohr) | Gamma optical error | Boundary optical error | Boundary lowest-mode error |
| --- | --- | --- | --- |
| 0.002 | 9.5585e-6 | 1.1971e-5 | 1.6413e-5 |
| 0.001 | 3.0850e-6 | 4.0039e-6 | 5.8983e-6 |

All errors are in Eh/Bohr²; all six satisfy the preselected absolute gate 2e-5.
The doubled cell's undisplaced energy divided by two agrees with the native
primitive energy within 8e-14 Eh/cell. The native lowest curvature is
-0.005430210 Eh/Bohr², while the independent PySCF 0.001-Bohr difference is
-0.005424312 Eh/Bohr². The negative curvature is therefore reproduced without
Fourier interpolation. It does not establish a physical instability of bulk Si.
With the same SZV basis and k=2³, increasing FFT from 31³ to 41³ changes the L
transverse pair from -71.4779 to -71.5419 cm^-1, and the largest IFC entry change
is 1.81e-5 Eh/Bohr². Basis, k/q and geometry convergence remain separate gates.
GTH-DZVP at FFT31³/k2³ gives a lowest pair of -84.8419 cm^-1; expanding the
basis alone also does not remove it. Example 08 records these independent
sensitivity cases, and example 09 plots them alongside the PySCF negative-
curvature check.

## Compilation boundary

Primitive finite-q actions pass their grid and SCF reference arrays as JIT
runtime arguments. The source is validated before construction, and cell and
sampling metadata are copied into an independent snapshot. Later changes to
the original SCF object do not mix its new settings with the snapshot arrays.
The public displacement source still checks a live reference before assembly;
both paths reuse the same array implementation and shared response solver.

For a fixed column displacement v, the row contraction is real-linear in u:
`B(u) = u.T @ Phi(q) @ v`, with `B(0) = 0`. Thus its ith derivative at zero is
exactly `B(e_i)`. The implementation streams these unit-row evaluations with
`lax.map`, reusing the single electronic response for v. It avoids constructing
an internal reverse graph solely to recover coefficients of a linear function.
AO/geometry derivatives inside each contraction still use AD, and differentiation
of the resulting column is retained. Outer reverse differentiation can still
store intermediates across rows; this is not a universal low-memory backward
guarantee.

Bloch-grid generation, one-body matrices and nonlocal overlaps batch their k
axis rather than copying the same graph once per k point. This avoids embedding
large reference arrays in the compiled action and reduces trace growth. It does
not make all runtime workspaces small or establish large-k material scaling.
The initial Si k=4³ attempt was stopped
after reaching about 195 GiB RSS; an isolated JAX constant-handling experiment
also failed under a 64 GiB virtual-memory cap. No experimental compiler option
is required or enabled by the library. The remaining large-case and GPU gates
are tracked in ROADMAP.md.

On the Intel Xeon Platinum 8457C node with JAX 0.10.0, CPU float64 and
OMP/OpenBLAS thread counts set to two, the final streamed implementation gave:

| Single L-point case, FFT31³/k2³ | SCF + force constants, including compilation | Process peak RSS |
| --- | --- | --- |
| GTH-SZV, 8 AOs | 185.63 s | 15.14 GiB |
| GTH-DZVP, 28 AOs | 313.82 s | 30.97 GiB |

The complete SZV raw matrix differs from the original implementation by at most
9.38e-15 Eh/Bohr². The small-case cold time did not improve overall. These are
completed single-point measurements on a shared node, not warm-call scaling
fits. The k4³ run converged its SCF but was stopped after the RSS monitor crossed
its 32 GiB trigger (sampled peak 45.4 GiB); it has no force-constant result. A
five-second sampling interval can overshoot the trigger and is not a hard RSS
cap. Increasing the permitted memory is not evidence of an algorithmic fix.

### Isolating the k4 workspace

The 2026-10-14 profile of revision `8da6036` uses the same Si cell, PBE,
GTH-SZV/GTH-PBE, FFT31³ and q=(0,0,1/2), now with 64 k points. SCF tolerances
are 1e-11 Eh for energy, 1e-10 for density and 1e-9 for the orbital gradient.
On `jy_gpu_1` (CPU float64, JAX 0.10.0, two OMP/OpenBLAS threads), SCF took
31.49 s and the reference-force construction another 24.00 s. The latter
finished at 4.87 GiB RSS after a process high-water mark of 15.72 GiB.
The first action finished XLA compilation before the 32-GiB RSS monitor
terminated it. Sampling every 0.2 s reduced overshoot; the process still reached
32.84 GiB. This run produced no force-constant column or hot-call timing.

Separate compile-only probes used runtime reference arrays and did **not**
execute the large action. XLA reported the following temporary buffer estimates;
these are not measured process RSS or additive allocations between experiments:

| Compiled calculation | Temporary bytes | GiB |
| --- | ---: | ---: |
| Production full-column action | 42,838,947,312 | 39.90 |
| Exact translated-orbital reuse trial | 42,704,768,304 | 39.77 |
| Split trial: electronic response only | 24,582,540,720 | 22.89 |
| Split trial: curvature/response contraction only | 43,078,288,840 | 40.12 |
| One dynamic bilinear row, including electronic response | 32,208,131,184 | 30.00 |

The single-row probe supplies independent runtime row and column displacements;
it does not specialize to a fixed direction with zero derivative channels.
The full action has approximately 0.91 GiB of array arguments. Its optimized
HLO contains many AO/derivative/shifted tensors of shape `(64,4,29791,8)` and
complex128 dtype, each approximately 0.91 GiB. Logical tensor sizes alone do
not determine peak liveness, but the single-row result excludes row batching
as the sole cause. The next memory boundary must include k-resolved geometry
and grid contractions, not just separate JIT calls around the linear solve.

Translated-orbital reuse passed its equivalence/AD checks but saved only 0.3%
of the estimated workspace. Splitting electronic response from contraction
also failed to lower the dominant workspace. Both production-code trials were
reverted; their diagnostic scripts/patches remain under ignored artifacts.
No compiler flags, additional caches or alternate physics paths were retained.

Reproduction records are in the isolated remote snapshot
`/home/yjiao/gradscf_checks/pbc-roadmap-01de656-y0uDNwuQ`:
`phonon_cpu/memory_phases.log`,
`phonon_cpu/silicon_memory_compile_before_translation/`,
`phonon_cpu/memory_compile_translation.log`,
`phonon_cpu/silicon_split_memory/` and `phonon_cpu/silicon_row_memory/`.
The probes use `artifacts/roadmap/bounded_cpu_profile.sh` with a 600-second
timeout, a 128-GiB virtual-memory ceiling and a 32-GiB sampled RSS trigger.
The single-row upload archive SHA256 is
`5d9247d0e37f1fbab05ef224745d320cf5bc3d8b4247a28d607441bee829137c`.

### Streaming shared response contractions

The shared `_fields` contraction now accumulates the weighted density and
gradient fields one k point at a time. `_project` evaluates each k-resolved
potential matrix in the same way. Both keep the original complex bilinear
formulas and spin/channel ordering; the implementation changes temporary
storage rather than the response equation or its solver. The common dtype
of density, both AO legs and quadrature weights is applied to the small
density block before multiplication, preserving mixed-precision semantics.

These contractions need grid-sized scratch space for one k point instead of
the full k mesh. The AO inputs, shifted AO arrays and geometry derivatives
still span all k points. Reverse differentiation may also retain loop
intermediates, so this is not a guarantee of bounded total forward/backward
memory independent of the k mesh.

The final implementation reduces the full-column XLA temporary estimate
from 42,838,947,312 to 25,068,718,408 bytes: 39.90 to 23.35 GiB (41.5%).
The final implementation passes eight
focused CPU checks covering complex non-Hermitian density blocks, LDA/GGA
values, JVP/VJP, mixed precision and scratch-space growth, in 12.77 s. The two
memory gates failed on the original full-k implementation, and both
mixed-precision regressions were observed failing before correction.
Eight independent q-kernel/displacement/geometry checks for HF/PBE also pass
in 154.84 s. Three complete-action snapshot/JVP/VJP checks pass in 507.19 s.
All 19 checks ran remotely on `jy_gpu_1`; no local numerical tests were run.

The final source/test overlay SHA256 is
`3a00eeaf26a4389243c79d215e6fab6c1f21f9f7d913d1c247277a2124469f94`.
Focused, physical and complete-action AD logs are `contraction_verified.log`,
`contraction_physical.log` and `contraction_ad.log` in the same remote snapshot.
The final compile-only estimate is recorded in
`phonon_cpu/silicon_final_memory/compile.jsonl`. It is not an RSS measurement.

A separate execution diagnostic retained the 600-second time limit but used
a 48-GiB RSS trigger. One k4³ column completed with response residual 4.21e-13;
its hot repeat matched within 1e-12. The cold column took 249.39 s and the hot
repeat 178.93 s. The process eventually peaked at 32.71 GiB RSS, below the
diagnostic trigger but above the original 32-GiB target. It reached the time
limit during the second column (exit 124), so no complete force-constant matrix,
k4³ frequencies or k-convergence result was obtained. See
`phonon_cpu/stream_execution48.log` and
`phonon_cpu/silicon_stream_execution/k4.jsonl`.

The timings come from a shared CPU node while other validation processes were
running. They are resource diagnostics, not an isolated scaling fit. There
is no completed original k4³ hot call for a speedup comparison. Remaining
work includes measuring AO/geometry and projection costs and assessing k
block sizes without giving up the demonstrated memory bound and AD checks.

The test gates were: three Bloch-grid/value/JVP tests; 23 shared-integral,
geometry, HF/OFDFT and phonon tests after batching; then three final snapshot/
constant-capture/JVP-VJP tests and three final independent physical checks after
row streaming. All ran on `jy_gpu_1`; no local numerical tests were run. The
final two gates took 508.20 s and 246.28 s, respectively. An earlier combined
run reached its 600-second limit and was rerun as these separate completed gates.

For background on compile-time array handling, see the
[JAX closed-over constants documentation](https://docs.jax.dev/en/latest/internals/constants.html).
The runtime-argument boundary here uses ordinary JAX APIs, without relying on
the experimental constant-handling switch discussed there.
