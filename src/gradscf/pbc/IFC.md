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

| Displacement step (Bohr) | Gamma absolute error (Eh/Bohr²) | Boundary absolute error (Eh/Bohr²) |
| --- | --- | --- |
| 0.002 | 9.5589e-6 | 1.1972e-5 |
| 0.001 | 3.0850e-6 | 4.0075e-6 |

All four satisfy the preselected absolute gate 2e-5 Eh/Bohr². The doubled
cell's undisplaced energy divided by two agrees with the native primitive
energy to 7.2e-14 Eh/cell. These checks establish agreement at matched coarse
settings; they do not establish convergence of the negative phonon branches.
