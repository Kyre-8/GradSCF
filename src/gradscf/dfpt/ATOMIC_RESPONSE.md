# Atomic response and bounded reference phonons

`atomic_problem` connects periodic KS/HF and Cell-based OFDFT to the same
stationary-response framework. Atomic positions are the input PyTree. Both
the induced state and explicit integral changes contribute to its density and
force responses. `pbc.phonon.Phonons` assembles harmonic force constants from
these responses and uses the shared dense eigensolver for numerical modes.
No new SCF iteration or response solver is maintained in this adapter.

## Moving overlap and finite-q sources

For row k+q and column k, write the MO-space first variations as F1 and S1.
With L_nm=(f_n-f_m)/(epsilon_n-epsilon_m), the generalized matrix-function
derivative is

\[
D^{(1)}_{nm}=L_{nm}\left[F^{(1)}_{nm}
-\frac{\epsilon_n+\epsilon_m}{2}S^{(1)}_{nm}\right]
-\frac{f_n+f_m}{2}S^{(1)}_{nm}.
\]

This follows by differentiating the generalized stationary density equations.
For two filled orbitals, L=0 but D1=-S1; the occupied-occupied metric response
therefore cannot be dropped. At finite temperature the coincident-energy limit
is the same shared stable Fermi derivative used in electronic response.

`QSource(fock, overlap, rho_spin)` supplies three complex arrays:

- `fock`: the fixed-density Fock variation, shape `(2,nk,nao,nao)`. For an atomic
  displacement this includes hcore, two-electron and XC basis/grid variations;
  hcore alone is insufficient.
- `overlap`: the spin-independent metric variation, shape `(nk,nao,nao)`.
- `rho_spin`: direct real-space density-envelope variation at fixed AO density,
  shape `(2,ngrid)`. The final physical density response adds this basis term to
  the contribution from the coefficient-density response.

The potential-only shortcut still supplies zero metric and direct-density
variations. The general `Problem.inputs` for finite q is now a `QSource` rather
than a bare AO matrix; pass geometry sources via `response(tangent=source)`.

For a KS reference on a compatible mesh:

```python
phonons = Phonons(mf, masses=masses_in_amu)
source = phonons.displacement(q, displacements_in_bohr)
response = dfpt.DFPT(mf, q=q).response(tangent=source.perturbation)
dh = source.hcore
ds = source.perturbation.overlap
```

The default displacement adapter constructs primitive k+q/k blocks directly,
as described below. The explicit `method='supercell'` oracle uses JVPs of the
existing supercell integrals and fixed-density Fock builder for cosine and sine
patterns, then folds their combination back to those same blocks. Both include
local/nonlocal GTH, moving Gaussians, Hartree, exchange and XC contributions.

## Stationary forces and their response

For KS/HF, let D be the spin density and let the ground-state matrices satisfy
F C=S C epsilon. The energy-weighted density is

\[
W=C f\epsilon C^\dagger=D F S^{-1}.
\]

The implementation symmetrizes the last expression numerically. DFD would
introduce f squared and would be wrong for fractional occupations. At fixed
temperature and spin populations, the force is

\[
\mathcal F_I=-\left.\frac{\partial E}{\partial R_I}\right|_D
+\operatorname{Re}\sum_{sk}w_k\operatorname{Tr}
\left[W_{sk}\frac{\partial S_k}{\partial R_I}\right].
\]

Here E is the internal energy in the explicit derivative; stationarity of
E-TS cancels the occupation-response terms in the first force. Density and W
are held fixed only for this *inner partial derivative*. Their response is
retained when differentiating the force along the self-consistent solution.
Thus the force-constant action includes D1, W1, second overlap/integral
derivatives and the ionic Hessian. This is the energy-weighted-density Pulay
structure discussed in the [FHI-aims DFPT manual](https://www.fhi-aims.org/uploads/manual/Ch3/S37.html).

OFDFT reuses its fixed reference whitening chart y and number constraint N(y,R).
Forces are minus the partial coordinate gradient of E(y,R)-mu*N(y,R).
The KKT response supplies both y1 and mu1, including the Gaussian metric when
present. Grid and periodic-Gaussian amplitude representations are supported.

```python
problem = dfpt.atomic_problem(mf)  # Or a converged Cell-based OFDFT calculation.
response = problem.response(tangent=cartesian_displacements)
force_constant_action = -response.values['forces']
```

The lattice, FFT mesh, Ewald image topology, exponents and contractions are
fixed; all atom-attached shells move. Coordinates/displacements use Bohr, forces
Eh/Bohr, and force constants Eh/Bohr squared. First density response supplies
the second energy derivative, following the stationary DFPT construction of
[Gonze and Lee, Phys. Rev. B 55, 10355 (1997)](https://doi.org/10.1103/PhysRevB.55.10355).
Taylor series remain disabled for this adapter until the active integral and
response primitives have the required higher-order jet coverage.

## Primitive-cell displacement sources

`dfpt.displacement_source(mf, q, displacement)` constructs the fixed-density
Fock, overlap, direct density and core-Hamiltonian derivatives on the primitive
cell. It does not build a supercell or rerun SCF. Displacements are real
Cartesian amplitudes in Bohr; q is fractional reciprocal. Both integer and
finite-temperature HF/semilocal KS references are accepted. The full equal-weight
k mesh must be closed under q; the source itself also accepts shifted and even
meshes. This does not relax the phonon geometry driver's restrictions below.
The caller's reciprocal representative is preserved, matching `DFPT(mf,q=q)`:
q and q+G give the same lattice displacement but their periodic density
envelopes differ by exp(-iG.r). Neither source path silently wraps this envelope.

For AO Fourier columns A_k and their center derivatives B_k, the overlap source
has two moving-basis legs,

\[
 S^{(1)}_{k+q,k}=B_k^\dagger A_k/\Omega
                  +A_{k+q}^\dagger B_{k+q}/\Omega.
\]

The kinetic source has the same form with the kinetic-energy multiplier.
The local ionic potential is differentiated at G+q. Nonlocal GTH includes two
AO legs and two projector legs, sharing the ground-state projector overlaps.
On the grid, with lifted periodic Bloch factors a and their derivatives b,

\[
 \rho^{(1),\mathrm{basis}}_{s,q}(r)=\sum_k w_k
 \left[b_{k+q}D_{s,k}a_k^\dagger
       +a_{k+q}D_{s,k+q}b_k^\dagger\right].
\]

Hartree and semilocal XC include both this induced field and both moving AO
projection legs. Exchange differentiates all four AO factors and the Ewald
metric correction; its FFT pair-potential action is shared with ground-state
exchange. Reciprocal folding phases are retained. The resulting source is fed
directly to the existing finite-q response solver:

```python
source = dfpt.displacement_source(mf, q, displacement)
response = dfpt.DFPT(mf, q=q).response(tangent=source.perturbation)
```

`Phonons.displacement(...)` now uses this primitive construction by default;
`method='supercell'` explicitly selects the independent full-supercell AD oracle.
The source facade checks reference freshness eagerly and supports JIT/JVP of
displacement amplitudes. It is not an array-level training adapter for varying
the captured SCF reference. Arbitrary external functional finite-q extensions
and OFDFT primitive sources are not inferred.

Conjugacy at a Brillouin-zone boundary also requires FFT convergence: a finite
G window need not be closed under a reciprocal shift. For the shifted two-k H2
test, the core-source adjoint defect decreased from 5.35e-4 at mesh 15 cubed to
1.41e-6 at mesh 31 cubed, and passed the 1e-8 absolute comparison at mesh 41
cubed. Sources are not silently symmetrized to conceal this cutoff effect.

## Primitive force constants and the commensurate reference

The phase convention uses cell translations only: u_l=exp(iq.R_l)u. The driver
solves cosine/sine responses with the shared linear solver, then projects

\[
\Phi(q)u=-\frac1{N_c}\sum_l e^{-iq\cdot R_l}\mathcal F_l^{(1)}.
\]

The dynamical matrix is M^(-1/2)*Phi(q)*M^(-1/2), with masses converted from amu
to electron masses. Returned modes are normalized in mass-weighted coordinates.
`frequencies` contains signed hbar*omega in Eh: negative values denote imaginary
curvatures. These are not positive boson energies for direct use in EP spectral
models. The current mode facade is numerical; differentiation of individual
degenerate phonon eigenvectors is not asserted.

The raw force constants, Hermiticity error, maximum linear residual and (at
Gamma) raw acoustic-sum-rule residual are retained. `asr=True` explicitly
projects uniform Cartesian translations out of Phi before mass weighting, only
at Gamma. It does not modify the reported raw force constants or replace mesh
convergence. Unrelaxed geometries give local curvatures, not an equilibrium
phonon benchmark.

Both paths retain explicit atom/grid/workspace limits. The independent
supercell reference additionally has these restrictions:

- The KS supercell must match a full, equally weighted, Gamma-centered k mesh.
  A fresh supercell reference is checked against the folded primitive density.
- OFDFT first checks primitive stationarity under current inputs and parameters,
  then verifies that the supercell density remains the translated primitive
  solution. It does not silently select a different electronic branch.
- Only odd diagonal repetitions are accepted, preserving the current odd FFT
  mesh convention. Even repetitions and general supercell matrices are pending.
- Defaults limit atoms to 24, grid points to 20,000, and the AO-pair workspace
  estimate to 2,000,000 elements. Gamma checks also account for its k count.
  Explicitly increasing a limit requires assessing the resulting workspace.
- Arbitrary external model replication is not inferred. Named KS functionals
  and built-in OFDFT kinetic functionals use their existing domains; OFDFT still
  requires local pseudopotentials. Atomic Gamma response can use existing
  external-functional adapters.

`Phonons.kernel(q, method='supercell')` retains the full supercell response
space and supports HF/KS and OFDFT. It remains the default, independent oracle.
`method='primitive'` now constructs both HF/KS geometry curvature and electronic
response in the primitive cell. It does not construct a replicated Cell, build
supercell integrals, or rerun SCF. The earlier intermediate geometry-contraction
path has been removed. Gamma reuses `atomic_problem`; nonzero q uses the
existing `finite_q_problem` and shared linear solver. A closed, equally weighted
k mesh is required, including shifted/even meshes; no real-supercell-density
or odd-replication restriction is imposed on this path.

To derive the native contraction, differentiate the stationary force once more.
Holding D and W fixed in the inner geometry derivative defines

\[
 L^{\rm geom}_{uv}(q)=\partial_{u,-q}\partial_{v,+q}
 \left[E(D,R)-\sum_{sk}w_k\operatorname{Tr}(W_{sk}S_k(R))\right]_{D,W}.
\]

The remaining chain-rule terms are density and energy-weighted-density
responses. Thus the bilinear force constant is

\[
 u^T\Phi(q)v=L^{\rm geom}_{uv}(q)
 +\sum_{sk}w_k\operatorname{Tr}\left[
 F_{u,q}^\dagger D^{(1)}_{v,q}-S_{u,q}^\dagger W^{(1)}_{v,q}\right].
\]

F_u here is the fixed-D Fock source. There is no extra factor of two or real
projection: the row already represents -q and the result can be complex.
For the column v, first solve for D1 and form F1=F_source+K_q[D1]. Differentiating
W=D F S^-1 gives, with p=k+q,

\[
 W^{(1)}_{pk}=(D^{(1)}_{pk}F_k+D_pF^{(1)}_{pk}-W_pS^{(1)}_{pk})S_k^{-1}.
\]

The code averages this with the equivalent derivative of S^-1 F D to match
the shared symmetrized W convention. It never symmetrizes a q block by its own
adjoint, since the adjoint belongs to the -q sector. This also preserves
fractional Fermi occupations; replacing W by D F D would be incorrect.

The fixed-D,W term uses JVPs of the shared Fourier AO and GTH projector builders.
The first AO variations live at k+q and k-q, and their mixed variation returns
to k. The kinetic/overlap contractions retain all four product-rule terms.
Hartree and semilocal XC contain the second density field plus the product of
two first-order fields and the corresponding kernel. Exchange and nonlocal
GTH each differentiate four factors, including same-factor second derivatives
and every distinct-factor cross term. The Ewald exchange metric correction is
included. Ionic Ewald curvature uses the same real/reciprocal image topology
as the ground-state evaluator; wrapping an atom by lattice vector T also
transforms its +q amplitude by exp(-iq.T).

For each column v, AD differentiates the real and imaginary parts of this
bilinear expression with respect to the real row amplitude u at zero. This
produces the whole complex column without a dense coordinate Hessian or a
second electronic solve for each row. Only the final 3N by 3N force-constant
matrix is assembled. The grid and AO workspaces are primitive-cell sized;
exchange still sums over pairs of k points and the full AO basis. This is not
an end-to-end timing or large-system scaling claim.

Native finite-q OFDFT phonons, arbitrary external finite-q models, Born charges,
macroscopic nonanalytic LO/TO corrections and general dispersion interpolation
remain outside this increment.

The compiled finite-q independent response has a regression for complex MO
frames against a full generalized-eigenproblem finite difference, at zero and
finite temperature and with/without moving overlap. Local JAX 0.8.1 CPU
miscompiled constant gather/conjugate/transpose of a closed-over complex frame;
a runtime `optimization_barrier` avoids that constant-folding path. It is an AD
identity, not a `stop_gradient`, and relaxed-reference response-loss derivatives
remain covered by the existing tests.

## Checks and example

Focused regressions compare moving-metric response with generalized-eigenproblem
finite differences; atomic density/force response with displaced SCF; and
finite-q force constants with re-converged supercell energy curvatures. They
cover zero/finite-temperature HF and grid/Gaussian TF+vW OFDFT. Tests also check
nonlocal GTH first/second geometry derivatives, physical grid-density response,
mass scaling, unequal-mass ASR translations and rejection of stale references
or requests beyond the size limits. Named PBE has an optional runtime test.

`examples/dfpt/05_atomic_phonons.py` uses H2/GTH-SZV/GTH-Pade in a 6-Bohr cell,
a 3x1x1 k mesh and a 31-cubed FFT mesh. It explicitly raises the small default
workspace limits for its three-cell reference. Its geometry is not optimized.
The raw Gamma ASR residual was 0.1147 Eh/Bohr squared at mesh 15 cubed and
5.46e-4 at mesh 31 cubed: derivative agreement at a fixed mesh must not be
confused with converged acoustic frequencies. No silent zero-force correction
is applied.

Validation on 2026-10-13:

- Native geometry increment: all numerical tests run on `jy_gpu_1`, never
  locally. The first dedicated suite passed **11 tests** in 116.27 seconds:
  Gamma/finite-q Ewald including atoms outside the primitive cell, fixed-D,W
  curvature components for HF (with/without Ewald exchange correction) and PBE,
  carbon/hydrogen nonlocal GTH curvature, and relaxed force constants with a
  guard forbidding supercell construction. The nonlocal curvature error was
  4.23e-16 Eh/Bohr squared; the HF exchange component difference at mesh 15
  cubed was about 1.5e-7 Eh/Bohr squared. Finite FFT quadratures of primitive and
  replicated cells need not agree exactly before mesh convergence.
- The additional boundary batch passed **3 tests** in 156.94 seconds: restricted
  H2 with q/-q conjugacy, unrestricted H3 at kBT=0.1 Eh, and a shifted two-k
  mesh at its zone boundary. These also forbid supercell construction. All used
  CPU float64 in the same remote environment. The example was rerun there;
  native force-constant Hermiticity and linear residual were respectively
  5.63e-14 and 5.70e-16 in its reported atomic-unit conventions.
- The affected-module regression passed **34 tests, no skips**, in 715.87
  seconds on `jy_gpu_1` (`jax_scf`, JAX 0.10.0, jax-xc 0.0.9, CPU float64;
  OMP and OpenBLAS thread settings were 2). It covered displacement sources,
  periodic integrals and geometry derivatives, forces/stress, both phonon
  paths, and periodic OFDFT forces:

  ```sh
  PYTHONPATH=src JAX_PLATFORMS=cpu python -m pytest -q \
    tests/dfpt/test_displacement.py tests/pbc/test_integrals.py \
    tests/pbc/test_gradients.py tests/pbc/test_forces.py \
    tests/pbc/test_phonon.py tests/ofdft/test_periodic_forces.py
  ```
- The carbon/hydrogen nonlocal curvature was additionally tested through
  `jax.jit`: **1 passed**, 13.13 seconds, error 4.65e-16 Eh/Bohr squared.

Earlier stages (before native geometry replaced the intermediate path):

- Primitive-source/electronic-response increment: local CPU JAX 0.8.1,
  float64, **70 passed, 9 skipped**, 441.48 seconds, with
  `tests/dfpt tests/pbc/test_integrals.py tests/pbc/test_gradients.py tests/pbc/test_phonon.py`.
  Against the independent supercell path, the maximum force-constant errors
  were 1.35e-9 (zero-temperature HF), 3.15e-9 (HF at kBT=0.1 Eh), and
  1.13e-13 Eh/Bohr squared (handwritten PBE at kBT=0.1 Eh). The carbon/hydrogen
  nonlocal GTH source comparison gave 1.56e-14 Eh/Bohr. These are fixed-grid
  implementation comparisons, not timing/scaling or material benchmarks.
- After the q-representative boundary fix, the complete displacement test file
  was rerun: **8 passed, 2 skipped**, 89.43 seconds. It includes q+G density
  covariance and the unchanged HF/PBE force-constant comparisons.
- Isolated `jy_gpu_1` / `jax_scf`, CPU JAX 0.10.0, jax-xc 0.0.9,
  float64: the production-PBE displacement/phonon tests, complex-frame
  independent-response tests and production-PBE finite-q response/gradient
  tests gave **13 passed**, 530.62 seconds. PBE force constants from primitive
  and supercell electronic responses differed by 1.14e-13 Eh/Bohr squared.
  The response-loss gradient was 3.463457417934749, versus the independent
  finite difference 3.463457418084736. These are small-cell implementation
  tests, not a converged material spectrum or GPU performance benchmark.
  The final q-representative/input-validation test also passed separately in
  that environment (**1 passed**, 16.57 seconds).
- Local CPU JAX 0.8.1, float64: `tests/pbc tests/dfpt` plus the OFDFT
  representation/periodic-force regressions gave **133 passed, 34 skipped**
  in 660.08 seconds. Skips concern optional XC/reference dependencies.
- An isolated snapshot on `jy_gpu_1`, environment `jax_scf`, CPU JAX 0.10.0,
  jax-xc 0.0.9 and float64 passed the production PBE atomic-response regression
  (**1 passed**, 120.23 seconds). It checks stationary forces against energy AD,
  density response against displaced SCF, and force-constant action against
  displaced forces. Tolerances are respectively 2e-7, 3e-6 and 5e-6 in atomic
  units. This is not a separate production-PBE finite-q phonon benchmark.

The bounded phonon tests cover zero/finite-temperature HF and both TF+vW OFDFT
representations. They are small-cell correctness gates, not scale benchmarks
or validation of every built-in kinetic functional. Run with CPU float64:

```sh
PYTHONPATH=src JAX_PLATFORMS=cpu python -m pytest -q tests/dfpt/test_atomic.py tests/pbc/test_phonon.py
PYTHONPATH=src JAX_PLATFORMS=cpu python examples/dfpt/05_atomic_phonons.py
```
