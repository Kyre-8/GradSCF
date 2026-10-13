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

The displacement adapter uses JVPs of the existing supercell integrals and
fixed-density Fock builder for cosine and sine displacement patterns. It folds
their complex combination back to primitive k+q/k blocks. Local and nonlocal
GTH, moving Gaussians, Hartree and XC contributions follow the same AD path.

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

## Commensurate phonon reference

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

This first driver is intentionally bounded:

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

Finite-q phonons currently use the full supercell response space, including for
OFDFT. The typed primitive q sources provide a route toward a more scalable
implementation, but the driver does not claim primitive-cell scaling. Born
charges, macroscopic nonanalytic LO/TO corrections and general dispersion
interpolation are not supplied by this increment.

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
