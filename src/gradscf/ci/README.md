# Single-reference configuration interaction

`gradscf.ci` implements real molecular CI with restricted and unrestricted references.
UHF and ROHF support, integral conventions and differentiation limits are
described in [Open-shell post-HF](../cc/OPEN_SHELL.md).
PySCF is used in the tests as an independent reference, not by the solver.
Numerical eigensolves and their response rules are owned by
[`gradscf.solvers`](../solvers/README.md); CI constructs the Hamiltonian action
and maps the common solver results to electronic-structure quantities.
Method references and their connection to the implementation are recorded in
[REFERENCES.md](REFERENCES.md), with [BibTeX entries](references.bib).

## Methods and interfaces

| Interface | Space and energy convention |
| --- | --- |
| `CIS(mf, singlet=True/False)` | Spin-adapted singles; excitation energies relative to HF |
| `CIS_D(mf)` | Canonical RHF singlet CIS(D); corrected excitation energies |
| `CID(mf)` / `UCID(mf)` | Reference plus all double substitutions, excluding singles |
| `CISD(mf)` / `UCISD(mf)` | Reference plus all singles and doubles, fixed (N-alpha, N-beta) |
| `CISDT(mf)` / `UCISDT(mf)` | Reference plus all excitations through triples |
| `CISDTQ(mf)` / `UCISDTQ(mf)` | Reference plus all excitations through quadruples |
| `CI(mf, max_excitation=k)` / `UCI` | General rank-truncated space |
| `UCIS(mf)` | Spin-conserving UHF singles; no total-spin projection |

Restricted determinant spaces fix N-alpha = N-beta; unrestricted spaces permit
different electron counts and different orbital frames. They do **not** select total
spin S: higher roots can be singlets, triplets, or higher-spin states. Only CIS
on closed-shell references exposes explicit singlet/triplet adaptation. This differs from the
spin-adapted restricted CISD amplitudes in PySCF; raw CI coefficient arrays are
not interchangeable. Truncated CI is generally not size extensive; see the CI
review by [Sherrill and Schaefer (1999)](https://doi.org/10.1016/S0065-3276%2808%2960532-8).

Quadratic CI is a separate nonlinear approximation: restricted `QCISD` and
`QCISD(T)` live in [`gradscf.cc`](../cc/QCISD.md), where they reuse the common
nonlinear amplitude solver. They are not aliases of variational `CISD` or `CCSD`.

```python
from gradscf import gto, scf, ci

mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")
mf = scf.RHF(mol).run()
myci = ci.CISD(mf).run()  # equivalently mf.CISD().run()
print(myci.e_tot, myci.e_corr, myci.converged)

singles = ci.CIS(mf, nroots=1, singlet=True).run()
corrected = ci.CIS_D(mf, nroots=1).run()
print(singles.e, corrected.e, corrected.correction)
```

All energies are in Hartree. `CI.kernel()` returns `(e_corr, coefficients)`;
`CIS.kernel()` and `CIS_D.kernel()` return `(excitation_energies, amplitudes)`.
`.run()` returns the solver object. Scalar CI facade attributes are used for one
root; `CIResult` always keeps an explicit root axis, with coefficient columns of
shape `(ndeterminants, nroots)`. CIS amplitudes have shape `(nroots, nocc, nvir)`
in the active space and unit spatial norm. PySCF restricted TDA amplitudes have
half that squared norm; the legacy GradSCF CIS(D) adapter converts explicitly.

`frozen=2` freezes the first two doubly occupied spatial orbitals.
`frozen=[0, 1, 8]` freezes arbitrary occupied or virtual spatial orbitals.
Frozen electrons remain in all determinants: their core energy and interactions
with active electrons are retained rather than removed by array slicing.

`CI(..., excitation_ranks=(0,2))` and the same argument to the space builders
select the CID space directly; the default `None` retains the usual hierarchy.
Determinant CI objects expose `spin_square(root=0)`, returning `<S^2>` and an
effective multiplicity. UHF-backed objects use their actual alpha/beta orbital
overlap; explicit unrestricted MO inputs must provide `overlap_ab`.
This diagnoses spin without projecting it or changing the selected roots.
See [CID and spin conventions](CID_SPIN.md).

## Functional and differentiable API

Construct determinant topology outside `jit`, then close over it:

```python
import jax
from gradscf.ci import CIConfig, make_ci_space, solve_ci

space = make_ci_space(nmo=4, nocc=2, max_excitation=3)
config = CIConfig(conv_tol=1e-10)

@jax.jit
def energy(h1_mo, eri_mo):
    return solve_ci(h1_mo, eri_mo, space, config=config).total_energies[0]

dh = jax.grad(energy, argnums=0)  # callable; supply real MO integrals
```

MO orbitals must be orthonormal and occupied orbitals must precede virtual ones.
`eri[p,q,r,s]` uses chemists' notation `(pq|rs)`. The one-electron Hamiltonian
must be symmetric, and the real ERIs must obey their usual permutation
symmetries. The low-level kernels check shapes and dtypes; physical consistency
of supplied integrals and the reference is the caller's responsibility.

For explicit inputs to a facade use
`CIReference(h1_mo, eri_mo, nocc, nuclear_repulsion=..., mo_energy=...)`.
SCF-backed restricted non-DF `CIS` instead shares the AO single-excitation
action with TD response, avoiding a complete MO-ERI transformation. Native
direct references retain their J/K provider without constructing an AO-pair
matrix. `CIS(..., max_memory=1024.)` budgets this path in decimal MB; frozen
spaces, singlet/triplet selection and unit-norm CIS amplitudes are preserved.
DF, explicit-integral inputs, UCIS and CIS(D) retain their existing routes.
See [direct response and validation](../tddft/DIRECT_RESPONSE.md) for its
fixed-basis AD contract, cache format, device behavior and measured limits.
For AO inputs, `integrals.molecular.ao2mo.transform_integrals` supports full ERIs, an s4
AO-pair matrix, or density-fitting factors with shape `(naux, nao, nao)`.
The SCF facade adapter is eager and accepts converged GradSCF `scf.RHF` (or `RKS(xc="hf")`),
`UHF` and `ROHF`; the latter two use `UnrestrictedReference` and `make_uci_space`.
use functional kernels inside `jax.jit`, `jax.grad`, and `jax.jvp`.

The default `gradient_mode="eigenvalue_only"` stops the iterative Ritz vectors
and differentiates their Rayleigh energies. At convergence and for an isolated
root, dE = c.T (dH) c. Coefficient derivatives are disabled in this mode.
`gradient_mode="implicit_eigenvector"` additionally solves the constrained
eigenvector response using the shared Hermitian response implementation. This
mode is necessary for coefficient-dependent objectives, including CIS(D).
The general CI/CIS(D) method-level validation covers **first-order** derivatives
of converged, nondegenerate roots. Isolated singlet CIS has separate small-system
second-response checks described below. Differentiable root tracking through
degeneracies is not a supported contract.

`solve_cis(..., response=EigenResponseConfig(...))` accepts the existing shared
energy/eigenpair response policy. For a bounded small problem it can explicitly
select `LinearSolverConfig(method="direct", max_dense=..., rtol=...)`; the
default remains the legacy GMRES response. Explicit policy and nondefault
legacy `gradient_mode`, `adjoint_tol`, or `adjoint_max_cycle` controls cannot be
combined. The `CISResult` represents individual roots, so `target="subspace"`
is rejected rather than returning a partial subspace result. First energy
derivatives use stationarity and require no response linear solve; second
derivatives and first amplitude derivatives do require one, including its
capacity and true-residual checks.

The dedicated H2/Cartesian 6-31G* tests compare a fixed-MO CIS parameter Hessian
with independent PySCF matrices and dense eigenvalue differentiation, and a
relaxed native-integral geometry HVP with reconverged native/PySCF gradient
differences. These checks do not extend the general CI, UCIS or CIS(D)
higher-response contract. For a multiroot request, every response branch that
AD evaluates must converge: an unused root's failed response can still produce
NaNs in a composed derivative. The caller must use a suitable shared response
solver, without weakening the convergence guard or silently replacing failed
responses.

Unconverged roots remain available for diagnostics; `residual_norms` and
`converged` are per-root arrays. Their energy derivatives are NaN. Check all
requested roots before differentiation. CI AD is with respect to its numerical
inputs; complete nuclear gradients also require SCF orbital response and the
chosen integral backend's derivative support. No automatic backend fallback is
introduced here.

Degenerate molecular orbitals inside an isolated occupied or virtual subspace
do not by themselves invalidate a CISD energy gradient. The nuclear-response
path uses a smooth transported HF frame and differentiates the CI Hamiltonian;
it does not differentiate individual canonical orbitals. This differs from a
true degeneracy of the selected many-electron CI root, which retains the
individual-root restrictions above. Native N2 checks and their basis/size
limits are recorded in [VALIDATION.md](VALIDATION.md).

## CIS(D) theory and validation

The singlet correction implements the unscaled CIS(D) method of
[Head-Gordon, Rico, Oumi, and Lee (1994)](https://doi.org/10.1016/0009-2614%2894%2900070-0):

```text
delta omega = <CIS|V|U2 HF> + <CIS|V|T2 U1 HF> - E_MP2
```

`U2` uses the excitation-energy-shifted doubles denominators; `T2` uses MP2
denominators. The second term contains disconnected triples. This is not
variational CISD, and is not merely an external-doubles EN2 correction.
The formula and subtraction are described in the
[Q-Chem manual, equations 7.38–7.40](https://manual.q-chem.com/5.1/sect-excorr.html).
The implementation reuses `tddft.cisd.restricted_cisd_second_order_correction`.
The independent test constructs the doubles and disconnected triples explicitly
in PySCF's FCI determinant basis and verifies all three contributions together.

`cis_d_correction(eri_mo, mo_energy, singles, nocc=..., frozen=...)` exposes
the pure JAX correction. Supply canonical RHF orbital energies and singlet CIS
amplitudes. Its result contains original energies, corrections, corrected
energies, the minimum absolute denominator per root, and `valid` flags.
Denominators below `denominator_tol` (default 1e-10 Hartree) produce invalid/NaN
results; no shift or clipping changes the theory. Correction gradients from an
energy-only CIS result are NaN to avoid returning incomplete derivatives.
The `CIS_D` facade enables amplitude response automatically and checks that
the Fock matrix is diagonal and consistent with the supplied orbital energies.

## Scope and computational limits

- Variational solves use compact alpha/beta excitation-rank blocks. Increasing
  particle/hole strings, one-group removal/creation maps and tiled integral
  contractions replace stored determinant-pair Hamiltonian connections. The
  complete one-/two-body operator is projected into the retained space; an
  intermediate single-excitation projection is not introduced.
- Full MO ERIs still require O(nmo^4) storage. Restricted and unrestricted
  Hamiltonian actions retain spatial/spin integral blocks without constructing
  a full spin-orbital ERI. The coefficient vector and Davidson subspace remain
  combinatorial; this does not turn high-rank CI into a polynomial method.
- The default vector capacity is 1,000,000 determinants (configurable). Creator
  and spectator string contractions are tiled, and vector columns are processed
  in batches. The default Davidson basis capacity is
  `min(ndet, max(40, 4*(nroots+1)))`. Initial guesses retain exact reference/low
  diagonal unit vectors and add separate full-support random columns. The old
  Slater–Condon connection implementation remains a small
  independent oracle, with its 1,000,000-connection bound; dense verification
  remains limited to 2048 determinants. Density matrices still use their
  separate connection implementation and capacity guard.
- GHF, complex orbitals, periodic CI, and automatic
  spin selection for general CI roots are not implemented.
- Named `CISD(T)` / `CISDT(Q)` corrections are not exported. They need a specific
  perturbative definition and reference implementation; these names are not
  inferred from the coupled-cluster hierarchy.

## Reduced density matrices

Determinant CI objects expose `make_rdm1(root=0)`, `make_rdm2(root=0)` and
`make_rdm12(root=0)`. The functional forms accept `(coefficients, space)` and
work under JIT/AD. The coefficients are normalized internally; zero or nonfinite
vectors yield NaNs. No extra diagonalization or integral construction occurs.
The facade requires a converged root and unchanged reference/rank/frozen inputs.
CIS/TDA amplitudes and CIS(D) corrected results are not determinant vectors and
do not inherit a density interpretation through these methods.

Restricted densities are spin summed. Unrestricted 1/2-RDMs return `(a,b)` and
`(aa,ab,bb)`, in their respective MO frames, with frozen electrons included.
The convention is `dm1[p,q] = <a_q^+ a_p>` and
`dm2[p,q,r,s] = <a_p^+ a_r^+ a_s a_q>`. Contract restricted ERIs with `dm2/2`;
the unrestricted `(aa,ab,bb)` contraction factors are `(1/2,1,1/2)`.
These are full fermionic densities, not individually pair-symmetrized ERI
derivatives. Their trace and contraction are `Tr(dm1)=N` and
`sum_r dm2[p,q,r,r]=(N-1)*dm1[q,p]` in the real spin-summed representation.
Use `gradient_mode="implicit_eigenvector"` for integral derivatives of a density;
the default energy-only mode intentionally stops CI coefficient response.

The implementation stores static operator connections (up to 4,000,000 per
density rank) and full MO density arrays. It remains a small-system reference
implementation; transition densities, AO output and automatic total-spin selection
are future work. See [forward parity and remaining gaps](../cc/FORWARD_PARITY.md).

## Reproduce the checks

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=1 \
  python -m pytest -q tests/ci
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  python examples/ci/restricted_ci.py
```

Tests use H2 (0.74 Angstrom), asymmetric linear H4 (z = 0, 0.8, 1.9, 3.1
Angstrom), and LiH (1.6 Angstrom), with STO-3G, RHF, CPU, and float64. Energy
comparisons to PySCF use absolute tolerances of 2e-9 Hartree (1e-8 for the native
end-to-end H2 example); Hamiltonian elements and FCI limits use 2e-12 Hartree.
Directional finite differences use a 1e-4 parameter step with approximately
2e-7 derivative tolerance. These are validation tolerances, not error estimates
for the electronic-structure approximation.

For theoretical citations, numerical-method references, software attribution,
and the limits of these validations, see [REFERENCES.md](REFERENCES.md).


Full determinant spaces for active-space work are implemented separately in
[`gradscf.fci`](../fci/README.md), using alpha/beta string contractions and the
same shared numerical solvers. The CI module retains the reference/rank-truncated
hierarchy; its fermionic phase helper is shared with FCI.
