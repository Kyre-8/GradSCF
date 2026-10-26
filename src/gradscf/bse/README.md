# Static molecular GW-BSE

`gradscf.bse` implements static GW-BSE for finite molecules, real closed-shell
orbitals or real collinear unrestricted references, with integer occupations. Both TDA and full BSE have matrix-free
Davidson paths and bounded dense reference paths. Singlet
and triplet energies, transition moments and length-gauge oscillator strengths
are supported. First-order response of isolated roots includes QP energies,
screening energies, factors, amplitudes and supplied dipoles. The numerical
eigenproblem and screening linear solves belong to `gradscf.solvers`.

This implements P0/P1 and real full-BSE forward/amplitude response from
[the development plan](../gw/BSE_PLAN.md). The dense reference is described in
[FULL_BSE_PLAN.md](FULL_BSE_PLAN.md); the matrix-free metric-response extension
is described in [MATRIX_FREE_BSE.md](MATRIX_FREE_BSE.md).
Spin-flip, complex/periodic static references, general dynamical BSE kernels
and degenerate-cluster properties are not exposed by the static solver.
[Exciton–vibration response](ep_coupling/README.md) is a separate fixed-bath
effective-Hamiltonian layer: restricted TDA projection, neutral Fan/DW,
spectral matrices and dynamic absorption. Its full-matrix resolvent can be
differentiated at internal degeneracy without individual eigenvector AD;
the BSE-result adapter retains the static solver's amplitude-response guards.

## Public workflow

```python
from gradscf import bse, gto, scf, gw

mol = gto.M(atom="O 0 0 0; H 0 -.757 .587; H 0 .757 .587", basis="sto-3g")
mf = scf.RHF(mol, conv_tol=1e-12).run()
mygw = gw.GW(mf, nw=100).run()
response = bse.BSE(mygw, nroots=3, singlet=True).run()
print(response.e)                     # Hartree
print(response.oscillator_strength())  # dimensionless
```

`kernel()` returns `(energies, (X,Y))`; `run()` returns the object. Amplitudes
have shape `(nroots,nocc_window,nvir_window)`. TDA has `Y=0`; full BSE returns
both amplitudes. Request full BSE explicitly with
`bse.BSE(mygw, tda=False, solver="davidson", max_space=40).run()`.
Use `solver="dense", max_dense=256` for the independently validated reference.
The Davidson path does not materialize A or B and ignores the dense capacity
limit; it still obeys auxiliary/factor limits and its subspace capacity.
An explicit `BSEReference(qp_energy, screening_energy, mo_factors, nocc,
dipole_mo=...)` can replace the GW facade. These explicit inputs must describe
one common orthonormal orbital frame; they are not verified by running GW.

`occupied` and `virtual` select the optical window. `screening_occupied` and
`screening_virtual` independently select the screening window, defaulting to
all occupied/virtual orbitals. Omitting core transitions from the optical
window does not automatically remove their screening contributions.

For G0W0 and evGW0, GW-backed input uses the original MF spectrum for W0. No QP re-screening is
silently introduced. QP energies of every selected level must be covered by
both `qp_computed_mask` and `converged_mask`; the legacy GW result marks
unrequested MF-filled levels converged, so that flag alone is insufficient.
`GWResult.screening_energy` records the actual pole spectrum used in W by the
restricted/unrestricted CD drivers. A restricted or unrestricted result can be converted with
`BSEReference.from_gw_result(result, mo_factors=..., nocc=..., dipole_mo=...)`.
This preserves QP coverage/convergence masks and the recorded screening spectrum;
missing metadata is rejected. For converged evGW, that spectrum equals the final
QP spectrum, including any frozen unrequested MF levels. Factors and dipoles
must be in the returned orbital frame. The GW facade now accepts `method="g0w0"`, `"evgw"`, or `"evgw0"`. This supports fixed-frame evGW output as
BSE input. The low-level evGW drivers also offer opt-in
[outer implicit response](../gw/OUTER_RESPONSE.md); their resulting arrays can
be passed directly to `run_bse` without stopping this dependence. qsGW/scGW
results without this provenance still require a separately justified explicit
reference and are not automatically adapted.

The eager interface checks source fingerprints. SCF/GW inputs, GW settings,
explicit arrays or BSE window/configuration changes invalidate cached results.
Failed reruns invalidate property access and reset convergence status. Missing
dipoles are completed through GradSCF integrals without repeating SCF; completing
an identical lazy dipole payload does not invalidate GW. Changing dipoles requires
a new BSE snapshot, while leaving the underlying QP calculation reusable.

## Kernel and normalization

For metric-whitened real factors, `(pq|rs)=sum_P L[P,p,q] L[P,r,s]`,

```text
Pi_PQ(0) = -4 sum_ia L_Pia L_Qia / (eps_scr[a]-eps_scr[i])
epsilon = I-Pi(0)
W[pq,rs](0) = L[:,p,q]^T solve(epsilon,L[:,r,s])
A[ia,jb] = (eps_qp[a]-eps_qp[i]) delta_ij delta_ab
           + kappa (ia|jb) - W[ij,ab](0)
kappa = 2 (singlet), 0 (triplet)
```

This uses **full W**, not the contracted W-v used by GW self-energy routines.
The exchange-like term uses bare Coulomb integrals; the attractive direct term
uses W. QP and screening spectra remain separate arguments. Zero-frequency
screening is the gapped imaginary-axis limit; no optical broadening parameter
enters its denominator. The spectral broadening helper in `tools.spectra` can
be used after the discrete excitations have been calculated.

Vectors obey `sum_ia (X[ia]**2-Y[ia]**2)=1`, reducing to unit X norm in TDA.
For singlets, `mu = sqrt(2) sum_ia (X[ia]+Y[ia]) dipole[ia]` and `f=(2/3)*Omega*sum_xyz mu**2`
in atomic units. Triplet electric-dipole transitions from a singlet ground state
have zero strength without SOC. This differs from the half-norm restricted
TD-SCF amplitude convention; existing TD-SCF property routines are not called
with incompatible BSE amplitudes.

The kernel convention follows the [2020 BSE perspective](https://arxiv.org/abs/2006.09440)
and the inspected [QuAcK GW-BSE driver](https://github.com/pfloos/QuAcK/blob/2236bfcda24ff0971358f5107636b506dd201bb1/src/GW/RGW_phBSE.f90).
See [REFERENCES.md](REFERENCES.md) for attribution and the distinction between
formula definitions and executed software comparisons.

## Functional differentiation

```python
space = bse.make_bse_space(nmo, nocc, occupied=occ_indices, virtual=vir_indices)
screening_space = bse.make_bse_space(nmo, nocc)
config = bse.BSEConfig(nroots=3, gradient_mode="implicit_eigenvector")

result = bse.run_bse(qp_energy, screening_energy, mo_factors, space,
                     screening_space=screening_space, config=config)
strengths = bse.oscillator_strengths(result, dipole_mo, space)
```

Construct static topology/configuration outside JIT. Explicit functional inputs
with omitted masks mean user-supplied energies; composition with a GW function
should pass its actual QP coverage/convergence masks. A fixed-rank factor input
can be differentiated without invoking the eager host spectral ERI factorization.

Default `gradient_mode="implicit_eigenvector"` supplies complete first-order
isolated-vector response. The optional energy-only mode keeps forward optical
values but invalidates their derivatives instead of returning only the energy
or dipole part. Screening parameter response is included. Neglecting the
functional derivative of W in deriving the standard static BSE kernel does not
mean stopping dW/dtheta when differentiating this specified model.

An extra root guards the selected spectrum. `response_valid` requires converged
guard/requested roots, positive selected frequencies and resolved internal and
boundary gaps. Exact/near degeneracies keep inspectable forward results but
invalidate individual-root derivatives. TDA cluster projector observables are
a later interface; there is no implicit root-count expansion or gap broadening.
In TDA, `stable` denotes positive selected frequencies, not an independent
global stability certificate for the starting mean field. Negative TDA roots
are retained as diagnostics and rejected for optical property evaluation.

Full BSE checks `A-B` and `A+B`, with absolute stability tolerance 1e-10
Hartree. In dense mode, `stability_margins` reports their minimum eigenvalues
and a passing result has `stability_certified=True`. In Davidson mode these
are **lowest Ritz estimates**, with `stability_residual_norms`; positive,
converged estimates set `stable=True` but `stability_certified=False`. They
are numerical screening, not a rigorous global proof that no lower unstable
mode was missed. Inspect both fields when certification matters. TDA reports
no global certificate and `stability_margins=None`.

Dense full-BSE vectors use Cholesky-Hermitian reduction; Davidson uses a
structure-preserving projected Hamiltonian/metric pencil and a metric-constrained
implicit amplitude solve. Nonpositive/unresolved stability checks give NaN
physical outputs and false convergence/response status. No imaginary mode is
converted into a real excitation. `response_valid` certifies the primal root
and gap prerequisites; a subsequent failed adjoint solve still returns NaN.
A full-BSE request containing any unresolved internal/boundary degeneracy
invalidates derivatives for the **whole requested root set**, consistently for
JVP and VJP. A smaller isolated prefix can still be differentiated. Forward
values for stable degenerate roots remain available.

The numerical contract is first-order response in the supplied MO data and
GW+BSE composition on valid locally isolated branches. Eager facades use host
validation; use the functional drivers inside JAX transformations. Nuclear/basis
response also requires differentiable upstream inputs. This does not provide a
Matsubara-to-real-axis continuation or root tracking across GW pole crossings.

## Storage and solver boundaries

The default Davidson path does not form the transition-space A/B matrices or
four-index W. A/B actions screen contracted trial vectors in orbital blocks;
for wider trial blocks they instead screen a bare factor slab and contract it
in auxiliary blocks. Neither path caches full screened virtual-pair or
occupied-virtual factors. Reverse mode rematerializes these block intermediates.
The bare MO factor input remains resident.

Static screening defaults to the bounded dense/direct solve. Select a
matrix-free auxiliary dielectric explicitly with the shared linear config:

```python
from gradscf.solvers import LinearSolverConfig

response = bse.BSE(
    mygw, tda=False, nroots=3,
    screening_config=LinearSolverConfig(
        method="gmres", rtol=1e-11, atol=1e-13, restart=20, maxiter=100,
    ),
).run()
```

This applies `epsilon v = v + 4 L_ov ((L_ov.T v)/gaps)` without an
`naux x naux` array. GMRES uses the unpreconditioned physical residual metric. Multiple RHS columns are solved
sequentially inside the shared implicit solve, bounding Krylov storage by the
restart size rather than the number of RHS. True primal/transpose residuals
are checked; failed screening does not trigger a dense fallback. The screening
config is independent of the dense/Davidson excitation-solver choice. Its
`maxiter` counts restart cycles. `gw.solve_static_screening` exposes the
screening solution, residual norm and convergence/status for explicit callers;
`screening_valid` in BSEResult continues to describe input validity.

The direct method and all existing auxiliary/factor capacity limits remain
available. GMRES saves dielectric storage but introduces repeated inner solves;
its runtime and convergence depend on the screening spectrum and RHS. Neither
this path nor orbital blocking removes the full bare MO factor input or the
cost of the preceding GW calculation.

Defaults: `max_aux=1024`, `max_factor_elements=20_000_000`, `max_space=40`,
`block_size=16`. The first two are checked before the GW facade transforms AO
factors to MO factors. They are capacity controls, not measured peak-memory
bounds; screening, factors, workspaces and AD intermediates coexist. The legacy
GW calculation before this adapter has its own storage requirements.

`solver="dense"` is a bounded oracle with `max_dense=256`, checking capacity
before materializing A/B. Its stable RPA reference uses n-by-n matrices rather
than a doubled physical matrix. Full-BSE Davidson instead allocates paired
bases proportional to `ntrans*max_space` and projected matrices bounded by
`2*max_space`. Its metric adjoint uses checked diagonally preconditioned GMRES.
The capacity must fit the requested roots, an available guard root and two
correction slots (unless it already spans the full transition dimension).
Increase `max_space` or `max_cycle` if a requested or guard root fails; there
is no hidden dense fallback or automatic relaxation of tolerances. Davidson requires
space for the requested roots plus its guard root and uses deterministic
full-support guesses (seed 0 by default). Shared direct solves factor once per RHS block; shared GMRES solves RHS columns
sequentially, with checked implicit and transposed solves in both modes. True RI
versus spectral-ERI-factor benchmarks, GPU scaling and out-of-core bare factor
storage remain future work.

Run [water_tda.py](../../../examples/bse/water_tda.py) or
[water_full.py](../../../examples/bse/water_full.py) for native examples.
Executed checks and limitations are in [VALIDATION.md](VALIDATION.md).


## Frequency-dependent optical properties

After a successful calculation with dipoles:

```python
alpha = response.polarizability(omega, eta=.01)  # a0^3, (...,3,3)
sigma = response.absorption_cross_section(omega, eta=.01, unit='Mb')
sigma_z = response.absorption_cross_section(
    omega, eta=.01, polarization=(0., 0., 1.), unit='Mb')
alpha_static = response.polarizability()  # omega=eta=0
```

`omega` is a real scalar or 1D array in Hartree; `eta` is a Lorentzian
half-width in Hartree. Polarizability includes both signs of the poles:

```text
alpha_ab(w) = sum_s mu_sa mu_sb [1/(Omega_s-w-i eta) + 1/(Omega_s+w+i eta)]
sigma(w) = (4 pi w/c) Im Tr(alpha(w))/3
```

A polarization selects a normalized real direction instead of the average.
Cross sections use a0^2 by default (`unit='au'`); `Mb` denotes 1e-18 cm^2.
The low-level functions with the same names accept `(result,dipole_mo,space,omega)`
and retain the validated isolated-root JVP/VJP contract. Only computed roots
contribute: a truncated spectrum is not a complete polarizability or a TRK
sum-rule certificate. Triplet electric-dipole strengths remain zero without SOC.

Default screening windows are inherited from the GW snapshot; an explicitly
supplied BSE screening window intentionally selects a different screening model.
Optical windows never implicitly remove screening transitions. Mutable input
lists are copied into immutable reference-window tuples.

See [the native spectrum example](../../../examples/bse/molecular_spectrum.py),
[GW controls](../gw/README.md), and the MolGW numerical comparison in
[VALIDATION.md](VALIDATION.md). MolGW uses the same resonant/antiresonant optical
construction in [m_spectra.f90](https://github.com/molgw/molgw/blob/b831818d7a845c36f295d036dc8ceef59daf9991/src/m_spectra.f90).

## Unrestricted spin-conserving response

`BSE(ugw).run()` selects spin-conserving response (`singlet=None`) automatically.
Explicit `singlet=True/False` is reserved for restricted spin adaptation.
The same facade accepts `BSEReference(..., nocc=(nalpha, nbeta))`, with energies
shaped `(2,nmo)`, MO factors `(2,naux,nmo,nmo)` and dipoles `(2,3,nmo,nmo)`.
Both channels must use the **same AO auxiliary metric**; separately factorized
spin ERIs do not define the cross-spin Coulomb vertices.

The functional entry point remains `run_bse`; construct its static space with
`make_bse_space(nmo, (nalpha, nbeta))` and use `BSEConfig(singlet=None, ...)`.
Occupied/virtual windows are paired tuples, e.g. `occupied=((4,), (3,))`.
Amplitudes are `(X_alpha, X_beta)` and `(Y_alpha, Y_beta)`, each block shaped
`(nroots, nocc_window_spin, nvir_window_spin)`. Empty transition channels are
supported for explicit references, although UGW currently requires occupied
and virtual orbitals in each spin.

The common static dielectric is

```text
epsilon_PQ = delta_PQ + 2 sum_(sigma,ia) L^sigma_Pia L^sigma_Qia / gap^sigma_ia
A_ia(s),jb(t) = delta_st delta_ij delta_ab gap^QP_ia(s)
               + (ia_s|jb_t) - delta_st W(ij_s,ab_s)
B_ia(s),jb(t) = (ia_s|jb_t) - delta_st W(ib_s,aj_s)
```

Same-spin direct actions reuse the restricted factorized kernels. One joint
low-rank charge action supplies the intra- and inter-spin exchange terms.
The screening state is shared; neither full spin-orbital factors nor dense A/B
are built for Davidson. Dense/GMRES screening and dense/Davidson eigen-solves
retain their existing residual and first-order response contracts.

Transition moments sum the two spin amplitudes **without** an extra sqrt(2).
In the closed-shell limit, symmetric/antisymmetric spin combinations reproduce
the restricted singlet/triplet spectra and strengths. Unrestricted roots
conserve M_S and are not automatically eigenstates of S^2; this interface does
not claim spin-pure multiplicities or supply an excited-state S^2 diagnostic.

See [the OH example](../../../examples/bse/oh_unrestricted.py) and
`tests/bse/test_unrestricted.py` for the independent scalar-loop oracle,
closed-shell reduction, optical JVP/VJP, windows and QP provenance checks.
The spin-resolved static kernel follows Monino and Loos, JCTC **17**,
2852–2867 (2021), DOI: [10.1021/acs.jctc.1c00074](https://doi.org/10.1021/acs.jctc.1c00074).

## Response linear-solver controls

The forward eigenproblem and its differentiated state equations have separate
convergence checks. `BSEConfig.response_config` accepts the existing shared
`LinearSolverConfig`; the facade accepts the same keyword:

```python
from gradscf import bse
from gradscf.solvers import LinearSolverConfig

response = bse.BSE(
    mygw, tda=False, solver="dense", nroots=6,
    response_config=LinearSolverConfig(method="direct", rtol=1e-10, max_dense=256),
).run()
```

This configures state response for both TDA and full BSE. `screening_config`
continues to configure W, and `solver` continues to select the forward
eigensolver. Omitting `response_config` preserves the previous checked GMRES
settings. Its `max_dense` bounds the actual response equation: N for TDA or
the dense stable-RPA reduction, and 2N for the structured RPA metric equation,
where N is the optical excitation-space dimension. It is not multiplied by
the number of requested roots.

The same keyword is available on shared `solve_rpa` and `solve_stable_rpa`.
No BSE-local linear solver is introduced. Both primal and transpose solves
retain true-residual checks; failed solves still invalidate derivatives.
Changing this setting on a completed facade invalidates its cached optical
properties. A converged forward eigenpair and a valid isolated-root gap alone
do not certify that the subsequently requested response solve will converge.
