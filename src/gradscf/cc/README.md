# Ground-state coupled cluster and quadratic CI

`gradscf.cc` provides real molecular CCS, CCD, CCSD, CC2, LCCD and LCCSD, plus
the conventional CCSD(T) and Urban CCSD+T(CCSD) corrections. Real restricted
`QCISD` and canonical `QCISD(T)` are also available, with their own equations
and triples weighting; see [QCISD.md](QCISD.md). Numerical iterations, DIIS and implicit
root/adjoint solves are owned by `gradscf.solvers`. The CC module defines the
Hamiltonian blocks, amplitude representation, residuals and energy contractions.

Real UHF/ROHF-based `UCCSD` and `UCCD` are also available. `CCSD(mf)` and
`CCD(mf)` select the equations from the reference. See
[Open-shell post-HF](OPEN_SHELL.md) for spin-block amplitudes, frozen orbitals,
implicit response and the current in-core limits. The remaining methods and
Urban triples and non-CCSD/CCD model examples below apply to restricted
closed-shell references. Canonical UHF CCSD(T), UCCSD/UCCD Lambda and spin-resolved
densities are available through the same functional and facade methods.

Method citations are in [REFERENCES.md](REFERENCES.md). Some contraction code
is adapted from PySCF 2.9.0 under Apache-2.0; see [NOTICE.md](NOTICE.md).
No PySCF runtime calls are made by the CC kernels.

## Public facade

```python
from gradscf import gto, scf, cc

mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")
mf = scf.RHF(mol).run()
mycc = cc.CCSD(mf).run()  # also mf.CCSD().run()
print(mycc.e_tot, mycc.e_corr, mycc.converged)
e_corr, t1, t2 = mycc.kernel()
l1, l2 = mycc.solve_lambda()
print(mycc.converged_lambda)
e_triples = mycc.ccsd_t()
print(mycc.e_tot + e_triples)

urban = mycc.triples(variant="ccsd+t(ccsd)")
print(mycc.e_tot + urban.energy, urban.min_abs_denominator)
dm1 = mycc.make_rdm1()  # spin-summed MO density, including frozen cores
dm2 = mycc.make_rdm2()  # chemists' order; includes the CC Lambda state
```

`QCISD(mf).run()` (also `mf.QCISD().run()`) uses the shared nonlinear engine with
the quadratic CI residual. Its `.qcisd_t()` adds the QCI triples correction.
`CCS`, `CCD`, `CC2`, `LCCD`, and `LCCSD` use the same constructor convention;
`RCCSD` explicitly requires a restricted reference. `CC(mf, method="cc2")` selects a model
explicitly. Unsupported names are rejected, not mapped to CCSD. The facade
accepts converged closed-shell GradSCF `scf.RHF` (or `RKS(xc="hf")`) or explicit
`CCReference(h1_mo, eri_mo, nocc, nuclear_repulsion=...)` data for restricted models.
CCSD/CCD additionally accept UHF/ROHF and explicit `UnrestrictedReference`.
GHF, complex orbitals and DFT references are not accepted.
Changing the reference, method or frozen space invalidates post-processing until
`kernel()` is run again. Reference-array mutations are detected by content hashes;
SCF-backed references also track the completed SCF state.

Energies are Hartree. External amplitudes are `t1[i,a]`, `t2[i,j,a,b]`, with
restricted pair symmetry `t2[i,j,a,b] = t2[j,i,b,a]`. The internal vector stores
independent lower-triangular `(ia,jb)` components with sqrt(2) off-diagonal
weights. It does not introduce redundant antisymmetric coordinates into the
implicit Jacobian. Initial t2 is projected onto that symmetry; absent initial
amplitudes use zero singles and MP2 doubles.
Restart amplitudes are promoted to the common dtype of the integrals and supplied
amplitudes before packing, so a float32 checkpoint can restart a float64 solve.

Frozen orbitals use PySCF-like conventions: `frozen=1` freezes the lowest occupied
spatial orbital; `frozen=[0, 5]` selects arbitrary occupied/virtual orbitals.
All occupied electrons, including frozen ones, contribute to the reference
Fock matrix and HF energy. Amplitudes live only in the active space. Freezing
all excitations returns the reference energy with empty/zero amplitudes.

## Functional JAX interface

```python
import jax
from gradscf.cc import CCConfig, run_cc, triples_correction

config = CCConfig(method="ccsd", conv_tol=1e-11, residual_tol=1e-10)

def total_energy(h1, eri):
    result = run_cc(h1, eri, nocc=2, config=config)
    return result.total_energy

energy_and_gradient = jax.jit(jax.value_and_grad(total_energy, argnums=(0, 1)))
```

MO integrals use chemists' notation `(pq|rs)` and an orthonormal orbital basis;
occupied orbitals must precede virtual orbitals. Orbital counts, frozen indices
and configuration are static. `integrals.molecular.ao2mo.transform_integrals` handles full
AO ERIs, s4 pair matrices and density-fitting factors. This implementation
materializes full MO integrals even for DF inputs; it is not a low-memory DF-CC
implementation. The reference facade adapter is eager; use the functional API
inside JIT and differentiation.

Results contain total/correlation/reference energies, t1/t2, infinity-norm
residual, energy change, iteration count, forward convergence, denominator
diagnostics and a numerical method identifier. Default convergence requires
both `residual_tol=1e-9` Hartree and `conv_tol=1e-10` Hartree. The residual norm
uses the independent, norm-preserving packed coordinates. Preconditioning,
damping and DIIS affect the trajectory but do not redefine the physical residual.
`level_shift` (default zero, Hartree) subtracts one shift from singles iteration
denominators and two from doubles denominators, equivalent to shifting virtual
levels in the preconditioner. Initial doubles use these shifted denominators.
`CCResult.min_abs_denominator` refers to this iteration preconditioner. The minimum
physical triples denominator is reported separately by `evaluate_triples`.
Shifts do not enter the converged CC or Lambda equations, and there is no
triples-denominator-shift option.

## Forward and backward equations

CCSD uses `R_mu = <Phi_mu| exp(-T) H exp(T) |Phi0> = 0`. CCS and CCD restrict
the cluster operator and residual projection to singles or doubles. LCCSD/LCCD
retain `R(0) + DR(0) T` and the corresponding linear energy; CC2 uses its own
approximate amplitude equations rather than a generic rank truncation.

The public residual driver converges the right amplitudes, then attaches a root
derivative. JVP solves `J dT = -dR`; energy VJP solves
`J.T lambda = -dE/dT`. The Jacobian is applied by JAX JVP/VJP, not materialized.
Forward iteration and the initial guess are stopped in implicit AD.

`solve_lambda(h1, eri, result, nocc=..., frozen=..., config=...)` exposes the
same model's adjoint equation. Its packed dual is converted to conventional
restricted amplitudes through the pairing
`2*l1*R1 + (2*l2 - l2.swap(a,b))*R2`. Forward and Lambda convergence are reported
separately. Pass the model's configuration when using a non-CCSD functional solve.

For CCSD(T), add `triples_correction(h1, eri, result, nocc=...)` to the returned
CCSD total energy **inside the differentiated function**. This retains the
CCSD amplitude response and the direct dependence of the correction; a CCSD-only
Lambda is not substituted for the derivative of the combined objective.
The default (T) path requires a canonical active Fock matrix and resolved denominators.
It streams symmetry-unique virtual triples and caps their count at 20,000 by
default. It does not allocate the full six-index triples amplitude tensor.

`evaluate_triples(...)` returns a `TriplesResult` containing the correction,
applied connected/singles components, minimum physical denominator, current
CCSD residual, canonicality error and a validity flag. The components sum to the
selected correction: Urban's variant has a zero applied singles component.
Even an empty triples space requires a matching valid state. The scalar
`triples_correction(...)` and `mycc.ccsd_t()` retain their default conventional-(T)
meaning; the functional scalar also accepts the explicit `variant` keyword.

`make_rdm1(h1, eri, result, nocc=..., config=...)` differentiates the stationary
CC Lagrangian with respect to a real symmetric one-electron perturbation.
It includes Lambda, restores frozen-core occupations and retains T/Lambda response
under an outer derivative. This density is orbital-unrelaxed and belongs to the
selected CC model, not CCSD(T). The facade solves Lambda automatically and rejects
a failed response. This is not a nuclear-gradient interface.

`make_rdm2` returns the real Hermitian part of the CCSD left/right two-particle
density, `dm2[p,q,r,s] = <a_p^+ a_r^+ a_s a_q>`, including frozen-core terms.
Restricted output is spin summed; UCC output is `(aa,ab,bb)`. Their two-electron
energy contraction weights are `1/2` (restricted) and `(1/2,1,1/2)` (unrestricted).
Its contractions preserve fermionic symmetries; this is not an eightfold
symmetrized ERI derivative. CCS/CCD use the corresponding restricted cluster
space. CC2/linearized/QCISD-model 2-RDMs and triples-corrected densities are not implemented.
The current 2-RDM path materializes dense spin-orbital intermediates.

For real UHF/ROHF `make_rdm1` and `make_rdm2` use spin-resolved MO frames;
`solve_lambda` returns `(la,lb)` and `(laa,lab,lbb)`. Canonical UHF `(T)` streams
distinct spin-orbital virtual triples and retains the converged-amplitude
response. For noncanonical or ordinary ROHF inputs, explicitly select
`mycc.ccsd_t(orbital_basis="semicanonical")`. This general-reference formula
includes `F_vo*T2` and uses the shared tensor-sum inverse, with implicit response
that remains defined at within-block orbital degeneracies. It is algebraically
equivalent to explicit semicanonicalization, without differentiating individual
orbital eigenvectors. The opt-in reference implementation stores six-index
moments with `max_triples_elements=2_000_000` per tensor; it is not streamed.
The additional `algorithm="streamed"` option uses bounded moment panels with
full Fock-block JVP/VJP through second order, controlled by `max_workspace_elements`.
`CorrelatedEnergy(mf, "ccsd(t)")` reuses the molecular HF response and this
correction for complete nuclear gradients and Hessian-vector products. The
streamed option explicitly rejects third-order AD; its remaining spin-block memory costs
are documented with the formulation.
Its Cartesian-spectrum singularity guard is conservative. See
[the formulation and limits](SEMICANONICAL.md).
See [the forward comparison and roadmap](FORWARD_PARITY.md).

The OpenMolcas `CCSDT` program name must not be confused with fully iterative
CCSDT. Definitions and the inspected revision are in [OPENMOLCAS.md](OPENMOLCAS.md).

Invalid denominators, nonconverged right states and failed adjoint solves are
not silently converted to valid zero derivatives. Inspect result convergence
and residuals; invalid implicit responses and low-level (T) corrections yield
NaNs, while the facade rejects a failed correction or noncanonical input to the
default canonical path explicitly.

## Scope and validation

This is an in-core reference implementation for small systems. Full MO integrals,
CC intermediates and DIIS histories can be large. GPU performance, large-basis
scaling and orbital optimization are not validated here. Native N2/STO-3G
CCSD(T) has separate complete-gradient and two-direction nuclear HVP checks;
these do not establish general large-system or unrestricted nuclear Hessians.
Response requires a converged nonsingular branch; root crossings and
strong-correlation failures are not regularized into a different method.

Tests use H2 (0.74 Angstrom), asymmetric H4 (z = 0, 0.8, 1.9, 3.1 Angstrom),
and H2O (H at y = +/-0.757, z = 0.587 Angstrom), with RHF/STO-3G, CPU and
float64. Tests cover random residuals, independent determinant BCH equations,
PySCF energies/amplitudes/Lambda/(T), frozen spaces, two-electron FCI, fragment
additivity, energy and amplitude response, DIIS-trajectory independence and
failure diagnostics. Typical energy tolerances are 2e-9 Hartree; residual
comparisons use 1e-11–1e-12 Hartree; finite-difference derivatives use a 1e-4
step and 3e-7 absolute tolerance. These test tolerances do not estimate the
physical approximation error.

Executed commands and measured results are recorded in [VALIDATION.md](VALIDATION.md).

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=1 \
  python -m pytest -q tests/cc tests/solvers/test_nonlinear.py
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  python examples/cc/restricted_ground.py
```

Further stages include CC3, CCSDT/CCSDTQ, higher perturbative/renormalized
corrections, other spin references, RDMs/properties and local/explicitly correlated
methods. They are not represented by placeholder APIs in this implementation.

## EOM-CCSD spectra

Real restricted EE singlet and IP/EA doublet sectors are available through
`EOMEE`, `EOMIP`, `EOMEA` and the functional `run_eom` interface. The first
version offers dense and bounded-subspace Davidson non-Hermitian solvers in `gradscf.solvers`,
with biorthogonal left/right states and isolated-energy first-order JVP/VJP.
See [equations, API, conventions and limits](eom/README.md).
