# Full configuration interaction

`gradscf.fci` is an independent real, common-spatial-orbital FCI engine. It
solves complete alpha/beta occupation-string spaces for arbitrary integer
`(Nalpha,Nbeta)`, including odd electrons and fully polarized sectors. It does
not require a reference determinant or an excitation-rank cutoff.

```python
from gradscf import gto, dft, fci

mf = dft.RKS(gto.M(atom="H 0 0 0; H 0 0 .74", basis="sto-3g"), xc="hf").run()
result = fci.FCI(mf).run()
print(result.e_tot)                 # Total energy, Hartree
print(result.ci.shape)              # (n_alpha_strings, n_beta_strings)
dm1, dm2 = result.make_rdm12()
print(result.spin_square())         # <S^2>, effective multiplicity
```

## Integral interface for active-space methods

```python
solver = fci.FCISolver(nroots=2, gradient_mode="implicit_eigenvector")
energies, ci = solver.kernel(h1, eri, norb, (nalpha, nbeta), ecore=constant)
dm1, dm2 = solver.make_rdm12(ci[0], norb, (nalpha, nbeta))
tdm1, tdm2 = solver.trans_rdm12(ci[0], ci[1], norb, (nalpha, nbeta))
```

`kernel()` returns total energies, including `ecore`, rather than correlation
energies. One root returns a scalar and one CI matrix; multiple roots return
an energy array and an array `(nroots,n_alpha_strings,n_beta_strings)`.
Alpha strings are the first axis, beta strings the second. Each spin's bitstrings
are in increasing integer order, matching PySCF FCI vectors. `nelec` can be a
pair or a total electron count; the latter defaults to the lowest-|Ms| sector.
`spin` specifies `2*Ms`, not a projection onto a particular total S.

The eager facade offers a PySCF-shaped subset: `kernel`, `energy`,
`make_hdiag`, `absorb_h1e`, `contract_1e/2e`, RDM/transition-RDM methods and
`spin_square`. An optional `ci0` is a matrix or stack of matrices in the same
string ordering; dense solves accept but do not use it. These methods are
intended to be called by future CAS drivers, not as a drop-in implementation of
every optional PySCF FCI argument or addon.

`FCI(source, core=..., active=...)` accepts a converged GradSCF common-orbital
RKS/ROHF object or `FCIReference(h1,eri,nelec,ecore)`. `core` is a count or a
sequence of doubly occupied orbital indices. `active` preserves its supplied
orbital order; excluded non-core orbitals are empty. Core folding is also
available as `fold_core(h1,eri,core=...,active=...,ecore=...)`.
For SCF sources the core J/K and constant energy are evaluated in AO space;
**only active orbital columns are transformed into the four-index MO ERIs**.
Active-space determinant, link and workspace limits are checked before that
transformation. Returned RDMs describe the active electrons only. No SCF rerun
or orbital optimization is hidden in FCI; the constant includes the nuclear
repulsion and the folded doubly occupied core.

## Hamiltonian and density conventions

With real chemists' integrals `(pq|rs)` and spin-summed `E_pq`,

```text
H = sum_pq h[p,q] E_pq
    + 1/2 sum_pqrs eri[p,q,r,s] (E_pq E_rs - delta_qr E_ps)
```

The Hamiltonian acts through one-spin string maps and tensor contractions.
There is no full determinant-pair connection table and no spin-orbital
`(2*norb)^4` embedding. `contract_hamiltonian` takes the physical h1 and ERIs.
As in PySCF, **`contract_2e` instead contracts an E E tensor**; for a complete
Hamiltonian use `contract_2e(absorb_h1e(h1,eri,space,fac=.5), ci, space)`.
The default `fac` of `absorb_h1e` is 1.0. Integral inputs and its output
are full `(norb,norb,norb,norb)` tensors; PySCF's packed output must be
restored to this layout when comparing tensors. Do not interpret a raw `contract_2e`
call on bare ERIs as the physical two-electron Hamiltonian alone.

```text
dm1[p,q] = <q^+ p>
dm2[p,q,r,s] = <p^+ r^+ s q>
E = sum_pq h[p,q] dm1[q,p] + .5 sum_pqrs eri[p,q,r,s] dm2[p,q,r,s] + ecore
```

Diagonal RDM functions normalize their CI input. Zero/nonfinite states return
NaNs. Transition RDMs are bilinear in the provided bra and ket, without
normalization or sign matching. Spin-resolved diagonal 2-RDMs return
`(aa,ab,bb)`; transition 2-RDMs return `(aa,ab,ba,bb)`, matching PySCF.
RDMs are fermionic matrix elements, not permutation-symmetrized ERI gradients.
`spin_square` contracts spin-string actions without allocating the 2-RDM.
General fixed-Ms roots and arbitrary CI inputs need not have integer S.

## JAX and shared solver response

```python
from gradscf.solvers import EigenSolverConfig, EigenResponseConfig

space = fci.make_fci_space(norb, (nalpha, nbeta))  # Static; outside jit
cfg = EigenSolverConfig(method="davidson", nroots=1, max_subspace=40)
result = fci.solve_fci(h1, eri, space, ecore=constant, config=cfg,
                      response=EigenResponseConfig(target="eigenpairs"))
dm1, dm2 = fci.make_rdm12(result.coefficients[0], space)
```

All diagonalization and physical first-order response remain in
`gradscf.solvers.solve_hermitian`; FCI has no private Davidson or adjoint solver.
The default energy-only response supplies valid isolated-root energy
derivatives. Coefficient-dependent objectives require `target="eigenpairs"`
(or the facade's `gradient_mode="implicit_eigenvector"`); an energy-only
coefficient response is deliberately invalidated instead of silently returning
an incomplete RDM gradient. `ecore` participates in the energy derivative.

For a **complete isolated subspace**, use `EigenResponseConfig(target="subspace")`
and optional flattened CI-space `probes`. The result exposes `energy_sum` and
`projection`, with `total_energies=coefficients=None`. Internal degeneracy is
allowed here; numbered root/vector derivatives require isolated roots.
Forward roots/RDMs remain inspectable at degeneracy, but basis-dependent
individual-root derivatives are invalid. No full degenerate-state RDM-response
interface or higher-order derivative contract is claimed.

Supplied integrals must be real, finite and have their physical permutation
symmetries. Invalid inputs, failed convergence and unresolved derivative gaps
are reflected in the result's convergence/response flags and invalid derivatives.
Facades reject stale inputs and properties after failed reruns. Explicit functional
calls under JIT/AD assume fixed electron counts, orbital dimension and string
ordering; nuclear derivatives also require the orbital/integral backend response.

## Resources and current scope

Defaults: 1,000,000 determinants, 4,000,000 orbital-pair/string-map entries, and
8,000,000 elements per dominant contraction tensor. The workspace limit also
bounds active MO ERI and returned 2-RDM tensors. Davidson vectors are processed
in bounded column blocks and their contraction intermediates are rematerialized
in reverse mode. These are **array capacity checks, not a bound on total peak
RSS**: several buffers, numerical solver storage and compiler memory coexist.
Dense reference solves retain the shared `max_dense` guard (2048 by default).

FCI guesses emphasize the lowest diagonal determinants, with small deterministic
full-support perturbations plus additional random columns. This keeps the
search from being confined to a guessed spin or spatial invariant sector while
retaining useful reference determinants in larger product spaces.

This JAX reference implementation stores full active MO ERIs and orbital-pair
excitation tensors. Its speed/memory are not claimed to match an optimized C
FCI implementation. Separate alpha/beta orbital frames, complex/GHF/PBC FCI,
point-group/total-spin projection, selected CI, 3/4-RDMs, CASCI/CASSCF drivers and
state-averaged orbital optimization are not included in this increment.

Examples: [ground state](../../../examples/fci/ground_state.py),
[frozen-core active space](../../../examples/fci/active_space.py),
[model-parameter derivative](../../../examples/fci/response.py),
[H2 bond stretching](../../../examples/fci/bond_stretch.py).
See [validation](VALIDATION.md) and [references](REFERENCES.md).
