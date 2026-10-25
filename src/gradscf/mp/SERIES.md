# Residual-driven MP series

`MP(mf, order=p)` computes all corrections E2 through Ep from one physical
residual. The generic engine supports real restricted/unrestricted HF frames,
integer occupations and fixed frozen spaces. Canonical frames remain the
default; `orbital_basis="noncanonical"` retains full occupied/virtual Fock blocks.
It has no fixed order
ceiling, but it has explicit representation-dependent capacity limits.
It does not claim convergence of the series at lambda=1.

## Physical partition and gauge

Normal ordering with respect to the fixed HF determinant gives

$$
H(\lambda)=H_0+\lambda V_N,\qquad
H_0=E_\mathrm{HF}+F_N.
$$

In canonical orbitals, each retained determinant D has the Fock excitation gap

$$
\Omega_D=\sum_{\sigma p}\epsilon_p^\sigma
\left(n_{Dp}^\sigma-n_{0p}^\sigma\right).
$$

All occupied electrons, including frozen cores, contribute to the physical HF
Fock and reference energy. Frozen orbitals restrict excitations; they are not
removed from reference determinants. The default operator uses the shared CI
compact particle/hole string contractions on the retained space. The
canonical `algorithm='tensor'` contracts dense antisymmetric blocks; its
noncanonical variant uses the compact direct action with the same packed
output coordinates. See [TENSOR_BLOCKS.md](TENSOR_BLOCKS.md). No FCI product-space
enumeration, physical dense Hamiltonian or diagonalization is required.

Use intermediate normalization, c0(lambda)=1. Write the unknowns as
y=(c_exc, delta E), c=(1,c_exc), and W=H-H0. The residual is

$$
F_D=(\Omega_D-\delta E)c_D+\lambda(Wc)_D,\quad D\ne0,
\qquad F_E=\lambda(Wc)_0-\delta E.
$$

Here Q excludes the reference determinant. Its known zero-point solution is
y0=0. The zero-point Jacobian is the block operator diag(Q F_N Q,-1), reducing
to diag(Omega_exc,-1) in canonical coordinates. The reference must be isolated from all
retained excitations. Internal degeneracies among excited gaps are permitted.
The reported minimum denominator covers these excitation gaps, unlike the
optimized MP2 diagnostic that covers allowed doubles only.

## Taylor lifting

For normalized coefficients y(lambda)=sum lambda^n y_n, each step evaluates

$$
b_n=[\lambda^n]F\left(\sum_{j=1}^{n-1}\lambda^j y_j,\lambda\right),
\qquad J_0 y_n=-b_n.
$$

`jax.experimental.jet` extracts the source coefficient with
`factorial_scaled=False`. The known zero-point Jacobian applies the complete
normal-ordered Fock operator and a minus sign on the energy coordinate.
The same checked shared linear solver is used at every degree. Its diagonal
preconditioner is the exact inverse for a valid canonical reference. Both
primal and transpose solves retain the shared residual checks. No independent
iterative or adjoint solver lives in MP.

The linear nodes W Psi_j are computed once per state coefficient and shared
with later residual degrees and Rayleigh extraction. Only the remaining
polynomial/quotient graph is lifted by Taylor AD. This preserves all outer
integral and full-Fock derivatives. The linear F_N term has zero source at degree
n when the unknown nth state coefficient is zero; only its Jacobian action is
needed by that step. Thus no Hamiltonian action is traced through `jet`.
For the noncanonical path, the shared direct builder constructs W from the
normal-ordered quartic interaction and cross occupied/virtual Fock terms.
Same-block Fock kernels and the reference constant are omitted before tracing,
rather than generated in H and subtracted again. The complete occupied/virtual
Fock operator remains live in the residual Jacobian and Rayleigh functional.
See [GRAPH_REUSE.md](GRAPH_REUSE.md) for the
coefficient alignment, measured CPU speedups and scope.

The determinant residual is only a generic polynomial in state, energy and
coupling; no separate MP2, MP3 or higher-order correction formula is coded.
Returned wavefunction rows contain Phi0, Psi1, ... in `calculation.space`
coordinates: determinant order for `series`, packed quasiparticle blocks for
`tensor` (different ordering and phases; see the tensor coordinate convention).
Their reference components are (1,0,...). They are coefficients
under intermediate normalization, not individually normalized physical states.

## Complete rank limits and Wigner's rule

A two-body operator changes excitation rank by at most two. Thus the full kth
wavefunction correction has rank at most 2k, including disconnected pieces.
The engine retains every allowed spin-conserving determinant through 2k,
without dropping selected classes such as triples or quadruples.

For energy order p, default k=floor(p/2). Let

$$
\chi_k(\lambda)=\Phi_0+\sum_{j=1}^{k}\lambda^j\Psi_j.
$$

The Rayleigh quotient

$$
\mathcal E_k(\lambda)=
\frac{\langle\chi_k|H(\lambda)|\chi_k\rangle}
     {\langle\chi_k|\chi_k\rangle}
$$

agrees with the true eigenvalue through order 2k+1. The state error is
$O(lambda^(k+1))$; stationarity makes the energy error $O(lambda^(2k+2))$. Taylor
AD of this one functional therefore gives every requested energy coefficient.
In code, its lambda-independent E_HF offset is removed before Taylor lifting:
the quotient uses F_N + lambda W and E_HF is added once to the final total.
This exact energy-origin shift preserves every requested coefficient and its
physical parameter response while avoiding high-order cancellations of E_HF
times the state norm.
Increasing `wavefunction_order` above the default computes more complete state
coefficients and increases the rank/capacity requirement accordingly.

Each shared node uses exact rank support: `W Psi_j` has input rank at most
`2j` and output rank at most `2j+2`. The final node `W Psi_k` appears only in
the Rayleigh numerator at degree `k+1`. For energies through degree p its bra
can have degree at most `p-k-1`, hence rank at most `2(p-k-1)`. Projecting this
node onto those output ranks leaves every requested coefficient unchanged,
including its outer parameter derivatives. The projector is applied to the
whole `H-E_HF-F_N` action, and its transpose exchanges input/output projectors.
No coefficient magnitude enters this pruning.

| Energy request | Default complete state order | Maximum retained rank |
| --- | ---: | ---: |
| MP2 / MP3 | 1 | 2 |
| MP4 / MP5 | 2 | 4 |
| MP6 / MP7 | 3 | 6 |
| MP8 / MP9 | 4 | 8 |

Physical particle/hole counts can reduce the actual rank. At sufficiently high
order the retained space can coincide with the entire finite determinant space;
the algorithm still does not perform an FCI eigensolve. Wavefunctions in this
representation grow combinatorially. Direct actions avoid the complete
determinant-pair connection table and spin-orbital four-index integrals. Full
spatial MO integrals are still required. The canonical tensor backend still
has dense antisymmetric temporary blocks, within its explicit capacity budget.

## Public usage

```python
from gradscf import mp

calculation = mp.MP(mf, order=6, with_coefficients=True).run()
print(calculation.corrections)  # E2, E3, E4, E5, E6, Hartree
print(calculation.e_corr, calculation.e_tot)
print(calculation.wavefunction_coefficients.shape)  # (k+1, ndet)
print(calculation.space.ranks)
```

For JIT/AD, use `run_mp(h1,eri,nocc=...,config=MPConfig(order=6))`.
`algorithm='auto'` selects specialized R/U MP2 and restricted MP3 when available,
otherwise the generic series path. `algorithm='series'` forces the generic path
from MP2 onward. `algorithm='tensor'` uses the same Taylor engine with the
experimental block action, from MP2 onward. `algorithm='direct'` explicitly selects a specialized kernel
and rejects unsupported orders or coefficient requests. The existing MP2/MP3
facades remain convenient aliases; UHF MP3 uses the generic engine.
Noncanonical MP3 always uses the series engine. Explicit `algorithm='direct'`
with noncanonical MP3 is rejected rather than silently selecting another method.

`t2` always contains first-order amplitudes, and SS/OS energies decompose E2
only. `with_t2=False` omits amplitude extraction, not internal response states.
`with_coefficients=True` returns the complete state rows through k. Capacity
checks precede action construction and eager MO transformations. Unresolved
gaps or invalid references invalidate values and generic JVP/VJP with NaNs.

In noncanonical mode, the occupied--virtual Fock blocks must vanish within
`canonical_tol`; off-diagonal occupied/occupied and virtual/virtual entries
remain part of H0. Frozen/active blocks must be decoupled and spectrally
separated. This excludes freezing part of an exactly degenerate manifold.
The array derivative contract keeps the selected frozen subspaces fixed;
general moving frozen-core nuclear response requires a separate projector
response and is not implemented here.

The shared linear solve differentiates A x=b as A dx=db-(dA)x, retaining the
off-diagonal Fock response at internal orbital degeneracy. Numerical small-block
eigenvalues are used only to diagnose the excitation spectrum, outside AD;
they do not replace the live Fock operator or stop its physical response.

Outer JAX gradients/HVPs differentiate integral, denominator and response
dependence. Upstream HF/integral/basis response must remain on the graph for
complete physical parameter derivatives; lambda expansion itself holds that
reference fixed. Eager objects are preparation interfaces, not traced solvers.

## Verification and references

`tests/mp/test_series.py` compares individual orders against an independent
full-space PySCF Hamiltonian action plus Rayleigh--Schrodinger recurrence.
The reference does not use `jet`, production CI connections or a fitted energy
curve. Tests cover rank limits, state coefficients, R/U inputs, frozen spaces,
JIT, gradients/HVPs, empty spaces and invalid responses. The H6 example is
[taylor_series.py](../../../examples/mp/taylor_series.py).
The water [comparison](../../../examples/mp/compare_water_pyscf.py) independently
converges both HF references and compares E2--E8 in STO-3G. Its higher-order
PySCF-action recurrence is a reference construction, not a native PySCF MP8 API.

- [Wigner's (2n+1) rule in MBPT](https://doi.org/10.1080/00268978000100121),
  Molecular Physics 39 (1980): finite-order energy from lower-order wavefunctions.
- [Rayleigh--Schrodinger partitioning by excitation order](https://doi.org/10.1002/qua.560230446),
  International Journal of Quantum Chemistry 23 (1983): order-dependent spaces.
- [JAX Taylor mode](https://docs.jax.dev/en/latest/jax.experimental.jet.html):
  normalized versus factorial-scaled coefficients. The initial validation used
  JAX 0.8.1; the 2026-10-20 response increment used JAX 0.10.0.
- [Method references](REFERENCES.md) and [validation](VALIDATION.md).
