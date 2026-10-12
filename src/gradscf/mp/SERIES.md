# Residual-driven MP series

`MP(mf, order=p)` computes all corrections E2 through Ep from one physical
residual. The generic engine supports real canonical restricted/unrestricted
HF frames, integer occupations and fixed frozen spaces. It has no fixed order
ceiling, but it has explicit representation-dependent capacity limits.
It does not claim convergence of the series at lambda=1.

## Physical partition and gauge

Normal ordering with respect to the fixed HF determinant gives

$$
H(\lambda)=H_0+\lambda V_N,\qquad
H_0=E_\mathrm{HF}+F_N.
$$

For each retained determinant D, its Fock excitation gap is

$$
\Omega_D=\sum_{\sigma p}\epsilon_p^\sigma
\left(n_{Dp}^\sigma-n_{0p}^\sigma\right).
$$

All occupied electrons, including frozen cores, contribute to the physical HF
Fock and reference energy. Frozen orbitals restrict excitations; they are not
removed from reference determinants. The default operator is generated from the
shared CI Slater--Condon machinery on the retained space. The experimental
`algorithm='tensor'` instead contracts complete excitation blocks; see
[TENSOR_BLOCKS.md](TENSOR_BLOCKS.md). No FCI product-space
enumeration, physical dense Hamiltonian or diagonalization is required.

Use intermediate normalization, c0(lambda)=1. Write the unknowns as
y=(c_exc, delta E), c=(1,c_exc), and W=H-H0. The residual is

$$
F_D=(\Omega_D-\delta E)c_D+\lambda(Wc)_D,\quad D\ne0,
\qquad F_E=\lambda(Wc)_0-\delta E.
$$

Its known zero-point solution is y0=0. The zero-point Jacobian is
J0=diag(Omega_exc,-1), so the physical reference must be isolated from all
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
`factorial_scaled=False`. The Jacobian action comes from `jax.linearize`;
the same checked shared linear solver is used at every degree. Its diagonal
preconditioner is the exact inverse for a valid canonical reference. Both
primal and transpose solves retain the shared residual checks. No independent
iterative or adjoint solver lives in MP.

The linear nodes W Psi_j are computed once per state coefficient and shared
with later residual degrees and Rayleigh extraction. Only the remaining
polynomial/quotient graph is lifted by Taylor AD. This preserves all outer
integral and gap derivatives; see [GRAPH_REUSE.md](GRAPH_REUSE.md) for the
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
Increasing `wavefunction_order` above the default computes more complete state
coefficients and increases the rank/capacity requirement accordingly.

| Energy request | Default complete state order | Maximum retained rank |
| --- | ---: | ---: |
| MP2 / MP3 | 1 | 2 |
| MP4 / MP5 | 2 | 4 |
| MP6 / MP7 | 3 | 6 |
| MP8 / MP9 | 4 | 8 |

Physical particle/hole counts can reduce the actual rank. At sufficiently high
order the retained space can coincide with the entire finite determinant space;
the algorithm still does not perform an FCI eigensolve. Wavefunctions in this
representation and their connection table grow combinatorially. This is an
in-core generic reference engine, not yet a tensor-factorized high-order backend.
The unrestricted CI adapter also materializes spin-orbital integral tensors.
The opt-in tensor backend avoids that adapter and its connection table, but
still has combinatorial state growth and dense antisymmetric temporary blocks.

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

`t2` always contains first-order amplitudes, and SS/OS energies decompose E2
only. `with_t2=False` omits amplitude extraction, not internal response states.
`with_coefficients=True` returns the complete state rows through k. Capacity
checks precede Hamiltonian connections and eager MO transformations. Unresolved
gaps/noncanonical inputs invalidate values and generic JVP/VJP with NaNs.

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
  normalized versus factorial-scaled coefficients. The tested runtime is JAX 0.8.1.
- [Method references](REFERENCES.md) and [validation](VALIDATION.md).
