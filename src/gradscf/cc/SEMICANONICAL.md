# General-reference triples through a tensor resolvent

The explicit option `orbital_basis="semicanonical"` extends real molecular
CCSD(T) to ROHF-based unrestricted amplitudes and to noncanonical occupied/virtual
orbital frames. It uses the general-reference spin-orbital formula attributed to
Watts, Gauss and Bartlett (1993), [doi:10.1063/1.464480](https://doi.org/10.1063/1.464480).
The working moments were adapted from
[PySCF 2.9.0 gccsd_t_slow](https://github.com/pyscf/pyscf/blob/v2.9.0/pyscf/cc/gccsd_t_slow.py);
source attribution and hashes are in [NOTICE.md](NOTICE.md).

This means unrestricted CCSD on the supplied ROHF determinant. It does not
introduce a separately spin-adapted ROCCSD formulation or imply equivalence to
every software package's ROCCSD(T) convention. Complex and spin-mixed GHF inputs
remain unsupported.

## API and backward compatibility

```python
mycc = cc.CCSD(mf).run()  # real HF, UHF or ROHF source
correction = mycc.ccsd_t(orbital_basis="semicanonical")
details = mycc.triples(orbital_basis="semicanonical")
```

The same keyword is accepted by `cc.evaluate_triples` and
`cc.triples_correction`. The default `orbital_basis="canonical"` retains the
previous strict canonical checks and streamed virtual-triples implementation.
There is no automatic algorithm switch or reference-orbital modification.
Only conventional `(T)` supports the new option; restricted Urban +T(CCSD)
continues to use its original canonical path.

An additional explicit algorithm selects bounded response through second order:

```python
correction = mycc.ccsd_t(
    orbital_basis="semicanonical", algorithm="streamed",
    max_workspace_elements=2_000_000,
)
```

`algorithm="auto"` retains the existing paths, including the six-index tensor
reference for semicanonical inputs. Its `max_triples_elements` limit is unchanged.
The streamed option supports conventional real restricted/unrestricted CCSD(T),
first- and second-order JVPs and VJPs, including HVPs and reverse-over-reverse
directional curvatures. Third-order AD raises explicitly. Numerical eigenvectors
are used only as a factorization frame, not differentiated.

## Forward equation

Let `o` and `v` denote active **spin** orbitals in the existing occupied-alpha,
occupied-beta, virtual-alpha, virtual-beta ordering. Frozen electrons contribute
to the Fock matrix before its active projection. Define the tensor operator

\[
\begin{aligned}
(\mathcal D X)_{ijkabc}={}&
\sum_l(F_{il}X_{ljkabc}+F_{jl}X_{ilkabc}+F_{kl}X_{ijlabc})\\
&-\sum_d(F_{ad}X_{ijkdbc}+F_{bd}X_{ijkadc}+F_{cd}X_{ijkabd}).
\end{aligned}
\]

The first sum is in the occupied block, the second in the virtual block.
In a basis that diagonalizes these blocks, `D` reduces to
`eps_i + eps_j + eps_k - eps_a - eps_b - eps_c`. Hence solving in the original
frame is algebraically equivalent to explicitly rotating the integrals and
amplitudes to semicanonical orbitals, dividing by these sums, and rotating back.

Using the fully antisymmetrized connected moment `W` and disconnected moment
`V` from the attributed formula,

\[
\mathcal D X=W,\qquad
E_{(T)}=\frac{1}{36}(W+V):X.
\]

`W` contains the two-electron-integral/T2 contractions. `V` includes **both**
the `T1 * <ij||ab>` contribution and `F_vo * T2`. The latter cannot be dropped
for a general reference: independent alpha/beta occupied–virtual Fock blocks
need not vanish for ROHF. No division by diagonal entries of an untransformed
noncanonical Fock matrix is substituted for the tensor inverse.

The resulting energy is invariant under separate active occupied and virtual
orthogonal rotations. Such rotations must preserve the frozen/active boundary.
They need not preserve an arbitrary choice of eigenvectors inside a degenerate
Fock block. Restricted t1/t2 are mapped to the existing spin-orbital representation
and evaluated with the same formula.

## Solver ownership and derivatives

`gradscf.solvers.solve_tensor_sum` owns the numerical factorization, true-residual
checks and implicit response. It accepts a tuple of symmetric factors and a
tensor RHS, without constructing the full Kronecker matrix. The triples code
passes three occupied factors and three negative virtual factors.

The numerical solve diagonalizes the small factors, but AD differentiates

\[
\mathcal D\,dX=dW-(d\mathcal D)X.
\]

The eigenvectors used by the numerical factorization are stopped *inside*
`custom_linear_solve`; the operator and RHS remain differentiable. Therefore
their response is retained without eigenvector gap denominators. A repeated
occupied/virtual eigenvalue is allowed as long as the tensor-sum inverse is
resolved. This is not the same as stopping the physical orbital dependence of
the correction. Right CC amplitudes retain their existing implicit response.
The streamed kernel has independent tensor-oracle first- and second-response
checks. The molecular adapter below separately verifies nuclear gradients and
two nuclear HVP directions, including the upstream HF/CCSD/integral response.

### Bounded-panel derivative

The streamed option fixes virtual indices `b,c` and forms only the panel
`W[a,i,j,k]`, together with its disconnected moment. The occupied and virtual
blocks are numerically diagonalized to evaluate the inverse. For a weighted
scalar

\[
E=\frac1{36}(\alpha W+\beta V):X,\qquad
X=\mathcal D^{-1}W,\quad Y=\mathcal D^{-\mathsf T}(\alpha W+\beta V),
\]

the implemented differential is

\[
dE=\frac1{36}\{(\alpha\,dW+\beta\,dV):X
              +Y:dW-Y:(d\mathcal D)X\}.
\]

The moment pullback receives `(alpha X+Y)/36` and `beta X/36`.
Antisymmetry makes the three occupied-axis contractions equal, as well as the
three virtual-axis contractions. For each fixed `b,c`, they contribute

\[
\overline F^{oo}_{il}=-\frac3{36}\sum_{ajk}Y_{aijk}X_{aljk},\qquad
\overline F^{vv}_{ad}=+\frac3{36}\sum_{ijk}Y_{aijk}X_{dijk}.
\]

These are full matrices, including off-diagonal response inside a repeated
eigenvalue block. Differentiating only canonical denominators would lose those
entries. The custom JVP accumulates these cotangents and the integral/amplitude
cotangents in a panel loop, then contracts them with input tangents. Transposing
this linear expression supplies the first VJP without retaining every triples
panel. The `F_vo*T2` term and the converged CC amplitude response remain in the
computation graph.

### Second response with bounded panels

Write `Z = D^-1 V`, so the weighted adjoint above is `Y = alpha X + beta Z`.
Differentiating the two residual equations gives

\[
\mathcal D\,\dot X=\dot W-\dot{\mathcal D}X,
\qquad
\mathcal D\,\dot Z=\dot V-\dot{\mathcal D}Z.
\]

The custom JVP of the complete value/gradient evaluator computes these
responses and differentiates the moment pullbacks and Fock contractions.
It returns the directional value and `H dot(theta)`, with all integral,
amplitude and Fock dependencies live. The numerical frame stays fixed while
forming this differential; the full `dot(D)` replaces any eigenvector
derivative. Degenerate occupied/virtual eigenvalues therefore do not introduce
eigenvalue-gap denominators.

Four axes of `dot(D) X` lie inside the `aijk` panel. The fixed virtual axes
also contribute

\[
-\sum_d\dot F_{bd}X_{ijkadc}
-\sum_d\dot F_{cd}X_{ijkabd}.
\]

These neighbouring panels are recomputed and accumulated one at a time.
Virtual antisymmetry allows `X[:,b,d] = -X[:,d,b]`; the same relation holds for
`Z`. Each response RHS is solved with the common four-axis `solve_tensor_sum`,
using factors `(-diag(e_v)-(e_b+e_c)I, diag(e_o), diag(e_o), diag(e_o))`.
No new numerical or adjoint solver lives in CC.

The outer and neighbouring-panel loops are checkpointed so reverse-over-reverse
AD recomputes panels rather than retaining a complete panel history. Storage
remains a fixed number of four-index panels and the input/gradient blocks;
the extra neighbouring-panel recomputation can increase work by a factor
proportional to the virtual-space dimension. This is a time/memory tradeoff,
not a measured general scaling improvement. Third response requires additional
resolvent rules and is explicitly unsupported by this implementation.

### Complete molecular geometry response

For a converged native-integral GradSCF RHF/UHF reference, the existing
correlated-energy adapter composes the same molecular input response and CCSD
solution with the streamed correction:

```python
import jax
from gradscf_tools.correlated_response import CorrelatedEnergy

energy = CorrelatedEnergy(mf, "ccsd(t)")
value, nuclear_gradient = jax.jit(jax.value_and_grad(energy))(energy.coordinates)
hvp = jax.jit(lambda coordinates, direction: jax.jvp(
    jax.grad(energy), (coordinates,), (direction,))[1])
nuclear_hvp = hvp(energy.coordinates, direction)  # direction: (natom, 3)
```

The map rebuilds AO integrals, relaxes HF, transports the occupied/virtual
frame, transforms MO integrals, solves CCSD, and adds `(T)` inside AD. Nuclear
repulsion is included once in the CCSD total. Both the CCSD and triples
validity flags must hold. The N2/STO-3G test covers exact pi-orbital degeneracy
and compares with independently reconverged PySCF energies, finite differences,
and its analytic gradient using **CCSD(T) Lambda** amplitudes. Ordinary CCSD
Lambda amplitudes cannot substitute for that reference. The wrapper remains
an in-core small-system route without frozen orbitals or density fitting;
the streamed `(T)` response supports second order. N2/STO-3G bond and mixed
Cartesian HVPs are compared with independently reconverged PySCF CCSD(T)
analytic-gradient differences at two step sizes, including mixed Hessian
symmetry and translational checks. A full molecular Hessian matrix and
unrestricted nuclear Hessians were not part of that validation.

## Limits and diagnostics

- With `algorithm="auto"`, the semicanonical reference path materializes
  six-index `W`, `V`, solution and work arrays.
  `max_triples_elements=2_000_000` limits `(nocc_spin*nvir_spin)**3` **per tensor**
  before moment construction. Two million float64 elements are 16 MB per array;
  this is not a peak process/device-memory bound. AD and simultaneous arrays
  require more memory. This path is a correctness reference, not a low-memory
  replacement for the default canonical streaming path.
- With `algorithm="streamed"`, `max_workspace_elements` limits the
  `nvir_spin*nocc_spin**3` elements in **one** moment panel. Several panels,
  four-index arrays and two sets of their cotangents coexist; this is not a
  peak-memory bound. The restricted bridge creates only the `ooov`, `oovv`,
  `ovvv` spin blocks, Fock blocks and amplitudes, without a full spin-MO ERI or
  spin `vvvv` array. Retained blocks still have up to a factor of 16 spin
  redundancy relative to spatial tensors. This does not remove the underlying
  CC integral/residual memory costs or implement fully spin-adapted low-memory
  CCSD(T).
- The common solver conservatively requires **all Cartesian eigenvalue sums**
  to be separated from zero, including repeated-index and spin-forbidden tensor
  entries. It may therefore reject an otherwise nonsingular antisymmetric
  physical subspace. It does not silently pseudoinvert or clip the physical
  correction. Restricting the inverse to exterior-power spaces is future work.
- `min_abs_denominator` in this mode refers to that Cartesian spectrum.
  `canonical_error` still reports the input active Fock off-diagonal maximum;
  it may be nonzero in a valid semicanonical calculation.
- `connected_component` is `W:X/36`; the existing `singles_component` field
  contains the whole applied disconnected term `V:X/36`, including `F_vo*T2`.
- The current CCSD residual, model identifier, amplitude symmetries and solve
  validity are checked. Empty triples spaces still require a matching valid
  CCSD state. Failed low-level corrections yield `valid=False`/NaN; facade
  requests raise. No denominator shifts are applied.

## Independent validation

Tests explicitly diagonalize active alpha/beta occupied and virtual blocks in
NumPy, rerun PySCF UCCSD in that frame, and use its optimized UCCSD(T) correction.
They cover ROHF OH/STO-3G (0.97 Angstrom), all-electron, frozen-core and
spin-dependent frozen spaces; random occupied/virtual rotations; restricted
H4 canonical agreement; and fixed-MO finite differences with off-diagonal Fock
perturbations. A synthetic interacting model tests exact within-spin orbital
degeneracy and complete converged-amplitude response. The shared solver has an
independent dense Kronecker-matrix oracle at exact degeneracy.

Run `examples/cc/semicanonical_triples.py` for a native GradSCF UHF/ROHF example.
Commands and measured results are recorded in [VALIDATION.md](VALIDATION.md).
