# Experimental excitation-block MP action

`algorithm="tensor"` replaces the determinant-connection representation of W
inside the existing residual/Taylor engine. It does not introduce another MP
recursion, energy functional, linear solver or backward solver. It supports the
same real canonical RHF/UHF references and fixed frozen spaces. Every allowed
excitation through the requested rank is retained, including disconnected
contributions. This is an exact representation change, not a CCSD truncation.

```python
from gradscf import gto, scf, mp

mol = gto.M(atom="Li 0 0 0", basis="6-31g*", cart=True, spin=1)
mf = scf.UHF(mol, conv_tol=1e-12).run()
calculation = mp.MP(mf, order=6, algorithm="tensor",
                    with_coefficients=True).run()
print(calculation.corrections)  # E2 through E6, Hartree
print(calculation.space.size, calculation.space.dense_elements)
```

The pure JAX entry point is `mp.run_mp(h1, eri, nocc=..., config=MPConfig(...))`.
Defaults remain unchanged; selecting `tensor` is explicit.

## Coordinates and normalization

Introduce HF quasiparticles b_a^dagger = a_a^dagger for a virtual orbital and
b_i^dagger = a_i for an occupied hole. All b annihilate the HF reference.
Groups have the fixed order (virtual alpha, virtual beta, occupied-alpha holes,
occupied-beta holes). The block with alpha/beta excitation ranks (r_a,r_b) has
group counts s=(r_a,r_b,r_a,r_b). Its full tensor is antisymmetric within each
group, and its contribution to the state is

\[
 |\Psi_s\rangle = {1\over\prod_t s_t!}
 \sum_{X_0\ldots X_3} A_s(X_0,\ldots,X_3)
 b^\dagger_{X_0}\cdots b^\dagger_{X_3}|\Phi\rangle.
\]

Only strictly increasing index combinations in each group are stored in the
state vector. These independent coordinates are orthonormal:

\[
 \langle\Psi|\Psi\rangle
 =\sum_s {\|A_s\|_F^2\over\prod_t s_t!}
 =\|c_\mathrm{packed}\|_2^2.
\]

`wavefunction_coefficients` returns Phi, Psi1, ... in these coordinates, **not
the determinant ordering/phases of `algorithm="series"`**. Both use intermediate
normalization; the reference components are (1,0,...). For tensor coordinates:

- `space.blocks` is ordered by `(r_a+r_b, r_a, r_b)`.
- `space.orbitals` gives the four groups of physical MO indices in ascending order.
- Each block's `indices[t]` lists lexicographically ordered combinations of local
  indices in group t. Each group's empty combination has shape (1,0).
- `coefficients[block.offset:block.offset+block.size].reshape(block.shape)`
  uses C order, with the fourth group varying fastest. For returned state rows,
  first select the desired perturbation-order row.
- `space.expand` reconstructs antisymmetric tensors; `space.pack` extracts the
  independent coordinates. These are experimental space inspection helpers.

For mapping to a canonical determinant, let P be the concatenated alpha/beta
particle indices and I the concatenated hole indices (beta indices shifted by
nmo). Define eta from
`excite(reference_bits, I, P) -> (determinant_bits, eta)`. For r=r_a+r_b,

\[
 c_\mathrm{det}=\eta(-1)^{r(r-1)/2}c_\mathrm{packed}.
\]

The additional sign arises because the quasiparticle convention orders hole
creators forward, whereas the standard excitation operator reverses its hole
annihilators. No factorial multiplies the independent coefficients. Returned
`t2` retains the conventional occupied/occupied/virtual/virtual ordering and
signs for both backends; SS/OS energies still decompose E2 only.

## Generating the contraction graph

The HF-normal-ordered fluctuation operator is

\[
 W=\sum_{pq}(F_{pq}-\delta_{pq}F_{pp})\{a_p^\dagger a_q\}
 +{1\over4}\sum_{pqrs}\bar g_{pqrs}
 \{a_p^\dagger a_q^\dagger a_s a_r\}.
\]

Off-diagonal Fock elements are retained, even when negligible for a converged
canonical reference, so the action preserves the existing integral derivative
contract. Spin deltas form antisymmetrized physicists' integrals directly from
the restricted ERI or unrestricted (aa,ab,bb) spatial tensors. A full doubled
spin-orbital ERI is never allocated on this path.

For each spin/occupied/virtual assignment, replace physical operators by
quasiparticle operators, normal order them, and collect the fermion permutation
sign. No additional contraction terms are generated: W is already normal
ordered relative to the HF vacuum. Group creators in ascending group order
and annihilators in descending group order. Reverse the annihilator sequence
when assigning contraction axes. Integral leaves and kernels with identical
group counts are shared.

Let m and l be the numbers of creators and annihilators in the four groups.
For a source block s, the remaining counts are u=s-l, and the target is t=u+m.
The term contributes only if all u are nonnegative and the target is retained.
Contract the l kernel axes with the first l axes of each source group, then
interleave created and surviving axes by group. The remaining cross-group sign is

\[
 (-1)^{\sum_{a<b}u_a(l_b+m_b)}.
\]

Antisymmetrize the target by signed subset shuffles of the m created positions
among its t positions. The action multiplies the normal-ordered kernel by
\(\prod_a m_a!\); there is no additional l! factor. In implementation that
factor is incorporated once during kernel construction. Each contraction uses
`jnp.tensordot`, followed by the signed output permutations.

At Taylor degree j, Psi_j has no rank above 2j. The action skips those source
blocks and output ranks above 2j+2 exactly, without magnitude screening. The
resulting W Psi_j is reused by all later residual degrees and Rayleigh energy
extraction, as in [GRAPH_REUSE.md](GRAPH_REUSE.md). JAX differentiates the
integral kernels, gaps, contractions and shared checked linear solves. No
stop-gradient or separate hand-coded tensor adjoint is introduced.

## Capacity and limitations

`max_determinants` (default 5000) bounds independent state coefficients in
either backend. `max_tensor_elements` (default 2,000,000) bounds the sum of full
antisymmetric block sizes for one expanded state, before expansion-map
allocation. It is **not** a bound on total peak memory: integral kernels, cached
permutations, compiler temporaries and differentiated graphs also consume memory.
The space remains combinatorial. Full spatial MO ERIs are still prepared.

This first version does not factor disconnected states, stream density-fitted
integrals, optimize a general contraction tree, or guarantee a speedup. Expansion
and antisymmetrization can dominate when multiple same-spin holes and particles
are present. The exact-rank skip does not remove that remaining cost.

## Verification and reproducible timings

`tests/mp/test_tensor_action.py` compares complete action matrices at ranks
0--4 against the CI Slater--Condon operator, plus a complete sixfold-excitation
action on a random vector in six orbitals, including noncanonical Fock terms,
asymmetric UHF frozen spaces, norms and packing. It compares E2--E12, state
coefficients after signed permutation, standard t2 and SS/OS energies with the
existing series backend. Moving-gap tests cover JIT reverse gradients and HVPs;
invalid inputs reject both primal and gradient results. These integral-level
tests do not establish molecular nuclear gradients without upstream HF and
integral response.

Validation on the environment below: the MP suite, integral namespace/module
boundary tests and shared GMRES accuracy tests passed together (153 tests,
284.68 s). The subsequently added sixfold-excitation test passed separately
(1 test, 3.41 s). GPU execution and molecular nuclear gradients of this backend
were not tested in this experiment.

Run `PYTHONPATH=src JAX_PLATFORMS=cpu python examples/mp/benchmark_excitation_blocks.py`
from the checkout root. The example uses native GradSCF HF and the same full MO
integrals for both backends; all electrons are active. Three warmups precede 15
paired, alternating-order, synchronized executions. Timings cover compiled
energy corrections only. HF, integral transformation, graph lowering and
compilation are excluded and the latter two are recorded separately. Raw
samples, source hashes and compiler cost estimates are written to ignored
`artifacts/mp_blocks/benchmark.json`. Compiler byte estimates are not measured
peak memory.

Measured 2026-10-10: Apple M4 Pro CPU, 24 GB, macOS 26.6.2, Python 3.12.2,
JAX 0.8.1, float64, Cartesian 6-31G*, HF energy/gradient tolerances 1e-13/1e-11.
Geometries in Angstrom are included in the example and JSON. Maximum difference
between paired energy corrections was 8.33e-17 Hartree.

| System | Order | Series warm / ms | Tensor warm / ms | Series / tensor |
| --- | ---: | ---: | ---: | ---: |
| H3, UHF | 4 | 0.124 | 0.142 | 0.88 |
| H3, UHF | 6 | 0.145 | 0.181 | 0.80 |
| H3, UHF | 12 | 0.256 | 0.323 | 0.79 |
| H4, RHF | 4 | 1.245 | 0.946 | 1.32 |
| H4, RHF | 6 | 1.555 | 1.621 | 0.96 |
| H4, RHF | 12 | 2.661 | 3.490 | 0.76 |
| Li, UHF | 4 | 4.911 | 0.753 | 6.52 |
| Li, UHF | 6 | 6.295 | 1.238 | 5.08 |
| Li, UHF | 12 | 10.604 | 2.128 | 4.98 |

These are small-system representation comparisons, not asymptotic scaling or
end-to-end speedups. The series baseline already includes shared W Psi_j nodes.

For MP12, lowering plus compilation cost 1.80/3.00 s (series/tensor) for H3,
2.04/4.01 s for H4, and 3.10/3.01 s for Li. Thus warm execution gains do not
automatically imply gains for a single evaluation. H3 has 90 independent
coefficients versus 438 expanded elements; H4 has 784 versus 24,649; Li has
1,575 versus 10,545. The larger H4 expansion illustrates the remaining
permutation cost, but these three systems alone do not establish a general
performance model.
