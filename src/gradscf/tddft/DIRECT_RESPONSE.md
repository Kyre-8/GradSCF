# Bounded non-DF molecular response

The native restricted direct-SCF reference can supply an AO J/K action to
CIS, TDA and TDHF without constructing a full AO ERI, an AO-pair matrix or
complete MO ERIs. This is a storage change to the same single-excitation
equations, not a density-fitting approximation or an orbital-space truncation.

## Public use

```python
from gradscf import gto, scf, ci, tdscf

mol = gto.M(atom="O 0 0 0; H 0 -.757 .587; H 0 .757 .587",
            basis="6-31g*", unit="Angstrom", cart=True)
mf = scf.RHF(mol, integral_backend="native", jk_backend="direct").run()
cis = ci.CIS(mf, nroots=5, max_memory=1024.).run()
tda = tdscf.TDA(mf, nstates=5, max_memory=1024.)
tda_result = tda.kernel()
tdhf = tdscf.TDDFT(mf, nstates=5, max_memory=1024.)
tdhf_result = tdhf.kernel()  # An HF reference gives TDHF.
```

`max_memory` is decimal MB and defaults to 2000. Low-level response callers
can set the same value in `ResponseKernelOptions`. Restricted CIS singlets
and triplets share the AO action; unrestricted CIS and TDHF triplets are not
new capabilities of this increment.

Explicit `CIReference`/`solve_cis(h1, eri, ...)`, DF CIS and CIS(D) retain their
existing MO-integral routes. Explicit local PT2-feature requests and the
existing external-XC self-consistent solver still obtain their required
integral arrays; the default direct excited-state reference does not.

## Representation and action

The reference retains `NativeDirectBasis`: its shell topology is static,
while coordinates, exponents and coefficients are dynamic PyTree leaves.
Its two-electron supply is `get_jk(density)`, not an ERI array. The native
engine traverses shell quartets and contracts their local integrals directly.

The response path is

\[
X_{ia}\longrightarrow\delta P_{\mu\nu}
\longrightarrow J[\delta P],K[\delta P]
\longrightarrow (AX)_{ia},(BX)_{ia}.
\]

The existing A/B density orientations and projections are preserved: a
transition density is generally nonsymmetric and must not be symmetrized.
Restricted singlet and HF triplet actions retain their respective `2J-K`
and `-K` contributions. Noncanonical CIS uses the complete occupied/virtual
Fock blocks, `X Fvv - Foo X`. Frozen orbitals restrict the transition columns
without removing their contribution to the reference density and Fock matrix.
CIS spatial amplitudes retain unit norm; restricted TD amplitudes retain
their separate `sqrt(1/2)` convention.

Standard HF has no semilocal XC kernel. Its response bypasses all quadrature
feature projection, including an otherwise useless `grid * occ * vir` tensor.
General DFT grid response is unchanged by this HF-specific elimination.

## Workspace and cache lifetime

A static response plan reserves primary AO state, returned vectors, Ritz/QR
work, and a shared action-workspace portion. It derives the Davidson capacity
and vector microbatch from one budget. A requested capacity that conflicts
with the budget, or an insufficient minimum working set, fails before the
large solver allocation; requested and boundary roots are not dropped.

Hermitian Davidson's unspecified capacity is now bounded rather than `D`.
The shared default is `min(D, max(40, 4*(search_roots+1)))`; search roots include
the available spectral boundary root. Initial unit-vector columns are built
directly with one-hot arrays, without first constructing an identity of size D.
The existing RPA capacity policy remains in place.

AO actions process sequential vector microbatches, including a nondivisible
tail. Checkpointing surrounds the complete local action, so its AO work can
be recomputed during transposition. Only the required OV output rows are
assembled. Numerical eigensolvers and spectral-response mathematics remain
owned by `gradscf.solvers`; there is no new method-local iterative solver.

This budget covers the current direct-HF primary arrays. It is not a process
RSS cap: full-grid AO caches, pre-existing full/DF integrals, nested adjoint
work, external functional internals, XLA compilation and allocator residency
need separate accounting. General grid streaming and broader shared budgets
are later increments.

The restricted HDF5 cache stores direct providers as topology/parameter
arrays, plus projection matrices when applicable, and reconstructs the plan.
It never materializes ERIs to serialize a provider. Rewriting a cache group
also removes obsolete pair, DF and MO-block datasets, preventing old large
arrays from being reloaded after switching to direct representation. Existing
array caches remain readable. Cached references still use the existing
explicit functional-selection convention when constructing TD drivers.

## Differentiation and devices

Fixed-basis density JVP/VJP and density second response remain supported.
CIS energy and amplitude parameter responses use the existing implicit
eigensolver. A native guard accepts only strictly zero materialized basis
tangents, which can arise from `custom_root` even for a fixed basis. Nonzero
basis/geometry JVPs and active basis VJPs remain explicitly unsupported;
under JIT the exception can be wrapped as an XLA callback runtime error.
The new native response path does not stop physical basis dependence or
replace it by a zero derivative. Projected providers retain their existing
fixed-primitive, differentiable-contraction contract.

When response arrays move to CUDA, bounded density/J/K batches can cross a
CPU callback to the same native shell engine. The integrals are still
computed on CPU; this is not a GPU-native integral implementation. The CPU
backend must remain available. CUDA operator values and density gradients
were checked on GPU5; GPU6 was not used.

## Validation, 2026-10-20

Environment: `jy_gpu_1`/`c20`, Intel Xeon Platinum 8457C, Python 3.11.15,
JAX/jaxlib 0.10.0, PySCF 2.13.0, float64. Each job had 16 logical CPUs and
OMP/OpenBLAS/MKL limits of 8; the concurrent selections used at most 64 logical
CPUs. CPU tests hid every GPU. No local numerical tests were run.

The final focused selection passed **110 CPU tests in 328.51 s**, and the
two CUDA compatibility tests passed in **9.40 s** on GPU5. This is not the
whole repository suite. Coverage includes forbidden ERI construction,
bounded default shapes, noncanonical/frozen CIS, input/amplitude AD, strict
basis-AD rejection, cache roundtrips and rewriting, existing response/guard
regressions, and independent software comparisons. A 1060-AO/168004-transition
synthetic shape test checks allocation structure; it is not a completed
DABNA-2 electronic-structure calculation.

H2O/STO-3G, Cartesian, all electrons and all ten transitions were compared
with independently converged PySCF and non-RI stock MolGW 3.4, source
`b831818d7a845c36f295d036dc8ceef59daf9991`:

| Method | Maximum energy error vs PySCF / Ha | Maximum energy error vs MolGW / Ha |
| --- | ---: | ---: |
| CIS/TDA singlet | 8.24e-13 | 7.81e-9 |
| CIS triplet | 1.06e-12 | 3.78e-9 |
| TDHF singlet | 7.67e-13 | 8.89e-9 |

The largest singlet oscillator-strength errors are 3.72e-12 versus PySCF and
5.76e-9 versus MolGW. The MolGW fixture records its input, binary/source
identity and its own eV/Hartree constant; its YAML rounds excitation energies
to six or seven decimal places in eV. These are implementation consistency
checks, not basis convergence or experimental accuracy claims.

The runnable [benzene example](../../../examples/tdscf/direct_response.py)
was executed in fresh processes against baseline `8766fa5` and the optimized
source. Both used native direct RHF, Cartesian 6-31G*, 102 AOs, 1701
transitions, five roots, a 40-column subspace and a 1e-8-Ha TD residual
tolerance. CPU affinities were 0--15 and 48--63, respectively, on the same
shared host; the example separates SCF, reference construction, compilation,
warm action calls and the full TDA solve.

| Measurement | Baseline | Optimized |
| --- | ---: | ---: |
| Reference two-electron array cache / bytes | 220752072 | 0 |
| Compiled five-vector action temporary / bytes | 867678560 | 1251936 |
| Whole-job peak RSS / KiB | 4316836 | 1513560 |
| Warm five-vector action median / s | 0.1914 | 3.2715 |
| Full TDA driver after cache clearing / s | 22.02 | 172.44 |

All five excitation energies agree within 1.1e-15 Ha and both solves
converged. XLA's temporary estimate excludes opaque native workspace; RSS
includes the full example process. Memory is substantially lower, while
recomputed shell integrals make this direct path slower than retained packed
integrals on this system. The two timings are measured shared-host results,
not a claim of a general speedup. Bounded integral reuse and contraction
amortization are subsequent performance work.
