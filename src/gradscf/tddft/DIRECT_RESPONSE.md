# Bounded non-DF molecular response

Current SCF/TDDFT AO grid streaming, memory controls and external-model
boundaries are documented in [Molecular AO grid streaming](../integrals/grids/STREAMING.md).

The native restricted direct-SCF reference supplies an AO J/K action to
CIS, TDA and TDHF without requiring an AO ERI array or complete MO ERIs.
CPU response operators can reuse exact s8 integrals when the workspace budget
admits them; otherwise they contract shell-direct. This is a storage change
to the same single-excitation equations, not density fitting or an orbital-space
truncation.

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

SCF and TD `max_memory` is decimal MB and defaults to 60000 (60 GB). TD facades inherit
the SCF/reference budget unless explicitly overridden. Low-level response callers
can set the same value in `ResponseKernelOptions`. Restricted CIS singlets
and triplets share the AO action; unrestricted CIS and TDHF triplets are not
new capabilities of this increment.

Explicit `CIReference`/`solve_cis(h1, eri, ...)`, DF CIS and CIS(D) retain their
existing MO-integral routes. Explicit local PT2-feature requests still obtain
their required integral arrays. Restricted external-XC SCF and training
preserve direct providers without requesting pair ERIs.

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
Conventional weighted grid kernels use block-local occupied/virtual factors
and two-sided projections, without a full `feature * grid * occ * vir`
transition tensor. Streamed references also obtain their AO values and spatial
derivatives from the shared `grid_ao` supplier inside these grid blocks. The
dense small-system reference path retains its original arrays.

Real AO TDHF/TDDFT combines its two transition densities before the AO action:

\[
D=2C_vX^{\mathsf T}C_o^{\mathsf T}+2C_oYC_v^{\mathsf T}.
\]

For real spatial integrals, the linear map
\(R(D)=J[D]-\alpha K[D]/2\) obeys
\(R(D^{\mathsf T})=R(D)^{\mathsf T}\). The two TDHF rows therefore use one
integral traversal and two projections of the same AO result. Occupied/virtual
Fock terms remain separate. Fixed grid XC kernels apply once to `X+Y` and
contribute the same result to both rows. Opaque XC callbacks instead evaluate
`F(X)+F(Y)`, preserving legacy output clipping that is not linear. Custom
nonlocal A/B and low-rank actions retain their existing separate treatment.

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
RSS cap: explicitly requested full-grid AO arrays, pre-existing full/DF integrals, nested adjoint
work, external functional internals, XLA compilation and allocator residency
need separate accounting. The shared AO supplier additionally plans its
primitive/grid workspace using `grid_max_memory`; response blocks respect
both limits. These controls are working-array estimates, not an RSS cap.

### Budgeted integral reuse

`ResponseKernelOptions.integral_cache` accepts `"auto"` (default) and `"none"`.
The latter keeps the shell-direct action at any budget. In automatic mode,
only an ordinary CPU `NativeDirectBasis` is eligible. Projected providers and
the normal CUDA workflow retain direct J/K. Neither choice constructs a full
four-index array or a four-index exchange gather.

For \(P=N_{\rm AO}(N_{\rm AO}+1)/2\), the s8 cache occupies
\(C=8P(P+1)/2\) bytes. It is generated directly in this layout; an s4 buffer
is not constructed first. The action share is one half of the usable budget,
after the runtime reserve and known resident arrays. It pays for the cache,
native workspaces, vector microbatches and grid blocks together. The planner
charges two packed buffers, admits a cache only when the joint minimum fits,
and selects up to 32 vector columns. The second packed buffer covers the
zero-basis JVP/recomputation temporary. The per-vector estimate includes up to
eight native workers and three concurrent J/K calls in a mixed zero-basis /
density JVP, in addition to AO/projection arrays. This is a bounded estimate
for the supported forward and response paths, not arbitrary high-order AD.

The response closure owns the numerical integrals, and exposes the selected
`memory_plan` for measurement. The SCF reference and its HDF5 representation
remain lazy; no global numerical ERI cache is added. Creating an operator for
new basis parameters recomputes its integrals. Releasing the operator releases
its ownership, subject to JAX executable/allocator residency. Solver and action
builders use the same action share, so changing the requested root count does
not silently change integral-cache admission.

This policy follows the distinction between integral supply and consumers in
[PySCF's SCF implementation](https://pyscf.org/_modules/pyscf/scf/hf.html), the
separate integral-memory setting in the
[ORCA manual](https://www.faccts.de/docs/orca/6.1/manual/contents/essentialelements/integralhandling.html),
and the phase-specific data ownership described in
[the ORCA 6 software paper, section 2.3](https://doi.org/10.1002/wcms.70019).
The checked MolGW `m_eri_ao_mo.f90` uses explicitly allocated transformation
workspaces and releases them at the end of each transformation. Its dense
response matrices are not adopted here. ORCA documentation and the paper,
rather than unavailable numerical-core source, were used for this comparison.

The restricted HDF5 cache stores direct providers as topology/parameter
arrays, plus projection matrices when applicable, and reconstructs the plan.
It never materializes ERIs to serialize a provider. Rewriting a cache group
also removes obsolete pair, DF and MO-block datasets, preventing old large
arrays from being reloaded after switching to direct representation. Existing
array caches remain readable. Cached references still use the existing
explicit functional-selection convention when constructing TD drivers.

## Differentiation and devices

Density JVP/VJP, first and second coordinate response, and mixed
density-coordinate products are supported through `IntegralPlan.get_jk`
and its Native/Projected providers. Coordinates and the remaining integral
environment are separate differentiation inputs: primitive exponent and
coefficient derivatives remain explicitly unsupported. The projected
provider retains both coordinate and contraction-matrix dependence.
The raw flat-ENV helper retains its density-only boundary; geometry callers
use the canonical plan/provider interface.

The native geometry action contracts each derivative shell block immediately
into J/K matrices or a coordinate adjoint. It reuses the existing libcint
first/second derivative kernels without producing an AO four-index derivative
array. Its density transpose retains the coordinate-aware primitive, so mixed
second derivatives keep the coordinate contribution. Real third coordinate
derivatives are explicitly rejected.

Coordinate differentiation currently requires zero screening:
`direct_scf_tol=0.` for SCF and `ResponseKernelOptions(direct_tol=0.)` for
response. Nonzero screening remains available for forward/density-only use;
an active screened coordinate derivative raises an explicit error. No
derivative is silently replaced by zero. The geometry kernels are currently
serial CPU contractions; their large-system throughput is not benchmarked.

Traced coordinates bypass optional s8 construction and use the direct
geometry action. A CPU-prepared action later
used on CUDA selects the existing shell-direct CPU callback instead of the
portable packed GPU contraction. JIT-compiling the entire cache factory
explicitly for CUDA is unsupported; the packed-integral constructor is CPU-only.

When response arrays move to CUDA, bounded density/J/K batches can cross a
CPU callback to the same native shell engine. The integrals are still
computed on CPU; this is not a GPU-native integral implementation. The CPU
backend must remain available. An earlier snapshot checked CUDA operator
values and density gradients on GPU5. The new geometry path has CPU numerical
validation only; GPU6 was not used.

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
not a claim of a general speedup. This measurement motivated the budgeted
integral reuse and TDHF contraction sharing described above.

### CPU integral-reuse verification

The follow-up compared `7c5d51e` (shell-direct only) with the budgeted-cache
implementation using the same example, Python/JAX versions, CPU0--15 affinity,
eight-thread BLAS limits, float64 and numerical tolerances. Fresh processes ran
sequentially on `jy_gpu_1` (Intel Xeon Platinum 8457C); the host was shared with
other work, so these timings are not an isolated-machine performance guarantee.

| Measurement | Shell-direct `7c5d51e` | Budgeted s8 reuse |
| --- | ---: | ---: |
| Reference ERI arrays / bytes | 0 | 0 |
| Response integral cache / bytes | 0 | 110397048 |
| Compiled action temporary / bytes | 1251936 | 1248480 |
| Whole-job peak RSS / KiB | 1519088 | 1568412 |
| Warm five-vector action median / s | 3.77024 | 0.10772 |
| Full TDA driver / s | 201.471 | 20.382 |

All five roots agree within 1.6e-15 Ha; the HF total energy is unchanged.
The measured warm action and full solve are about 35.0 and 9.9 times faster,
respectively, while peak RSS remains about 1.5 GiB. These measurements include
the explicitly reported cache; a zero reference-array count does not imply
zero response storage. Native FFI worker workspace is not included in XLA's
temporary byte count.

The final focused CPU selection passed **124 tests in 312.68 s**, including
the same independent water PySCF/MolGW comparisons, cache admission/tail batches,
strict basis-AD rejection and fused TDHF derivatives. Source hashes were stable
during each run. No native rebuild or dependency installation was required.
Three CUDA checks were skipped after CUDA initialization failed on the current
server; this follow-up does not claim a new GPU numerical pass. Per the user's
final instruction, further validation focuses on CPU memory on `jy_gpu_1`.
General DFT grid streaming was outside that measurement. Its subsequent
implementation uses the shared AO supplier described above. Large-system
partial integral caches and the DF exchange workspace remain separate work.

## External neural functionals

An external scalar-energy model continues to use
`dft.custom_xc(input_fn=..., energy_fn=...)` and an explicit parameter PyTree.
The input callback receives physical quantities from the current SCF density;
the architecture stays outside the response and solver modules. AD computes
the AO potential and its Hessian-vector product from the same energy. A full
four-index AO Hessian is not needed by the production operator.

Direct J/K providers now reach the restricted external-functional SCF context,
its implicit reconstruction, the final energy/Fock evaluation, and training
total-energy evaluation. The existing `fixed_density`, `explicit`, and
`implicit` modes retain their meanings and shared solvers.

An opaque AO model or a model coupling different grid points must receive its
complete input. Local grid blocking does not split its energy into independent
pieces or replace its Hessian by pointwise blocks. The full AO kernel action
and the complete-grid HVP routes therefore remain intact. This includes models
using an AO density matrix, pooled grid features, attention, or other global
dependencies, provided their operations have JAX differentiation rules.
Architectures beyond the small regression models have not been benchmarked.

The new focused checks cover four-layer local and global networks, TDA/TDHF
values and parameter/density JVP/VJP, and all three training modes against a
dense small-AO oracle and finite differences. These are compatibility checks,
not large-molecule neural training or convergence evidence.

The callback's internal tensors and complete-grid small-feature HVP inputs
are not bounded by `grid_block_size`. Streamed external-functional inputs
generate density, gradient, kinetic-density and Laplacian features from the
AO supplier on demand. Initialization does not prepare a full-grid AO
Laplacian merely because a model could request that feature. Explicit
`DensityInputs.ao`/`ao_deriv1` requests still materialize full arrays, and legacy
local-HFX/PT2 caches require the dense path. Coordinate AD uses the same direct
provider as conventional XC. The AO supplier retains basis-parameter AD;
primitive exponent/coefficient AD of native direct integrals retains its
existing unsupported boundary, independently of network parameter derivatives.

## Large-system follow-up, 2026-10-20

The shell-direct implementation shares one call-scoped CINT optimizer and
assigns disjoint output AO rows to workers. It no longer allocates one full
J/K output per worker. A shell quartet can consequently be recomputed for
several row owners; bounded storage alone does not establish a speedup.
J-only and K-only calls use one native output and skip the other contraction.
The planner reserves optimizer primitive-pair storage, shell scratch and table
copies separately from the vector batch. Small-system packed contraction
still has its own worker buffers and retains the larger cache-workspace charge.

The CPU implementation snapshot passed **182 related tests in 400.31 s** with
stable source hashes and exit code zero. The subsequent optimizer/workspace
budget refinement passed **25 focused grid/cache tests in 35.06 s**. These
were separate snapshots; the latter adds fixed-workspace accounting without
changing the AO equations. Device-transfer contract tests also cover the
restricted overlap, Fock and AO Laplacian fields. No new CUDA numerical pass
is claimed in this follow-up.

For DABNA-2, Cartesian def2-SVP gives 1060 AOs, 194 occupied and 866 virtual
orbitals, and 168004 transitions. A matched-orbital action check used an
independently converged non-DF PySCF RHF checkpoint, float64, sixteen CPU
threads, a 1e-12 Schwarz threshold and no retained ERIs. It did not rerun
GradSCF SCF or restrict the orbital space.
The probe retained the earlier two-vector microbatch plan; the refined
fixed-workspace planner has not yet been benchmarked on this molecule.

| One-vector action measurement | GradSCF | PySCF |
| --- | ---: | ---: |
| JIT compilation / s | 4.912 | not applicable |
| First action / s | 252.229 | 38.172 |
| Warm action / s | 273.999 | not separately measured |

The maximum action difference was **1.15e-10 Ha**. The entire GradSCF probe
peaked at **1644892 KiB** group RSS and reached its 600-second limit during
the four-vector stage. The PySCF timing is one independent call, not a warm
timing baseline. These data establish one matched complete-space action;
neither the four-vector measurement nor a 20-root spectrum completed.
An independent PySCF 20-root TDA run also reached its 1800-second limit after
converging non-DF RHF to -2253.6589957497354 Ha. The full large-molecule
response throughput remains an open optimization target.

The rigid dimer input in `examples/tdscf/dabna2` contains 192 atoms, preserves
both monomer internal geometries, and has a minimum interfragment nuclear
distance of 2.561339 Angstrom. It is a computational benchmark, not an
optimized aggregate. Its electronic-structure calculation has not completed.

## Direct geometry response and TDDFT integration

The focused continuation reuses the existing `native/jk.py` and native
shell-direct forward kernel. It adds coordinate JVP/VJP products and connects
them through restricted and unrestricted molecular response. It does not add
a second J/K engine or method-local numerical solver.

The differentiable workflow rebuilds integral inputs/provider, grid, SCF
density, orbitals and Fock at each geometry, then calls the public
`tdscf.TDA(state, ...)` or `tdscf.TDDFT(state, ...)` interface. An already solved
fixed state is not a relaxed geometry calculation. The eager `mf.run()` facade
retains its NumPy freshness/cache management; this increment does not make that
entire management layer a JIT API. Differentiable SCF array-state execution and
the existing external-functional implicit SCF are the tested paths.

The UKS occupation validator keeps static shape checks and eager range/count
validation without converting traced occupations into Python floats. The
common assembly now accepts traced native-direct inputs with
`direct_scf_tol=0.`. TD operators keep the provider and its coordinate
dependence, including through projected contractions and both spin channels.

Focused CPU verification on `jy_gpu_1`, float64:

- **83 integral tests passed in 43.86 s**, including 21 direct-geometry tests,
  nonsymmetric/batched density response, Cartesian/spherical integrals,
  translation invariance, mixed second products and original integral AD.
- **11 distinct end-to-end nodes passed in split runs** through the public TD
  interface. They cover H2/H2+ Cartesian STO-3G and neutral HeH doublet/6-31G,
  level-0 atom-centered grids, HF, PBE and a four-layer global spin-coupled
  neural energy. HeH has two occupied alpha and one occupied beta orbital,
  with both occupied-virtual response channels nonempty. The nodes include one
  occupation/JIT contract test; ten perform full excitation-gradient checks.
- **23 external-network compatibility tests passed in 79.64 s** after the
  geometry change, including fixed-density, explicit and implicit training.
- **14 operator/control tests passed in 5.51 s**, including projected-provider
  forwarding of J/K selection, screening and worker controls.

End-to-end tests require convergence at the central and displaced geometries,
check that both density and orbitals change, compare direct/full excitation
energies within 3e-9 Ha, and compare coordinate JVP/VJP with the full-integral
oracle and central differences (2e-4 bohr displacement). Gradient comparisons
use 8e-7 Ha/bohr absolute and 3e-5 relative tolerance against the oracle;
finite-difference checks use 3e-6 Ha/bohr and 3e-4. The full-ERI calculations
are small independent oracles; the tested direct path does not request them.

These are first-order relaxed excitation-gradient checks. Integral-level
mixed/second coordinate response is separately covered; no complete TDDFT
Hessian, new CUDA geometry pass, large-molecule gradient benchmark, or primitive
exponent/coefficient AD is claimed.

## Static range-separated direct action

Named molecular RSH references can retain the native provider with empty full
and LR ERI leaves. The existing direct kernel accepts a static nonnegative
`omega` in inverse bohr; positive values select `erf(omega*r12)/r12`.
The shared SCF/response action computes

\[
J_0[D],\qquad K_{\mathrm{eff}}[D]
=a_{\mathrm{SR}}K_0[D]+(a_{\mathrm{LR}}-a_{\mathrm{SR}})K_\omega[D].
\]

Restricted response uses `J0 - Keff/2` for its spin-summed AO perturbation.
Unrestricted response uses `J0[Da+Db] - Keff[Ds]` for each spin. The effective
exchange already contains both coefficients, so it is not multiplied by the
short-range fraction again. RKS incrementally accumulates this same effective
exchange; UKS retains separate spin exchange actions. LR calls request K only.
Positive-omega actions do not reuse the ordinary Coulomb s8 cache.

The integral-level LR regression covers Cartesian/spherical J/K selections,
density and coordinate JVP/VJP, mixed second products and finite differences:
19 tests passed in 8.42 s on CPU. Twelve SCF assembly/algebra tests passed in
4.82 s, including an explicit prohibition on constructing full/LR ERIs.
Five public R/U TDA/TDDFT adapter tests passed in 1.06 s after fixing a guard
that had rejected the canonical unrestricted semilocal adapter.

The complete named functional also needs its semilocal response and any
additional energy contributions. These AO tests alone do not establish
relaxed TD gradients for every recognized RSH name. In particular, attempted
LC-wPBE end-to-end geometry tests encountered compiler resource limits:
restricted whole-pipeline JIT timed out at 600 s, a modular run timed out at
900 s, and unrestricted runs reached the 32 GiB process-group cap. Lowering
the LLVM optimization level was not established as a solution. Those records
are retained as incomplete validations, not numerical passes. The focused
functional-family acceptance uses omegaB97X-D, whose factored local expression
has separate higher-derivative regression coverage.

Geometry AD still requires zero direct screening cutoff. Omega itself and
primitive exponents/coefficients remain unsupported AD inputs. The external
scalar neural-XC contract is unchanged: an initial RSH label does not silently
add a separate exact-exchange term to an arbitrary supplied neural energy.

## Functional-family acceptance (2026-10-21)

All eight full-TDDFT/UTDDFT cases passed through the public driver with
whole-pipeline JIT, relaxed SCF, direct/full comparisons, coordinate JVP/VJP,
central differences and an independent PySCF 2.13.0 reference on the same
moving level-0 quadrature. JAX/jaxlib were 0.10.0, Python 3.11.15, float64,
on `jy_gpu_1` CPU. Each process used 16 logical CPU affinity entries; the
largest concurrent allocation was 64. All GPUs were hidden.

| Family | Functional | Restricted system | Unrestricted system | Max PySCF energy error (Ha) | Max PySCF FD derivative error (Ha/bohr) |
| --- | --- | --- | --- | ---: | ---: |
| LDA | SVWN (VWN5) | H2/STO-3G | HeH doublet/6-31G | 4.99e-11 | 7.22e-7 |
| GGA | PBE | H2/STO-3G | HeH doublet/6-31G | 2.66e-12 | 2.58e-7 |
| Hybrid | PBE0 | H2/STO-3G | HeH doublet/6-31G | 8.62e-12 | 1.56e-7 |
| RSH | omegaB97X-D | H2/STO-3G | H2+ doublet/STO-3G | 9.33e-15 | 2.06e-9 |

The differentiated coordinate is the second atom's z coordinate at 1.6 bohr;
each displaced point rebuilds the grid and solves SCF. The FD step is 2e-4
bohr. Direct/full energy, derivative and JVP/VJP maximum differences across
the eight cases were 1.67e-15 Ha, 1.41e-14 Ha/bohr and 3.61e-15 Ha/bohr.
These are numerical equivalence checks, not basis/grid convergence claims.
Classical SCF uses its array-state scan AD; the separate external-network
tests cover implicit SCF. The TD eigensolver retains its shared implicit
spectral response.

HeH has nonempty occupied/virtual spaces in both spin channels. H2+ has no
occupied beta orbital, so the RSH case does not establish that stronger spin
coverage. Its use avoids unsupported He CHG parameters. The original CHG
dispersion is included in GradSCF's complete recipe; it is a geometry-only
state-independent scalar and does not enter the Fock/TD kernel. Its absence
in the PySCF XC reference does not affect vertical excitation energies or
their derivatives; no CHG total-energy force comparison is claimed here.

The additional focused suites passed 19 LR-integral, 12 RSH-SCF, 5 public
adapter and 129 related regression tests. The latter retain 23 external-neural
tests for fixed-density, explicit and implicit training. All completed phase
records exited 0 with unchanged source hashes; 33 changed runtime source
files match the final remote snapshot. Interrupted/rejected LC-wPBE compiler
candidates remain excluded from these passing counts.

The final test is `tests/test_direct_excitation_geometry.py`, selector
`-k functional_families`. Validation records are retained locally under
`artifacts/response_direct_validation/dabna2/functional_family_results.json`.
The immutable final evidence archive SHA256 is
`65fea049db54dcca612b05433b2068db477e8ba7da803ebfdf5dc7776a158a36`.
