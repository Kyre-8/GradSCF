# Memory budgets and differentiable wavefunction calculations

`max_memory` is a static array-planning budget in decimal MB. Its default is
60000 MB; molecular MP, CI, CC and FCI objects inherit their SCF object's budget
unless explicitly overridden. The budget changes block sizes, optional caches
and solver capacities. It does not change the basis, excitation space, number
of requested states or convergence tolerances.

Start with [direct MP2 geometry response](mp/direct_geometry.py). This ordinary
Python example uses native direct RHF and noncanonical MP2 for N2/STO-3G at an
internal pi-orbital degeneracy. It computes a nuclear gradient and a stretch
Hessian-vector product (HVP); the actual CPU output is included at the end of
the file. All electronic-structure algorithms come from `src/`.

## Choosing the budget and integral representation

```python
from gradscf import gto, scf, mp

mol = gto.M(atom="O 0 0 0; H 0 .757 .587; H 0 -.757 .587",
            basis="6-31g*", cart=True)
mf = scf.RHF(mol, integral_backend="native", jk_backend="direct",
             max_memory=4000., conv_tol=1e-12,
             conv_tol_density=1e-11, conv_tol_grad=1e-10).run()
correlation = mp.MP2(mf).run()  # Inherits 4000 MB.
print(correlation.e_tot)
```

The explicit density and orbital-gradient tolerances make the reference
consistent with canonical MP2's Fock-matrix check. SCF energy convergence alone
does not guarantee the tighter stationarity required by that post-HF entry.

The shared planner reserves 20% for runtime/model overhead, then charges known
resident inputs, required outputs and working arrays. A minimum working set
that cannot fit raises a diagnostic instead of silently dropping orbitals or
switching to a full ERI representation. Existing determinant/element guards
remain additional checks where the corresponding method still requires them.

This is **not an operating-system RSS or VRAM limit**. Compilation, numerical
libraries, allocator reservations and arbitrary outer AD graphs need separate
measurement. CPU RAM and accelerator memory are different resources. A method
can fit its steady-state tensor plan while compilation has a larger peak.

Native direct J/K avoids a saved full AO ERI in SCF. Requested AO-to-MO blocks
then use the shared `gradscf.ao2mo` layer. This does not remove tensors required
by the correlation model: an output T2, a CI vector or a requested full 2-RDM
must still fit. Supplying a full AO/MO tensor explicitly also keeps that input
resident; slicing it later does not recover its preparation cost.

## Current method boundaries

Here `o`, `v` and `a` denote occupied, virtual and active orbital counts. The
table describes the existing real molecular implementations, not periodic
correlation or unrestricted support for every excited-state sector.

| Method or operation | Representation and reuse | Necessary storage or remaining restriction |
| --- | --- | --- |
| Restricted/unrestricted MP2 | Selected occupied-virtual blocks or existing DF factors; aa/ab/bb channels share the budget. | Noncanonical tensor-sum inverses retain O(o²v²) state. `with_t2=False` removes the requested output, not all derivative intermediates. |
| Canonical MP3 | Selected CC integral blocks and the existing residual-JVP expression. | First- and second-order doubles workspaces remain necessary; this is a separate implementation from general Taylor MP. |
| Higher MP | Shared Taylor operations, exact excitation-rank bounds and packed target blocks. | Wavefunction histories still grow with excitation space and order. A mathematically divergent series is not repaired by memory planning. |
| CCS/CCD/CCSD and existing approximate CC/QCI models | Selected blocks; virtual dressing contractions avoid a mandatory full dressed Wvvvv. | Dense bare vvvv remains the ordinary fast path. The explicit native virtual action trades recomputation for storage; it is not universally faster. |
| UCCD/UCCSD | Spatial-spin integral blocks and packed amplitude coordinates; energy and Lambda share a packed energy contraction. | The residual still expands dense spin amplitudes. The energy-callback optimization is not a fully blockwise UCC residual. |
| CCSD(T) | Existing streamed triples panels retain first- and directional second-response support. | Upstream integrals, T2 and response state remain live. Existing third-derivative and variant-specific restrictions still apply. |
| EOM-EE/IP/EA | Matrix-free right/left actions, shared non-Hermitian solvers and factored virtual dressings. | Current prepared EOM is restricted-reference CCSD. Native provider and supported outer-response known-array phases share the budget. Measurements apply to their recorded cases, rather than every EOM sector and device combination. |
| CIS and truncated CI | Selected integral kernels, compressed excitation/string topology and direct Hamiltonian action. | CI coefficients and Davidson state remain necessary. Internal orbital degeneracy and degeneracy of a selected CI root are different problems. |
| FCI in a fixed active space | Core folding and active ERIs; tiled string actions share the solver budget. | An active a⁴ ERI and the chosen full CI space remain required. This is not selected CI or orbital-optimized CAS. |
| RDMs and properties | Requested blocks or contracted observables avoid building an unrelated full spin 2-RDM. | Explicit full-RDM export still pays for that output. A fixed-MO property response is not automatically a relaxed nuclear derivative. |

Conventional dense array entry points remain available as reference paths.
Native selected transforms support moving-coordinate products through second
order; unsupported basis-parameter or higher mixed products must raise rather
than return zero. An older full-integral backend's derivative coverage does not
automatically extend to every streamed operation.

For geometry response, use the relaxed adapter as in
[direct_geometry.py](mp/direct_geometry.py). It includes moving integrals and
the stationary HF response. Differentiating a solver with respect to fixed MO
integrals answers a different question. Noncanonical MP2 avoids differentiation
of individual canonical orbitals inside a degenerate occupied or virtual space;
canonical eigenpair guards remain explicit. CI/EOM root crossings retain the
shared solver's isolated-root or complete-subspace response contract.

## Accelerator placement

For a traced GPU calculation, place the numerical arguments on the selected
concrete device and keep `jax.default_device(device)` around tracing/execution.
Place concrete arrays captured by the function on that target too. The planner
can query that supplied device without initializing unrelated GPUs. A string
such as `"gpu"` does not identify a device whose capacity can be checked.

An outer `jax.jit(..., device=device)` alone does not reliably expose its target
to Python planning code. A CPU array captured by a GPU-compiled function still
reports its original CPU device. This inference boundary remains explicit;
the current checks do not prove a device-memory bound for that ambiguous case.
Checks occur when a plan is built, not on every invocation of a cached executable.

## Reading measurements

Record the fixed geometry, basis, spin, orbital selection, dtype, thresholds,
backend, hardware and exact source alongside each result. Separate preparation,
compilation, first execution, warm execution and complete-process elapsed time.
Report compiler temporary estimates and measured process peaks separately.

The verified CPU MP2 gradient family illustrates the tradeoff. At the fixed
aniline/Cartesian 6-31G* point (119 AO, all 25 occupied and 94 virtual orbitals),
full/direct complete-child peaks were 23529724/5963260 KiB and warm times
178.587/605.247 s. Both passed the same PySCF energy/gradient oracle. Direct used
about 25.34% of the full peak but took 3.389 times the warm time. Water, ethane,
H3, NH and O2 small points instead had slightly higher direct process peaks.
These finite CPU measurements do not establish a GPU speedup or asymptotic law.

Related runnable examples are [restricted CC](cc/restricted_ground.py),
[open-shell CI/CC](cc/open_shell_ground.py), [CI](ci/restricted_ci.py),
[active-space FCI](fci/active_space.py) and [EOM](cc/eom_ccsd.py). The engineering
[acceptance ledger](../src/gradscf/WAVEFUNCTION_MEMORY_PLAN.md) distinguishes
component gates, finite whole-method/device measurements and capability limits.

The following public examples were executed in separate CPU/float64 processes
on 2026-10-24 (AMD EPYC 9654, JAX 0.10.0, native library built on the same node).
Source snapshot `40824fe` plus the documented example edits was verified through
actual import paths and file hashes. Times include compilation and verification;
the RSS column is the complete child-process peak.

| Example | Elapsed time (s) | Peak RSS (KiB) |
| --- | ---: | ---: |
| N2 direct MP2 gradient/HVP | 22.540 | 1317420 |
| Restricted CC hierarchy | 57.224 | 2317768 |
| Restricted CI and CIS(D) | 305.915 | 6439156 |
| LiH active-space FCI | 9.383 | 1234232 |
| H3 UHF/ROHF CI/CC | 106.969 | 3368568 |
| Water EOM-EE/IP/EA | 61.962 | 3654968 |
| RHF/MP2 code block above | 7.529 | 1154292 |

All seven final processes and Slurm steps exited successfully. The N2 results
agree with its output comments within 1.28e-13 Eh, 2.66e-10 Eh/bohr and
2.20e-10 Eh/bohr^2. The code block's canonical Fock error is 2.79e-12, below
MP2's unchanged 1e-8 threshold. The six other examples were retained after the
guide's SCF stopping criteria were tightened. These small examples demonstrate
interfaces and derivative consistency; large-system and GPU validation remain
separate acceptance gates.

On 2026-10-25, the final focused device audit passed 6 CPU and 11 CUDA tests
on one allocated V100-SXM2-32GB (JAX 0.10.0, float64). It covered R/U MP2 and
CC ground/Lambda joint responses, CI actions and contracted observables,
FCI Davidson energy/gradient/HVP, CI energy/coefficient derivatives, and MP4
with exactly degenerate occupied/virtual orbitals. Tests explicitly reject
nonfinite numerical outputs and check GPU placement. These selected cases
are additional evidence, rather than a full-suite or all-method GPU claim.

The existing three-orbital, one-occupied-orbital tensor MP4 benchmark (seed
2318) also ran through the public `mp.run_mp` entry on that V100. With an
explicit outer JIT, lowering took 0.680 s, compilation 4.437 s and first execution
6.610 ms. Five synchronized warm samples ranged from 0.381 to 3.791 ms;
the median was 0.414 ms. XLA reported 4,206,640 temporary bytes. The benchmark
reported peak child RSS of 1,065,900 KiB; one-second device observations reached
1,358 MiB. These fields measure different things, and sampling can miss short
device peaks. This tiny fixed-input point supplies CUDA timing evidence; it
does not establish a GPU speedup or a large-molecule memory limit.

The agreed engineering plan is closed with 34 parent tasks verified and the
optional disk backend explicitly deferred. Conventional dense paths remain
available. Larger required amplitudes, CI spaces, library allocations and
recomputation costs still determine which calculations are practical.
