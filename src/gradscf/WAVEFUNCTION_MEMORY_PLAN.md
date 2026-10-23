# Wavefunction memory optimization: engineering plan and task ledger

Created: 2026-10-21. Baseline: `1a7ee6c7d047d0beefe029a95201be176956e5f8`.
Branch: `perf/wavefunction-memory`. Worktree: `artifacts/side/wavefunction-memory`.

This is the engineering plan for **all existing molecular wavefunction methods**,
not an MP2-only optimization. MP2 geometry response is the first end-to-end
integration target because it exposes the shared integral and AD bottleneck with
the fewest method-specific intermediates. Completing that target does not close
this project. Checkboxes below mean the stated acceptance gates have passed;
source changes without numerical verification remain unchecked.

## 1. Objective, scope, and success criteria

Reduce avoidable memory growth in integral preparation, correlation kernels,
iterative solvers, properties, and geometry differentiation while preserving the
method, orbital space, convergence thresholds, and derivative definition.

In scope:

- RHF/UHF-based MP2, MP3 and recursive higher-order MP;
- CIS/CISD and other existing truncated-CI branches, their existing perturbative
  corrections, and FCI in a specified active space;
- CCS, CCD, CCSD, existing approximate/linearized CC and QCI branches, restricted
  and unrestricted implementations, and existing triples corrections;
- EOM-EE/IP/EA, left states, Lambda equations, RDMs and contracted properties;
- the shared moving-molecule/`CorrelatedEnergy` adapters, SCF orbital response,
  integral JVP/VJP, directional second derivatives, and supported mixed responses;
- byte budgets, cache lifetime, CPU/GPU execution and optional disk storage.

Retain existing ROHF-to-spin-reference entry points in CI/UCC and their explicit
restrictions. The first moving-geometry slice is RHF/UHF; it does not silently
add ROHF geometry response or relax UCIS's stationary-UHF requirement.

GW/BSE, periodic correlation, OFDFT, new correlation models, selected-CI,
local-correlation approximations and frozen virtuals are outside this project.
Shared infrastructure may be reused by those modules later. Existing DF is a
supported representation, not a compulsory replacement for exact non-DF paths.

Success requires all of the following for each migrated path:

1. Equivalent forward results and declared derivatives on fixed small inputs.
2. No unnecessary full AO/MO/spin ERI or hidden full ERI cotangent on that path.
3. A budget covering necessary resident state, live intermediates, solver state
   and AD workspace, with an explicit error when the minimum cannot fit.
4. Measured peak memory and warm execution time on a reproducible scale test.
5. Compatible public APIs, documented capability limits and an explicit dense
   reference path for validation. No silent dense fallback after a budget error.

A budget is a planning limit, not an operating-system RSS guarantee. Native
libraries, XLA compilation and allocator reservations must be measured separately.
No claim of a universal speedup or of fitting an arbitrary molecule in 60 GB is
made. Necessary T2, CI-vector and requested full-RDM storage remains necessary.

## 2. Baseline and evidence boundaries

The user-provided review is
`/Users/jiaoyuan/Desktop/GradSCF_tensor_memory_optimization_review_2026-10-21.md`.
It is source analysis, not a common-hardware benchmark. The reported **1.56 TB**
for an aniline MP2 geometry graph is a compiler allocation estimate for a specific
snapshot, not measured RSS and not a result for all wavefunction methods. Before
reusing that number, recover the precise input, checkpoint, compiler field and
source hash; otherwise label the new measurement a new baseline.

Confirmed baseline behavior, checked against committed source:

| Area | Existing implementation to retain | Remaining memory issue |
| --- | --- | --- |
| Moving molecular energy | Shared stationary orbital solver and implicit SCF response; moving overlap transport | Adapter requires full J/K; HF assembly evaluates full moving ERI |
| Correlated geometry | Noncanonical MP2 default; implicit correlation/SCF dependencies | Full AO to full MO conversion inside the differentiated map |
| Ordinary MP2 | OV-only integral preparation; canonical per-occupied contraction and rematerialization | Direct adapter can materialize AO-pair integrals; noncanonical amplitude solve still owns O(o²v²) state |
| Higher MP | Shared residual nodes and Taylor solver; packed excitation blocks | Expanded contraction blocks, all-target action and coefficient/history lifetime |
| RCC | Method-specific integral slices, shared implicit nonlinear solver and Lambda | Full upstream MO ERI, vvvv and dressed intermediates coexist |
| UCC | Spin-amplitude organization | Explicit `(2*nmo)^4` spin ERI, then active-space extraction |
| Triples | Existing streamed first/second-derivative path | Panel limit does not budget all resident data and AD workspace |
| EOM | Operator actions, shared non-Hermitian solvers | Ground-state integral cost, closure lifetime and sector-specific W intermediates |
| CI | Compressed particle/hole strings and direct Hamiltonian action | Kernel/RHS batches, integral preparation and RDM/cache workspace |
| FCI | Alpha/beta strings; active-space reference/core folding | Stacked E_pq C and weighted arrays proportional to nact²*Ndet*nvec |
| Properties | Existing Lambda and CI/RDM definitions | Full spin 2-RDM and multiple large bra/ket intermediates for small observables |

For UCC, `(2*nmo)^4` contains 16 times as many elements as a spatial `nmo^4`
array. This does not establish a 16-fold process peak. At DABNA-2's previously
discussed o=194, v=866, one full float64 o²v² T2 array alone is about 225.8 GB
(decimal). This project cannot make that all-orbital CCSD state fit a 60 GB
in-memory budget merely by optimizing caches.

Source anchors are relative to the worktree:

- `src/gradscf_tools/{molecular_response,correlated_response}.py`;
- `src/gradscf/scf/{reference,orbital_optimization}.py`;
- `src/gradscf/integrals/plan.py` and `src/gradscf/ao2mo/`;
- `src/gradscf/integrals/backends/native/{jk.py,autodiff/geometry.py}`;
- `src/gradscf/cc/{integrals,_equations,_intermediates,uccsd,triples,properties}.py`;
- `src/gradscf/cc/eom/{operators,_charged}.py`;
- `src/gradscf/{ci/direct.py,fci/hamiltonian.py,mp/mp2.py,mp/_tensor.py}`.

Inspect the local PySCF AO2MO/MP2/CC/FCI implementations before each algorithmic
change; record the exact version/hash in that task's evidence. PySCFAD is an
additional AD comparison, not a substitute for GradSCF derivative tests. Translate
mutable buffers and HDF5 use into explicit JAX dependencies rather than copying
them literally. No new external numerical result is asserted by this document.

Named branch coverage (a shared kernel pass does not imply every branch passes):

| Existing branch | Responsible tasks | Required distinction |
| --- | --- | --- |
| MP2/RMP2, UMP2, noncanonical MP2 | WF-05/06/07 | aa/ab/bb, resolvent response, with/without amplitudes |
| Standalone canonical MP3/RMP3 | WF-33 | CC-residual-JVP implementation, independently of series |
| MP series and tensor recursive backend | WF-23/24/25/26 | Per-order coefficients and actually exercised excitation ranks |
| CIS/UCIS, CID/UCID, CISD/UCISD | WF-18/20/22 | Noncontiguous CID ranks (0,2) and unequal-spin occupations |
| CISDT/UCISDT, CISDTQ/UCISDTQ, generic excitation_ranks | WF-18/20/22 | Rank-specific small oracle, no claim that CISD covers higher ranks |
| Existing CIS(D): CIS_D/cis_d_correction | WF-22 | Reference-state/denominator definition and existing guards |
| CCS/CCD/CCSD and UCCD/UCCSD | WF-09/10/12/13/15 | Restricted/spatial-spin conventions and frozen spaces |
| Existing CC2/linearized/QCI branches | WF-11 | Enumerate exported names at implementation; separate equations/tests |
| ccsd(t), ccsd+t(ccsd), qcisd(t) | WF-14/15 | Latter two currently reject semicanonical/streamed full-Fock; preserve rejection |
| EOM-EE/IP/EA and left states | WF-16/17 | Sector-specific intermediates and root-response contract |
| IP/EA Dyson orbitals/pole weights | WF-34 | Endpoint workspace, left/right columns and vector-response guard |
| FCI and active/core folding | WF-19/21/22/27 | Fixed CAS, spin sector, RDM and derivative conventions |
| Existing ROHF spin-reference conversion | WF-12/13/18/22 | Common spatial frame, occupation split, current UCIS/geometry restrictions |

The branch inventory is rechecked against public exports before changing each
family. Named CISD(T) and CISDT(Q) corrections are not exported at this baseline
and are not new-model deliverables of this project. No new theory variant is
introduced merely to fill this table.

## 3. Shared design and invariants

### 3.1 Representation and ownership

Keep the existing public dense reference arrays compatible. Do not change an
existing `.eri` array into an opaque object without an explicit new entry point.
Maintain the shared transformations in the dedicated `gradscf.ao2mo` package;
its block, native, reference, spin and response modules own distinct operations.
Reuse the native integral plan for integral generation. Avoid separate MP/CI/CC implementations of AO2MO,
byte accounting, response solvers or integral derivatives.

Every input representation must identify:

- static AO topology, integral symmetry/convention and spin convention;
- dynamic basis coordinates/parameters and MO coefficients;
- occupied/virtual/active/core index sets, dtype and target device;
- available blocks/actions and derivative order, with explicit unsupported cases;
- resident bytes, estimated working bytes, cache owner and release point.

Dense, pair-packed, DF and native-direct representations may supply the same
requested block or contraction. A consumer requests OVOV or vvvv*T, not a full
ERI followed by a slice. Required active-space ERIs and outputs are legitimate
resident data and must still be charged to the budget.

For native non-DF AO2MO, use shell blocks and reusable half transformations or
batched contractions. A separate full shell traversal for every MO pair is an
exact small reference implementation, not the production default. Reuse existing
direct J/K for the SCF part; it does not by itself implement an arbitrary
correlated two-particle-density contraction.

### 3.2 AD contract

For I(C,R), each block/action must propagate both the MO-coefficient and basis
coordinate dependence. In reverse mode, contract the incoming block cotangent
inside the block/shell traversal, accumulating coefficient and coordinate
cotangents. Do not scatter it into a full AO four-index cotangent first. In
forward mode, differentiate the same block action rather than assembling the
entire `(3*Natom,nao,nao,nao,nao)` derivative tensor.

Preserve the existing nested implicit equations. For SCF residual s(kappa,R)=0
and correlation residual r(t,kappa,R)=0,

    L = E + lambda^T r + z^T s
    r_t^T lambda = -E_t
    s_kappa^T z = -(E_kappa + r_kappa^T lambda)
    dE/dR = partial_R L.

The existing shared implicit solvers generate these adjoints. A new integral
provider must not add another manual Lambda/Z correction on top. CC energy is
not stationary with respect to its amplitudes; the HF stationary-value shortcut
cannot be applied to it. A CCSD(T) observable needs the triples contribution to
amplitude/orbital response; a plain CCSD Lambda/RDM is insufficient.

Internal occupied or virtual degeneracies must not force differentiation of
individual canonical eigenvectors. Preserve noncanonical resolvents and orbital
gauge covariance. Root degeneracy of a CI/EOM state is a different issue: retain
the shared solver's complete-subspace/invariant-observable contract. Do not
regularize physical singularities or modify denominators silently.

Initial derivative targets are forward, JVP, VJP and directional HVP on real
RHF/UHF molecular paths. Full Cartesian Hessian output is optional and costs
O((3N)²). Supported higher mixed-response paths remain regression gates; they
must not be disabled globally because one new streamed backend supports only
order two. Basis exponent/coefficient derivatives require their own capability
checks. Never return zero for a missing integral derivative.

### 3.3 Memory budget and lifecycle

Reuse `_memory.py` and the existing `max_memory=60000` MB convention (decimal MB).
Correlation objects inherit the SCF budget unless overridden. Extend the existing
public configuration minimally after inspecting each facade; do not add competing
`workspace_mb`, `memory_limit` and `cache_size` user knobs.

The planner accounts for simultaneous live storage:

    available workspace = effective device budget
                        - runtime reserve
                        - necessary resident data
                        - live solver and AD state
                        - retained caches.

Choose integral tiles, RHS batches, contraction batches and Krylov/DIIS capacities
from the remainder. Preserve convergence and request counts; reduce optional cache
and batches before rejecting a computation. A minimum that cannot fit must fail
early with the necessary-state size and limiting object. Existing element-count
guards remain safety checks until replaced by equivalent byte-aware checks.

Host RAM and device VRAM are distinct. Charge transfers, staging and overlapping
buffers explicitly. Do not initialize an unrelated GPU to inspect memory. Cache
static topology/plans freely within a bound; numeric caches are call-scoped or
versioned and budgeted. Do not persist JAX tracers, detach moving data, or copy
whole device arrays to host merely to hash cache keys. Use rematerialization only
where its measured recomputation cost is acceptable. Mark the last consumer of
large arrays so forward and backward do not retain them accidentally.

Out-of-core storage is a later optional backend. File I/O alone is not an AD
rule. Distinguish host staging, disk reads and device contraction; provide
derivative-aware recomputation or a separately tested boundary before enabling it
on geometry/parameter AD paths.

## 4. Execution order and dependencies

```text
WF-00 baseline/worktree/plan
  -> WF-01 direct stationary SCF input chain
  -> WF-02 shared budget + block contract
  -> WF-03 native partial AO2MO -> WF-04 block derivatives
      -> WF-05/06 R/U MP2 -> WF-07 CorrelatedEnergy geometry + HVP
          -> WF-08 first measured scaling gate
      -> WF-09/10 RCC blocks and actions -> WF-11 approximate branches
                                       -> WF-33 standalone canonical MP3
      -> WF-12/13 UCC spin blocks and residual
          -> WF-14 triples -> WF-15 Lambda/contracted properties
          -> WF-16/17 EOM EE/IP/EA -> WF-34 Dyson properties
      -> WF-18/19 CI/FCI preparation -> WF-20/21 action workspace
          -> WF-22 CI/FCI derivatives and properties
      -> WF-23/24/25 higher-MP target tiles and history -> WF-26 MPn AD
  -> WF-27 full-RDM export -> WF-28 optional disk backend
  -> WF-29 solver budget integration -> WF-30 CPU/GPU integration
  -> WF-31 public examples -> WF-32 final regression/scale/API audit
```

Budget integration begins in WF-02; WF-29 removes the remaining cross-module
inconsistencies after real consumers establish their requirements. Dense small
oracles stay available throughout. Independent read-only reviews can run in
parallel; edits to coupled numerical kernels are integrated one task at a time.
Do not skip directly to a large molecule before a slice's forward/AD gates pass.

## 5. TODO list and task acceptance

Status vocabulary: `pending`, `in_progress`, `implemented_unverified`, `verified`,
`blocked`, `deferred`. A check requires the verification specified for that task,
plus a plan review entry. A narrower completed subtask cannot close a larger task.

Planning constraint (2026-10-22): the existing WF-00 through WF-34 are the fixed
top-level scope. Each TODO may have **one child level and at most four children**.
No grandchildren or new top-level tasks may be added to work around this limit.
Implementation steps, bug fixes, test shards and retry attempts belong to their
existing TODO and evidence record; they are not new plan branches. Consolidate
related work before starting it and retain a finite acceptance condition.
Evidence may satisfy several related integration gates without repeating the
implementation or the same test campaign. Historical checkpoint labels in
section 8 are provenance, not additional active TODO levels.

Current acceptance snapshot (2026-10-24): twenty of the thirty-five parent tasks
are fully accepted (WF-00/01/03/04/05/07/08/09/18/19/20/21/22/23/24/25/26/27/33/34); thirty-two of the thirty-seven existing
children are checked. These are acceptance counts, not a percentage estimate of
engineering effort. Keep all other parent checkboxes open. In particular,
successful component derivatives do not establish a whole-method memory bound.

| Existing scope | Accepted frontier | Remaining acceptance boundary |
| --- | --- | --- |
| WF-02/03/04/05/06/08 | Shared AO2MO/DF factors, native block response and measured consumer conversions, restricted MP2 integral/energy budgets, R/U full/direct gradient family and aniline oracle | Unrestricted saved response-state accounting, cross-caller ownership and traced target-device capacity; direct remains opt-in |
| WF-10/11/12/13/14/15 | Restricted/spin virtual dressing actions; selected UHF facade, Lambda/RDM1, requested/contracted RDM2 and nonzero streamed-(T) geometry; five-block CCS and direct physical-coordinate boundaries | Efficient bare-vvvv policy, packed residual lifetime, complete phase budgets and nuclear two-body-property response |
| WF-16/17/34 | Shared EOM primal and isolated-pair response planners, factored EA virtual actions, native prepared EE/IP/EA, streamed Dyson, retained-state accounting and matched component measurements | Remaining native phase budgets and whole-method measurements |
| WF-18/19/20/21/22/27 | Selected CI through genuine U quadruples, UCIS/CIS(D) budgets and ROHF guards, joint FCI action/response budgets, scalar spin/spatial properties and relaxed CI/FCI/CIS(D) nuclear response | Final GPU and large-method measurements; arbitrary outer-AD/allocator bounds remain outside the known-array contract |
| WF-23/24/25/26/33 | Exact target bounds, packed canonical actions including actual sextuples, shared Taylor history/caller budgets, selected eager dispatch, standalone MP3 math/API/warm gates, pure-frame R/U MPn and moving native MP4 geometry second response | Complete cross-caller AD ownership and higher-order method-level size/performance acceptance |
| WF-29 | Hermitian/non-Hermitian primal, isolated-pair response, SCF implicit response and common linear/nonlinear phase planners; explicit CI and CC caller phases | Remaining cross-consumer ownership and whole-method reconciliation |
| WF-28/30/31/32 | Existing small oracles and component device checks remain usable | Evidence-based disk decision, final CPU/GPU audit, public examples/capability table and cross-family regression/scaling |

The damaged jy_gpu_1 stages remain paused. Allocations103505/n002 and104081/n006
are no longer active. The latest renewal is104443/n006: one RTX5090,64 logical
CPUs,512 GiB and48 hours, ending2026-10-25 21:29:47 Asia/Shanghai. The actual
affinity is96-127,288-319. Slurm assigns GPU1, remapped to process-visible GPU0,
UUID GPU-7aa3e8aa-431b-da04-7ba3-5c37ae5479d0; GPU6 remains forbidden.
The float64 CUDA/JIT/AD assertions completed and were read back from their JSON.
The SSH control connection exited255, so its exact srun exit was not independently
captured. This establishes readiness, not a numerical-method acceptance gate.
Keep pending implementations separate from accepted evidence.
Do not reuse damaged stages, c20 credentials or its compiled native library.

### Foundation and first complete geometry slice

- [x] **WF-00 — Isolated baseline, inventory and engineering plan.** Create the
  new worktree from committed code, preserve main WIP, document all methods and
  dependencies, define acceptance/evidence. Acceptance: Git/source inspection,
  plan completeness review; no numerical or performance claim.
- [x] **WF-01 — Direct stationary SCF input chain** (`verified`). Pass existing
  native `direct_basis` through the shared orbital optimizer with no dense ERI;
  support moving RHF/UHF HF inputs in the molecular adapter. Reuse native J/K and
  existing implicit response. Acceptance: direct/full energy, density response,
  geometry JVP/VJP and directional second derivatives on tiny R/U systems;
  fail-on-full-ERI instrumentation; unchanged dense-path regression. RSH/custom
  XC expansion is not required to close this HF task and must remain guarded.
- [ ] **WF-02 — Shared block and byte-budget contract** (`in_progress`). Add only the metadata,
  block/action requests, accounting and ownership required by MP2 and CC.
  Acceptance: dense/DF request equivalence, dtype/device retention, invalid
  representation errors, cache lifetime and minimum-memory diagnostics. Include
  an early target-device capability check; full device validation remains WF-30.

  - [x] **WF-02a:** budget the existing array-block entry point and bound DF
    auxiliary intermediates; validate nested AD and CPU/GPU7 execution.
  - [x] **WF-02b:** wire caller budgets/resident-state ownership into the MP/CC
    consumers as WF-05/09 migrate, rather than using the block default everywhere.
    Facade inheritance/overrides, known caller ownership and the explicit
    canonical orbital-solve relay are verified. Cross-channel saved AD state
    and downstream phase integration remain with their specific consumer gates.
    Historical checkpoint: WF-02b1.
  - [ ] **WF-02c:** complete the native block/capability contract with WF-03/04,
    including target-device capacity when JIT tracing hides concrete devices.
    Concrete default-device inheritance and ambiguous accelerator-label guards
    are verified. General outer-JIT target inference with captured CPU arrays
    is still unsupported; do not count a context-only test as that capability.
  - [x] **WF-02d:** establish the dedicated AO2MO domain package and migrate
    existing callers without redundant forwarding files; share byte-reservation
    validation in `_memory` without an integral-to-solver dependency or a second
    validator. Includes the verified historical WF-02e checkpoint.
- [x] **WF-03 — Exact native partial AO2MO** (`verified`). Generate requested MO tiles without
  full AO-pair or full MO ERI. Reuse shell traversal and half transformations.
  Acceptance: asymmetric coefficient sets, permutation symmetry, packed/dense
  oracle, frozen/core indexing, measured conversion time and tile workspace.

  - [x] **WF-03a:** native static shell-slice producer using the pinned shell
    fill driver, without a full AO tensor or a new integral engine.
  - [x] **WF-03b:** bounded compact half transformations with dynamic shell
    endpoints/static padded storage, coefficient response and coordinate products
    through order two on CPU/GPU7. Includes historical WF-03b1/WF-03b2a.
  - [x] **WF-03c:** multiple requests share traversal and first/second/half
    transformations; joint output/resident reservations, physical/native versus
    raw array charts, and single-request/multiple-request efficiency verified.
    Includes historical WF-03b2b1.
  - [x] **WF-03d:** integrate the shared transformations with the method
    consumers and measure conversion/scaling workspace and time. Reuse the
    corresponding WF-05/06/09/12/18/19 integration evidence; do not create a
    separate migration or benchmark campaign for the same consumer.
- [x] **WF-04 — Block integral JVP/VJP and HVP** (`verified`). Add coefficient/coordinate
  derivatives and contracted reverse accumulation. Acceptance: tangent/cotangent
  duality, dense AD, coordinate finite difference and HVP; structural/runtime
  absence of full ERI cotangent; explicit unsupported basis/mixed-order guards.

  - [x] **WF-04a:** local slab cotangents and coordinate products through order
    two, reusing the existing geometry primitive family; CPU and bounded GPU
    bridge verification. Preserve the existing full-integral higher products.
- [x] **WF-05 — Restricted MP2 integral/energy path** (`verified`). Connect the shared provider
  to OVOV/OV-factor consumption, preserving with/without-T2 behavior and public
  conventions. Acceptance: PySCF and dense GradSCF energy/T2, no full AO/MO
  materialization, non-DF and existing DF tests; byte-accounted resident T2.

  - [x] **WF-05a:** native direct SCF-to-OVOV and pure orbital preparation,
    existing dense/s4/DF compatibility and construction preflight.
  - [x] **WF-05b:** joint block/T2/solver ownership and measured method scaling.
- [ ] **WF-06 — Unrestricted and noncanonical MP2.** Supply aa/ab/bb channels
  without spin ERI, retaining the shared tensor-sum resolvent. Acceptance:
  closed/open-shell equality as appropriate, occupied/virtual rotation covariance,
  strict internal degeneracy, residual tolerance, JVP/VJP/HVP. Streaming the
  noncanonical inverse itself is a separate optimization if necessary-state
  storage still fits poorly; do not relabel an unchanged O(o²v²) solve as streamed.

  - [x] **WF-06a:** native aa/ab/bb preparation from saved UHF Focks,
    noncanonical rotation and coefficient JVP/VJP/HVP, mixed dtype regressions.
  - [ ] **WF-06b:** whole-caller spin/resolvent/AD residency and scaling.
- [x] **WF-07 — CorrelatedEnergy MP2 geometry integration.** Remove its obligatory
  full-MO conversion for R/U MP2. Keep moving S/h/J/K, stationary SCF, live C and
  correlation response. Acceptance: energy, full gradient, HVP and one small
  mixed perturbation against dense AD/finite differences; N2 pi-degeneracy test;
  direct geometry input must never call full ERI evaluation.
- [x] **WF-08 — First measured memory/scaling gate** (`verified`). Fixed water/aniline input
  and a modest increasing-size family, same basis/orbital space/tolerances.
  Acceptance: compiler estimates and observed RSS separated, compile/warm/full
  timing, byte accounting reconciled, no regression hidden by a changed model.
  Structural gate: the migrated path must contain no full AO/MO ERI or full ERI
  cotangent. On a case dominated by that old allocation, target at least 2x lower
  measured peak; a miss remains an open performance finding, not a numerical
  failure. Record warm-time cost, and investigate >20% regression before enabling
  a default unless the measured memory/time tradeoff is explicitly justified.

  - [x] **WF-08a:** repair the physical SCF-to-MO Fock transport exposed by
    aniline, preserving noncanonical response and explicit invalid-input guards.
    Small-system geometry/HVP checks close this correctness prerequisite;
    measured full-gradient/scaling performance and larger oracles remain open.
  - [x] **WF-08b:** complete the fixed-aniline selected-MP2 full-input geometry
    gradient, one warm execution and independent same-basis PySCF comparison.
    Keep the dense-AO input lineage explicit; direct input and increasing-size
    scaling plus total-budget closure are still separate acceptance gates.

### Coupled cluster, perturbative corrections and EOM

- [x] **WF-09 — RCC integral preparation** (`verified`). Request only required blocks; make
  orbital counts independent of `vvvv.shape`. Acceptance: frozen/core indexing,
  dense/DF/direct small equivalence, no transient full-MO conversion.

  - [x] **WF-09a:** remove the extra full active-space ERI copy in the
    existing MO-array path; derive virtual counts from the Fock dimension.
  - [x] **WF-09b:** pure live-C/Fock/reference-energy preparation into the
    existing seven-block container, one shared batch and one numerical ground
    body reused by the MO entry. Bare-vvvv storage stays explicit. Includes
    the verified historical WF-09b1 checkpoint.
  - [x] **WF-09c** (`verified`): prepared Lambda, properties, triples and
    EOM reuse one snapshot. Preserve their model/state/frozen checks and raw-MO
    APIs; the original dense RDM2/EA algorithms and provider guards remain.
    Historical implementation label: WF-09b2.
  - [x] **WF-09d** (`verified`): switch the eager SCF facade and relaxed molecular adapter
    after prepared consumers pass; validate full-space metadata, freshness,
    properties and absence of hidden complete-MO materialization.
- [ ] **WF-10 — RCC vvvv and intermediate actions.** Factor bare-vvvv contraction
  from dressing; tile tau/W consumers and release/recompute by lifetime.
  Acceptance: residual and energy term comparison, converged CCD/CCSD, JVP/VJP,
  Lambda, HVP and measured action workspace; no hidden dressed-vvvv cache.

  - [x] **WF-10a:** shared restricted virtual dressing action, with a static
    smaller-scratch choice for occupied-dominated spaces. Algebra/AD and action
    measurements include CC2's complete singles-pair construction. Retain
    factored inputs when expanding their outer product would cost more.
  - [ ] **WF-10b** (`in_progress`): remove the bare-vvvv residency through
    shared AO-source actions, joint amplitude budgets and measured converged
    calculations. The explicit native action passes component correctness;
    measured speed regressions keep the performance gate open and dense default.
- [ ] **WF-11 — Existing approximate CC/QCI branches.** Inspect separate CCS,
  CC2/linearized/QCI pathways and migrate applicable consumers. Acceptance:
  branch-specific energy/residual/derivative regressions; explicit exceptions
  rather than asserting that CCSD migration automatically covers every branch.

  - [x] **WF-11a:** restricted virtual consumers and raw first/second response
    in CCS, CCD, LCCD, LCCSD, CC2, CCSD and QCI; their existing model equations
    remain distinct. This does not close their AO-source or total-budget work.
- [x] **WF-33 — Standalone canonical MP3** (`verified`). Migrate its CC-integral preparation
  and residual-JVP implementation separately from recursive MP. Acceptance:
  E2/E3, with/without-T2, frozen orbitals, JVP/VJP/HVP, series-order-three cross
  comparison, and byte-accounted simultaneous first/second doubles workspace.
- [ ] **WF-12 — UCC spatial-spin integral blocks.** Remove full `(2*nmo)^4`
  construction, map allowed aa/ab/bb blocks lazily. Acceptance: signs, factors,
  antisymmetry, unequal occupations, frozen orbitals, existing ROHF conversion
  and residual parity.

  - [x] **WF-12a:** assemble the six existing UCC blocks directly from spatial
    spin sectors, eliminating full total/active spin ERIs.
  - [ ] **WF-12b** (`in_progress`): AO-source selected blocks, byte ownership
    and scale evidence. The pure producer, UHF facade and relaxed CCSD/(T)
    integration are verified; spin-vvvv/amplitude work, whole-phase byte ownership
    and shared method scaling remain with WF-13/14/15/29/32.
- [ ] **WF-13 — UCC residual/amplitude workspace** (`in_progress`). Reduce repeated dense
  spin-amplitude expansion using block consumers. Acceptance: UCCD/UCCSD forward,
  adjoints and geometry HVP, restricted-limit comparison; packed-coordinate
  normalization and dual factors preserved.
  The packed energy callback is verified and shared with Lambda; the residual
  still expands spin T2, so its existing budget and this parent remain open.
- [ ] **WF-14 — Triples total budget and AD integration.** Retain streamed (T)
  first/second derivatives; charge upstream integrals/T2/panels/adjoints and
  choose panels accordingly. Acceptance: canonical/semicanonical, off-diagonal
  Fock response and internal degeneracy; CCSD(T) gradient/HVP; explicit third-
  derivative guard only where that algorithm lacks it. Audit `ccsd+t(ccsd)` and
  `qcisd(t)` separately; preserve their current semicanonical/streamed rejection
  rather than implying an unavailable variant is now supported.
- [ ] **WF-15 — Lambda and contracted CC properties.** Reuse residual VJP/shared
  linear solver; compute contracted density/property contributions without full
  spin 2-RDM. Acceptance: one-/two-body observables, nonvariational response and
  geometry AD; include correction-specific response where part of the energy.
- [ ] **WF-16 — EOM-EE operator lifetime.** Reuse ground-state blocks, limit
  `linearize` closure residency and RHS batches. Acceptance: left/right actions,
  singlet/triplet where supported, roots and biorthogonality, derivatives and
  invariant clustered-root observables; no full EOM Hamiltonian.
- [ ] **WF-17 — EOM-IP/EA intermediates.** Audit Wvvvv*r2 and Wvvvv*t1/Wvvvo
  consumers together. Acceptance: each charged sector's right/left roots,
  residuals and gradients; batching and restart memory under the same budget.
- [x] **WF-34 — EOM Dyson properties.** Contract orbital/root tiles without
  stacking all `(o+v)*EOM-space` endpoints. Acceptance: IP/EA left/right Dyson
  columns, signed pole weights, norms and existing enabled/disabled vector-
  response guards; measured endpoint lifetime and common byte-budget coverage.

### CI and FCI

- [x] **WF-18 — Truncated-CI integral preparation** (`verified`). Supply needed blocks through
  the shared layer and retain compressed excitation/string topology.
  Acceptance: CIS/CID/CISD/CISDT/CISDTQ, unrestricted counterparts, generic
  `excitation_ranks`, open-shell/frozen cases, dense oracle and no determinant
  connection-table expansion on the scalable path. Exercise noncontiguous CID
  ranks independently from contiguous CISD.
  Retain ROHF common-frame conversion and explicitly test any branch that must
  reject a nonstationary ROHF occupied-virtual block.
- [x] **WF-19 — FCI active-reference preparation.** Retain core folding and only
  necessary active ERIs; remove upstream full AO-pair materialization.
  Acceptance: h_eff/E_core/ERI equality, active/frozen mapping, coefficient and
  geometry derivatives. A dense nact^4 input can remain a budgeted small-CAS
  choice; it is not a full total-MO tensor.

  - [x] **WF-19a:** native/projected CAS ERIs and AO core folding, fixed-frame
    coefficient/geometry products through two and allocation preflights.
  - [x] **WF-19b:** all reference representations and joint consumer budgets,
    verified for the stated real common-spatial CPU/known-array contract.
    Reuse representation/core-fold layers and the native relaxed closure;
    CUDA geometry and large-CAS scaling remain WF-30/32.
- [x] **WF-20 — CI action/cache budget** (`verified`). Replace fixed creator/spectator/RHS
  batches with byte planning and bound numerical/RDM caches by bytes.
  Acceptance: direct action/matmat parity, Davidson forward/response, no all-pairs
  connection build; repeated-call cache release and blockwise JVP/VJP.

  - [x] **WF-20a** (`verified`): replace Python-expanded multi-RHS batches
    with the existing shared compact column loop; preserve empty/small batches
    and validate values, coefficient/integral response and measured warm-time/
    workspace tradeoff. The later WF-20 closure verifies byte-selected widths
    and static-cache ownership; whole-method scaling remains separate.
- [x] **WF-21 — FCI Hamiltonian action tiles** (`verified`). Tile gather -> contraction ->
  transpose scatter instead of full E_pq*C stacks, rematerializing reverse tiles.
  Acceptance: alpha/beta signs, rectangular spin sectors, multiple RHS, transpose
  action and HVP; extra workspace scales with tile size, while input/output CI
  vectors and iterative subspace remain explicitly charged.

  - [x] **WF-21a:** alpha/beta/RHS Hamiltonian tiles, unit AD parity and
    measured action/reverse-residual memory and warm-time tradeoff.
  - [x] **WF-21b:** public facade preparation/action budgets and tiled
    capacity checks.
  - [x] **WF-21c:** joint solver state, operator workspace and AD budget.
    Current shared-owner gradient/HVP and insufficient-response-budget gates
    complete the known-array contract; compilation/allocator residency is separate.
    The prior WF-21b1/WF-21b2 labels remain only in their evidence records.
- [x] **WF-22 — CI/FCI properties and geometry integration** (`verified`). Stream bra/ket
  contractions and existing perturbative-CI corrections where applicable.
  Acceptance: energy/RDM identities and observable contractions, geometry
  gradient/HVP, root-cluster guards; necessary full-CI state size reported early.

  - [x] **WF-22a:** tiled FCI one-/two-RDM, transition and spin contractions,
    public budget propagation and measured first-reverse lifetime.
  - [x] **WF-22b:** CI perturbative properties and relaxed geometry integration.

### Recursive MP

- [x] **WF-23 — Exact target-rank restriction** (`verified`). Avoid residual target blocks
  that do not enter the requested Taylor/Wigner energy coefficient. Acceptance:
  per-order E2...E12 against current recursion, signs, open-shell/frozen cases;
  record which excitation ranks each case actually exercises (a two-electron
  high-order test cannot verify triples or quadruples contraction blocks);
  do not truncate required input waves or change the perturbation partition.

  - [x] **WF-23a:** apply the existing direct-path Taylor/Wigner target bound
    to canonical tensor actions before contraction. Retain every required
    wave coefficient; compare projected actions/response and complete energy
    series, including actual triples/quadruples/sextuples and frozen spins.
- [x] **WF-24 — Packed target-tile contractions** (`verified`). Keep increasing-index packed
  states and expand only contraction tiles. Acceptance: each excitation-block
  action against the current oracle, peak live buffers, JVP/VJP/HVP and costs.
  Dense contractions remain the explicit fast path when their complete workspace
  fits; the capacity fallback preserves the same required packed space.
- [x] **WF-25 — Taylor history lifetime/checkpoints** (`verified`). Share each W*Psi_j once,
  schedule waves/energies by last use and choose checkpoint/recompute policy.
  Acceptance: all Taylor coefficients and derivatives, with_coefficients True/
  False, no illegal removal of history required by later orders; measured time/
  memory tradeoff. Reuse the normalized Taylor primitive in
  `solvers/nonlinear/taylor.py`; MP retains its physical residual and Wigner-energy
  recurrence. Do not add a second engine.

  - [x] **WF-25a** (`verified`): stable whole-recurrence compilation at the
    public series entry, with physical inputs and tolerance dtypes left live.
    Measure cold and warm costs separately; preserve old coefficient, raw-input
    and nested-response gates. This removes eager dispatch fragmentation, not
    combinatorial state storage or the subsequent history-lifetime work.
- [x] **WF-26 — MPn geometry integration.** Connect blocks and lifetime policy to
  moving noncanonical R/U references. Acceptance: order-specific first/second
  responses and internal-orbital-degeneracy cases, justified capacity errors;
  mathematical divergence of an MP series must not be hidden by this optimization.
  Verified frontier: selected pure R/U frames with internal degeneracy and
  native relaxed RHF-series/UHF-tensor MP4, including two-step HVP FD checks
  and early capacity rejection (checkpoint6f8d70e). This is not a claim of
  convergence of arbitrary MP orders or the separate WF-08/32 size/time gates.

### Properties, storage, public integration and release gates

- [x] **WF-27 — Explicit full-RDM export** (`verified`). Provide budgeted block export or
  direct contraction; retain requested dense output for small cases. Acceptance:
  traces, symmetries, energies, normalization and block reconstruction; avoid
  creating full spin 2-RDM merely to extract spatial blocks.
- [ ] **WF-28 — Optional out-of-core backend.** Implement only after measured
  consumers justify it; specify storage dtype/chunking/version/lifetime and AD
  boundary. Acceptance: in-memory parity, bounded host/device staging, repeated
  reads, clear I/O failures, derivative tests for advertised modes. No claim that
  HDF5 is automatically differentiable; defer unsupported modes explicitly.
- [ ] **WF-29 — Solver and cache budget integration.** Coordinate Davidson,
  non-Hermitian Krylov, DIIS, Lambda and nonlinear-state residency with integral
  workspace. Acceptance: few/multiple-root and restart equivalence, minimum-
  subspace diagnostics, caller ownership, repeated-call release; retain shared
  forward/backward algorithms rather than method-local solvers.

  - [x] **WF-29a:** shared Hermitian primal planning and FCI solver/action
    reservations, including known bound source arrays and the original MB
    budget. Optional basis capacity must not starve useful action batches.
  - [x] **WF-29b:** bounded numerical GMRES probe with true-residual checks,
    finite-candidate retry and full recovery; retain the existing AD owner.
  - [ ] **WF-29c:** non-Hermitian, DIIS, Lambda/SCF and response-state ownership,
    phase-aware lifetimes and byte-controlled caches across all consumers.
    Shared non-Hermitian/EOM primal capacity, closures and action batches are
    verified. Shared packed DIIS and checked linear/root phase planning are
    verified. Isolated non-Hermitian and isolated/cluster Hermitian response
    phases and SCF assembly/forward/implicit response are verified; remaining
    cross-consumer ownership and whole-method reconciliation stay open.
- [ ] **WF-30 — CPU/GPU and nested-AD audit.** Run eager and JIT public entry
  points, native dispatch, device residency and transfer accounting. Acceptance:
  same numerical cases/dtype/seed, forward/JVP/VJP/HVP separately, actual peak
  VRAM/RSS and warm time; a CPU FFI sent to CUDA is a failure, not an acceptable
  runner-only workaround. GPU6 remains forbidden.
- [ ] **WF-31 — Public examples and capability table.** Thin readable examples
  using public facades, no CLI and no algorithm under reproducibility. Include
  memory/derivative limits, a noncanonical-degeneracy example, and output evidence.
  Acceptance: examples run from installed source; conventional array users keep
  compatible results and errors identify unsupported representation/AD order.
- [ ] **WF-32 — Final regression, scaling and API review.** Audit all in-scope
  families, old dense oracles and defaults before enabling the scalable default.
  Acceptance: the validation matrix below, recorded regressions/limitations,
  no duplicate engines, no unrelated files, focused code review and explicit
  merge/push instruction. No source-only or one-method pass closes this task.

## 6. Validation matrix and measurement protocol

Every numerical test runs remotely, never locally. The current 2026-10-23 user
instruction authorizes **jy_gpu_2 / Delta** via Slurm: one 5090/V100, 64 logical
CPUs, 512 GiB and 48 hours. The V100 nodes cannot supply 512 GiB; the admitted job is
104443/n006 with typed `gpu:NV5090:1`. Read the latest `lwcluster manual` before
operating; submit from `~/private/homefile/` to `home`. Execute all tests/builds
inside the allocation's checked srun steps, not on the login node. Use only
assigned Slurm GPU1/process-visible GPU0 by its recorded UUID; GPU6 remains
prohibited. The observed CPU set is96-127,288-319. Four CPU lanes may use
96-111,112-127,288-303,304-319; a GPU
test consumes a released CPU lane rather than adding host threads beyond 64.
CPU srun steps explicitly use `--gres=none --mem=120G --immediate=15`; otherwise
they inherit the allocation's sole GPU and unnecessarily serialize. This
behavior was observed and the CPU-only metadata step verified before use.
The resource-holder step reserves only 1 GiB and no GPU, leaving the remaining
allocation available to test steps. `--overlap` does not remove step-memory
reservations. Prior startup attempts104075/104078/104079 did not run numerical
tests; their scheduler/device-isolation failures are not physics failures.
Renewal attempts104434/104437 failed before computation because the global GPU
index was used after process remapping;104440 used an unsupported query field.
104443 uses the allocated process-visible index and records both numbering scopes.
Do not initialize unrelated GPUs or import c20 device/credential assumptions.
Keep individual tests bounded (normally at most 600 seconds for this campaign).
Existing jy_gpu_1 evidence remains valid only at its documented source version;
do not access its broken storage. No large-molecule campaign or dependency
installation follows automatically from this allocation. Record target-built
native provenance separately; never treat the old node's binary as portable.

| Gate | Inputs | Required checks |
| --- | --- | --- |
| Algebra/representation | Tiny deterministic real tensors; asymmetric coefficient blocks; R/U | Dense/packed/DF/direct equality, signs, dtype, action transpose, budget error |
| Integral/SCF AD | H2; nontrivial open-shell H3 or OH; moving Cartesian basis | Direct/full energy, relaxed density response, JVP/VJP duality, HVP and FD |
| Degenerate orbitals | N2 with exact pi degeneracy; arbitrary occupied/virtual rotations | Noncanonical energy/gradient/HVP covariance; no isolated-orbital requirement |
| First correlation slice | H2/6-31G, H2O/STO-3G or 6-31G, small open shell | R/U MP2 energy, T2/resolvent residual, gradient and HVP |
| CC/triples/EOM | Small R/U molecules and fixed seeds | Per-method energy/residual, Lambda, EOM left/right roots, total correction response |
| CI/FCI | Small fixed electron/orbital sectors and specified CAS | Actions, roots, RDM contractions, derivatives and complete-cluster quantities |
| Recursive MP | Fixed small stable systems plus existing difficult-series examples | Every coefficient, wave when requested, rank restriction and derivative parity |
| Scaling | Fixed basis/space along a modest size family; fixed CAS sizes for FCI | Peak RSS/VRAM, compile/warm/solve times, tile size, actual budget coverage |
| Public/device | Eager and JIT, CPU then authorized GPU | No host materialization/CPU-FFI leakage; same numerical/AD gates |

Reuse existing tests before adding focused missing cases, especially
`tests/test_molecular_correlated_response.py`, `tests/test_scf_orbital_*`,
`tests/mp/`, `tests/cc/test_streamed_triples_*`, `tests/ci/` and `tests/fci/`.
Record the exact selected test node IDs; a focused pass is not a full-suite pass.

Tolerances belong to each test with a physical unit and reference. Initial
targets for well-converged tiny float64 cases are ~1e-9 Hartree energy, 1e-7
Hartree/bohr gradient and 1e-6 Hartree/bohr² HVP agreement; adapt only with a
documented conditioning/finite-difference study, not to hide a mismatch. Verify
SCF/correlation stationarity and eigen-residuals before comparing derivatives.
Use at least two FD step sizes for a new derivative identity. Preserve seeds,
occupations, orbital windows, basis/cart convention and all convergence settings.

For every benchmark, save: input/checkpoint SHA256, source commit plus dirty-file
manifest, Python/JAX/reference/native-library versions and hashes, hardware,
backend/device, dtype, environment/affinity, command, exit code, convergence,
compiler allocation field and value, observed host/device peaks, compile time,
warm action time after synchronization, solve time, result errors and exclusions.
Do not compare a cold JAX run with a warmed reference as a speed ratio. Necessary
output memory must be included; source tensor shapes alone are not peak evidence.

## 7. Definition of done, review loop, and rollback

For **each completed task**, before starting the next dependent task:

1. Check the implementation against that TODO's full scope and acceptance.
2. Review correctness/specification, then code quality/reuse. Resolve findings.
3. Review the evidence, remaining limits, dependencies and forward/AD/device
   matrix in section 8. Update the checkbox only when acceptance is satisfied;
   otherwise record narrower progress. Explain changes without removing scope.
4. Keep the diff focused and commit a reviewable change when verified. Do not
   merge or push the engineering branch without a user instruction.

Keep this review inside the fixed TODO structure: merge a newly discovered
implementation detail or failed test into the owning item, record its evidence,
and continue toward that item's existing acceptance. Do not grow another level
of plan or split off extra TODOs for each experiment. Starting a further planned
part does not reopen an already accepted foundation unless new evidence fails it.

If a new path fails physics/AD or worsens peak/time, keep the old explicit small
reference available and leave the new default disabled. Do not raise tensor
limits, change basis/grids, drop roots/orbitals, freeze moving coefficients or
weaken convergence as an implementation fix. Distinguish necessary representation
limits from avoidable workspace. A deliberate approximation or scope change
requires its own explicit decision and numerical validation.

## 8. Progress and evidence ledger

### 2026-10-21 — WF-00: source baseline and isolation

- Status: `verified` for worktree/inventory/plan only.
- Created `perf/wavefunction-memory` from `1a7ee6c`; main's existing modified and
  untracked files are outside this worktree and were not imported.
- Read the supplied desktop review and checked the shared molecular adapter,
  orbital optimizer, native/direct J/K connection and method-specific source
  boundaries. Existing performance numbers above retain their evidence labels.
- Numerical checks: none for WF-00; none are needed for a plan-only change.
- Plan review: covers all existing molecular wavefunction families, properties,
  nested AD and device/storage budgets. First implementation is WF-01; WF-02
  through WF-34 remain open. The first complete acceptance target is WF-07/08,
  not completion of the entire project.
- Independent plan review identified standalone MP3, EOM Dyson properties and
  named high-rank CI/triples branches as coverage gaps. Added WF-33/34 and the
  branch map above; retained existing unsupported-mode guards. No numerical work
  was claimed by that review. Device checking and measurable performance gates
  were also made explicit. Plan review outcome: the revised scope remains the
  full wavefunction project, with the HF input-chain task first.

### 2026-10-21 — WF-01: started

- Initial status: `in_progress`. Focused direct/full stationary-HF regression
  tests were written and observed failing remotely before the production input
  chain changed. Final verification and review are recorded below.
- RED observed on `jy_gpu_1` (`c20`), CPU affinity `80-95`, GPU disabled, timeout
  600 s: `python -m pytest -q tests/test_scf_orbital_direct.py::test_direct_stationary_hf_matches_full_state`.
  Exit 1, exactly 2 expected failures (RHF/UHF: unexpected `direct_basis`
  argument), 26.02 s wall time. Both full-ERI oracle solves were stationary.
  Baseline manifest: 1056 verified source/test files, base `1a7ee6c`; reused
  native library SHA256 `3b0658cd5381c6c9e170f7d50c271e7521e1200581ab49b3c97b47451ac0e766`.
  Remote evidence: `wavefunction-memory-20261021/red/{red_result.json,red.xml,red.log}`
  under `/home/yjiao/gradscf_checks/`. Local copies are ignored artifacts, not
  new source files. Production edits began only after this expected failure.
- Implementation now exists in `scf/orbital_optimization.py` and
  `gradscf_tools/molecular_response.py`; `eri=None` plus dynamic direct basis
  replaces the full HF ERI only on the explicit new path. Existing dense defaults
  and downstream correlation/GW consumers are unchanged. No new solver was added.
- GREEN snapshot 1: 25 new tests passed in 152.68 s (155.55 s process wall),
  exit 0; 20 existing orbital/geometry/UHF response regressions passed in
  189.18 s (198.25 s process wall), exit 0. Same host, affinity and disabled GPU
  as RED; JAX 0.10.0, PySCF 2.13.0, pytest 9.0.3. Observed test-process peak RSS:
  1,828,548 KiB / 3,159,568 KiB respectively. These are test-suite peaks, not a
  method scaling benchmark or evidence of a quantified memory reduction.
- Specification and independent code-quality reviews passed. Specification
  review requested an additional energy finite-difference check of g dot v at
  both step sizes; this test-only change awaits a focused rerun of the two R/U
  geometry cases. Status therefore remains unchecked until that final gate.
- Remote evidence: `wavefunction-memory-20261021/green1/{results.json,direct.xml,regression.xml}`.
  Both production file hashes are unchanged for the final focused rerun.
- Final gate passed directly on c20, CPU affinity `56-71`, GPU disabled,
  timeout 600 s: the updated R/U geometry test passed both cases, exit 0,
  56.80 s pytest / 59.01 s process wall, peak RSS 1,557,584 KiB. At both
  3e-4 and 1e-4 bohr it checks energy FD against g dot v and gradient FD against
  HVP. Evidence: `green2/{geometry-direct-result.json,geometry-direct.xml,geometry-direct.log}`.
- Completion review: WF-01 is `verified` (25 new + 20 existing unique cases;
  two geometry cases rerun after the test-only review improvement). CPU forward,
  density JVP/VJP and coordinate gradient/HVP passed with no-full-ERI traps.
  GPU, basis-parameter AD, DFT/RSH direct in this adapter, and post-HF integral
  storage are not covered by this completion. WF-02 is next; WF-03/04 are still
  required before claiming scalable MP2 geometry. The overall plan is unchanged.

### 2026-10-21 — Test allocation preparation

- The user initially authorized a 48-hour allocation. CPU-only job 11786 started
  with 64 logical CPUs; no GPU was requested. GPU0 was already assigned elsewhere,
  while the single common GPU type did not provide a direct GPU7 request.
- The user then explicitly cancelled the Slurm workflow. Job 11786 was verified
  `CANCELLED`, ending 2026-10-21T17:18:58 (scheduler time); there is no continuing
  resource hold. The unused scheduler helper was removed from this worktree.
  Submission/script/cancellation records remain ignored execution artifacts.
- The attempted final geometry check inside Slurm exited before numerical work
  because its step description lacked CPU IDs. This is a runner failure, not a
  failed geometry derivative. The focused test is rerun directly with 16 CPUs
  and all GPUs disabled. Algorithm work continues without Slurm changes.

### 2026-10-21 — WF-02: bounded AO2MO increment started

- Status: `in_progress`. Extend the existing `transform_eri_block` rather than
  introduce another provider hierarchy. First bound auxiliary-axis DF transforms
  by the shared byte budget, retaining dense/s4 small reference paths and charging
  their resident arrays/output. This does not remove the resident AO/DF input.
- Reviewed local PySCF 2.9.0 `mp/mp2.py::_ao2mo_ovov`, `mp/dfmp2.py` auxiliary
  batching, and `ao2mo/outcore.py::half_e1`. Exact source SHA256 values are saved
  in ignored `artifacts/wavefunction-memory-20261021/pyscf-source-review.json`.
  This is an algorithm/source comparison, not a numerical speed comparison.
- Native partial AO2MO has no existing block ABI at this baseline. WF-03 still
  needs shell-slab/half transformations; evaluating one J/K per MO pair will not
  be introduced as a production shortcut. WF-04 must supply its contracted
  coordinate/coefficient derivatives before the geometry adapter consumes it.
- Focused tests are written before production changes: budget errors, auxiliary
  temporary shapes, asymmetric blocks, dtype, JVP/VJP and HVP. WF-02 remains open
  until its shared contract and integration criteria are met.

### 2026-10-21 — WF-02a: verified bounded array-block increment

- Initial RED: 2 expected failures for the missing `max_memory` argument, exit 1,
  2.71 s, CPU `56-71`, GPUs disabled. Evidence: `wf02-red/red-result.json`.
- Extended the existing `integrals/molecular/ao2mo.py`; no provider hierarchy,
  alternate solver or method-local integral implementation was added. The public
  entry validates and dispatches; `_eri_block_size` plans static storage and
  `_transform_df_eri_block` performs the exact auxiliary-block contraction.
- Initial numerical tests passed, but quality review found an eager-tail budget
  bug: three full outputs could coexist while only two were charged. The revised
  minimum-budget test failed on frozen source `c5fe99e6...db39a4` before the fix
  (exit 1, 2.84 s, no budget rejection before contraction). Evidence:
  `wf02-tail-red/red-result.json`. This was a forward-workspace defect, not AD
  history storage. Fixed DF accounting to three outputs; dense/s4 remain two.
- Final source SHA256:
  `fbb7847f01a6528ffd47725072e75c296ff4f4151bb2019e0c224bde38c334aa`.
  Test SHA256:
  `c80d3ec64dc9b78969a9d36a87e2bd4aa9889efa14786b1bc856962a9a421f14`.
  The frozen manifest verified 1061 source/test files; native library is unchanged.
- Final commands (all direct execution on c20, no Slurm):

  ```text
  python -m pytest -q -p no:cacheprovider tests/integrals/test_ao2mo_memory.py
  python -m pytest -q -p no:cacheprovider tests/mp/test_integrals.py tests/mp/test_mp2.py tests/mp/test_noncanonical.py
  ```

- Results: 18 new CPU cases passed (5.42 s pytest / 6.38 s process); the same
  18 cases passed on physical GPU7 (9.09 / 10.16 s); 39 related CPU MP cases passed
  (99.15 / 104.76 s). Every group exited 0. CPU affinity `56-71`, GPU-test CPU
  affinity `72-87`, at most 32 logical CPUs concurrently. GPU7 was selected by
  UUID `GPU-b7c1537a-ac7b-2381-a910-866b9afb9ae8`; GPU6 was never visible.
  JAX 0.10.0, float64 enabled (explicit float32/mixed cases also tested), seed
  812/813. Tests include factors and four coefficient matrices in JVP/VJP/HVP,
  two FD steps, remainder/empty axes, dtype, shape and pre-allocation guards.
- Observed host test-process peak RSS: CPU 833,692 KiB; GPU test 1,345,332 KiB;
  MP regressions 4,282,268 KiB. These are test-suite host peaks, not method-level
  memory savings, device peak VRAM, or warmed speed comparisons. Final evidence:
  `/home/yjiao/gradscf_checks/wavefunction-memory-20261021/wf02-green2/`.
- Specification re-review and independent quality re-review passed; the tail
  finding is closed. Readability review retained two small helpers and advised
  against further cosmetic changes.
- Plan review: WF-02a is verified. The outer WF-02 task remains `in_progress`:
  caller-level budget inheritance, native capability/ownership and unknown-JIT-
  device capacity need WF-02b/c. Native exact non-DF half transformations and their
  geometry adjoints are still WF-03/04; no DF approximation was introduced to
  replace a non-DF model. Full AD/Krylov/process budgeting remains WF-29, with
  measured end-to-end scaling required at WF-08/32. Overall scope is unchanged.

### 2026-10-21 — WF-03a / WF-04a: verified native shell windows

- Reused the pinned `GTOnr2e_fill_s1` driver and existing first/second geometry
  contractions. Static eight-endpoint shell windows produce local AO tensors;
  reverse products consume only those local cotangents. The eight inverse
  symmetry boxes restrict contributing shell traversal without removing the
  multiplicity of derivative emissions. Existing full-integral third/fourth
  products and vendor sources are unchanged.
- RED: the missing `shell_slice` entry failed remotely before implementation.
  The first build exposed the installed JAX FFI Span API (`begin()`, not `data()`);
  three mechanical call-site fixes resolved that build error. The rebuilt native
  library SHA256 is
  `276875681817b5a47561ed3cf9e47137bebb11a4ac292f1cd9fa740c0bccaa42`.
  Existing compiler and LP64 BLAS were reused; no dependency was installed.
- Initial GREEN: 21 new CPU cases, one physical-GPU7 case, and 112 existing
  native value/geometry/Hessian/higher/direct-response cases passed, all exit 0.
  Pytest times were 10.24, 3.72 and 211.33 s. Evidence is in remote
  `wavefunction-memory-20261021/wf03-native/` (slice and native-regression logs,
  XML, result JSON). The 112-case group peaked at 7,671,624 KiB host RSS.
- Review then found that guaranteed shell/cache int32 overflow could be checked
  after declaring a CUDA output. Two metadata-only tests reproduced this before
  the correction (exit 1). The shared capacity helper now rejects this before
  primitive/output declaration; full ERI thresholds are preserved. Final
  `ffi.py` SHA256:
  `7b60a9b7de00f5a7d4b6ed843b9ed5cf86235bd94223d41c477fe9ea642565f9`.
- Final GREEN on manifest `wf02b-green1.json` (1120 verified source/test files):
  23 shell-window cases plus 21 existing value cases passed, exit 0, 12.53 s
  pytest / 13.80 s process, 1,053,464 KiB RSS. Updated GPU7 coverage includes
  eager primal/VJP as well as JIT primal/JVP/VJP/nonlinear HVP: one case passed,
  exit 0, 6.49 / 7.68 s, host RSS 1,942,660 KiB. Evidence is in remote
  `wavefunction-memory-20261021/wf02b-green1/`. CPU affinities were `56-71` and
  `72-87`; at most 32 logical CPUs concurrently. GPU6 was never exposed.
- Tests cover asymmetric windows/nonzero origins, cartesian and spherical s/d
  shells, Coulomb and long-range ERIs, coordinate/coefficient response, two FD
  steps, tangent/cotangent duality, nested no-full-AO-shape inspection, and
  explicit unsupported primitive-basis/third-order guards. Independent spec and
  quality reviews passed, including the capacity fix.
- Completion review: only WF-03a and WF-04a are verified. Their CPU shell engine
  has a bounded CUDA callback bridge; this is not a native GPU integral kernel
  or a measured GPU speedup. Static windows alone do not make production AO2MO
  scalable. WF-03b next adds padded dynamic slabs and compact half-transform
  loops, avoiding one Python/JIT graph per shell pair. Outer WF-03/04 and the
  end-to-end MP2 and total-AD budget gates remain open. The full project scope
  and sequence remain valid.

### 2026-10-21 — WF-02b1: verified caller budget inheritance

- Added one shared `resolve_memory_mb` helper. MP/CC/CI facades inherit an SCF
  source's decimal-MB budget unless explicitly overridden; explicit MO references
  use the shared 60000 MB default. Kernels forward the current method field and
  adapters revalidate it before transformation. No numerical solver/config or
  physical-state key was changed. Restricted/unrestricted full wrappers and MP2
  selected blocks forward this same budget; CIS retains its AO response planner.
- RED: inheritance and full-wrapper forwarding tests failed on the frozen prior
  source (missing field/keyword), exit 1. Final manifest `wf02b-green1.json`
  verified 1120 files. On c20 CPU, float64, GPUs disabled: 91 new cases passed
  (10.97 s pytest / 12.39 s process, 1,245,236 KiB peak RSS); 36 MP integral,
  MP2 and MP3 regressions passed (107.32 / 112.92 s, 4,168,324 KiB); five focused
  CI/CIS(D)/RCC/UCC/QCISD caller/transform cases passed (48.02 / 51.89 s,
  2,573,608 KiB). All these groups exited 0. Evidence is in remote
  `wavefunction-memory-20261021/wf02b-green1/`.
- An optional broad CI group hit its 900 s timeout (exit 124); a broad CC group
  was stopped after the relevant five-case caller group passed (exit -15).
  Neither is reported as a completed regression suite. Broader method validation
  belongs to its later algorithm changes and WF-32, rather than repeating
  expensive unchanged numerical kernels for keyword propagation.
- Specification and independent quality reviews passed. Review confirmed the
  existing config/cache semantics and identified no duplicated budget system.
- Completion review: WF-02b1 is verified; WF-02b remains open for simultaneous
  correlated states, DF Qia, spin blocks, solver/AD ownership (WF-05/09/29).
  This does not impose a hard process-RSS or VRAM ceiling. The plan remains a
  full wavefunction project; the next shared consumer uses native AO2MO blocks.

### 2026-10-21 — AO2MO package boundary: user-directed adjustment

- The user requested a dedicated `gradscf.ao2mo` package alongside `cc`/`ci`.
  Split public exports, array/DF blocks, native slab contractions, reference
  transformations, spin utilities and response slices into focused files.
  Integral evaluation and its native geometry primitives remain in `integrals`.
- Migrate existing source/examples/tests to the canonical package and remove
  obsolete forwarding locations. This is an organization change within WF-02/03,
  not a new integral engine or an additional correlation method. Numerical and
  module-boundary regression gates remain required before claiming completion.

### 2026-10-21 — WF-03b1: verified compact dynamic shell slabs

- Added five additive native FFI targets using dynamic int32 shell endpoints and
  static padded storage. Actual AO bounds and storage strides share the existing
  window/traversal code. Forward holds one actual window plus the padded result;
  first/second coordinate products use the same native contraction family and
  ignore finite cotangents outside the actual window. No vendor source changed.
- RED: missing dynamic entry, exit 1, 2.33 s. Incremental native build passed in
  4.80 s with existing compiler/BLAS; native SHA256
  `3eeaa2e5118bd2ecf140d0d1228b0ee0e17e5c1276da9d45c8f35b7c08cd41b9`.
- First numerical run found a nested AD defect: 52 passed, one failed. HVP
  through a differentiated loop reached its retained raw primal without a JVP.
  Added an internal bounded-primal JVP reusing the existing linear geometry
  product. The public unsupported-basis guard is preserved. Final geometry.py
  SHA256: `6a89e14995a4f5f7b0ad0782a65b3650a098bbf85b9eb2f83f622e79ff14f656`.
- Final CPU: 30 dynamic plus 23 static cases passed, exit 0, 15.24 s pytest /
  16.46 s process, 1,049,600 KiB peak RSS. Static and dynamic-padded GPU7 cases
  both passed eager/JIT primal, JVP, VJP and nonlinear HVP, exit 0, 7.68 / 8.86 s,
  1,971,960 KiB host RSS. All 112 original native value/geometry/Hessian/higher/
  direct-response regressions passed again, exit 0, 213.53 / 222.65 s,
  7,840,684 KiB RSS. These are test-process observations, not method benchmarks.
- Evidence: remote `wavefunction-memory-20261021/wf03-native/` slab-cpu1/2,
  slab-gpu2 and slab-legacy2 logs/XML/result JSON; local frozen overlays and
  manifests `wf03-slab-green1/2`. CPU affinities `56-71` and `72-87`, at most
  32 CPUs concurrently, GPU7 selected by UUID; GPU6 remained invisible.
- Independent specification and quality reviews passed after the JVP fix.
  Tests include trace reuse, graph size for 3 versus 300 loop iterations,
  changing windows/tails, padding and local cotangents, finite differences,
  and first/second coordinate products. Raw-ENV entry remains CPU value-only;
  the differentiable entry uses a bounded CPU shell callback on CUDA.
- Completion review: WF-03b1 is verified. Compact JAXPR is not a measured bound
  on all reverse residuals. The consumer must budget padded/actual buffers,
  transfers, optimizer and half transforms and verify rematerialization. Those
  are WF-03b2/04 and total caller/AD ownership remains WF-29. The user-directed
  AO2MO package migration and five tiny native transform tests have an initial
  combined GREEN, but await their independent reviews and conversion measurements.

### 2026-10-21 — WF-02d / WF-03b2a: verified dedicated AO2MO package

- Added `gradscf.ao2mo` alongside CC/CI, with focused `blocks`, `native`,
  `reference`, `spin`, and `response` implementations and ten public exports.
  Existing source/examples/tests use the canonical package. Removed the former
  molecular file; no compatibility forwarding file or extra backend engine was
  retained. Existing integral-namespace function aliases retain their behavior.
- Native block requests consume dynamic shell slabs. Two compact loops share
  the partial `PQrS` transformation, performing the final r-to-MO contraction
  once per r tile. Real ERI pair symmetries choose smaller outer coefficient
  spaces; tail gathers preserve AO offsets. Entire contributions are recomputed
  during AD. This is not one complete shell traversal per MO pair.
- Specification review found projection allocation before the budget check and
  a complex-transform cast that discarded imaginary parts. Both were reproduced
  remotely (two RED failures), then corrected using static shape/dtype metadata
  and an explicit real-transform guard. Quality review found omitted matmul
  input-promotion buffers; its expansion-projection counterexample also failed
  before the fix. The planner now counts these buffers before allocation, and
  execution shares promoted transforms by dtype without changing matmul precision.
- Final native source SHA256:
  `1de5fc4294430bb31f58ec10063c1364186c6af0794d1c637562b0806cb26fce`.
  Final CPU: 124 cases passed, exit 0, 22.17 s pytest / 24.12 s process,
  1,651,028 KiB peak RSS (namespace, twelve native AO2MO cases, existing array/DF
  blocks and post-HF budget propagation). One GPU7 case passed eager energy,
  JIT coefficient/coordinate gradient and mixed HVP against the CPU dense oracle,
  exit 0, 7.25 / 8.37 s, 2,096,316 KiB host RSS. GPU6 was never exposed.
- Final package validation uses the staged package-only FCI import change,
  excluding the separately developed FCI algorithm. The frozen science-file
  overlay is `ao2mo-final-overlay.json`; remote evidence is
  `wavefunction-memory-20261021/ao2mo-final/`. Forty-one existing MP/CI/CC caller
  cases also passed after the import migration (146.94 / 153.80 s; before the
  final promotion-only fix). No local numerical test or build was run.
- Independent specification and quality re-reviews passed. Native cases cover
  unequal coefficient spaces, Cartesian/spherical d shells, projected float32/
  float64 inputs, empty MO output, allocation guards, geometry finite differences
  at two steps and first/second response. Test-process peaks are not measured
  end-to-end correlation memory savings.
- Completion review: WF-02d and the kernel-only WF-03b2a are verified. Outer
  WF-02/03/04 remain open for caller ownership, conversion measurements and
  broader derivative/device gates. Existing MP/CC/CI adapters can still request
  full AO-pair data; WF-05/06 now connects direct SCF and selected MP2 blocks,
  reusing live converged Focks rather than reconstructing them. The overall
  engineering plan remains unchanged beyond the requested package boundary.


### 2026-10-21 — WF-05a / WF-06a: verified native direct MP2 preparation

- Added a pure `mp.integrals.from_orbitals` entry taking live coefficients,
  full MO-frame Focks, occupations and the total HF energy. It requests only
  selected OVOV blocks or existing DF Qia factors. No full MO/spin ERI or
  separate orbital diagonalization is introduced. Frozen and noncanonical
  reference checks remain in force.
- Eager RHF uses saved integral inputs and Fock; UHF reads its existing solved
  reference payload directly. Neither rebuilds the source nor recomputes J/K.
  Native/projected direct providers reach the shared AO2MO package. The legacy
  JAX-direct reference retains packed construction with a metadata preflight.
- RED: the original native adapter reached forbidden packed construction.
  Initial GREEN exposed ten UHF test-fixture errors: standard UKS has no saved
  `_scf_inputs`, so the actual existing reference payload was used. A second
  RED reproduced two legacy-direct compatibility failures and two mixed-dtype
  carry failures; one all-float32 oracle cancellation issue was also corrected
  in the test without loosening production validity tolerances.
- Final production SHA256:
  `a54596cfd568ae086b2de140a811d7abd190339cf6ea3965179e9d5ff25a4121`.
  Twenty-five new CPU cases passed, exit 0, 63.61 s pytest / 68.00 s process,
  3,280,700 KiB peak RSS. Cases include independent PySCF R/U MP2 energies,
  frozen/with-T2 controls, full-integral/rebuild traps, pure coefficient gradient
  and HVP, occupied/virtual rotations and float32/mixed-float64 data.
- All 130 existing MP integral/MP2/noncanonical/post-HF budget cases passed,
  exit 0, 104.03 s pytest / 110.19 s process, 4,290,760 KiB peak RSS. Remote
  snapshot `wavefunction-memory-20261021/mp2-direct-green2/` contains
  `direct-mp2-cpu2` and `mp2-legacy-regression` logs/XML/result JSON. Numerical
  runs used CPU56-71, float64 default, JAX0.10.0/PySCF2.13.0, GPU invisible.
- Independent static implementation review passed. Completion review closes
  WF-05a/06a only. Each transform still has a separate budget; retained sibling
  spin blocks, Qia, T2, resolvent and AD/solver state require WF-29 integration.
  Full relaxed molecular geometry is being verified under WF-07. These test
  process peaks are not a measured MP2 scale comparison.


### 2026-10-21 — WF-09a: verified restricted CC block extraction

- The MO-array entry now gathers each of its seven required blocks directly
  from the original ERI. It no longer first copies the complete active-space
  four-index tensor. Frozen cores still contribute to the full Fock/HF energy;
  selected occupied/virtual ordering and arbitrary raw-integral derivatives
  remain unchanged. Virtual dimension is derived from Fock, independently of
  `vvvv`, preparing consumers for a later virtual-pair action.
- RED: seven failures and five passes, exit 1, 7.52 / 8.97 s. Six cases detected
  the forbidden active-four-index copy; one detected the old vvvv-dependent
  orbital count. Final twelve cases passed, exit 0, 9.29 / 10.70 s,
  1,102,328 KiB peak RSS, including float32/64, frozen occupied/virtual spaces,
  empty O/V spaces and raw block gradient/HVP parity.
- Twenty-six existing RCC/CC2/QCISD cases passed, exit 0, 216.64 s pytest /
  223.18 s process, 3,519,240 KiB peak RSS. They include residuals, energies,
  amplitudes, Lambda, frozen spaces, triples and enabled model responses.
  Remote `wf09a-green1/` and local `artifacts/wavefunction-memory-20261021/wf09a/`
  retain commands, hashes and JUnit evidence. CPU96-103, no GPU, no local numerics.
- Production SHA256:
  `4cfd8a9f07879a4404a519be09ef67b673efb69166bfaaf45683720d22fd2649`.
  Independent specification/quality review passed. The original complete MO
  input and vvvv block still exist: this closes WF-09a only, not WF-09 or WF-10.
  No large-molecule process-memory improvement is inferred from this unit test.


### 2026-10-21 — WF-19a: verified native active-FCI preparation

- Active-space preparation requests only nact^4 MO ERIs from the shared AO2MO
  package. Closed cores are folded in AO space with D=2*Cc*Cc.T, V=J-.5*K,
  Ecore+=Tr[D*(h+.5*V)] and h_active+=Ca.T*V*Ca. Existing native J/K is reused
  with zero screening; no second integral engine or full total-MO ERI is added.
  The pure helper retains coefficient, hcore, core energy and provider leaves.
- Molecular max_memory inherits the source or an override. Static active/core
  arrays and minimum native tiles are checked before J/K/integral work. Review
  exposed two further preparation holes: legacy packed construction and explicit
  reference core folding reached allocation before budget checks. Both failed
  new remote traps (2 failed, exit 1), then received metadata-only preflights.
- Final 32 cases passed, exit 0, 54.17 s pytest / 57.54 s process,
  2,804,556 KiB RSS on CPU88-95, JAX CPU/float64, GPU invisible. Cases cover
  reordered CAS/frozen core, independent same-frame dense Hamiltonian/FCI/RDM1,
  inherited/overridden/stale budgets, ordinary/projected provider coefficient
  and coordinate products through two, finite differences and early errors.
  The initial run had 16 passes and 14 ROHF fixture errors: ROHF does not expose
  direct-SCF controls. Its fixed-frame test now runs existing full SCF then
  adapts the integral representation exactly; no direct ROHF-SCF claim is made.
- Final reference.py SHA256:
  `24c0e483165505906a9cf374e3530005b6df44375bd921654b12a1ad630d108c`.
  Evidence: remote `wf19-green2/logs/active-fci-cpu2.*`, mirrored under local
  `artifacts/wavefunction-memory-20261021/wf19-green2/`. Independent review passed.
- Completion review closes WF-19a only. The pure preparation test is not a
  relaxed molecular FCI geometry test; full public solver, RDM, sibling-array
  lifetime and AD budget integration remain WF-19b/21b/22/29.

### 2026-10-21 — WF-21a: verified tiled FCI action and measured AD residency

- Hamiltonian contraction now visits alpha/beta/RHS tiles: gather E*C locally,
  contract the orbital-pair kernel, then transpose-scatter the contribution.
  Whole beta/RHS tiles use complete row access. Partial tiles retain explicit
  tail masks before and after contraction. Checkpoint the complete additive
  body; retain the original full excitation helper only for unmigrated RDMs.
- max_workspace_elements bounds one P*A*B*R tile. max_memory counts known
  h/g/coefficient/maps, three output versions, six tile buffers and preparation
  scratch, with the common reserve. Core absorption's extra ERI buffers were
  added after a real preflight RED. This is not a total eigensolver RSS limit.
- Final 38 CPU cases passed, exit 0, 19.19 / 21.00 s, 1,383,504 KiB RSS.
  Coverage includes unequal spin strings, vacuum/full occupancy, multiple RHS,
  remainder tiles, mixed dtype, transpose action, JVP/VJP/HVP and allocation
  traps. Independent specification/quality review of the final row path passed.
  Production SHA256:
  `35cdc09415282f184b5f137bdb653e326ce65c9bf528842718922fb77da75683`.
- Matched source/inputs, CPU72-87, seed2104, float64, JAX0.10.0, noGPU:
  nact=12 and 853,776 determinants, default 60,000MB/8M tile-element policy:
  old/new warm action 0.490625/0.407317 s; process RSS 3,536,780/928,992 KiB;
  compiled temporary bytes 2,950,649,856/199,973,792. Sigma agrees within
  6.94e-17 over nact8/10/12. A 64MB tile policy instead needs 0.598595 s and
  853,036 KiB: lower memory with a measured 22% timing cost.
  This is the complete /usr/bin/time peak; the runner phase snapshot was
  766,872 KiB and must not be substituted for the process peak. Tiny nact8 has
  18.2% overhead on a 31-repeat audit; no universal speedup is claimed.
- Reverse audit takes raw ERI and CI-vector gradients of a fixed bilinear
  probe. Old/new gradients agree within 3.56e-15. Saved numerical residuals
  retain one input c, g, probe and maps; scan history is only scalar indices,
  not number_of_tiles times the CI state. At nact8, old/wide/one-row compiled
  gradient temporary bytes are 7,604,832/4,014,456/235,000. Warm gradients cost
  0.002956/0.003983/0.005051 s: recomputation trades time for memory here.
- Evidence and exact commands/hashes are under remote/local
  `wf21-row-measurements/`; the local `WF21_ROW_MEASUREMENTS.md` explains policies
  and all process exits. Thin benchmark runners call src only. These measurements
  cover one action and its first reverse derivative, not a complete FCI solve,
  RDM or relaxed geometry response. WF-21b/22/29 remain open.


### 2026-10-21 — WF-12a: verified requested UCC spin blocks

- UCC integral preparation now computes full-reference alpha/beta Focks and
  HF energy directly from spatial aa/ab/bb ERIs, including all frozen cores.
  A 29-line internal AO2MO spin helper gathers only requested allowed spin
  sectors; beta-alpha uses the exact pair transpose of ab. Six existing dense
  antisymmetrized CC blocks replace complete total-spin/active-spin ERIs and
  the complete antisymmetrized intermediate. Packed amplitudes and equations
  remain unchanged. The largest requested vvvv block is still dense.
- Preserve the raw definition W[pqrs]=g[prqs]-g[psqr], even away from physical
  spatial ERI permutation symmetry. No symmetry assumption changes integral
  cotangents. Unequal spins, separately frozen occupied/virtual orbitals, empty
  sectors and mixed dtypes retain the original ordering/conventions.
- RED: one forbidden full-spin call failed while 25 independent dense-oracle
  cases passed (16.96 / 18.55 s, exit 1). Final 26 focused cases passed,
  exit 0, 47.33 / 50.54 s, 2,597,220 KiB RSS; twelve existing UCC/UCCD cases
  passed, 114.50 / 120.18 s, 3,900,960 KiB RSS; nine relevant Lambda/RDM/(T)
  and response cases passed, 248.67 / 258.07 s, 4,672,404 KiB RSS. Total 47
  GREEN cases include OH/6-31G PySCF residual/amplitude/(T) comparisons.
  CPU96-103, noGPU, existing dependencies; no local numerical execution.
- Final uccsd.py SHA256:
  `634c282f2aed26b89570812056aaa88525bf47a31ebd9eed53e056c8033c92bb`;
  ao2mo/spin.py SHA256:
  `8c8f5a915db9387e5c0b9a21ad6feaae771e4c9c23107fbc7c85cdfd07c9d329`.
  Independent review passed. Commands, hashes, JUnit and result JSON are
  retained under local `wf12a/{red,green1}` and remote corresponding snapshots.
- Completion review closes WF-12a only. Spatial full MO integrals, returned
  spin blocks and unpacked amplitudes remain resident. Test-process peaks
  combine many compilation signatures and do not establish a matched single-
  system speedup or memory saving. AO-source preparation, block-sparse residuals,
  full geometry tests and total budget remain WF-12b/13/29/30.


### 2026-10-21 — WF-06 response gate: fix pre-existing nested diagnostic batching

- A complete mixed Raman regression exposed JAX's `vmap out_axes[0]` failure
  in `jvp(jacfwd(dipole))`. Isolated controls showed the original full-MO
  workflow fails too. Initial hypotheses about selected AO2MO/live Focks/the
  stationary HF energy were disproved; no such physical dependency was removed.
- A pure tensor-inverse toy localized the failure to convergence/residual
  auxiliary outputs of JAX0.10 `custom_linear_solve(has_aux=True)`. Using those
  outputs in a physical validity mask contaminated the nominally unbatched
  primal under nested batching. The same solution with diagnostics recomputed
  from its operator/RHS passes the transform.
- The shared tensor-sum solver now returns only its solution from the opaque
  solve. Every numerical primal/tangent/adjoint solve still checks its true
  residual internally and returns NaN on failure. Public residual/convergence
  are recomputed from the physical returned solution, adding one operator
  application. No tolerance, SCF/MP2 equation, jacfwd API or gradient dependency
  changed. A failed NaN solution now also reports a NaN public residual instead
  of the old possible finite diagnostic of its unmasked candidate.
- RED: three pure diagnostic-mask cases failed, exit 1, 4.05 / 5.12 s. Final
  three new plus three old tensor-sum cases passed, 10.50 / 11.79 s, exit 0,
  1,048,644 KiB RSS. Dense Kronecker and two-step FD comparisons pass; singular
  systems retain invalid forward/gradient/nested response with a nonzero mixed
  derivative fixture. Both original and selected full Raman cases passed,
  respectively 45.26 / 47.15 s and 42.43 / 44.19 s, exit 0, including scalar
  reverse-cubed, tensor response, symmetry and two-step FD. All 155 MP2/
  noncanonical/native-direct/orbital/budget regressions passed, 164.21 / 171.84 s,
  exit 0, 4,622,624 KiB RSS. CPU72-87, noGPU, no local numerical execution.
- Source SHA256:
  `f0b5139fee94f19881aaab1695660f723cfe8fdb2f004e8fce1220e56658d3c5`.
  Independent code review passed. Evidence and withdrawn hypotheses are in
  local `raman-diagnostic/DIAGNOSIS.md` and `downloaded/`, with remote
  `raman-diagnostic`, `raman-legacy-green` and `raman-green` snapshots.
- Completion review: this closes the nested diagnostic-mask defect only;
  it does not close the complete geometry/scaling/whole-budget engineering
  project. The additional check cost is explicit; it is not replaced by an
  unchecked tangent/adjoint solve to improve a timing result.


### 2026-10-21 — WF-07: verified complete selected-block MP2 geometry response

- CorrelatedEnergy now consumes the live solved C/Fock/HF energy and shared
  MP2 orbital preparation instead of obligatory full MO tensors. Noncanonical
  R/U MP2 and internal orbital degeneracy use the existing tensor resolvent.
  Explicit jk_backend='direct' removes moving full AO ERIs as well; default
  full HF remains available. max_memory inherits the source for transformation
  planning, without claiming a total SCF/solver/AD process ceiling.
- RED was the missing direct constructor entry. The new three-case CPU suite
  passed complete energy, nuclear gradient, potential response, mixed HVP,
  two-step FD, translational invariance and N2 strict pi degeneracy, with full
  AO evaluation/full MO transformation traps. Independent original MP2 R/U/N2
  PySCF gradients pass. The pre-existing nested Raman defect was isolated and
  fixed in the shared solver as recorded above, preserving the tensor API.
- Full GPU integration exposed two genuine native boundaries: missing CUDA
  lowering for the nuclear geometry VJP, then a CPU FFI value sent to CUDA.
  Reused the existing bounded CPU-product bridge for first/second one-electron
  products and raw one-electron values. Static NumPy tables are cached; GPU
  arrays are not retained as native metadata. Full AO ERI transfer is not added.
  Raw ENV still has no AD rule, and cubic/quartic full products remain CPU-only.
- Final combined snapshot: seven CPU MP2 cases passed, three unrelated CC cases
  deselected, exit 0, 256.19 s pytest / 260.52 s process, 2,356,020 KiB RSS.
  Five GPU7 cases passed, exit 0, 57.67 / 59.80 s, 2,135,092 KiB host RSS:
  full direct MP2 nuclear gradient/HVP plus overlap/kinetic/nuclear/dipole
  geometry products. Every GPU output leaf is asserted device-resident and
  compared with a CPU dense oracle. Native CPU value/geometry/Hessian/higher/
  coefficient/exponent/direct-response regressions: 169 passed, exit 0,
  293.42 / 304.77 s, 8,433,060 KiB RSS. No local numerics were run.
- Final source hashes: workflow
  `30bf3ca6952a05d1a3c191a8043124bde1c9836c594cb35219458e57c00b7670`;
  native ffi `0973bae736e734ccc18a58178e58980cd6c6baaf5bec527afe1e82955f75cbc5`;
  geometry `ab0a22c1311eb2c76a9c9e77d13cbfae187cff859c54db414eea32e799da1103`.
  Independent reviews passed. Remote `correlated-final1/` and
  `correlated-gpu-green3/` logs, manifests and local integration archive retain
  commands/environment/source lineage. CPU groups stayed within the project
  64-CPU limit; GPU7 was UUID-selected after an idle check, GPU6 hidden.
- Completion review closes WF-07's numerical integration gate, not WF-08 or
  the whole project. This bridge is native CPU integration compatible with CUDA,
  not a GPU integral kernel or evidence of GPU acceleration. Water/direct has
  significant warm overhead, and the aniline performance gate remains open:
  old actual gradient failed its finite-derivative check; direct hit a bounded
  timeout. Neither is a successful matched aniline benchmark. Their compiled
  allocations are estimates, not a replacement for valid complete results.


### 2026-10-21 — WF-21b1 / WF-22a: verified FCI facade and density tiles

- FCISolver/solve_fci now pass the resolved source/override budget to their
  existing Hamiltonian action. A metadata forecast of active h/g, preparation
  scratch and necessary CI states replaces the old P*Ndet capacity rejection
  before molecular transformation. Shared dense/Davidson numerical algorithms
  are unchanged. Budget changes invalidate cached facade properties.
- FCI densities reduce determinant tiles directly into requested one-/two-body
  moments. Same-state bra/ket actions are reused; transition left/right actions
  remain distinct. Complete beta tiles use the verified whole-row gather
  strategy. The checkpointed carry contains only moments, not CI-state history.
  Spin exchange is reduced directly without forming a two-RDM. Existing real
  PySCF index/normalization conventions and necessary nact^4 outputs remain.
- New low-budget property tests found four public methods using the RDM default
  rather than their facade budget (4 genuine DID-NOT-RAISE failures). All RDM,
  transition and spin facade entry points now pass the same max_memory. The
  obsolete full-excitation-stack capacity helper was removed once no caller
  remained. No extra solver or duplicate density implementation was retained.
- REDs: missing solver budget/full-stack precheck 19 failures; original RDM
  stacks/low tile cap 6 failures; property budget bypass 4 failures. Final
  combined API/action/RDM/forward/response/nested suite: 75 passed, exit 0,
  189.66 s pytest / 198.12 s process, 4,639,596 KiB RSS, CPU112-119. It includes
  independent PySCF densities/spin/energies, unequal bra/ket, raw CI gradient/HVP
  and two-step FD, multiple roots/cluster guards, active/core mapping and stale
  budgets. Earlier separate 19-new/46-existing suites also passed.
- Matched RDM benchmark on CPU96-103, JAX0.10.0, float64, seed2201, default
  60,000MB/8M policy: nact12 / 853,776 determinants, old/new warm
  1.571182/0.865614 s, whole-process RSS 9,084,120/1,360,120 KiB, compiled
  temporaries 6,891,679,872/454,064,440 bytes. Compiled output allocation is
  unchanged at 500,008 bytes. All public outputs agree within 1.94e-14 over
  nact8/10/12 and a nonidentical-bra/ket transition comparison.
- Small-case costs are retained: nact8 forward 4.090/5.200 ms, nact10
  transition 53.349/55.926 ms. The two-tile reduction adds small-loop overhead;
  avoiding full E stacks provides the measured larger-space benefit. There is
  no universal speedup claim. nact8 reverse warm improves 5.234/3.744 ms,
  compiled temporaries 20,150,088/10,297,672 bytes, with raw CI gradients within
  1.33e-15; total small reverse process RSS is essentially unchanged.
- Saved-residual/JAXPR inspection on this first-reverse case retains one CI
  array, fixed maps/probes, normalization scalars and two tile indices. No full
  E stack or per-tile complete-CI history is retained. All ten benchmark
  processes exited zero. Exact input/output hashes, source versions, compile/
  first/warm samples and time-v RSS are in local `wf22-rdm-efficiency/` and
  remote corresponding results. These are numerical-action measurements,
  not a complete FCI solver or molecular geometry benchmark.
- Final hashes: api.py
  `bd6c938f3b69eda7829386dd8ac9f450355289d7e238a7e76c636beb1494f6c8`;
  solver.py `52027295001530fee2cacb78ba1a51c6f22d18fa636e2a045e03f32e762eb931`;
  rdm.py `5d4919d003c8e2fd7c5bee64ca0c7cec87120a6badfe5beea5157332fb16d2d5`;
  Hamiltonian after unused-guard removal
  `e14ddbe62738fc88890c6451c7f1d7e9a807304252f980faa942c4783f0df499`.
  Independent math/device review passed; its whole-row gather and actual-link
  trap suggestions were applied before final validation. No local numerics.
- Completion review: WF-21b1/22a are verified. The same nominal budget reaches
  preparation, actions and densities, but retained solver/adjoint state is not
  yet jointly reserved; that remains WF-21b2/29. The shared planner conservatively
  reserves three CI output versions even for density reductions; avoid a second
  planner, and refine that over-reservation with common ownership in WF-29.


### 2026-10-21 — WF-31a: public native-direct N2 geometry example

- Added `examples/mp/direct_geometry.py`, a linear, no-CLI public example:
  native direct RHF, implicit noncanonical MP2 total energy, nuclear gradient
  and stretch HVP at exact internal pi degeneracy. It retains the complete
  STO-3G orbital space and checks the stationary branch and translation sum.
  Existing basis-parameter examples remain separate.
- Executed once against the verified `correlated-final1` source on CPU112-119,
  exit 0, 29.71 s process, 1,248,536 KiB peak RSS. Total energy is
  -107.651843995332 Hartree, z gradients +/-0.323413809 Hartree/Bohr and stretch
  HVP z components -/+2.096236923 Hartree/Bohr^2. Recorded actual output is
  included as comments at the script end. Script body SHA before output comments:
  `aa7dcdc1396ac7b1d16445a90c5d0c151553267a24015d08acfe01be6a302592`.
  Evidence: remote `direct-geometry-example/{example.log,result.json}`.
- Updated MP storage/geometry documentation to describe the new native-direct
  path and canonical AO2MO package. Higher MP geometry, necessary T2 storage
  and unsupported native basis derivatives retain their stated boundaries.
- Completion review: this is only WF-31a; final capability table/examples for
  the remaining migrated wavefunction methods await those methods' gates.


### 2026-10-21 — WF-10a / WF-11a: efficient restricted virtual actions

- A single restricted action contracts virtual dressings before they become
  a dense four-virtual tensor. A static smaller-scratch choice keeps a small
  transient W for occupied-dominated spaces. Neither choice caches W.
  CCSD supplies tau, QCI zero singles/T2, and CC2 its two singles factors.
- Measuring the complete CC2 term found a real 74% low-virtual slowdown in
  the first implementation: forming the outer product lost its rank-one
  contraction order. The retained internal `(L,R)` input delays that product
  until the large-virtual branch. A dot plus axis transpose reproduces both
  small-W dressings by index relabeling, with no ERI symmetry assumption.
- Final source: 58 focused checks and 6 named physical/core checks passed,
  exit 0. These include seven models in both orbital ratios, independent
  factors, raw nonsymmetric integrals, float32/64, JVP/VJP/HVP and finite
  differences, CC2/PySCF, CCSD Lambda, QCI density/adjoint and water triples.
  CPU80-87; pytest 107.21/60.38 s; process RSS 2,439,656/2,340,360 KiB.
- Twelve matched independent CPU processes exited zero. Complete CC2 warm
  times old/new: o32/v8 0.120907/0.120746 ms, o8/v32 3.712/3.212 ms,
  o8/v64 124.031/43.918 ms. The low-virtual case is equivalent speed and has
  identical 524,288-byte compiled scratch. Generic dense-X o8/v64 improves
  189.239/40.748 ms, scratch 384/162 MiB. Requested results agree to 1e-15.
- CC2's large-virtual process RSS is higher despite smaller compiled scratch;
  no universal RSS or complete-CC speed improvement is claimed. Bare vvvv,
  AO-source preparation, EOM/UCC and total-state budgets remain open.
  Evidence: local `wf10a-cc2/`, with rejected policy/HLO traces, final source,
  all commands, hashes, XML and outputs. Final helper/caller/QCI hashes start
  `bec87a18`, `12f915f3`, `111d3d1d`. Independent algebra/AD review passed.
- Completion review closes only WF-10a/11a. The parent tasks still require
  bare-integral actions, AO-source consumers and complete method budgets.


### 2026-10-21 — WF-29a: one Hermitian/FCI primal array budget

- The shared planner adjusts existing Davidson capacities and column batches,
  reserves dense/primal arrays before identity/guess/absorption allocation,
  and passes integer solver bytes to the FCI action under the original MB
  value. The runtime reserve is applied once; no new eigensolver is added.
- Useful guess/trial/restart/action batches take precedence over an optional
  larger Ritz basis. The initially larger m23/g2/b2 policy cost 8.111 ms on
  the seed2902 six-orbital/(2,2) fixture. At the same 0.5MB, m17/g4/b6 costs
  4.338 ms with 254,792-byte compiled scratch. The same-source 4MB control
  costs 5.302 ms/979,272 bytes; both take 10 iterations and agree within
  7.99e-15 Hartree. Process RSS remains dominated by the runtime baseline.
- Known source h/g, existing SCF payloads, declared basis dataclasses and
  parameter PyTrees remain charged after active-space preparation, with
  object-identity deduplication. No reference builder or global cache is
  inspected. Explicit dense matrices are normalized once and charged even
  when supplied as lists/tuples. Independent review found the basis/list
  omissions; four genuine RED failures preceded these minimal fixes.
- Final frozen version: 16 affected focused checks and 87 named Hermitian/FCI
  forward/response/HVP/nested checks passed, no skips, exit 0; CPU104-111.
  The latter took 249.63 s pytest / 258.36 s process, RSS 4,942,484 KiB.
  Evidence: `wf29-review-evidence/`; earlier rejected policies/measurements in
  `wf29-evidence/`. Final FCI/eigen APIs begin `8c0e8b88`/`b347e39c`;
  planner `4a63bf51`. Independent repair review passed.
- Completion review closes the known-array primal increment. Preparation
  scratch is conservatively co-reserved; phase-optimal lifetimes, full
  response/AD, other method caches and traced-device capacity remain open.
  This is an array planner, not a whole-process or allocator hard limit.


### 2026-10-21 — WF-29b: true-residual GMRES work, preserving SCF behavior

- Actual MP2 stationary-state profiling located the small direct-water
  overhead in SCF work. Zero/one Newton-refinement experiments passed narrow
  tests but regressed complete gradients; both were rejected and the original
  three refinements restored exactly. No SCF forward policy is changed.
- A difficult archived anchor spent 635 of 680 response actions in upstream
  incremental GMRES despite a well-conditioned physical operator. Finite-only
  warm retry did not fix that cost. A one-cycle probe also regressed the
  healthy control; the retained two-cycle probe preserves its 37 actions.
- The existing opaque numerical owner now true-checks after at most two
  incremental cycles, then uses the existing full-budget batched retry and,
  only if needed, full incremental recovery from the best finite candidate.
  User tolerances, physical residual guards and the internal 0.1 margin stay
  unchanged. Raw numerical zero-RHS handling must remain inside that AD owner.
- Healthy default direct warm old/new: 2.02294/2.23909 s, identical work and
  bit-identical values/gradients. Full-selected warm: 0.39702/0.17401 s. The
  archived direct stress case improves 7.72736/1.34180 s and 680/37 actions;
  its rejected one-refinement SCF policy is not restored. New direct/full
  gradients agree within 1.89e-13 Hartree/Bohr on this fixed water case.
- Verified 52 CPU and 22 GPU executions, all process exits zero and no XML
  skips/failures/errors: accuracy21, shared24, complete MP2 geometry/HVP/Raman7,
  then accuracy21 plus complete MP2 GPU geometry/HVP1. GPU7 was selected by
  UUID after an idle check; every output leaf of the complete case is on GPU.
  GPU6 was hidden. Final GMRES/test hashes begin `1fbdf181`/`444250f5`.
  Evidence: `MP2_EFFICIENCY_AUDIT.md`, `mp2-efficiency-evidence/`.
- Completion review closes this numerical-solve increment. It does not close
  WF-08: a same-input aniline/full-selected rerun still failed the finite
  native-derivative guard (exit 1, 113.87 s); direct remains under a bounded
  run. Neither failure nor an allocation estimate is a completed scale result.


### 2026-10-21 — Current combined numerical checkpoint

- Current restricted action, Hermitian/FCI budget, GMRES and MP target code
  were frozen together in `joint-wavefunction1`. Twelve focused integration
  checks passed, no skips, exit 0: FCI budgeted density gradient/HVP and nested
  guards, mixed-dtype column tails, CCSD Lambda and factored raw CC response,
  complete direct MP2 relaxed response and sextuple MP coefficients.
  CPU104-111; 203.01 s pytest / 209.58 s process; RSS 3,972,892 KiB.
- The exact full source manifest and unchanged native library SHA are recorded
  alongside commands/XML. This clears a narrow composition gate, not all
  combinations or the final project regression. MP broad-suite total-time
  limits and large-aniline diagnostics remain explicit open items.


### 2026-10-21 — WF-02e: common reservation validation

- Moved the existing nonnegative-integer byte validator unchanged to `_memory`
  and updated solver/FCI imports. This enables AO2MO caller reservations without
  creating an integral-to-solver dependency. No numerical equation or budget
  formula changed. Twelve existing reservation/minimum/capacity checks passed,
  19 deselected, exit 0; CPU104-111, 13.07 s pytest / 14.65 s process,
  1,218,584 KiB RSS. Evidence: remote `memory-validator-move1/` manifest/log/XML.
- Completion review closes only this shared utility move. The next native
  multiple-request transformation and all-consumer ownership remain separate
  WF-03b2b/02b increments, with their own numerical and efficiency gates.


### 2026-10-21 — WF-23a: terminal tensor targets, narrow checkpoint

- Reuse the existing direct-path rank bound for canonical tensor actions.
  W Psi_k contributes at lambda^(k+1); energy through p needs only bra degrees
  a<=p-k-1 and hence output ranks <=2(p-k-1). Skip those contraction targets
  before allocation. All required waves and the packed coordinate layout stay
  intact; no integral symmetrization, tolerance or perturbation partition changes.
- Thirteen focused checks passed, exit 0, 104.87 s pytest / 107.75 s process,
  RSS 2,740,276 KiB: action/state JVP/VJP/HVP, structural skips, six-orbital
  MP6 sextuples, spin/frozen spaces and extra wavefunction orders. Three old
  moving-gap/integral response and invalid-input checks passed separately,
  exit 0, 505.01 s pytest / 530.71 s process. The combined checkpoint also
  exercises sextuple wave coefficients with the current shared solvers.
- Seed2325 six-orbital/three-occupied single action, matched CPU64-71/f64:
  old/new warm 25.520903/4.703314 ms; compiled scratch 142,740,872/12,411,144
  bytes; requested outputs agree exactly and discarded targets are zero.
  This is not complete MP timing or a whole-process memory claim.
- Runtime limits remain visible: the original tensor/series whole groups
  exceeded 600 s, and a forward-only eager shard exceeded 300 s. A real
  six-orbital high-wave test also exceeded 600 s in eager execution; fresh
  whole-recurrence JIT wrappers then passed the identical two cases in 31.39 s.
  The old forward comparison is compiled as a complete recurrence,
  retaining every E2...E12 coefficient and wave assertion. No tolerance or
  test case was removed. Its full 28-case forward shard passed separately,
  exit 0, 543.50 s pytest / 552.28 s process, RSS 7,350,028 KiB. Together with
  the three response cases above, this completes the original tensor-action
  file. Separate series/noncanonical groups are not marked passed.
- Evidence: local `wf23-review-evidence/`; additional completed response shard
  and completed forward records in remote `wf23-green1/4`. Independent math,
  raw-chart and focused-evidence review passed. Final source hashes begin
  `19911d5a`/`85ad6cb7`. Completion review closes WF-23a at this checkpoint;
  complete packed-state contractions and history scheduling remain separate
  WF-24/25 tasks. Later compilation-boundary changes need their own validation.


### 2026-10-21 — WF-03b2b1: native multi-request reuse

- Added one shared AO2MO batch entry point. Static space IDs identify orbital
  coefficients; native physical eightfold symmetry chooses equivalent small
  halves and restores requested output axes. One bounded q/r shell traversal
  shares first, second and half transformations within the call. There is no
  cross-call cache. Dense/s4/DF paths reuse only identical ordered requests,
  preserving their off-manifold input derivatives and original dtype.
- Batch outputs, coefficient/source residency, known caller reservations and
  unique half workspaces share the original MB budget with one runtime reserve.
  Shape validation precedes planning/contraction; normalized array sources are
  forwarded once. This is a forward-array plan, not a complete AD-process cap.
- Native/frozen/asymmetric/projected values, structure, coefficient/geometry
  JVP/VJP/HVP, finite differences and array raw-chart checks: CPU 70 passed,
  one explicit GPU-only skip; GPU7 four passed. A later array-normalization
  regression first failed all six dense/s4/DF x f32/f64 cases, then final-source
  CPU 32 and GPU7 nine passed without skips. The final narrow run does not
  relabel the earlier native/device measurements as rerun. Native kernel SHA
  stayed `48e52f64...7608`; final blocks/test hashes start `ff0782ab`/`97e4a108`.
- Matched water, Cartesian bases, seed6205, f64, 60000 MB, CPU80-87 with
  OMP4/BLAS1: seven separate requests versus one batch reduced slab calls
  14 to 2. Warm times: 19 AO/6-31G* 97.282 to 17.554 ms; 65 AO/cc-pVTZ
  2.09394 to 0.44779 s. Maximum block differences are 3.33e-16/1.94e-16.
  Single-OVOV controls retain identical tiles, scratch and outputs, with warm
  new/old ratios 0.99655/0.99285. No full-method speedup is claimed.
- At 65 AO, sharing increases RSS 1,170,440 to 1,233,348 KiB and XLA scratch
  165,981,040 to 228,210,144 bytes. This measured speed/workspace tradeoff is
  explicit; it is not a universal memory reduction. Consumer reservations and
  smaller tiles still govern whether a batch is affordable. Timing predates
  the array-only shape/normalization fixes; the native kernel is unchanged.
- Root reviewed the equations, ownership, source hashes, exact process exits,
  XML and comparison records. GPU7 UUID selection and idle pre/post checks are
  recorded; GPU6 was hidden. Evidence: local `ao2mo-batch/` manifests, logs,
  XML and small comparison JSON. Completion review closes this batch foundation;
  RCC/UCC consumers, bare-vvvv actions and whole-method budget gates remain open.


### 2026-10-22 — WF-08a: physical Fock transport at the SCF boundary

- Aniline's invalid MP2 forward state was traced to a virtual-Fock skew norm
  of 1.40e-12, above the unchanged 1.49e-13 Hermitian-solver threshold. The
  candidate inverse was otherwise finite with a 2.09e-14 true residual, and
  the SCF orbital gradient was 2.20e-12. Relaxing either guard would address
  the wrong layer. Actual raw AO/MO skew measurements after the fix are
  6.26e-14/1.43e-12, consistent with transport roundoff.
- The shared physical SCF adapter now computes sym(C.T F C). The full
  off-diagonal occupied/virtual Fock response remains live. For physical
  F=F.T and its physical perturbations this preserves the first and second
  derivative. Generic Hamiltonian transformations and explicitly supplied
  from_orbitals Focks are unchanged: invalid nonsymmetric inputs still fail.
  No SCF refinement, equation, tolerance or solver diagnostic was modified.
- TDD: eight failures/three passes before the fix; eleven focused checks pass
  after it, 40.49 s pytest / 43.85 s process, RSS 2,315,260 KiB. Twelve existing
  R/U/direct/full/N2 geometry, PySCF, finite-difference, HVP and nested tensor
  diagnostics pass separately, four deselected, 229.12 s pytest / 233.51 s
  process, RSS 2,472,412 KiB. CPU64-71, f64, GPU hidden; no local numerical
  tests. Raw logs, commands, exact exits and frozen source manifests were
  reviewed; these runs did not request JUnit XML.
- Same fixed aniline input, Cartesian 6-31G*, 119 AO/25 occupied/94 virtual,
  gives valid MP2 with correlation -0.9546311957491757 Eh and a 7.00e-15
  true inverse residual against 1.74e-10. Three SCF refinements remain enabled.
  A first complete finite gradient returned in 259.53 s, with 23,480,720 KiB
  RSS; its original three-warm process was deliberately stopped before it
  could meet its 600 s cap. The runner is being corrected to save each finite
  result immediately and request one warm repeat. This partial run is not a
  completed warm benchmark or medium-system PySCF gradient comparison.
- Final production hashes start `77d84747`/`8027ccad`/`7b6e0063`; only the three
  physical adapter sources changed in the frozen validation stage. Its older
  AO2MO/scheduling base is identified explicitly in the manifests; current
  combined-branch validation is still required. Evidence:
  `wf08-aniline-forward-audit/`. Completion review closes the correctness
  prerequisite only; WF-08 performance, direct scaling and total-memory
  acceptance remain open.


### 2026-10-22 — WF-08b: completed aniline gradient and warm reference

- Corrected the thin runner to save every finite first/warm result immediately
  with its fingerprint. An interrupted partial run stays incomplete. The
  subsequent same-input full-AO/selected-MP2 process completed one warm sample,
  exit 0 in 557.91 s: compile 14.00 s, first 263.36 s, warm 259.80 s; observed
  RSS 23,488,236 KiB and XLA scratch 22,467,944,088 bytes are separate metrics.
  CPU64-71, OMP4/BLAS1, f64, no GPU; 60,000 MB input budget. This is one warm
  sample, not a statistical scaling result, and it remains slow.
- An independent PySCF 2.13.0 full/Cartesian 6-31G* calculation used the same
  fixed 14-atom coordinates, hcore guess, 50 electrons, no frozen/DF, SCF
  energy/gradient targets 1e-12/1e-10 and four threads in the same affinity.
  HF seed difference is 1.59e-12 Eh; MP2 total/correlation differences are
  4.04e-12/1.83e-13 Eh. All 42 gradient components agree within 4.56e-8
  Eh/Bohr; translation residual is 3.82e-12 Eh/Bohr. Original acceptance
  thresholds 3e-9 Eh and 3e-7 Eh/Bohr are unchanged. HF takes 2.40 s and
  PySCF MP2 energy/gradient 13.11 s; those are distinct from JAX compile time.
- Source/input/library hashes match the previous isolated Fock-fix stage;
  this does not validate the later combined AO2MO batch branch. Root reviewed
  the actual result JSON, process exits, oracle differences and the 16-file
  evidence archive (195,681 bytes, SHA `98be5c04...b518`). Evidence:
  `wf08-aniline-forward-audit/`. Completion review closes this one-molecule
  numerical/performance observation. WF-08 direct/scaling/total-budget work
  and the large speed gap remain open, without repeating the failed 3-warm run.


### 2026-10-22 — WF-09b1: selected RCC preparation and one ground body

- `cc.integrals.from_orbitals` prepares the existing seven-block container
  from live C, full pre-freezing Fock/reference energy and one AO source.
  A single shared batch includes known caller/Fock/coefficient reservations.
  Native data promote Fock/energy to f64; raw array inputs retain their chart.
  Bare vvvv is still resident. The original raw `prepare_integrals` is unchanged.
- Extracted `run_cc_from_integrals` from the original ground entry. AST
  comparison verifies that its complete numerical body is unchanged; old
  `run_cc` now prepares then calls the same solver/implicit-response owner.
  No facade or post-CC consumer has switched to this preparation yet.
- Initial source `3868f544`: all 46 new cases pass (42 forward/solver/structure/
  budget plus four raw/native coefficient/geometry JVP/VJP/HVP/FD), then 41
  existing RCC, CC2/LCC, QCI, Lambda/(T)/density and virtual-action regressions
  pass. These geometry tests hold C/F/E as independent supplied inputs; they
  are not complete SCF-relaxed CC nuclear gradients. All process exits zero.
- A later shape-before-budget regression failed three scalar-source cases,
  then reused the shared AO2MO source validator. Final source `9abcb69a`
  passes 37 overlapping array/response/shape/reservation cases. No native
  mathematics or prepared numerical body changed. Do not add overlapping
  phases as independent test coverage. CPU80-87 only, no local numeric/GPU test.
- Isolated matched RHF-frame preparation compares full-MO AO2MO then block
  selection with the new preparation, not seven redundant independent
  transforms. Water/Cartesian 19 AO: warm 21.017 to 17.731 ms; scratch
  2,691,816 to 1,273,568 bytes; RSS essentially unchanged. At 65 AO: warm
  0.477329 to 0.554637 s (+16.2%), scratch 364,736,040 to 228,244,000 bytes,
  RSS 1,314,344 to 1,251,212 KiB. This explicit speed/memory tradeoff remains
  below the investigation threshold; it is not a universal speedup or 2x RSS
  reduction. Timing used the earlier source; only array validation changed.
- Maximum block/Fock/reference differences are 1.09e-13/1.90e-12/3.56e-13 Eh.
  Root reviewed the code, raw XML/process exits, AST assertions and hashes.
  Evidence: `rcc-prepared/`, forty small JSON/XML/log files, no large arrays.
  Completion review closes this pure foundation. Prepared post-consumers,
  facade/relaxed geometry integration, bare-vvvv actions and joint amplitude/
  AD budgets remain WF-09b2/10b and subsequent tasks.


### 2026-10-22 — Plan convergence rule

- Applied the user's limit to every active TODO: one child level, at most four
  children. The existing thirty-five parent items are unchanged; no new parent
  was introduced to hide deeper work. Folded repeated foundation checkpoints
  into WF-02/03 and flattened RCC integration into WF-09a through WF-09d.
- Existing evidence and verified progress are preserved. Old multi-part labels
  in this historical ledger remain identifiers for their original frozen runs,
  not active child plans. The executing prepared Lambda/property work is now
  part of WF-09c; CI column scheduling and MP staging remain WF-20a/WF-25a.
- Local structural inspection checks only the Markdown: thirty-five unique
  parents, checkbox indentation zero or two spaces, and maximum four children
  per parent. No numerical code or method scope changed. Further fixes and
  test shards must converge on these existing acceptance conditions.


### 2026-10-22 — WF-20a: compact CI multi-column lifetime

- Replaced the Python list of four-column action results plus concatenate
  with the existing shared `apply_columns` loop. It writes one final output
  and executes one static batch body. Empty and at-most-four-column inputs
  retain their existing direct action; rank projections, scalar MP actions,
  integral kernels and numerical/eigen-response owners are unchanged.
- Structural RED: one failure, twelve numerical passes. All fifteen new
  R/U value, tail, coefficient/integral JVP/VJP/HVP cases then passed, exit 0,
  330.64 s pytest / 343.59 s process. Two GPU7 cases passed in 149.18 s,
  including an explicit device assertion on every response output leaf.
  GPU6 was hidden and GPU7 was idle before and after the UUID-selected run.
- Six existing solver/PySCF/FD/vector/rotation cases were selected. Their
  combined process hit its 600 s total cap after four completed cases; that
  process is not marked passed and produced no complete XML. Only the two
  remaining cases were rerun separately: vector response 331.75 s and multiroot
  rotation 105.16 s, both exit 0 with complete XML. No threshold or case changed.
- Matched synthetic nmo12/nocc4/CISD1425, f64, seed2312, CPU72-79,
  OMP4/BLAS1, five warm samples, old/new isolated processes: at 24 columns
  warm 3.57555/3.61390 ms (+1.07%), XLA scratch 8,304,040/2,231,864 bytes,
  observed RSS 3,342,196/2,228,380 KiB and compile 38.80/13.36 s. Maximum
  output difference is 2.67e-15. Single-column scratch/output is unchanged;
  warm 0.62723/0.65838 ms (+4.97%). This is a sigma-action measurement,
  not a converged-CI or whole-gradient scaling claim.
- The original stage used the pre-relocation shared memory module. Root
  verified its `apply_columns` AST is identical, then integrated the current
  `_memory` validator import in a separate frozen stage: four structural/
  empty/tail checks pass, 34.95 s, exit 0; all current hashes match locally.
  Final CI/column-test hashes start `4d35c458`/`97ee14e5`. Opt-in GPU assertions
  were added after the original CPU run; its original test hash is retained.
- Root reviewed the code, independent math/dtype/closure/AD review, raw logs,
  XML, process exits, comparison JSON and source lineage. Evidence:
  `ci-columns/`. Completion review closes compact column scheduling only;
  byte-selected creator/spectator/RHS widths, caches, integral preparation
  and whole-state/AD budget ownership remain in the existing WF-18/20/29 scope.


### 2026-10-22 — WF-09c: prepared Lambda and one-body partial

- Restricted raw Lambda now prepares then calls `solve_lambda_from_integrals`;
  R/U share the same original VJP/linear-solve/packed-dual numerical body.
  No additional solver or integral copy is introduced. A prepared density
  partial perturbs only active Fock/diagonal energies and the full occupied
  reference trace, retaining the supplied ERI objects and live T/Lambda response.
- Raw density still prepares at sym(h), including off-manifold h, then returns
  the original h dtype. Prepared Fock data are not symmetrized. Tests preserve
  frozen core/virtual mappings, electron count, one-body contractions and
  failed-state diagnostics; no derivative through either state is detached.
- Twenty-six new distinct cases and twelve old R/U regressions pass across
  recorded shards: four restricted models, packed duals, full/frozen joint
  h/g JVP/VJP/HVP with reconverged finite differences, raw first/second
  response, mixed dtype, empty spaces and PySCF density/Lambda references.
  The first batch retains its real exit 1 after seventeen passes and an LLVM
  allocation failure; the failed f32 case passes in a fresh process with the
  same source, test and tolerances. Other completed shards exit 0. Cumulative
  JIT resource pressure is a hypothesis, not an observed tensor-capacity cause;
  no global VM setting changed. No dependency skips were recorded.
- Root and an independent read-only reviewer checked the math/chart, code,
  XML/exits and matching source hashes `012b0164`/`837b791b`/`59ed2410`.
  Evidence: `wf09b2-prepared-properties/`, thirty-one small files, archive SHA
  `55c9df1e...f2c3`. All tests used CPU80-87, no GPU or local numerical tests.
- This closes the executed Lambda/1-RDM part only, within the fixed WF-09c.
  Prepared triples/EOM/RDM2 consumers remain; facade/relaxed geometry and
  complete CC/Lambda/AD memory ownership remain WF-09d/15/29. No new child
  plan or broader geometry/performance completion is implied.

### 2026-10-22 — WF-25a: stable MP recurrence compilation

- The public series entry validates and canonicalizes static occupation/frozen
  metadata, then reuses one private compiled recurrence. Physical h/g/nuclear
  inputs and both numerical tolerances stay live, including weak/strong float32
  comparison semantics. Defaults, required waves, Taylor coefficients and the
  existing shared implicit linear solve are unchanged. Only thirty-five
  production lines change; no second Taylor engine or method-local solver.
- Fifteen focused CPU cases, six target/oracle cases and four GPU7 cases pass.
  Device assertions cover every MP result/T2/wave and JVP/VJP/HVP output leaf.
  All thirteen original noncanonical cases pass, including MP4/MP6 strictly
  degenerate internal orbital spaces, first/second response and invalid poles.
  The original UHF MP test and water HF/MP2...MP8 public example also pass.
- Original series coverage maps all sixty-four cases: fifty-three have direct
  JUnit passes; eleven are completed sequential progress points in a saved
  aggregate process that later hit its 600 s cap. Their IDs are inferred from
  the recorded serial collection order, not synthesized into passing XML.
  Remaining cases were run separately. An actual missing-fixture failure was
  retained, then passed after copying the unchanged shared fixture. Three SSH
  255 events are transport failures; recovered process records establish exits.
- The unchanged final H6 cached-response case timed out at 600 s, then passed
  its single 1200 s gate: 859.90 s pytest / 887.14 s process, RSS 10,876,072 KiB.
  Compile logs show that cold cost remains substantial. No timing improvement
  is inferred from that validation run.
- Matched synthetic three-orbital/one-occupied MP6, float64, default T2 retained,
  CPU88-95, OMP4/BLAS1: public series warm 175.615 -> 0.659 ms and tensor warm
  171.810 -> 0.396 ms; first calls 15.981 -> 13.467 s and 11.738 -> 1.455 s.
  These measure small-input eager-dispatch elimination, not molecular scaling
  or removal of combinatorial coefficient/history storage. Outer-JIT cold
  times remain comparable. No tolerance, excitation space or output was reduced.
- Root reviewed code, raw XML/process exits and the explicit coverage mapping;
  all ninety-nine evidence files and current source hashes match. Production
  `series.py` SHA starts `899bc6ce`; evidence is
  `mp-series-staging-evidence/` and `mp-series-staging-summary.json`. All numerical
  work was remote; GPU6 was hidden. This closes WF-25a only: packed contractions,
  history scheduling and full molecular block/AD integration remain WF-24/25/26.

### 2026-10-22 — WF-10b: explicit native virtual action, performance open

- `ao2mo.actions.VirtualERI` contracts general nonsymmetric virtual RHSs via
  Cv X Cv.T -> native exchange K -> Cv.T K Cv. No complete AO/MO/vvvv is
  materialized. A single pytree retains live basis/coefficient response;
  constrained multi-block reverse loops rematerialize density/K. Nested basis
  projections are explicitly rejected. Raw dense integral charts and the small
  virtual-space fast branch remain unchanged.
- Restricted `from_orbitals(..., vvvv='direct')` prepares six shared blocks and
  this action in the existing ten-field container. `vvvv='dense'` remains the
  default. The original forward byte plan charges native/bridge/projection/RHS
  workspace and caller reservations; it is not a complete solver/AD/process cap.
- Final CPU72-79, float64: twenty provider/import/tail/geometry/coefficient/RHS
  checks pass, plus 109 raw/default-dense/implicit ground response regressions.
  XML has zero failures/errors/skips; process exits are zero, 50.50/384.54 s,
  with RSS 1,818,712/5,462,980 KiB. No native build, local numeric test or GPU
  was used. Separate combined-consumer and target-device gates remain open.
- The physical action uses the existing water/cart cc-pVTZ RHF frame, 65 AO
  and sixty actual virtual orbitals, input SHA `6212593e...ea0e`, 512 MB and
  identical 4x4x60x60 RHSs. All 57,600 outputs agree with L-infinity 3.511e-12
  (atol 2e-11, rtol 2e-10). Five-sample warm dense/direct medians are
  14.192/397.500 ms, RSS 1,119,300/654,520 KiB, XLA scratch
  103,680,000/1,082,944 bytes. The action is **28.008x slower**. Its native
  exchange stage costs 392.865 ms, including integral traversal and density
  contraction; density/projection cost 0.487/0.547 ms. This is not a timing of
  the isolated integral kernel. The earlier C25x48 overcomplete probe is kept
  only as algebraic stress evidence, not physical virtual-space scaling.
- Water/STO-3G CCSD converges in thirteen iterations on both paths, residual
  about 7.866e-11 and total-energy difference 2.842e-14 Hartree. Tiny-space
  direct warm time is also 17.437x slower; its RSS is slightly higher. These
  observations justify retaining explicit opt-in, not enabling it by default or
  closing the efficiency criterion. No subsequent expensive molecule was added.
- Root and an independent read-only reviewer checked the action/dense charts,
  tail/AD and budget boundary. Root verified six-file hashes, XML/exits, the
  physical per-element report and input lineage. Evidence: `wf10b-native-action/`, final
  action SHA `042b1243`, native library `3eeaa2e5`. The remaining joint budget,
  provider consumers and performance work stays within WF-10/14/15/16/17/29;
  no new plan branch is created for these findings.

### 2026-10-22 — WF-08: bounded forward/response timing diagnostics

- Retain two thin test runners: whole MP2 molecular compile/first/warm/RSS
  measurement, and component timing using the actual C/Fock cotangents and
  existing source solvers. Inputs, units, backend/environment, exact command,
  source hashes, incomplete phases and finite first/warm outputs are persisted.
  Instrumentation changes neither equations nor solver options. The optional
  captured diagnostic operator is restricted to at most ninety-six variables.
- Forward counts wrap only the existing nonlinear minimizer's GMRES, with
  start/action/end scalar callbacks and restoration in `finally`. They add no
  HVP or residual evaluation. Water baseline/counted states and histories are
  identical; three solves use 42/42/41 actions. Fixed aniline uses 84/84/43
  actions (211 total), three unchanged polishing steps, and the same final
  stationary norm 2.20359e-12. Its instrumented first state costs 76.608 s;
  there is no warm sample. Solver `info=0` is not a residual certificate.
- The source polishing history records selected-shift outer relative residuals
  4.975e-8, 3.392e-8 and 4.930e-5 for RHS norms around 4.4...5.8e-12. This does
  not establish inner 1e-11 relative accuracy or expose every failed shift.
  A possible normalization/roundoff floor is only a source-informed hypothesis.
  No inner/outer tolerance, refinement count or regularization was changed.
- A native-direct aniline diagnostic hit its actual 600 s process cap with no
  forward-GMRES callback. One final 120 s localization also timed out (exit124):
  lowering 14.330 s, compilation 13.079 s, then first stationary-state execution
  before any GMRES callback. Thus the unfinished stage is execution before
  GMRES, not compilation. No direct stationary/MP2-gradient/warm-time success is
  implied. An SSH255 is retained separately from recovered real process exits.
- Root checked thirty-two evidence entries and unchanged source/native hashes;
  the runner SHA starts `35ea7445`. Evidence: `wf08-forward-count/` and the
  existing `wf08-aniline-forward-audit/`. All numeric work used remote CPU64-71,
  no GPU. WF-08 remains open: this bounded diagnosis is not an additional
  subplan, a completed direct benchmark, or a reason to repeat long jobs blindly.

### 2026-10-22 — WF-09d: shared solved-HF source prerequisite

- Move the existing eager RHF/UHF source reading into the internal small
  `ao2mo.scf` module, not MP or CC. One six-field NamedTuple carries C, full MO
  Focks, full occupations, reference energy, the existing AO hcore and exactly
  one AO representation. MP delegates to it and its original orbital preparation;
  no complete MO integral, J/K evaluation, provider hierarchy or solver is added.
  SCF freshness, occupations, HF restrictions, physical Fock transport and
  native/projected/dense/s4/DF/legacy-JAX selection/preflight remain in force.
- Independent review identified mutable public `e_tot` as an unsafe authority
  for the new CC consumer. A six-case RED demonstrates the alias/missing-field
  failures. The helper now reads RKSResult.total_energy or the solved UHF
  reference.mf_energy, rejecting absent/None records. It retains hcore by
  reference for MO-density dtype metadata without transforming a full h_MO.
- Final twenty-six narrow cases plus two existing actual R/U physical-Fock
  workflow cases pass: exit0, 32.87 s pytest / 35.23 s process, RSS 2,084,716 KiB.
  Seventeen existing direct MP2/PySCF/legacy-JAX cases previously passed in an
  invocation that also had two genuine old-workflow failures; that invocation
  stays exit1. Only the committed physical-Fock workflow dependency was then
  supplied and those two cases rerun successfully. Neither the old failed run
  nor two initial array-comparison test errors are relabeled as clean passes.
- Root reviewed the extraction and authoritative-energy fix, all sixty archive
  entries, raw XML/exits and local/final source hashes (`9ab61768`/`5eea2af9`/
  `f9a86c4c`). Evidence: `ao-source-evidence.tar.gz` (SHA `14ae5385...7312`) and
  `ao-source-summary.json`. CPU88-95 only, no local numeric tests or native build.
  This closes the common source prerequisite within WF-09d, not its remaining
  CC facade/geometry integration or joint memory ownership; no new child TODO.

### 2026-10-22 — WF-10b/30: finite CUDA action component

- Two native/projected virtual-action cases pass on GPU7 with joint geometry,
  coefficient, RHS and projection value/JVP/VJP/HVP against CPU ERI oracles.
  Every input and output response leaf is explicitly checked on its requested
  device; full AO ERI fallback is forbidden. Production code is unchanged.
- The UUID-selected GPU7 has 15 MiB/no compute app both before and after;
  GPU6 is never selected. CPU104-111, preallocation disabled, exit0, 4.94 s
  pytest / 5.59 s process, zero failures/errors/skips. Test SHA `8a89a965`,
  actions `042b1243`, native `3eeaa2e5` match root inspection and XML. Evidence:
  `wf10b-native-action/cuda-component-manifest.json` and companion logs/XML.
  This is the finite action/AD device gate, not full CC CUDA/geometry scaling,
  a total GPU budget guarantee, or closure of the open performance criterion.

### 2026-10-22 — WF-09c: remaining prepared consumers accepted

- Prepared triples/scalar and EE/IP/EA build/run entries share the original
  numerical bodies and solver ownership. RDM2 consumes full-frame metadata and
  right/left states without integral preparation; its original dense full-spin
  allocation and dtype-promotion behavior remain explicit. Lambda/1-RDM were
  accepted in the earlier checkpoint. No method-local solver is introduced.
- Independent/root review exposed two semicanonical-auto raw-chart differences:
  damaged T2 changes the packed spin residual norm, and raw ERI pair-exchange
  breaking changes the two UHF Focks independently. The raw API retains its
  original UCC dispatch. Its complete diagnostics/dtypes, a critical validity
  threshold and the directed g[0,0,2,3] VJP match the original UCC oracle.
  The new prepared general entry declares its physical single-Fock/pair-symmetric
  T2 chart and rejects states outside it; it does not construct spin vvvv merely
  to reproduce a diagnostic for an unsupported damaged state.
- Thirty new unique cases plus ten existing reference cases pass in eighteen
  bounded batches, all exit0/no skips, CPU80-87. These cover six triples choices,
  JVP/VJP/HVP and reconverged FD, all EOM sectors' energy/left/right vector
  response, full/frozen RDM2 outer AD, capacities, invalid states/denominators,
  explicit AD-order guards and PySCF references. The longest process is 355.06 s.
  Initial missing-entry and real chart-guard RED failures remain recorded.
- Unchanged canonical contractions/loop AST match the original and both tested
  prepared versions (`7339a878`). The final changed prefix, six-case value/
  diagnostic/dtype matrix and semicanonical response pass separately; unchanged
  EOM/RDM files and test drivers have matching hashes. This reuses valid
  evidence without rerunning an identical canonical campaign for a new filename
  hash. Root reviewed the sixty-nine evidence entries, exact exits/XML, source
  hashes and independent review. Evidence: `wf09c-prepared-consumers/` and archive
  SHA `b5436b75...5c01`; final triples/EOM/properties `25f1ecc8`/`a21ff091`/
  `7c194b56`. No local numerical work, native build, GPU or extra TODO was used.
- Closure applies to WF-09c's reuse contract. Dense EA Wvvvv/RDM2, virtual-provider
  compatibility beyond ground actions, total-budget ownership and full method
  scaling remain their existing WF-10/14/15/16/17/27/29 gates. Facade/relaxed
  geometry integration is the dependent WF-09d, with its own measured evidence.

### 2026-10-22 — WF-09d: restricted SCF facade and relaxed response

- Restricted SCF CC now prepares one selected snapshot and reuses it for ground,
  Lambda, density, triples and EE/IP/EA. Explicit MO/UHF/ROHF references retain
  their raw path. Full nocc/nmo/frozen and density dtype metadata remain separate
  from the active container. Failed rebuilds clear the old state before validation.
- The relaxed restricted CCSD/(T) adapter transports live C, Hermitian F and HF
  energy into the same pure preparer. Native direct removes moving full AO ERIs;
  neither route constructs complete MO ERIs. Bare-vvvv stays dense. Streamed (T)
  keeps its second-order AD contract and original third-order rejection.
- Fourteen new facade cases pass across fixed shards, including a real native
  direct SCF -> CC -> Lambda/density path that forbids complete AO/MO integrals.
  Fifty final metadata/legacy executions verify budget forwarding, energy-alias
  safety, normalized HF names, frozen/raw/UHF compatibility, stale states and
  existing eager EOM/FCI-limit references. Counts overlap the narrow helper
  campaign; they are executions, not fifty additional physical molecules.
- Both new H4/STO-3G geometry cases pass final value/gradient/JVP/HVP against
  complete-MO oracles, 3e-9 Hartree energy tolerance, duality/translation and FD
  at 3e-4/1e-4 Bohr; (T) is explicitly nonzero. Final processes 154.43/202.12 s,
  RSS 2,742,480/2,983,392 KiB. The unchanged CCSD third-order mixed Raman test
  passes separately, 233.75 s, RSS 3,481,588 KiB. These are validation runtimes,
  not warm benchmarks or a whole-process limit from the 32/128 MB array budgets.
- Root and independent review resolved the authoritative-energy issue and
  checked raw paths/metadata/AD ownership. Final changed code hashes and exact
  XML/exits match; original REDs/earlier assertions remain recorded, alongside
  the stronger final geometry reruns. Evidence: `wf09d-rcc-integration/`, all
  numeric work remote CPU64-71/96-103, no GPU/build/local test. Newer WIP UCC/
  non-Hermitian planner changes were excluded from this frozen acceptance.
- All four WF-09 children now meet their restricted-preparation scope. Shared
  total solver/AD ownership, unknown external JIT target capacity, unrestricted
  facade migration and large-method scaling retain their existing separate gates.

### 2026-10-22 — WF-12b: pure UHF source preparation accepted

- A small `cc.unrestricted_integrals` module maps four C subspaces through one
  shared AO2MO request batch into the existing six SpinCCIntegrals blocks.
  Beta-alpha terms retain the raw gab pair-transpose chart, including arbitrary
  nonsymmetric AO/Fock inputs. Inputs, requests, six final blocks and assembly
  buffers share the original MB reservation. Dense spin-vvvv remains explicit.
- `run_ucc_from_integrals` reuses the original packed-amplitude solve. The raw
  preparer and the full space/guess/solve/result body have matching AST against
  the committed baseline. No new nonlinear/linear solver or facade path.
- Thirty-two new cases and thirty-two old raw/PySCF/implicit/invalid/empty cases
  pass, exit0/no skip/timeout. Native bare/projected, unequal spins/frozen spaces,
  dense/s4/DF, float32/64 and joint C/F/E/source/geometry JVP/VJP/HVP are covered.
  Root and independent review checked charts, axis order, dtype and byte owners;
  all forty-five evidence files and final source hashes match (`94051cb5`/
  `0559d7ef`). CPU88-95 only, no local numeric/GPU/build/install.
- Matched 4 AO/8 MO, DF6 algebraic prep, f64, seed6514, fifteen warm samples:
  raw/selected 0.16363/0.13937 ms, XLA scratch 260,417/57,153 bytes, same
  236,888-byte output and max difference 1.39e-17. Compile 0.493/0.712 s and
  first execution 3.56/15.36 ms worsen; end-process RSS 1,230,444/1,242,380 KiB
  does not decrease. Both include the same oracle/setup/compiler. This is not
  physical molecular scaling or a total-UCC memory/AD claim.
- Evidence: `ucc-orbitals-evidence.tar.gz` SHA `f4906e1f...b07e`, summary and
  static proof alongside it. The fixed producer slice is accepted; no more
  experiment branches are attached. The existing WF-13/14/15/29/32 gates retain
  residual/consumer adoption, amplitudes, adjoints and combined measured scaling.

### 2026-10-22 — WF-13: contract spin virtual dressings before expansion

- The spin residual applies its virtual intermediate directly to tau. The ovvv
  and oovv dressings first contract the trial pair into batch*o*v/batch*o**2
  arrays; no large dressed-vvvv array is formed. Occupied-dominated spaces keep
  the original smaller transient W. Bare spin-vvvv remains a dense input.
  Signs/indices do not assume raw ERI, amplitude or trial-pair symmetries.
- Both branches reuse the same explicit tau, preserving its independent AD
  and its two live appearances in the residual. The original explicit W is
  retained as the oracle. The nonlinear/implicit/adjoint solvers, packed
  coordinates, physical equations and dtype promotion remain unchanged.
- Twenty-four new cases pass: float32/64, mixed precision, empty spaces,
  arbitrary leading dimensions, raw JVP/VJP/HVP, independent tau response,
  and a fail-on-dressed-W residual trap. Thirteen existing cases pass, including
  UCCSD/UCCD, unequal spins/frozen/empty channels, PySCF residual/amplitudes,
  implicit response, and full H3/STO-3G relaxed geometry gradient/HVP with two
  finite-difference steps. All final processes exit0; no skips/timeouts. The
  missing-action RED, old residual trap and one fixed tau-branch RED are saved.
- Matched f64 action, o=8/v=64, fixed seeds6521/6525, CPU64-71, fifteen warm
  samples: explicit/factored median 385.635/42.697 ms, XLA scratch
  402,653,184/155,189,248 bytes, output 2,097,152 bytes unchanged, maximum
  difference 3.89e-16. Isolated process RSS is 1,333,388/1,220,480 KiB;
  compilation 0.0462/0.0715 s and first action 0.4059/0.0631 s. This is a seeded
  algebraic action, not physical molecular scaling or the total CC speedup.
- Root and independent review checked the old raw chart, branch choice, dtype,
  tau ownership and test oracle. All thirty-four evidence entries, raw XML,
  process exits and final patch hashes match; raw UCC's solve tail is AST-identical
  to the newer prepared-bridge body. The frozen source manifest records this
  baseline explicitly rather than claiming a test of concurrent unrelated WIP.
  Evidence: `wf13-spin-action/`, archive SHA `30cfdc97...4359`; final kernels
  `3d1e53ca`/`f1c2169d`. All numeric work remote CPU64-79, no GPU/build/install.
- This finite residual-action slice is accepted within the existing WF-13.
  Dense spin-vvvv, packed-amplitude expansion and joint method/AD budgets remain
  open; there is no new TODO branch or claim that the whole parent is complete.

### 2026-10-22 — WF-29c/16/17: shared non-Hermitian primal budget

- The shared non-Hermitian planner retains requested/guard roots, iteration
  limit, seed, convergence and response settings. It selects the largest legal
  Krylov capacity within the original configuration and byte budget, preserving
  mandatory algorithm/initial-column minima, and bounds both right and left
  action columns. Dense real arrays/SVD and complex eigenspaces have separate
  byte widths. The original MB budget receives its single 20% reserve once.
- EOM keeps its public three-return operator API. A small private context
  accounts for ground/integral payload, actual linearized/charged arrays,
  action constants and conservative known JAX temporary sums. Raw selected
  payload is shape-checked before eager preparation; the reviewed allocation
  window was first reproduced and then repaired. Unknown allocator/library
  state and implicit-response Krylov arrays are not claimed as covered.
- The facade dynamically inherits CC's max_memory, supports explicit override
  and reset, includes the resolved budget in freshness checks, and clears stale
  results on failed rebuild. Low-level max_memory=None retains the original
  mathematical path without budget traces. No eigensolver or adjoint math changed.
- Twenty-eight new plus forty-nine existing cases pass in seven final batches,
  all exit0/no skips/timeouts: shared dense/Davidson, mandatory root/capacity
  guards, right/left batching, JVP/VJP and clustered-root response, EE/IP/EA roots
  and derivatives, facade inheritance/cache and raw preflight. Four RED batches
  retain seven expected failures. Longest final runner 213.98 s, remote CPU80-87.
- Matched n=256 two-sided primal pair, identical input SHA/config/tolerances:
  capacity 32->14, action width2, root max difference 1.39e-15; known solver
  reservation 659,900->386,228 bytes. Warm medians 5.805/6.717 ms (+15.7%);
  sampled RSS 709,533,696/709,636,096 bytes is essentially unchanged. Both fresh
  processes exit0. This is capacity control, not a whole-EOM memory/speedup win.
- Root and independent read-only review checked the raw preflight fix, source
  lineage, all final seventeen source/test hashes, raw XML and actual exits.
  Evidence: `wf29c-eom-memory/`, archive SHA `feec3d39...cbd8`, final planner/
  EOM kernels `077edee6`/`ddb0da15`. Concurrent MP/UCC WIP is excluded from the
  frozen baseline. No GPU, local numeric work, build or dependency installation.
- This primal slice is accepted within the existing WF-29c/16/17; no additional
  task branch is created. Charged Wvvvv algorithm/lifetime changes, Lambda/DIIS,
  implicit-response storage and joint method/AD/process budgets remain open.

### 2026-10-22 — WF-25/29: exact canonical Taylor linear solves

- A static explicit exact-diagonal declaration on the common LinearOperator
  permits a stopped numerical diagonal factor inside the existing checked
  custom_linear_solve. The original live action owns all AD; primal, tangent
  and transpose RHS still receive true-residual checks. Invalid/nonfinite/zero
  factors reject even zero RHS. Block broadcasting and RHS/output dtype are
  preserved; no denominator shift/clamp or second response solver is added.
- Only canonical MP declares this exact structure. Noncanonical MP retains its
  full block-diagonal Fock action even at a numerically diagonal point, including
  its off-diagonal first/second response. Matrix inputs and preconditioners do
  not imply exact structure. The old generic solve/as_operator ASTs are unchanged;
  the MP kernel differs only by this single static keyword, not its equations,
  Taylor recurrence, coefficients, orbital space or physical tolerances.
- Seventy-three fixed cases pass: fifteen new common solve and five new MP
  cases, twenty-four original MP and twenty-nine old shared-solver cases.
  Coverage includes vector/block, float32/64/mixed dtype, zero RHS response,
  invalid and stale factors, transpose/second response, R/U series/tensor raw
  H/G JVP/VJP/HVP, coefficients/T2/waves, strong/weak tolerance boundaries,
  independent E2...E12 and true-pole rejection. All final groups exit0.
- The API-missing RED is retained. A separate four-case observer failure was a
  test's reused outer JIT, not a demonstrated numerical discrepancy; that group
  was interrupted with exit2. Separate lambda wrappers repaired only the test
  tracing boundary, then all five MP cases passed with unchanged physics/tols.
- The original H6 cached-response test (SHA `0cfdaf43`) was attempted once in a
  fresh process, unchanged input/fixture/assertions and recorded compile/cache
  flags on CPU88-95, with a fixed
  595-second timeout plus 5-second kill bound. Actual exit124 at 595.044 s;
  cold test body did not complete and warm repeat never began. There is no
  completed H6 value/gradient/HVP, warm time, speed ratio or terminal RSS result
  from this attempt. A running VmHWM observation of 8,177,584 KiB is explicitly
  partial; the 596,264 KiB initial phase snapshot is not a measured peak.
- Root and independent review checked numerical-factor versus AD ownership,
  the unchanged raw charts, structural declaration, zero-RHS/transpose tests
  and finite scope. All numeric work is remote CPU64-71/88-95, no GPU, builds or
  installs. The correctness slice is accepted; H6 performance and the existing
  Taylor history/packed-space/joint-budget gates remain open. No cache-warning
  investigation, additional attempt, extra TODO or automatic timeout extension
  follows this finite measurement.
- Root verified all sixty-one archived evidence entries, six final patch/test
  hashes, raw XML and process exits. Evidence: `exact-diagonal-evidence.tar.gz`
  SHA `94b37c66...d26a`, summary and static proofs alongside it; final common
  operator/solve/MP kernels `b5936aed`/`1d168384`/`2a5696d1`. The fixed workflow
  ended with no active owned runner or reserved CPU, without merging/pushing.

### 2026-10-22 — WF-29: shared linear and packed nonlinear phase budgets

- Shared plan_linear/plan_nonlinear keep the original MB budget and one 20%
  reserve. Only optional GMRES restart and DIIS history shrink; solver method,
  tolerances, iteration limits, damping and physical convergence are unchanged.
  Mandatory DIIS history is two, except an explicitly requested original one.
  Exact diagonal/Cholesky factors do not pay for unnecessary generic work.
- solve_linear/checked_linear_solve/solve_nonlinear/attach_root accept optional
  max_memory/reserved_bytes/operator_min_bytes. Known visible inputs are charged
  locally; callers declare physical closure/residual workspace. DIIS rings end
  before implicit root response, which uses the same budget in its linear phase.
  Custom tangent solvers keep their own structure and ownership.
- Nine new budget/early-allocation/AD cases and fifty-five existing numerical
  cases pass. The original architecture check exposed a pre-existing private
  JAX import in the eager solver. A guard-only repair keeps old public API support
  or uses the JAX extension trace-state API, rejecting JIT/grad before physical
  map evaluation even for closed constant seeds. Its eight eager cases and the
  architecture check pass. The failed original batch is retained, not reported
  as all passing. Older JAX behavior is branch-simulated; no old runtime installed.
- Root/independent review checked the phase lifetimes, minima, original AD
  owner, guarded eager compatibility and unchanged numerical map/DIIS. All
  thirty-two archived entries and final seven source/test ASTs match. A local
  documentation-only clarification describes complete declared action scratch,
  including generic direct-solve column batches; no numerical change followed.
  Evidence: response-memory/common-evidence.tar.gz SHA 82f242cc...da02 and
  common-root-verification.json. All numeric work remote CPU64-79, no GPU/build.
- This closes the common phase-planning implementation, not the remaining
  per-method physical-state/closure integration or arbitrary outer AD-tape and
  allocator bounds. Those remain in their original method and final-audit TODOs.

### 2026-10-22 — WF-12/14/15: selected UHF consumers and relaxed response

- UHF SCF CC reuses the shared source adapter, selected six-spin-block producer
  and existing packed UCC solve; Lambda, local one-RDM and triples share that
  snapshot. Explicit MO/ROHF references retain the original raw path and guards.
  The raw UCC right solve/residual, left equation and triples numerical tail are
  AST-identical. RDM local Hermitian h-alpha/h-beta partials preserve their old
  separate dtype chart, full occupied traces, active/frozen map and dual factors.
- CorrelatedEnergy now prepares UHF CCSD/(T) directly from live C, saved Focks
  and total HF energy, including native direct inputs. It no longer requires a
  complete moving MO ERI for UCC. Its MP2 branch is AST-identical to the baseline.
  Native block AD, stationary SCF and the original CC/(T) response owners remain.
- Ten new plus twelve old cases pass in six final groups, all actual exit0 and
  no skip. LiH+/STO-3G tests enforce a nonzero triples correction, native direct
  no-full-AO/MO traps, complete gradient/JVP/HVP and two finite-difference steps.
  Fixed-frame Lambda/density includes arbitrary raw h/g joint first/second
  response; frozen, UCCD and original R/U/ROHF/triples guards are covered.
- Root and independent review found no remaining correctness issue in these
  seven source files. All seventy-three archived entries and local frozen
  source hashes match; executed-group JSON/source snapshots and raw XML are
  preserved. Evidence: ucc-scf-prepared-correctness.tar.gz SHA82911a97...d88f,
  source-manifest and correctness-summary. Geometry processes take 87.23/107.90
  seconds; these are validation runtimes, not converged-method warm benchmarks.
- The matched pipeline pair is unfinished: raw exit0 was observed, while
  selected never produced a completed result before node reboot/SSH loss.
  Recovery found no selected result/process; later authentication/I/O errors
  prevented a reliable retry. No matched performance, final RSS or scalability
  claim is made. All tests were remote CPU80-95 with GPU disabled.
- This accepted integration uses the existing WF-12/14/15 tasks; bare spin-vvvv,
  packed amplitude lifetime, contracted two-body properties and the later common
  solver-budget snapshot remain open. Unrun CCS/spin-block tests are excluded
  from this correctness checkpoint. No new TODO or subplan is introduced.

### 2026-10-22 — Local recovery checkpoint; remote validation still unavailable

- The engineering branch remains at ec194bf. The user's pause and subsequent
  resume preserve all uncommitted work and the existing accepted evidence.
  A unified read-only probe returned c20, uptime114.11 seconds, a readable
  project directory and the existing native library. This established neither
  stable storage nor readiness for numerical work: the first CC stage copy
  failed with `Structure needs cleaning` and `Input/output error`; subsequent
  root/CI/MP transfers returned SSH authentication exit255. No numerical job
  started in this recovery attempt. The user then explicitly selected local
  preparation until they notify a stable repair. No further remote probes,
  transfers or test retries are scheduled meanwhile.
- The two old CI/FCI groups without result JSON/XML and the old compact-MP AD
  group remain reboot-interrupted, not passing or numerically failing. The
  partially created packed-MP green4 and CC staging directories must not be
  reused without a new verified source snapshot after repair. No deletion,
  filesystem repair, credential change or native rebuild was attempted.
- WF-34 retains its earlier seven-case preliminary Dyson GREEN evidence, but
  the complete task remains open. Independent review identified missing
  resident owners: the separate packed Lambda adjoint, additional EOM arrays,
  ordinary reference dataclasses and known SCF payload/cache/parameter arrays.
  Five new parameterized cases are prepared for structured shape-only accounting,
  retained-state early rejection and facade deduplication. They have not run;
  the proposed common structured-accounting helper has not been implemented.
  Original H2/FCI residues, physical response/guard regressions and matched
  endpoint lifetime measurements still require execution.
- WF-23/24's current shared-CI-action packed factory is not the earlier
  six-forward-case snapshot. Current forward/dtype, actual rank-six target,
  joint AD, complete-series and matched performance gates remain unverified.
  WF-25's history/budget tests, WF-26's selected noncanonical R/U orbital-entry
  tests and WF-33's selected canonical RHF MP3 tests are prepared but unrun;
  those three production bridges/budget changes have not been implemented.
  There is no unchanged H6 rerun or claimed remedy for its previous timeout.
- WF-10/11/13's CCS lifetime and packed-spin-coordinate tests likewise remain
  unrun. The proposed model-specific CCS preparation/early return, elimination
  of per-action dense spin amplitudes, bare-vvvv efficiency policy and whole
  CC/Lambda/triples phase accounting are not implied by ec194bf's correctness
  checkpoint. WF-18/19/20/21/22/27 similarly retain their component evidence
  while newest joint ledgers, selected CIS(D), public/native geometry and
  matched whole-method measurements remain open.
- Local-only verification parses changed Python files with the standard
  library AST and checks `git diff --check`; it imports no numerical package
  and is not a numerical/AD test. The plan still contains exactly35 parents,
  thirty-seven children, one child level and at most four children per parent;
  no acceptance checkbox changes. A source snapshot, binary diff and SHA256
  manifest are preserved under the local wavefunction-memory artifact root.
  No merge, push or new software installation is performed.

### 2026-10-22 — Authorized Delta V100 allocation and target admission

- User explicitly moved the testing campaign to jy_gpu_2 and authorized one
  V100, 64 CPU threads, 256 GB host memory and two days after reading the manual.
  The latest available lwcluster/0.8.0 manual was read in full before submission;
  live queue, nodes, budget and storage were inspected. No c20 credential or
  compiled library was transferred. Slurm job 103505 passed the site's checks
  and runs in home on n002, from 2026-10-22 11:53:40 to 2026-10-24 11:53:40
  Asia/Shanghai. Requested memory 256G is 262144 MiB. CPU binding is exactly
  64 logical CPUs (0-31,40-71), not a claim of 64 physical cores.
- Only the assigned device 0 is used: Tesla V100-SXM2-32GB, UUID
  GPU-b720fb52-9628-c12c-2a38-bee0ad7a3cfe, driver 550.127.05. Separate CPU
  and CUDA float64 JIT/value-and-grad checks pass with zero gradient error
  and outputs on the intended devices. Existing environment: Python 3.11.15,
  JAX/jaxlib 0.10.0, NumPy 2.4.4, SciPy 1.17.1, PySCF 2.13.0, pytest 9.0.3,
  jax-xc 0.0.9 and CUDA 12 plugin/PJRT 0.10.0. No dependency installation occurred.
- All 1719 source/test/package-data files were SHA-verified from ec194bf plus
  the recorded WIP overlay. Initial archive SHA 7653bbf3...54d948. The old target
  native library SHA bdbf4a3a...e248f lacks ERI slice/slab symbols. An isolated
  offline build of the pinned current sources with installed CMake/GCC/BLAS
  exits 0 in 10.934 s; new library SHA eb9b1f1a...15193ed7. The original library
  remains intact. Upstream-header return warnings and NFS clock-skew warnings
  are retained, not described as a warning-free build.
- Target admission is verified by seven CPU cases (actual exit 0, process
  13.990 s) and two V100 cases (exit 0, 12.391 s): values/JIT, coordinate JVP/VJP,
  nonlinear HVP, static/dynamic padded slabs and GPU output residency. Raw XML
  confirms 9 passes/no skips, and source hashes are unchanged within each group.
  These gates validate the rebuilt target interface, not whole-method memory,
  performance or all of WF-30. Root verified the 28-entry admission archive
  SHA 670e8d81...0f0ff6 and wrote admission-verification.json under the local
  delta-v100-20261022 artifact directory.
- The initial srun invoked outside homefile was rejected before computation;
  the first build wrapper exited 127 before Python because /usr/bin/time is
  absent. Both are retained as runner failures. Use stdlib wall timing instead.
  Observed CPU steps inherited the sole GPU and queued unnecessarily; lost
  SSH connections did not terminate their pending launchers. Exact owned PIDs
  were verified/cancelled before replacement. An explicit no-GPU step with
  gres=none, mem=60G, immediate15s succeeds immediately inside the same allocation.
  Subsequent CPU groups use four disjoint 16-thread masks and ten-minute step
  caps, without increasing the 64-CPU/256G allocation. GPU tests are serial.
- The existing method owners resumed their finite unverified gates in private
  snapshots; they do not mutate the shared initial input. Old accepted source
  evidence is retained rather than rerunning entire suites. The damaged
  jy_gpu_1 stages remain untouched. No method parent checkbox is closed by
  this infrastructure/admission slice, and no merge or push is performed.

### 2026-10-22 — WF-29/34: structured ownership and streamed Dyson correctness

- The common shape-only tree_storage_bytes visits declared dataclasses,
  mappings, tuples/lists and namespaces without array values, host copies or
  arbitrary object reflection. It deduplicates aliases, protects cycles and
  recursively excludes already charged array descendants. Distinct view objects
  remain conservative charges, not measured allocator storage.
- Dyson's low-level owner now charges complete ground/Lambda/EOM state arrays,
  including the separately retained packed Lambda adjoint and guard arrays.
  The facade charges its reference, prepared integrals and declared SCF owner,
  excluding the low-level states. This includes ordinary reference dataclasses,
  AO/grid/SCF caches and external parameter arrays. The existing row-tile planner
  receives the same original budget; no physical or response equation changes.
- Five expected RED cases were observed first (exit 1, 8.191 s): two missing
  structured-accounting APIs, two retained-state omissions, and facade aliases
  plus distinct source arrays. The final group has 36 passes, exit 0, no skips,
  process 292.048 s: 15 common budget cases, 10 streamed/retained Dyson cases
  and 11 original endpoint/H2/FCI residue/response/guard regressions. Raw-input
  endpoint tests cover float32/64, independent PySCF values and joint T/Lambda/
  left/right-vector first/second response with two finite-difference steps.
- Root verified raw XML, all five executed source/test files against the local
  frontier, immutable per-run hashes and actual exits. Independent peer review
  found no remaining correctness/alias-accounting issue. All numerical work
  ran on n002 within job 103505, CPU0-15 and no GPU. Evidence archive SHA
  ad6a5710...5073a59 and retained-state-verification.json are stored beside
  the admission evidence; RED is retained in the earlier admission archive.
- This accepts the finite correctness/known-state slice within WF-29/34. The
  fixed 32-row ceiling has not yet received matched warm/RSS/compiler-workspace
  measurement; arbitrary outer AD tape/allocator state and whole-EOM scaling
  are not bounded by this helper. The parent checkboxes remain open, with no
  new TODO level or claim that the complete project is finished.

### 2026-10-22 — WF-11/13/15: model-aware CCS and physical spin boundaries

- CCS reuses the unchanged restricted R1 body and returns before the discarded
  doubles equation. The SCF producer requests five blocks and retains None for
  unused oooo/vvvv; nocc comes from ovov. Other models retain their seven-block
  selection and full numerical equation. Removing the static CCS early return
  leaves the original restricted equation AST unchanged.
- SpinAmplitudeSpace reuses the original independent-index ordering and packed
  metric while packing/unpacking physical alpha/beta blocks directly. Init,
  restart, result and Lambda boundaries no longer create a full spin-amplitude
  tensor just to select physical blocks. Same-spin projection, mixed dtype,
  empty/unequal sectors, level-shift denominators and Lambda's dual factor4
  are preserved. The residual still expands into the original full-spin chart;
  no claim that its working tensors or spin-vvvv storage are removed follows.
- Thirty-nine distinct cases pass in four final groups, all exit0/no skips:
  24 CCS/coordinate cases, two CCSD/CCD left/right joint JVP/VJP/HVP cases and
  13 directly affected original model/property/restart/frozen cases. Required
  RED groups were observed first. An initial integration test accidentally
  reused its outer JIT executable and falsely passed its allocation trap;
  that record is excluded. A forced fresh trace then produced the expected
  RED before the boundary connections changed. Two independent reviews found
  no remaining correctness issue.
- Matched coordinate-only benchmark, seed6551, occupations(8,7), virtuals(80,79),
  float64, Xeon Gold6230/n002, CPU56-71: old/direct-block warm medians
  20.827/4.061ms, XLA temporary95,068,488/5,472,488 bytes, maximum output
  difference0. Primary process peak RSS before the independent oracle was
  825,532/741,404KiB, while current RSS increased by8,552KiB. These measurements
  belong to the coordinate boundary, not a complete CC residual or solve.
- Root verified all75 payload hashes, ten final source/test hashes, raw XML
  and39 unique passing cases. Twelve static proofs preserve the original spin
  index/pack/unpack, residual/intermediate/property/triples and ground bodies.
  Archive412aaee6...af8248 and cc-delta-root-verification.json preserve exits,
  exact commands, source-before/after maps and the excluded cache-test record.
  All calculations ran inside allocation103505 on CPU56-71 without GPU.
- This finite slice stays within the existing WF-11/13/15 parents. Full spin
  residual storage, compact bare-vvvv supply, joint solver/AD phase accounting
  and whole-method measurements remain open. No new TODO/sublevel, merge or
  push is introduced.

### 2026-10-22 — WF-18/23: shared direct-action dependency isolated

- The optional checkpointed creator/spectator action and existing sigma's
  byte-planned RHS entry are a prerequisite of the packed MP candidate. They
  remain in CI's shared direct action; no separate MP normal-ordering engine
  or eigensolver is introduced. Existing default calls retain their numerical
  action and raw nonsymmetric integral chart.
- A true clean archive of HEAD96b6a804 was tested with only `ci/direct.py` and
  `tests/ci/test_action_budget.py` overlaid. All1158 Python source/test hashes
  matched that exact baseline plus the two overlays, before and after. Root
  rechecked the original JSON exit0 and JUnit4/4, with no skips/errors.
  Delta103505/n002, CPU40–55, x64, process122.171s, child peak1918332KiB;
  physical H and normal-ordered W joint JVP/VJP/HVP and early budgets pass.
  This is dependency acceptance, not a whole-method memory-cap claim.
- Evidence: delta-packed-mp-factory-r1/clean-ci-dependency, archive SHA256
  c3cc534f44a378ab0362423b4c8ec3ab8da99fc2fa114f7ec9f932ed19134aa3.
  No parent/child TODO is added or marked complete by this prerequisite.

### 2026-10-22 — WF-34 accepted: static Dyson groups and real baseline

- Batched occupied and virtual endpoint rows use separate static branches of
  the same endpoint equations. This removes full-T2/Lambda broadcasts from
  `vmap(cond)` without duplicating the physics. The block budget charges the
  larger actual slice panels as well as packed endpoints. Tail overwrites
  cover the same rows; checkpointing and response-validity guards remain.
- Actual Delta CPU0–15 exit0, process280.218s:38/38 pass with no skips, covering
  PySCF IP/EA endpoints in float32/64, joint T/Lambda/right/left JVP/VJP/HVP,
  two finite-difference steps, molecular FCI residues/moments, vector-response
  rejection, retained state/facade ownership, and the new no-broadcast lower.
  Root verified ten payload hashes and six exact current source/test files;
  an independent peer reviewed static grouping and unchanged equations.
- Earlier explicit-vmap diagnostic timings were not the original production
  baseline. They remain diagnostic only and do not support an old-version
  speed or memory claim. The matched benchmark instead loads the exact
  ec194bf properties.py (SHA256 f606220a...0abf0b) in fresh processes, raising
  only its max_elements allocation guard. The current and old methods use the
  same complete seeded no16/nv128/two-root float64 inputs, CPU0–15 on n002
  (Xeon Gold6230), JAX0.10.0; no SCF or converged EOM is timed.
- Original -> static-streamed IP warm median45.088 ->40.965ms; XLA temporary
  184604612 ->109070792B; process peak1018084 ->843796KiB. EA warm median
  250.040 ->154.325ms; XLA temporary977714244 ->167879560B; process peak
  1973628 ->924760KiB. Compile IP1.038 ->0.146s, EA1.833 ->0.149s. All four
  processes exit0; nine warm samples each. Maximum output differences are
  4.93e-16/IP and9.44e-16/EA. These are endpoint components, not complete CC
  timings or whole-process budget guarantees for arbitrary shapes.
  Other project tests may run concurrently on disjoint logical CPU masks;
  these timings are one paired observation, not exclusive-node performance.
- Exact JSON/XML/commands, before/after source hashes and frozen old code are
  in delta-v100-20261022: static-group archive3c4a8355...bd834 and true-baseline
  archive642de76b...9c192, with root verification JSONs. WF-34 closes without
  a new TODO; WF-16/17 charged intermediates and solver response remain open.
- The same unchanged static-group source additionally passes six selected
  endpoint cases on the allocated V100, actual exit0/process12.807s, no skips:
  float32/64 IP/EA PySCF values and both sectors' joint amplitude/vector HVP
  finite differences. Runtime metadata verifies backend gpu/cuda:0. Root
  checked XML and three exact source hashes; archive4bb63d30...a3e1d.
  This does not replace whole-method GPU scaling or a device-capacity audit.

### 2026-10-22 — WF-23/24: packed MP capacity fallback verified

- Packed tensor actions reuse the committed CI creator/spectator action, the
  original MP normal-ordered kernels and determinant-map phases. Only contracted
  kernel axes are antisymmetrized. Live integral/kernel/vector arguments and
  exact input/output rank bounds remain; there is no new excitation threshold,
  perturbative formula or normal-ordering engine. Canonical cases whose dense
  maps fit keep the old dense path; otherwise the identical packed space is
  available without first expanding all antisymmetric slots.
- Root verified the13 distinct raw JUnit cases, JSON exits0, unchanged source
  manifests, seven exact current source/test files and an independent peer
  review. Four immutable groups on Delta103505/n002 CPU40–55 cover full T6->T6
  action, R/U raw joint integral/vector JVP/VJP/HVP and canonical series. Their
  process times were159.556,330.865,58.725 and425.425s. Mixed-float32 summation
  tolerance is explicit; strict float64 comparisons remain at2e-13.
- Same no3/nmo6/full-rank6/400 coefficients/f64/seed2423 component benchmark:
  dense->packed warm22.664->7.341ms and XLA temporary142740872->1015656B.
  However compile3.360->75.454s and complete child peak918324->3011136KiB worsen
  (the packed benchmark's earlier internal sample is3011128KiB).
  The original default dense path is retained. This is a capacity fallback,
  not a general cold-start or peak-RSS improvement, and does not resolve the
  earlier H6 timeout. Recurrence history/AD budget integration and moving-frame
  molecular MPn acceptance remain in the existing open parents.
- Evidence delta-packed-mp-factory-r1/evidence.tar SHA256
  678a9b72f1b77f3de2f70eabb1120c43fd8bbc72c4a0d3afd8d88aa1edd2cfc0.
  Minimal shared CI dependency is separately accepted in9395b23. No parent
  checkbox or new child is introduced by this finite slice.

### 2026-10-22 — WF-02/19: reference transform reservation forwarding

- The existing small-system full-MO reference wrapper now forwards the caller
  reservation to its block transform, including the live one-body input,
  output, matmul intermediate and necessary dtype-promotion buffers. The raw
  C.T h C and ERI numerical contractions are unchanged. The shared validators
  reject an invalid reservation or unaffordable one-body minimum before ERIs.
- Missing forwarding was reproduced as six actual TypeError failures. The
  same finite fix plus old native active-space/core-budget cases gives12/12,
  exit0, no skips; joint nonsymmetric C/h/G first/second AD is included.
  Delta103505, CPU16–31, x64, process15.220s, peak1259652KiB. Root verified
  all13 archive payloads, original exits/XML, before/after source equality and
  the exact current wrapper/new test bytes. Independent peer identified the
  original missing keyword before the geometry tests proceeded.
- Evidence ci-owner-checkpoint/review-reference-reservation-20261022.tar.gz,
  SHA2560d2d1490b1307dd8d5879fe63ebd7d6a16ad265dce704ffbee8453932341cca3.
  The separate simultaneous lifetime of three unrestricted full reference
  transforms is not claimed solved. Parent/child checkboxes are unchanged.

### 2026-10-22 — WF-21/22: FCI response ledger and requested density blocks

- The FCI solve forwards the original MB budget and its actual retained
  Hamiltonian/string-link arrays to the existing Hermitian response owner.
  The focused new forwarding case and nine existing joint-ledger cases have
  actual Delta exit0/XML evidence in review-cisd-ledgers-20261022. The old
  Hamiltonian, eigensolver and density numerical bodies are retained.
- Requested 2-RDM and transition blocks reuse the existing determinant tiles
  and reductions. Only selected pair rows/columns are accumulated; aa, ab, ba
  and bb stay independent for rectangular transition outputs. The same-spin
  delta correction, coefficient normalization and AD owner remain unchanged.
  A full nact**4 density is not made just to return a small requested block.
- The missing old c20 raw evidence was recovered by a new immutable Delta
  group:4/4 pass, actual exit0/process8.536s, peak890616KiB, CPU0–15 and source
  unchanged. Irregular rectangular axes, independent transitions, empty-spin
  sectors and joint first/second response match slices of the original full
  density. An independent peer reviewed the index and spin conventions.
- Root verified all26 payload hashes of property-frontier-evidence.tar.gz
  (SHA25648228351df15313e04c4c23226b07ead39c1258b996fba5bd9e17d41794a170f),
  including raw4-case FCI and16-case CI XML, exact exits and source maps. The
  CI16 correctness result belongs to its unchanged0def5804 properties source;
  CI property sigma/facade reservations and mixed-amplitude CIS(D) buffers are
  separately identified open budget fixes. They are not accepted by those16
  numerical cases. Complete property/AD residency and whole-method performance
  remain open in the existing parents; no new TODO or checkbox is introduced.

### 2026-10-22 — WF-12/13/14/15: compact bare spin-virtual consumers

- UHF selected preparation may retain only the existing AA/AB/BB spatial
  virtual blocks, with the original spin assembly's permutations and signs
  applied on contraction. No zero-padded full spin vvvv or dressed spin W is
  built on this path. The public UHF facade now selects it; explicit low-level
  dense input remains a reference. The same residual, dual-coordinate Lambda,
  density and streamed-triples equations are reused without a new solver.
- Twenty distinct nodes pass across the immutable groups: float32/64 raw
  charts, unequal/frozen/empty sectors, joint C/AO/RHS HVP, mixed per-request
  dtype storage, public CCD/CCSD, Lambda/density outer AD, triples diagnostics
  and both relaxed CCSD/(T) native geometry with two Hessian FD steps. Root
  checked raw XML/JSON, all165 payloads and seven final source/test hashes.
  Original spin terms, five numerical modules and six intermediate functions
  were independently compared with96b6a80; nineteen stored static proofs and
  peer review preserve the remaining equations and benchmark frontier.
- Retain the real history: spin-consumers-basic exits1 with six passing nodes
  and a2.5MB pressure test rejected by the existing AO planner. The corrected
  test window3MB is above its compact minimum (~2.61MB), below dense-spin need
  (>6MB), and passes with the density regression (exit0/process82.888s).
  Production code, physical input and tolerances were not changed to repair
  that test window. CCSD geometry exits0/process112.348s; (T)137.716s.
- Same fixed25AO, nocc(3,2), nvir(22,23), f64/seed6555 residual component on
  n002/CPU56–71: dense/blocks warm14.652/6.599ms; actual integral payload
  37330008/10686072B; XLA temporary49497800/33516929B. Complete validation-child
  peaks1252980/1200116KiB, maximum action difference8.67e-19. Other logical CPU
  groups may run concurrently; this is one component pair, not whole-CC scaling.
  Benchmark and final source differ only in static per-request promotion
  accounting; the original same-dtype numerical kernels are unchanged.
- Evidence cc-spin-virtual-evidence.tar.gz SHA256
  089a77f30486dec00b60512b0dc4086ba32ce8d6291ec4f7ccc9dc36d2b704f6 and
  cc-spin-virtual-root-verification.json. CorrelatedEnergy's production keyword
  forwarding, packed residual amplitude lifetime, caller/solver phase budgets
  and method scale family remain open. No parent/child is closed by this slice.

### 2026-10-22 — WF-23/24/25/26: Taylor history and shared-array budget

- Pure `run_series` now validates static max_memory=60000 decimal MB. Its
  finite y/Psi/W-Psi histories are mandatory even when final coefficients are
  not requested. Requested wave/T2 copies, rank-view kernels/maps and the
  original linear solver's full direct-column or GMRES workspace are charged
  under the same MB budget. Existing exact-diagonal and implicit solve owners
  retain full off-diagonal response; no coefficient/rank/physical threshold is
  dropped. Dense actions may checkpoint only their internal AD intermediates.
- Actual REDs caught early dense-map construction before budget planning and
  mixed-spin dtype undercount. The minimal fix first plans packed metadata,
  forecasts each missing int32-index/int8-sign dense map, then materializes it
  only when affordable, reusing existing blocks/maps. The common physical
  result dtype covers all spin Focks. The public metadata helper's default
  result stays compatible; this slice does not yet forward eager MP overrides.
- Nine unique passing nodes, plus the original nested HVP repeated on final
  hashes, are verified from raw XML/JSON. Seven initial cases exit0 in564.072s;
  two allocation/dtype cases exit0 in62.183s; final same-case HVP exits0 in
  433.904s. All use Delta103505/n002, CPU40–55, x64; source maps are unchanged.
  Earlier six initial nodes retain their exact earlier frontier rather than
  being described as a new all-nine aggregate on final source. Root compared
  seven current files and six Taylor/Jacobian/energy definitions against true
  4a5276c ASTs. The final nested-response model/tolerances were not reduced.
- Same existing nmo3/nocc1/order4/f64/canonical-tensor outer-JIT forward point,
  true4a5276c -> final source: compile0.54372 ->0.55800s, seven-warm median
  0.14788 ->0.14419ms, XLA temporary6112B unchanged, terminal-child peaks
  766492 ->761544KiB. Input/config/flags match and E2–E4 differ by zero. The
  node is nonexclusive. This only checks small default forward overhead, not
  full response scaling or a process-memory ceiling.
- Evidence delta-series-memory-r1/evidence.tar SHA256
  f04a5d20386b19e7acfd790f044dbce12e00a57b52cb498e15b8195d806e42bf,
  HANDOFF.json and root-verification.json. Eager run_mp/MP-budget forwarding,
  selected AO-source R/U geometry, standalone MP3 and final cache/size/device
  audits remain open. The same35 parents/37 children are retained.

### 2026-10-22 — WF-18/23: selected CI producer checkpoint for shared MP use

- The pure C/F/reference-energy/AO producer returns compressed kernels and
  diagonal data in CIIntegrals. It requests only group blocks needed by the
  existing retained-rank sigma action and diagonal, preserving all occupied
  electrons in the supplied Focks/reference energy. Frozen indices restrict
  correlation only. Rank-one spaces skip unused virtual-pair diagonal blocks.
- This exact producer and selected-integral test file are frozen in
  review-ucis-supply-20261022 (archive62e88d5c...83bf12b). Sixteen focused
  cases exit0/process304.782s on Delta CPU16–31: R/U all existing ranks,
  noncontiguous CID/frozen selections, normal-ordered parts/transposes,
  nonsymmetric raw charts and joint C/F/E/AO second response. Root already
  verified the archive's raw XML/JSON, file hashes and unchanged source maps;
  both files still match those executed bytes. Independent review found no
  new sigma/normal-ordering formula or spin-phase change.
- The producer's imports use the committed shared direct kernel and memory
  helpers. It is checkpointed independently so MP can reuse that data without
  depending on uncommitted public CI/property/workflow edits. It does not
  close CI's public, full-phase property or frozen-geometry acceptance gates.

### 2026-10-22 — WF-10/11: measured native workers and dense-first policy

- The exact native virtual action selects four workers only when their actual
  native workspace plus one RHS fits, otherwise retaining the one-worker
  minimum. Both worker and RHS capacity come from the same original MB budget.
  Restricted preparation keeps dense by default; explicit auto first preflights
  dense and only native budget failures select six blocks plus the existing
  exact exchange action. Invalid input/unsupported-source errors are retained.
  CCS continues to omit its unused oooo/vvvv blocks.
- Six new budget/policy nodes and eight existing geometry/coefficient/RHS/
  projection JVP/VJP/HVP, tail and CC-model nodes pass (actual exits0 in12.860
  and56.064s). Root verified49 payload hashes, three final source/test files,
  fourteen unique XML cases, source-before/after equality, and unchanged raw
  preparation/denominator/integral-type ASTs. The real RED and cwd-filter setup
  rejection remain distinct from successful numerical records.
- One prepatch worker-causality probe uses the same65AO water/cc-pVTZ RHF
  frame with60 virtuals, f64, CPU56–71 on n002 at512MB. Setup and dense/w1/w4
  child exits are all0. Warm dense20.108ms, native1 489.147ms, native4 279.938ms;
  four workers improve native latency1.75x yet remain13.9x slower than dense.
  XLA temporary103.68MB dense versus1.083MB native; primary peaks1167144KiB
  versus681004/685000KiB. Native workspaces4233792/10621224B and maximum action
  difference6.52e-12 are recorded. This is a memory fallback component, not a
  full CC speedup. Other logical CPU groups may be active concurrently.
- The512MB causality timings do not establish the new tight-budget policy;
  its separate six-node gate does. Evidence rcc-action-evidence.tar.gz SHA256
  9e2b81c22c9c1b690051d2308490e68051bd1123581484c8207606f1f29ded64 and
  rcc-action-root-verification.json. Public auto and coupled caller/solver/AD
  ownership remain separate gates; no parent or new TODO is closed here.

### 2026-10-22 — WF-17: one EA virtual action for both original consumers

- The charged virtual dressing reuses the ground virtual action. With
  A[...,k,l]=sum_cd ovov[k,c,l,d]*X[...,c,d], the remaining charged terms are
  A*t2+A*t1*t1. Removing W*t1 from cached Wvvvo allows its contribution and
  W*r2 to share X[j,c,d]=r2[j,c,d]+t1[j,d]*r1[c]. This is exact on the original
  nonsymmetric raw chart; t1=0 still retains its parameter derivative. The
  original explicit Wvvvv/Wvvvo are kept as the small-system reference.
- The action retains only its three needed bare blocks through a small tuple;
  it does not capture the complete ground integral snapshot. Shared closure
  accounting flattens those actual payload leaves. Existing root/primal/AD
  owners remain unchanged. Native prepared EOM is still explicitly gated;
  enabling it and complete caller/response ownership are separate acceptance.
- Two no-cache REDs precede20/20 actual GREEN cases (process482.960s): raw
  float32/64, joint T1/T2/G/right-vector JVP/VJP/HVP at zero/nonzero singles,
  and the original EE/IP/EA memory/roots/response regressions. A separate
  unused-owner RED leads to six final-source passing cases (process33.094s),
  including that owner check, the four raw numeric cases and EA closure-budget
  tracing. There are21 unique passing nodes overall, with source frontiers
  preserved. Root verified57 payload hashes, five final source/test bytes and
  the executed benchmark; peer review and original-intermediate AST proofs
  preserve all untouched equations. Local PySCF2.9.0 rintermediates/eom_rccsd
  were inspected (SHA256c7f3311b...c903666 /9dd17a37...98ef9a).
- Matched original69a4369 charged module versus new action on identical
  supplied o8/v64/f64/seed6562 fields, CPU0–15/n002: prepare including its JIT
  first call1.0875 ->0.4294s; seven warm action median32.960 ->32.674ms;
  complete validation-child peaks2214932 ->1220056KiB. Action-only XLA scratch
  increases160433152 ->196346880B; the lifecycle gain comes from removing the
  retained dressed tensor and its preparation. Maximum output difference is
  2.78e-17. This is a seeded component, not a converged EOM calculation or an
  exclusive-node scaling result. Both child exits0 and exact old bytes were
  verified; no rewritten surrogate is called a baseline.
- Evidence delta-v100-20261022/eom-virtual-evidence.tar.gz SHAa8e0581f...165d024
  and eom-virtual-benchmark-evidence.tar.gz SHA2c2a8fb4...d3b979f, with root
  verification/static-proof JSONs. WF-17 and the same fixed TODO tree stay open.

### 2026-10-22 — WF-06/20/21: bounded declared static array caches

- A small private LRU retains only declared topology/connection/map array
  payloads, with both entry and byte limits. Oversized results are returned
  without retention and do not flush small entries. Six CI/FCI factories now
  have a combined 256,000,000-byte payload ceiling; known action closure maps
  receive an explicit weight. Python keys/headers, construction temporaries,
  compiled executables and allocator state are outside this cache ceiling.
- The original sigma/link-map mathematical bodies are unchanged. The existing
  one-body operator also forwards the caller's same MB/reservation to primal
  and transpose actions; it introduces no new derivative solver.
- Actual four-node cache RED exits1, followed by four metadata GREEN cases
  (exit0/2.221s) and four existing raw-ERI/one-body/FCI H/G/C second-response
  cases (exit0/71.951s), Delta103505 CPU16-31/x64. Root checked all30 payload
  hashes, exact five final files, unchanged before/after maps and raw XML.
- Evidence review-array-cache-20261022.tar.gz SHA256
  6d163eccd01e6893ad0e011d794e8a10b5893864023e47da9db9b540d86e6c23,
  root-late-frontiers-verification.json. This is a static array-cache boundary,
  not an entire-method or process limit. No parent closes and the existing
  fixed TODO hierarchy is unchanged.

### 2026-10-22 — WF-19/22: FCI caller/property and active-preparation ledger

- FCI property methods pass one original MB budget and the declared retained
  source/input/result arrays to the existing determinant-tile reducer. The
  shared tree-storage helper replaces a duplicate source walker. Normalized
  properties keep their original coefficient input charged during reduction;
  no new RDM/link-map numerical body or eigensolver is introduced.
- Active preparation keeps the parent's overlap/parameters/reference arrays
  reserved while only excluding identical numerical leaves owned by the
  callee. NumPy/sequence inputs conservatively forecast a possible backend
  copy; existing JAX/abstract/traced arrays do not pay that extra copy.
- Five property-ledger real RED cases lead to14 passing ledger/block nodes
  (exit0/9.143s); six unchanged RDM numerical/raw-AD cases exit0/12.353s.
  Two active-parent metadata REDs lead to five passing allocation/core-fold
  cases (exit0/6.432s). All use Delta103505 and source maps remain unchanged.
  Root checked exact four checkpoint source/test files and all relevant raw
  XML/exits. Earlier successful frontiers are retained separately.
- Evidence review-property-final-20261022.tar.gz SHA256
  fc7bb21367e2b45b9d6f7035f0728e43832ce1a1e032821e5969d6d0f383380d
  and review-frame-active-final-20261022.tar.gz SHA256
  9cdfda0488d967d5f01631e0b756a0e38ccd0dc61c2bed8168050176472ea7c0,
  root-late-frontiers-verification.json. These are known-array phase budgets;
  full process/XLA residency and size/device acceptance remain separate.
  The same fixed TODO hierarchy remains open.

### 2026-10-22 — WF-18/22: selected CI/FCI relaxed geometry and explicit frame

- CorrelatedEnergy now composes the existing stationary HF solve with the
  shared selected CI producer/sigma solver or required FCI active-space tensor.
  AO overlap/integrals, transported Focks, orbitals and reference energy remain
  live. No second SCF/eigen derivative engine or full-MO pre-adapter is added.
  CID and higher existing CI ranks keep their original retained-space meanings.
- Six fixed native geometry points initially pass5 and fail1: H2/6-31G
  restricted full/frozen CISD, H3/STO-3G unrestricted full/frozen CISD, H2 full
  FCI and LiH/STO-3G one-core FCI. The unequal-frozen H3 HVP differed from two
  gradient finite differences by5.918e-5/5.920e-5, despite full-MO/selected AD
  agreeing. The scalar AD Hessian remained bilinearly symmetric1.38e-16.
- A fixed initial-HF occupied/virtual anchor projected into the current
  stationary subspaces, followed by Cholesky orthonormalization, defines an
  integrable local orbital frame for selected CI/FCI models. On the same H3
  point the two HVP errors fall to1.680e-8/1.865e-9. Density/SCF energy remain
  unchanged at1e-12. This explicitly defines a local frozen-orbital model; it
  does not identify different arbitrary recentered frames away from the anchor.
  Rank loss stays nonfinite, with no clipping/canonicalization fallback.
- Production frame/UCC forwarding/dtype/rank-loss four cases exit0/311.617s;
  original restricted-frozen and FCI-core points on the final frame each pass
  (346.056s/53.908s). Unfrozen branches do not use this frame. MP/CC frozen/
  core/active workflow rejection is retained. Only the already verified UCC
  bare-block keyword changes the old MP/CC call body; root verified exact AST
  equality after removing that keyword and the new guarded CI/FCI branch.
- Source freshness RED2->GREEN2 precedes moving integral preparation. Root
  checked109 earlier payloads plus the37 final frame/active payloads, exact
  five source/test files and raw XML/exits. Evidence review-geometry-frontier,
  review-frame-causal-ucc and review-frame-active-final-20261022 archives,
  root-geometry-frame-verification.json and root-late-frontiers-verification.json.
  Existing small-point geometry success is not a large-CI performance or
  process-memory guarantee. The fixed TODO hierarchy remains unchanged.

### 2026-10-22 — WF-16/17/29: native prepared EOM and eager owner propagation

- The already checked VirtualERI provider now passes the prepared restricted
  EOM boundary. Unknown nonarray providers remain rejected; original spin,
  model, root, response and capacity guards stay in place. The existing EE
  residual and IP/EA actions consume it without a second eigen/AD solver.
- Three actual RED nodes hit only the former dense-vvvv interface rejection
  (exit1/87.003s). On exact8b1a689 plus the declared guard/test overlay, four
  nodes pass (exit0/131.080s): each EE/IP/EA right and transpose action and root
  matches dense, and EA coefficient response includes ground CC response plus
  an independent finite difference. This uses the existing supplied H2/3-21G
  algebraic frame, x64, CPU0-15, Delta103505; dense solving is only the tiny
  reference gate, not a claim about large EOM matrix-free scaling.
- Two eager-boundary REDs reveal missing retained SCF/reference/Lambda owners.
  The facade now forwards those declared arrays with the same inherited MB
  budget, excluding only child-owned ground/integral leaves. Both metadata
  cases and the original cache-invalidation/failure-cleanup regression pass
  (exit0/29.539s). No result or settings are silently reused after failure.
- Root verified41 payload hashes, original equations and before/after maps,
  exact final API/tests and raw XML/exits. A final operator doc-only edit
  explicitly retains the native provider's separate forward scratch budget;
  physical native-phase and implicit-vector-response budgets remain open.
  Independent review found no changed formula or broadened unknown provider.
- Evidence eom-native-evidence.tar.gz SHA256
  7983f129a6130f29c7ca7ffc918b2113a5623dc7d4746a775d6b106c6f83da70
  and eom-caller-evidence.tar.gz SHA256
  ac30a7cae3ecaba42773ef7d636ac00d187e72877dce27cd538d5f86b856d08c,
  eom-native-root-verification.json and eom-caller-root-verification.json.
  These seven unique component/boundary gates do not close a total-process
  budget, GPU or whole-method size gate. The fixed TODO hierarchy is retained.

### 2026-10-22 — WF-05/06/29: shared tensor resolvent budget entry

- The existing tensor-sum inverse accepts optional max_memory/reserved_bytes.
  It checks original inputs, possible common-dtype copies, individual-axis
  eigensystems, complete spectrum and transform/residual working arrays before
  diagonalization. It never constructs a full Kronecker matrix. The current
  custom-linear-solve, true-residual and singular/degenerate response bodies
  remain exactly unchanged; caller/AD/allocator/library scratch is not silently
  relabeled a complete process cap.
- Three actual missing-interface REDs precede nine passing cases on Delta
  CPU0-15/x64 (exit0/11.618s): early exhaustion, mixed promotions, budgeted
  repeated-eigenvalue second response and the six existing numerical/shape/
  singular/nested-response gates. Source maps are unchanged and no nodes skip.
  Independent review confirms the input-promotion accounting and preserved
  numerical body; root also proves that body AST equals the original.
- Evidence tensor-budget-evidence.tar.gz SHA256
  da6f23eb984f7fed4ec5d590a504838f3aead944056110ef08b84de7edffaf92
  and tensor-budget-root-verification.json; all16 payloads and exact final
  source/test bytes were checked. Method callers still need their joint spin/
  T2/source reservations; this prerequisite alone closes no parent task.

### 2026-10-22 — WF-18/20/22/27: selected public CI and affordable spatial densities

- SCF-backed variational R/U CI and UCIS now consume the shared selected
  kernels. Common-frame ROHF is explicitly opt-in and retains its saved two
  spin Focks; UCIS still rejects a nonstationary occupied-virtual spin block.
  Pure/explicit-reference numerical conventions and existing rank/frozen
  definitions are retained. Explicit raw solver budget forwarding remains an
  identified WF-20 integration boundary rather than a claimed completion.
- SCF CIS(D) prepares four selected spatial integral blocks for the existing
  shared correction. Original singlet normalization, denominators, amplitude
  response and complete energy-body AST are unchanged. Common dtype forecasts
  include both amplitudes and excitation energies before spin workspace exists.
  Actual selected/API/old-CIS(D) and joint raw-chart HVP frontiers retain their
  own before/after source maps; the final energy-dtype RED1->GREEN2 is separate.
- Spatial densities use the original fermionic connection/scatter formula
  only after a shape-based table/construction/output budget preflight. The
  affordable path remains fast; otherwise the shared sigma derivative and
  raw-integral normal-ordering correction avoid the density connection table.
  Direct observable contraction exports no RDM. All caller/root arrays share
  the same MB budget. The seventh32MB cache raises the combined declared
  array-payload ceiling to288,000,000B, excluding Python/XLA/allocator state.
- Ten finite density/cache gates have valid passing frontiers. The first
  aggregate exits1 with9 passes and one newly added coarse-step HVP failure;
  the unchanged production source then passes that same failed case using
  steps3e-5/1e-5 and unchanged tolerances (exit0/5.829s). Errors2.199e-5/
  2.442e-6 scale9.006 with the squared step ratio9, supporting the retained
  coarse1e-4 error2.443e-4 as finite-difference truncation. No production fix
  or tolerance relaxation was used to resolve that diagnostic.
- Original ec194bf versus final affordable path, same nmo6/nocc2/93det/seed771
  point, x64/CPU16-31/n002: lower0.1042->0.1052s, compile0.1377->0.1313s,
  XLA scratch131080B unchanged, terminal-child RSS638148->637596KiB,
  seven-warm median0.125989->0.056637ms, output difference0.0. The node is
  nonexclusive and the old formula is identical; no universal speedup is
  claimed. The earlier sigma-only regression remains archived and is not
  advertised as a performance benefit. FCI requested-block comparison stays
  a same-source functional comparison, with possible compiler output pruning.
- Root verified final source matches, original connection/norm/scatter/spin
  and CIS(D) energy ASTs, raw exits/XML and all51 density-final payloads.
  Evidence review-density-first-20261022.tar.gz SHA256
  ecaa6a38a64a1e2bc784bf055f559395bc8e64e0f2df830fa5ba78b8309b30eb,
  root-ci-density-final-verification.json and root-ci-final-source-matches.json.
  Earlier selected16/UCIS10, property/NaN/stale/raw-HVP and budget frontiers are
  in their previously verified archives. No raw datasets or executables enter
  the checkpoint; fixed TODO depth/count constraints remain unchanged.

### 2026-10-22 — WF-19 acceptance review

- WF-19a's original32 native/projected fixed-frame CPU cases retain their
  actual exit0/XML/source maps; they are not described as Delta reruns. Recent
  Delta dense/s4/DF core folding, NumPy/backend-copy and bound-source budgets,
  shared solver/action ownership and wrapper second response complete WF-19b.
  The native H2 full-FCI/LiH core-FCI relaxed closures add full gradient/HVP
  and two finite differences without a pre-full-AO/MO preparation.
- The acceptance is layered over the shared transformation/core-fold contract,
  not a repeated Cartesian product of every representation and geometry.
  The real common-spatial CPU/known-array WF-19 scope is accepted. CUDA FCI
  geometry, large-CAS scaling and complete compiled/AD process residency remain
  the explicit WF-30/32 gates. Acceptance counts become6/35 parents and27/37
  children; they are not engineering-effort percentages or a new TODO tree.

### 2026-10-22 — WF-05/06/29: joint MP2 channel arrays and shared graph estimate

- Prepared MP2 accepts optional max_memory/reserved_bytes. One known-array
  plan retains all aa/ab/bb outputs together and budgets the largest current
  channel; canonical row work and the unchanged noncanonical complete tensor
  inverse remain distinct. The latter forwards the same original MB budget
  and visible caller data to the shared tensor-sum solver. Raw direct/RHS
  aliases are excluded only by object identity.
- Review finds that unmasked channel T2 arrays survive while masked return
  arrays are made. A new actual metadata RED (296B counted versus592B needed)
  is fixed by charging both copies. This changes only static accounting; no
  energy, denominator, amplitude or validity formula changes.
- EOM's existing recursive primitive-output estimate moves to one shared
  metadata helper for EOM/CC reuse. It counts nested declared array outputs
  conservatively, without assuming view/allocator reuse or initializing any
  device. It is no second solver, and opaque runtime/library/AD state is not
  silently claimed as fully modeled.
- Initial five missing-interface REDs precede11 passing MP2 budget/rotation/
  degeneracy cases (exit0/14.230s). The separate T2-owner RED and missing
  shared-helper RED remain recorded. Final five MP2 cases plus the nested
  helper and three existing EOM closure-budget cases pass9/9 (exit0/40.949s),
  Delta CPU0-15/x64, unchanged source maps. Fifteen unique nodes have passing
  frontiers; this is not one all-fifteen aggregate on a single final test file.
- Root verifies37 payloads, six exact final source/test files and original MP2
  numerical/public-dispatch ASTs. Independent review checks promotion, spin
  residency and alias handling. Evidence mp2-graph-budget-evidence.tar.gz SHA256
  1dccd653c5555ebf5c561f726f28e1e62ded30ad6df4b8fb3161da6a0d1fe8ab,
  mp2-graph-budget-root-verification.json. Eager/raw source forwarding, joint
  preparation and full method/GPU/process measurements remain explicit gates;
  the same parent/child TODO hierarchy is unchanged.

### 2026-10-22 — WF-23/25/26: selected normal-order kernels and moving MPn

- A pure noncanonical C/F/total-HF-energy/AO entry prepares the shared selected
  CI kernels once and reuses them at every needed Taylor rank. Prepared F_N is
  already reference-subtracted; its HF trace is not subtracted twice. Signed
  tensor/determinant coordinates, full frozen-electron reference contributions,
  target-rank bounds, finite histories and requested coefficients are retained.
  Both raw and selected entries call the same extracted residual/Taylor tail;
  root proves six original mathematical definitions equal69a4369.
- Five original pure-frame gates have passing evidence: early capacity,
  explicit canonical-chart rejection, separate frozen-spin forward, plus R/DF
  and U/s4 joint C/F/E/AO value/JVP/VJP/HVP with all original result/coefficients/
  output-flag assertions. Three have pytest XML; the two expensive original
  response tests run unchanged closures in separate bounded raw/selected
  processes. Raw/selected source/fixture signatures match. No combined pytest
  GREEN is invented. The initial missing-asset collection exit2 and actual
  five missing-interface REDs are distinct.
- The retained original combined R case times out at570.590s. In separate
  processes R raw/selected complete286.830/483.505s, peaks7663328/11625212KiB;
  U completes164.261/229.540s, peaks4998484/6150208KiB. These cold second-response
  cases show no performance or RSS gain. Their total cold cost and simultaneous
  executable holding matter; no unmeasured unique timeout cause is asserted.
- CorrelatedEnergy uses this entry for higher-order noncanonical MP series/
  tensor methods. Existing MP2, canonical, CC, CI/FCI, dipole and MP/CC frozen/
  core/active rejection branches remain unchanged. Two real direct-workflow
  REDs precede the same H2/6-31G RHF MP4-series and H3/STO-3G UHF MP4-tensor
  native geometry gates. Each raw/selected phase exits0 under570s and final
  comparison_complete is true: full value/gradient/JVP/HVP, zero net force,
  JVP pairing and HVP finite differences at3e-4/1e-4Bohr with original tolerances.
  Complete AO/MO materialization traps are enabled on the selected path.
- H2 raw/selected lower26.132/26.137s, compile45.676/55.894s, one warm12.208/
  59.028ms, terminal-child peaks2663840/3023760KiB. H3 lower26.687/28.502s,
  compile49.772/64.145s, one warm14.653/66.320ms, peaks2776696/2869540KiB.
  These are single observations and distinct input representations, not
  a matched production scalability/speedup claim.
- Root verifies all six archive hashes/payload records, raw exits/XML, source
  before/after maps, three final production bytes and unchanged Taylor/legacy
  definitions. Evidence phase-ru-evidence.tar SHA256
  61707993952f57447699a801add589ad927f93c119c64f692b4b4454ae717f82 and
  geometry-evidence.tar SHA256
  fc3a6c0f4ee7d2c2cd9589829386b1d8d014d2e8a8078ccb6202b9926f5a9d33,
  both HANDOFF.json and delta-series-geometry-r1/root-verification.json.
  Eager/raw override forwarding, standalone canonical MP3, size/scaling and
  GPU/native phase budgets remain their existing tasks; no TODO is added.

### 2026-10-22 — WF-10/12/14/15/29: CC caller and physical-action phase budgets

- Raw/prepared RCC/UCC ground, Lambda, triples and one-RDM consumers accept
  the same optional MB budget and known parent reservations. Public facades
  retain visible SCF/reference/state owners. Preparation checks promoted full
  raw UCC copies, selected payloads and held initial/right/left states before
  their allocation. Native tiles use the maximum of actual planned primal/
  response phases, not their sum or requested-but-reduced solver capacities.
  Opaque producer reservations remain conservative rather than inferred aliases.
- Residual estimates use the real integral operand dtype. The same existing
  numerical DIIS/linear/root owners select optional capacities; models, physical
  coordinates, tolerances and maxiter are unchanged. Bare native action worker
  metadata is shared, not reimplemented. Public restricted preparation keeps
  the dense-first auto policy; no whole-method performance claim follows.
- A matched44-MO/8-occupied Lambda point exposes a real undercount: the old
  39,782,400B hand envelope allowed123.521238MB although compiled scratch was
  172,720,752B. The shared graph estimator now charges actual declared forward/
  transpose output arrays, including layout/pack intermediates. That original
  budget is rejected before GMRES. Actual saved VJP arrays stay charged by
  their declared PyTree leaves; no new AD owner is added.
- Seventeen unique original caller nodes have actual passing frontiers across
  the documented shards. Historical aggregate exit1 with seven passes/two
  metadata-spy failures remains; fixing only the spy's accumulated abstract
  VJPs gives five final-source passes (including two reselected numeric nodes).
  A further raw restricted semicanonical/auto (T) bridge preflight precedes
  its complete same-spin T2 allocation: metadata RED1->GREEN2, including the
  original all-diagnostics numeric oracle (30.587s). None math is unchanged.
  There are18 unique caller gates plus that existing physical regression.
- Final same-input Lambda baseline/budgeted children both exit0, x64 CPU56-71,
  n002,41904 physical coordinates, rtol1e-10/maxiter100. The legal342.198168MB
  plan keeps214,723,944B declared action scratch and chooses restart40->12.
  Five-warm medians605.727->618.124ms; XLA scratch182132144->172720752B;
  terminal peaks1428460->1420280KiB; maximum solution difference4.29e-13.
  The node is nonexclusive, this is a component on a fixed right state, and
  neither a speedup nor meaningful whole-process RSS gain is claimed.
- Root verifies274 delivery payloads, raw XML/exits,12 final source/test maps
  and the unchanged core residual/amplitude equations. Evidence
  cc-caller-budget-evidence.tar.gz SHA256
  3acc85f4958b887c8b77b66e5a34e6d86d7e7b16f01527d14d20732d2a7a103c,
  cc-caller-budget-raw-bridge.tar.gz SHA256
  4c6cadb1ac6a5e549fcd89c64e68f921410b4ab4cd34f4cff48bb9ce4036c33f,
  and cc-caller-budget-root-verification.json. Compiler/library/allocator/
  external AD storage, spatial two-RDM/contracted two-body work and complete
  method/device/size reconciliation remain their existing acceptance gates.

### 2026-10-22 — WF-18/20/22/29: explicit-MO CI budget handoff

- Raw CI/CIS/UCIS/CIS(D) accept the original optional MB budget and known
  caller reservations. None retains the original raw mathematical path. The
  existing normal-order/Fock/kernel/diagonal functions adapt raw MO tensors
  without identity AO2MO or a full spin ERI. UCIS stationarity uses spatial
  spin Focks, checked against the old independent full-spin formula with
  unequal occupations and nonsymmetric raw ab input. Both retained CI
  diagonals are charged in the shared action reservation.
- Facades retain visible SCF/reference/source owners and potential backend/BA
  transpose copies. AO-CIS passes the existing TD action's planned share to
  the common Hermitian solver. Adding that internal metadata changes none of
  the TD planner's four existing capacity/cache/grid decisions. CIS(D) shares
  its existing spin/root forecast; its physical correction formula is unchanged.
- Five final-source groups have actual exit0, unchanged before/after maps and
  no skips: metadata11/5.580s; raw CI/UCI value and joint first response2/
  437.206s; CIS/UCIS response2/68.399s; existing physical/TD4/37.084s;
  water native direct-CIS energy/amplitude response and FD1/32.933s. These20
  unique cases run on Delta job103505/n002, CPU16-31, x64. They are not warm
  or large-system scaling measurements. The extra four-method HVP diagnostic
  times out at570.502s with child-9/launcher124; its source/logs are retained
  and do not establish general raw CI second-response acceptance.
- Root verifies all131 archived payloads, five raw terminal/XML groups and
  nine exact final files. Evidence ci-owner-checkpoint/review-ci-raw-budget-
  20261022.tar.gz SHA256
  aea37ae57e4fa1e4df4191ea8fa6da744c6648033eddd1126d9710b6ac9bf707,
  RAW_BUDGET_REVIEW.md and current-checkpoint-root-verification.json. Static
  Hamiltonian-oracle guards and spin_square's default density budget remain
  documented boundaries. Compiler/allocator/external AD storage, GPU, complete
  method scaling and remaining parent acceptance gates stay open.

### 2026-10-22 — WF-30: native virtual action on the allocated V100

- The two existing native/projected VirtualERI CUDA gates exit0/7.183s on
  job103505/n002, physical GPU0, Tesla V100-SXM2-32GB. They compare value,
  JVP/VJP/HVP jointly in nuclear coordinates, coefficients, RHS and projection
  against the CPU dense oracle, prohibit full AO materialization and check
  every actual output leaf resides on CUDA0. GPU6 is not exposed.
- Root verifies11 archived payloads, actual terminal result, two-case JUnit
  XML, unchanged source maps and the explicit CUDA runner environment. The
  admitted native library SHA remains
  eb9b1f1ae3fd0f79b9c20708a9c1dae2fc18d8f69e2249a24badf07b15193ed7.
  Terminal-child CPU RSS1231068KiB is not a VRAM measurement. No whole-method
  GPU performance/scaling claim follows. The subsequent shared bare-worker
  metadata extraction is separately CPU-verified, not a new CUDA method run.
- Evidence native-action-v100-evidence.tar.gz SHA256
  28045804383faa10ab30dc78ec9d5d64ed5d47145e640312e3163735a2c388fb,
  current-checkpoint-root-verification.json. WF-30 remains open for its full
  method/device/transfer/peak-memory matrix; no new TODO is added.

### 2026-10-22 — WF-29c: isolated non-Hermitian pair-response budgets

- EOM relays the original MB/resident/action plan into the existing shared
  pair-response owner, without a second primal plan or larger action batch.
  Generic non-Hermitian solves retain their input owners before dense local
  rebinding. Right(n+1) and left(n) response phases are planned independently;
  budgeted root columns run sequential checked linear solves. None retains
  the old vmap. Normalization, cluster/complex-root and direct n+1 guards,
  root equations and primal pairing are unchanged; no EOM matrix is added.
- Declared forward/transpose graph arrays reuse the shared estimator; known
  opaque fixed action scratch is added rather than swallowed by max. The
  original3x3 two-root family checks direct/GMRES values, first/second response
  and two FD steps. Existing tiny prepared H2/STO-3G IP/EA residue responses
  retain ground/Lambda/vector dependencies. A frozen-vector counterexample
  initially used a symmetry-blind trace weight (two real failures retained);
  the complete residue resolves that test blind spot with unchanged physics.
- Nineteen unique nodes have actual passing frontiers on job103505/n002,
  CPU56-71/x64, with raw results/source maps/XML. Final integration2 passes
  in142.740s; final metadata5 in14.279s; strengthened physical-minimum1 in
  11.330s. Initial missing-interface RED5, trace-observable failures and fixed-
  scratch RED1 remain archived. These are response/component gates, not a
  large-method RSS or speed benchmark; WF-29c remains open for other consumers.
- Root verifies264 payloads,16 exact final source/test/dependency hashes,
  all nine batch exits/XML and unchanged-math AST proofs. Evidence
  eom-response-evidence.tar.gz SHA256
  e3ef41bb3040235ce8a968bd085f5a7f4d625c64b33f1beb0a00d79deb2dffb2,
  summary.json/static-proofs.json and mp3-eom-root-verification.json.

### 2026-10-22 — WF-01/08/30: preserve direct workers and verify moving MP2 on V100

- MolecularEnergy dropped mf.direct_scf_workers and implicitly selected the
  optimizer's one-worker default. Two real R/U metadata failures reproduce it.
  The adapter now forwards the seed's worker count for direct J/K, retaining
  zero screening; full-ERI calls and every physical equation are unchanged.
  This restores requested parallelism, not a measured explanation of the
  complete aniline runtime. No integral cache is introduced for moving geometry.
- Final CPU group exits0/95.427s, five tests: R/U forwarding, existing R/U
  native HF value/density/geometry-gradient oracles and existing H2/3-21G
  MP2 joint geometry/AO-potential gradient/HVP with two FD steps. Source maps
  are unchanged; terminal-child RSS1588508KiB. No new warm timing is inferred.
- The same frozen source passes all five existing GPU cases in67.844s:
  complete native-direct H2/3-21G MP2 value/gradient/HVP plus overlap/kinetic/
  nuclear/dipole geometry products. CPU dense outputs are the oracle and
  every returned actual leaf is checked on CUDA0. Full AO materialization is
  forbidden for direct MP2. Job103505/n002 uses only its allocated V1000;
  GPU6 is not visible. CPU-child RSS2178360KiB does not measure peak VRAM.
- Root verifies30 payload hashes, RED2/CPU-GREEN5/GPU-GREEN5 raw exits/XML,
  before/after source maps, two exact current files and explicit CUDA runner
  environment. Evidence molecular-workers-evidence.tar.gz SHA256
  2a042281cfdaef1c95afa5ffe628483d368022d5c57dd7db190ea1c0216ba70e,
  molecular-workers-root-verification.json. GPU size/transfer/VRAM and the
  direct-input warm scaling campaign remain the original open gates.

### 2026-10-22 — WF-33: selected standalone canonical MP3 kernel

- One selected CC preparation supplies the seven existing blocks; MP2 borrows
  its ovov/Fock/energies without a second transformation. The original zero-
  amplitude CCD residual JVP and E3 contraction remain one unchanged tail.
  Raw None preparation order is preserved. Optional planning forecasts source,
  selected blocks, simultaneous first/second doubles and declared graph arrays
  before actual transforms. Full-frame canonical/frozen guards remain live.
- The original seven gates have passing frontiers: four guards in20.022s and
  three complete raw/selected C/F/E/AO value/JVP/HVP comparisons, in bounded
  separate processes, covering array/no-frozen/T2, s4/frozen1/no-T2 and DF/
  frozen[0,3]/T2. Signatures and all original assertions match; no combined
  seven-case pytest GREEN is invented. Independent Taylor order-three and two
  existing default/JIT/PySCF cases pass3/81.559s on the same frozen kernel.
- Independent review reproduces two further REDs: strong64 reference energy
  incorrectly promoted32 interaction/T2, and Fock casts preceded the first
  capacity check. Interaction precision now depends only on C/F/AO; reference
  energy retains its separate dtype (native remains64). ShapeDtypeStruct
  forecasts casts before allocation. New2 pass15.153s; affected original
  reservation/full-Fock/Taylor3 pass44.488s. Old float64 tolerances and the
  original seven test bodies are unchanged. All batches use job103505/x64 CPU.
- Root checks both owner archives, six phase terminals/signatures, guard XML,
  three default/cross-oracle XML, the two-RED/five-GREEN review records and
  exact final bytes. The final energy tail equals26ab61e AST; raw/plan/view
  definitions equal the tested initial candidate. Evidence delta-mp3-orbitals-r1/
  HANDOFF.json; mp3-final-oracle-evidence.tar.gz SHA256
  0801d4d2035b3a77c67f12b96219dc0b5c4a8e8cfaea6f839827e52ef801122e;
  mp3-review-fixes-evidence.tar.gz SHA256
  61c2f9527039e7062c61d4410362e74570dfaaaaeb01cf6bfc8ad5dffc00787e.
  mp3-eom-root-verification.json and mp3-review-fixes/root-verification.json
  retain the actual boundaries. These are small pure-frame cold-response
  gates, not native/GPU, warm speed or whole-method memory gains. Eager source
  dispatch and caller ownership remain the existing integration work; WF-33
  is not yet closed solely by adding this kernel.

### 2026-10-22 — WF-29c: Hermitian isolated/cluster response ownership

- The original MB/resident plan follows a static hashable carrier through the
  existing closure-converted custom JVP and single root. Only retained source/
  Ritz/diagnostic/parameter arrays survive into the response budget; numerical
  Davidson capacities are not charged again. Response planning is lazy: a
  stationary first energy derivative and full-span projector/trace retain their
  original paths. None retains the old vmap and exact mathematical chart.
- Isolated roots retain n-dimensional checked solves, reused sequentially
  inside the original root tangent solve. Complete degenerate clusters retain
  the full n*k projected equation, with no internal-gap division or separated
  cluster solves. The shared graph counter and plan_linear account for actual
  direct assembly/transpose arrays and normalization. Only optional GMRES
  restart changes; tolerances, maxiter, validity and direct-capacity guards stay.
- Actual RED4fail/1pass precedes final metadata5/15.170s, isolated2/41.025s
  and exact-degenerate cluster1/27.872s, all exit0/no skips on CPU56-71,
  job103505/n002/x64. The eight unique accepted nodes include the original
  batched/inner-JIT third-response oracle, joint JVP/VJP/HVP and FD, cluster
  projector/trace, gap/NaN and full-span guards. No performance gain is asserted.
- Root verifies41 archived payloads, all four exits/XML/source maps,17 current
  source/test/dependency bytes and13 unchanged-math/None/guard proofs; independent
  review finds no blocker. Evidence hermitian-response-evidence.tar.gz SHA256
  6528fd645d486aa214a2d08ff2f8768dc54629f2f08878ca2c2485c5b76e15b2,
  HANDOFF.json/static-proofs.json and mp-callers-hermitian-root-verification.json.
  RPA wrappers do not yet relay these optional controls; arbitrary outer AD,
  opaque callbacks and process/allocator peaks remain outside an array cap.
  No new TODO is added and WF-29c retains its remaining caller/SCF gates.

### 2026-10-23 — WF-05/06/26/29/33: shared MP preparation and caller ownership

- The shared AO2MO DF-factor helper tiles auxiliary scratch while retaining the
  requested Qia output. MP2 preparation and consumption retain source, all
  three UHF outputs, Fock, T2 and resolvent owners under the original MB budget.
  Raw-array, SCF and moving-molecule callers relay that budget and reservation.
  Eager noncanonical series and canonical MP3 use the already tested selected
  orbital kernels. Canonical higher-order series retain their original chart;
  small dense maps are restored only after the existing byte preflight.
- Selected series keep the lambda-independent HF energy outside the interaction
  dtype, preserving float32 corrections with a float64 total-energy shift.
  Shared full-UHF transformation preflights aa/ab/bb together. Selected triples
  retain the still-live AO/MO Focks. No MP, CI, CC or triples formula is replaced.
- The original 23-node campaign has actual passing frontiers in five bounded
  shards. Review fixes have 15 unique passing affected nodes across a first
  14-pass/1-fail batch and one successful targeted rerun; no aggregate 15-pass
  XML is invented. Joint U-packed planning requires 4060 working bytes: the old
  0.005 MB case now asserts rejection, then tests the same physical oracle at
  0.0051 MB. Other budgets and original float64 tolerances remain unchanged.
  Four legacy observation spies first fail on removed entry-point names and
  then pass in 9.656116 s, without weakening their assertion ASTs or adding
  deprecated aliases. All production manifests remain unchanged within runs.
- These are CPU/x64 gates on the previous allocation 103505, not new 5090 tests
  or method scaling. Root and independent review verify the final 19 files
  against HANDOFF and archived bytes. Final P2/spies archive SHA256:
  143ae3f9faa3cffc84fa4b8972a4426a413aeaf0f9a89433aa21e3c329e6a839;
  original archive 7c06fd94c60f8004b5de872afcc6a4303b645bf3e854b1c1b3bde82b1344aca6.
  Original failures, mixed-dtype checks and caller lineage remain recorded in
  delta-mp-callers-r1/HANDOFF.json and review-p2/HANDOFF.json. Complete outer-AD,
  GPU, warm scaling and process-memory gates stay open. No TODO is added.

### 2026-10-23 — Remote resource renewal

- The refreshed Delta manual is lwcluster 0.8.0, updated 2026-09-29. V100 nodes
  have less than the newly requested 512 GiB; n006 supports the requested
  single RTX 5090, 64 logical CPUs and 512 GiB. Allocation 104081 runs from
  2026-10-23 09:54:40 to 2026-10-25 09:54:40 Asia/Shanghai, physical GPU0 only.
- The actual CPU affinity is 0-31,192-223. CUDA/JAX 0.10.0 float64 JIT produces
  the expected scalar 14.0 on CUDA0, exit 0, in 52.137341 s including first
  runtime initialization. This is a readiness check, not a method benchmark.
  GPU6 is never initialized. Metadata and scripts are preserved in
  artifacts/wavefunction-memory-20261021/delta-nv5090-20261023/.
- Earlier holder attempts are retained as resource-orchestration failures:
  104075 held the GPU, 104078 could not query a GPU from a CPU-only step, and
  104079 held all step memory. None ran a numerical test. The accepted holder
  uses no GPU and 1 GiB; scientific steps stay within the same allocation.
  Source/library provenance and existing acceptance gates are unchanged.

### 2026-10-23 — WF-15/27/29: requested CC density and observable boxes

- Keep one original Gamma2 equation source and the original sixteen O/V
  assembly segments. A small index-expression adapter propagates requested
  free-axis selections; contracted indices remain complete. Four-axis boxes
  serve full spatial export, static unequal/duplicate/empty selections, spin
  aa/ab/ba/bb and direct observable contraction. There is no full spin 2-RDM
  extraction, second density formula, numerical cache or new solver. The
  necessary O^2 V^2 spin T/Lambda bridge is retained and budgeted.
- The public CC methods accept make_rdm2(indices=None, spin=None) and
  expectation_value(h1=None, eri=None). The original MB budget and known
  source/state/output/operator copies remain in their respective Lambda and
  density phases. Impossible exports fail before Lambda/spin expansion.
  One-body dispatch and other original CC methods retain their mathematical
  ASTs. Source attribution remains in cc/NOTICE.md.
- Actual accepted frontiers comprise 14 selection/caller/joint-state AD cases,
  five original R/U full/frozen PySCF density/trace/energy oracles and three
  final scalar-budget/implicit-chain cases. All groups exit 0 on the previous
  allocation 103505/n002/CPU16-31; 22 distinct cases are not 22 full workflows.
  Final three cases alone use the corrected scalar-dtype byte fee. Their
  CC -> Lambda -> observable HVP perturbs fixed-MO one-body integrals, not
  nuclear coordinates. Original tolerances, true REDs and invalid orchestration
  records remain preserved. Root and independent review verify 173 payloads,
  the raw exits/XML/source maps and the exact four source/four test files.
- At the single fixed nmo=4/nocc=2/seed713 component point, historical/new full
  XLA temporaries are 176641/65537 B, warm medians 0.233/0.262 ms and compile
  times 0.350/2.707 s. Complete child RSS increases. The earlier 0.194/0.397 ms
  result is also retained: this demonstrates reduced declared temporary arrays,
  not overall speed or process-memory improvement. No GPU, size scaling or
  nuclear-property response is inferred. Those original gates remain open.
- Evidence: ci-owner-checkpoint/CC_RDM2_REVIEW.md and
  review-cc-rdm2-20261022.tar.gz, SHA256
  988ea4e2f34e1e358322436abeb9f08ffc92dc240c45f610d56a849958ac3113.
  This closes the finite correctness/known-array slice inside the existing
  tasks, without adding TODO branches or closing the full project.

### 2026-10-23 — WF-01/04/08/30: parallel native coordinate pullbacks

- Geometry construction now retains the requested direct-J/K worker count.
  The original serial v1 ABI and equations remain. A new VJP v2 target splits
  the first shell index across the existing FFI thread pool; each worker owns
  only its integral scratch and ENV-sized coordinate accumulator. Fixed-order
  reduction retains all first/second-coordinate terms. No worker-sized J/K or
  ERI copy is introduced. JVP remains serial; unsupported higher/basis
  derivatives retain explicit errors. CPU-backed CUDA dispatch is unchanged.
- The shared native scratch estimate now also bounds nine-component Hessian
  blocks and ENV partials. A metadata-only 64-contraction counterexample first
  fails, then passes after this fee correction; no huge numerical array is
  allocated by that test. Independent C++ review checks partition coverage,
  asynchronous buffer lifetimes, deterministic reduction and exception paths.
- On the previous n002 allocation, the same fixed-density aniline/119-AO
  Cartesian6-31G* input (seed807, rank25 density, x64, zero screening) has warm
  coordinate-gradient medians 110.053092 s at old one worker, 108.492497 s at
  old eight workers and 19.622632 s at new eight workers. All 42 components
  are identical. The approximately 5.53x improvement is this native J/K VJP
  component only, not complete SCF/MP2. Child RSS changes from 736648 to
  742536 KiB for old/new eight workers; XLA temporaries remain 457280 B.
  Existing CPU23, CPU end-to-end5 and CUDA5 gates pass; the original two
  worker-forwarding failures remain archived. Root and independent review
  verify the 99-payload archive eaaa24bf9e0fa2f8a2f2102fc322d5fbbbf1c9201481fc1093d9e158f1fe4ff4
  and the separate scratch-budget archive d13cdf774e3db527efbdbfb75227ea540cd6302e2661bf642542ce66fe2949b8.
- The new n006 target independently rebuilds the final source in 9.529163 s,
  then passes CPU24 in 10.033081 s and actual CUDA5 in 99.314905 s, all exit0,
  unchanged source, no skips. CUDA0 output leaves are checked; GPU6 is absent.
  These concurrent admission runs are correctness gates, not timing comparisons.
  The new build produces library SHA1e63dc2b2a12f55adb3401679e42c75cf2761720faffd99c7a77c4b78a625c01;
  upstream compiler/clock-skew warnings remain in the log. Root verifies all
  32 admission payloads, the exact12 current source/test files, exits and XML.
  Evidence: delta-nv5090-20261023/native-admission-evidence.tar.gz, SHA256
  4a601571c8d9b7af9561c4140ee07bd287e79b613feec2975be809bedb0a916d.
  Whole-method warm scaling and peak-memory gates remain open; no TODO is added.

### 2026-10-23 — WF-29c: SCF assembly, fixed forward and implicit-response ownership

- CorrelatedEnergy passes its original MB budget into MolecularEnergy, HF
  assembly, the existing orbital minimizer and the existing root/checked linear
  response. None retains the original path. The forward preflight preserves
  L-BFGS20, two stages, 24 backtracks, five polishing steps and the original
  dense-Hessian/GMRES split; it rejects insufficient fixed capacity instead of
  changing the algorithm. Only the existing implicit GMRES restart may adapt.
- Known source owners, saved VJP leaves, both response orientations and opaque
  native scratch are accounted by phase. Independent review reproduces two
  small omissions, then verifies their fixes: retained initial_energy and the
  original MoleculeSpec coordinate/charge arrays. Existing CE source ledgers
  also retain that spec. No physical formula, residual sign, regularization,
  reference anchor or AD owner is changed; 17 static proofs remain true.
- Final CPU22 are 11 metadata, five budgeted scalar/moving-S/explicit/implicit
  controls and three original native cases each for RHF and UHF. All exit0,
  no skips, on104081/n006/CPU16-31, with unchanged before/after source maps.
  The same final source additionally passes the existing complete MP2 CUDA
  value/gradient/HVP case, exit0/96.649308 s on GPU0/CPU192-207. This verifies
  budget introspection during actual CUDA tracing; GPU6 is not exposed.
- Root checks all 109 indexed CPU payloads plus the archive index (110 files),
  four raw exits/XML/maps, current seven source/four test files and 16 separate
  CUDA payloads. Actual RED9, owner RED2 and an earlier collection-resource
  failure remain recorded. Evidence scf-ownership/HANDOFF.md, root-verification.json;
  CPU archive SHA256 fd668d966ed3f1cb83d72ba8000deee7aebcd0fbbfc75c2f6d1638d37a767687;
  CUDA archive SHA256 a5c529db0721219a77503839ca83f7c0b6bb6381ae4974a71b179a01eee4626e.
  No compiler/allocator/outer-explicit-tape hard cap or speed gain is asserted.
- Acceptance review closes WF-04's established integral-response contract and
  WF-27's explicit full/block/contracted RDM export contract, using their existing
  CPU/CUDA and CC22/CI/FCI evidence. This does not close nuclear CC properties
  or full-method scaling. WF-33's mathematical/API gates are complete, but its
  required warm measurement remains open within the existing final benchmark
  work. No TODO, child level or model matrix is added.

### 2026-10-23 — WF-33/32: measured standalone MP3 warm gate

- Reuse the already accepted four-orbital/two-occupied array case, seed2426,
  float64, frozen=None, with_t2=True, max_memory64 MB. The thin runner calls
  only the existing selected MP3 source entry and compares the complete result
  against its saved accepted bundle with the original dtype checks and
  atol3e-11/rtol3e-9. No new method, formula, oracle or model matrix is added.
- Actual exit0/5.676390 s on104081/n006/CPU208-223, GPU hidden. Lowering is
  0.224310 s, compilation0.193096 s, first execution0.026270 s; five synchronized
  warm samples have median0.157196 ms. XLA temporary arrays are5760 B; complete
  child peak738200 KiB includes startup/compiler/verification, distinct from
  the earlier self-report666720 KiB. This is a nonexclusive fixed component
  point, not scaling or a speedup. Source/input maps remain unchanged.
- Root verifies14 archive members, the original fixture/setup/reference and
  current source/runner, actual exit and accepted-output comparison. Evidence
  delta-mp3-orbitals-r1/warm/HANDOFF.json and root-verification.json; archive
  SHA2565acf0ca26b4627ea45a15c5d031f31f8fe642c75a72f3987eead4e374c5d9f9e.
  This supplies WF-33's last required warm record; its established functional,
  AD and caller gates are unchanged. WF-33 is accepted; WF-32 stays open.

### 2026-10-23 — WF-08: combined direct aniline attempt, not a completed gradient

- The same14-atom/119-AO Cartesian6-31G* input uses all25 occupied/94 virtual
  orbitals, noncanonical MP2, no DF/frozen spaces, zero screening, original
  60000 MB budget and eight direct workers. A fail-on-full-ERI guard covers
  seed preparation and AD tracing. The entire source is frozen17b7b30; only
  the thin runner adds explicit worker selection and records it in the report.
- Seed mf.run completes in33.534554 s; compilation takes29.470212 s and reports
  3020471744 B XLA temporaries. The adapter's first execution still includes
  its orbital solve/polishing, correlation and reverse response. It does not
  finish before the actual570.172754 s child cap (SIGKILL, wrapper exit124).
  There is no first result, gradient or warm sample. Partial child peak2112080
  KiB is not a completed workflow peak or evidence of a memory-reduction ratio.
- Root verifies10 archived payloads, actual terminal/partial report and exact
  before/after source, geometry, runner and library hashes. Evidence
  delta-nv5090-20261023/aniline-direct-timeout-evidence.tar.gz, SHA256
  47049944da99e5bd83a00839e1598db60bd6de6749ead07e0b08478e7a8d4f75.
  The earlier fixed-density J/K speedup does not establish full MP2 speed.
  Do not repeat the long attempt or raise its limit without new localization.
- Source review identifies two finite profiling directions: invariant Fock/
  Hessian linearization during existing polishing, and serial slab geometry
  VJP plus cross-slab/rematerialization work. Same-slab MO requests already
  share intermediates. The reference-BLAS link does not justify blaming BLAS:
  external dgemm_ is used by ECP, not this all-electron shell path. A bounded
  diagnostic probe belongs to the existing WF-08, not a new TODO branch.

### 2026-10-23 — WF-08/29: stationary seeds enter the existing Newton refinements

- A temporary native probe and the existing state/count runner localize the
  initial aniline delay. In the 180 s caller-level prefix, stage0 starts with
  half-gradient norm5.90616372548961e-11, below the requested1e-10, but the
  initial L-BFGS gate uses1e-11. The scan has not ended at timeout; GMRES has
  not started. Native target timings alone could not establish that attribution.
  Root verifies all28 indexed caller-probe payloads and the original mathematical
  AST after removing the two scalar observers. Archive SHA256
  d6eee49f3f0c2f285492c1ca42f8e22a076216d4688d3171738b3b6266538d7a.
- Change only the initial L-BFGS gate to the declared gradient tolerance.
  Already stationary seeds retain the original three actual Newton refinements;
  nonstationary scan controls, inner0.1-tolerance rule, backtracking, shifts,
  iteration limits, final stationarity and implicit AD owner are unchanged.
  The finite explicit trajectory follows the new entrance decision. A shifted
  scalar quadratic verifies its analytic finite response7/8, while a separate
  well-conditioned case approaches the full stationary response. Both retain
  nonzero JVP/VJP and directional second response.
- Actual RED is3 expected assertion failures/2 passes, exit1/8.788978 s.
  After the single-line fix,18 unique CPU cases pass: five new entrance/response
  cases, five existing metric/implicit/finite-explicit controls, six native
  RHF/UHF state/density/geometry cases and two relaxed direct MP2 mixed-response
  cases. The final test's same-branch FD step is separately rerun:5 passes,
  exit0/8.286340 s. The existing complete CUDA MP2 gradient/HVP case passes,
  exit0/163.753526 s, with every actual output leaf on assigned GPU0. No native
  kernel or library is changed. Tests run only on104081/n006, CPU0-15 or GPU0
  with host192-207; GPU6 is never exposed.
- Root and independent spec/quality review verify the sole mathematical AST
  delta, final source/test hashes,75 indexed GREEN payloads, six original
  exits/XML/source maps and GPU UUID. Evidence stationary-seed-green/HANDOFF.json
  and root-verification.json; archive SHA256
  c0f1c6bfad663985fad3a22f5cd18a18a6592593bb07414d3942e668ba3e761b.
  This accepts a finite WF-08 prerequisite. A separately bounded postfix
  aniline state/count run remains diagnostic; the full direct gradient, warm
  timing and scale gate stay open. No parent or child is added or closed here.

### 2026-10-23 — WF-21: current FCI action/solver response budget accepted

- Reuse the original four-orbital, nelec(2,1), seed2901 joint-ledger fixture.
  The original0.055 MB window still passes its forward capacity/useful-batch
  checks, but correctly rejects AD under the current shared response planner:
  minimum working182496 B exceeds available32605 B. The older nine passing
  nodes used an earlier shared response source and did not close this frontier.
- One bounded1 MB diagnostic observes the existing gradient and HVP planner
  calls without changing the numerical equations. The largest declared phase
  is11395 resident+261248 working=272643 B. At0.5 MB, one20% reserve leaves
  388605 B after resident storage, enough for that phase. Preserve the0.055 MB
  AD rejection explicitly, and compare the original value/density/gradient/HVP
  loop at0.5/1 MB with its unchanged physical inputs and tolerances. No FCI or
  shared-solver production source is changed.
- The actual three-node closure passes, exit0/52.637213 s, no errors/skips,
  on104081/n006/CPU16-31 with unchanged source maps. Root verifies27 archived
  payloads, current six production sources and test, original XML/exits,24
  diagnostic planner records and the preserved scientific test AST. Evidence
  fci-current-response-7753/FINAL_VERIFIED.json and root-verification.json;
  archive SHA256f35bad31794ae9391bbd485074203932587647ddcb1c786497543946739b05f5.
- Independent spec/quality review combines these current-owner gates with
  WF-21a's sign/rectangular/RHS/transpose/AD and measured action/reverse tiles,
  and WF-21b's public preflight/state reservations. WF-21c and WF-21 are accepted
  for the declared-array/action/shared-solver contract. Nuclear properties,
  CPU/GPU whole-method scaling and an operating-system RSS cap are separate
  WF-22/30/32 concerns. The0.5 MB planner window is not a process-memory claim.

### 2026-10-23 — WF-20: existing CI action/cache contract accepted

- Inspect the existing implementation before adding another optimization.
  Creator/spectator/RHS widths already use the shared byte budget and compact
  column loop. The action closure contains static topology/plans; physical
  kernels, constants and CI vectors remain explicit arguments. Seven bounded
  static array caches total288000000 B of declared payload capacity. Dynamic
  method/prepared/result arrays remain in caller, solver and property ledgers.
- Preserve the original block-AD, compact-matmat, Davidson/PySCF, allocation
  guard and cache-release gates. CI direct/solver/API and the raw response test
  are byte-identical to their earlier accepted source. Complete the current
  shared-Hermitian-owner frontier with the four original CI/UCI/CIS/UCIS
  fixed-MO/integral response nodes, with32 MB versus the original unbudgeted
  reference. All four pass, actual exit0, no errors/skips, on104081/n006/CPU16-31.
  Three sequential child durations are136.408998/169.449148/42.782153 s.
- Keep the initial110 s timeout: no node or complete XML had finished, and the
  earlier accepted CI cold case itself took180.701 s on another node. That
  attempt established an insufficient test cap, not a numerical or performance
  regression. The later bounded shards complete the same inputs and assertions.
- Root verifies35 archived payloads, raw XML/exits, current eight production
  files and test,584 tracked-Python before/after entries including545 production
  Python files matching the frozen input. Independent spec/quality review closes
  the original finite WF-20 criteria. Evidence fci-current-response-7753/HANDOFF.md,
  CI_FINAL_VERIFIED.json and root-ci-verification.json; archive SHA256
  7beb2a7be2c111c44a2ddea80c4094783c55c39ae8ad2041da14d39bbb1836be.
  This is an evidence closure with no additional CI production change. Nuclear
  properties, outer-AD/full-method residency and scaling retain their separate
  WF-22/29/30/32 gates. No speedup or operating-system memory cap is asserted.

### 2026-10-23 — WF-23/24: rank bounds and packed capacity fallback accepted

- Recheck the original finite clauses against existing raw evidence before
  adding further kernels. WF-23 has13 focused,28 forward and3 response passing
  nodes in the recorded groups, with actual exit0. Order12 compares all energy
  coefficients; separate nodes exercise actual triples/quadruples/sextuples,
  frozen/open-shell spaces and extra wave orders. Required input histories and
  the perturbation partition are preserved. The six mathematical definitions
  remain AST-identical, including the normalized helper now shared with Taylor.
- WF-24's four archived groups contain13 unique passing nodes, covering full
  T6-to-T6 and T6-to-T2 actions, R/U joint input/state response, mixed dtype,
  frozen spaces, no-expansion guards and packed series. Current packed source
  and its action test match the executed bytes; reachable CI helper bodies are
  unchanged apart from static-cache decorators. The earlier capacity node used
  test-file54d14119 and the later response group usesb9d45c33; the capacity AST
  and all scientific assertions are equivalent. Do not claim one test-file
  revision was used by all four groups.
- The same400-coefficient nmo6/no3/rank6, seed2423/f64 component changes dense
  to packed warm22.664 to7.341 ms and XLA temporary142740872 to1015656 B.
  Compilation3.360 to75.454 s and complete child RSS918324 to3011136 KiB
  increase;3011128 KiB was the benchmark's earlier internal sample. Retain
  dense contractions when affordable and report this as a capacity tradeoff,
  not a universal process-memory reduction.
- Root checks the original XML/exits, archive49 members, source/test lineage,
  reachable helper ASTs and measurement records; independent spec/quality review
  accepts WF-23/24. Evidence mp-checkpoint/green/rank-packed-root-verification.json;
  packed archive SHA256678a9b72f1b77f3de2f70eabb1120c43fd8bbc72c4a0d3afd8d88aa1edd2cfc0.
  Whole-method size/performance and final device coverage remain WF-30/32.

### 2026-10-23 — WF-25: shared normalized Taylor and checkpoint cost gate

- Move the existing two-line normalized `_jet` helper into the shared Taylor
  module and use it from both MP and `root_taylor`. The generic root solver and
  the specialized MP physical recurrence retain their existing equations.
  Root/peer AST checks confirm the helper is unchanged, every other MP function
  is unchanged, and inlining the helper restores the original root-Taylor AST.
- New tests derive two budgets from the actual traced planner, then confirm
  both real differentiated calls use dense actions with checkpoint off/on.
  Both with_coefficients modes retain the complete required histories and match
  all outputs and JVP/VJP/HVP; every compared numerical leaf must be finite.
  Together with the original root-Taylor tests, the final5 cases pass, actual
  exit0/51.174993 s, no errors/skips. The old helper-missing RED is retained.
- The matched MP4/nmo3/no1/seed2318/f64 interaction-scale AD workload uses separate fresh CPU
  children on104081/n006/CPU16-31. Off/on inputs, configuration and full saved
  numerical outputs are identical. Off/on nominal budgets are0.026445/0.022125
  MB. Lowering is1.16558/1.24771 s, compilation0.96510/1.17614 s, first execution
  3.13886/5.63282 ms and seven-sample warm medians0.238478/0.248866 ms. XLA
  temporaries are10241/11808 B; complete child peaks797592/809864 KiB differ
  from internal samples739308/750444 KiB. This small forced-on point is slower
  and uses more memory; it demonstrates no memory saving. The ordinary60 GB
  budget selects off for this point.
- Root and independent review verify45 payloads, all three actual exits and
  source maps,5-case XML, final four files, policy flags and finite identical
  numerical outputs. Evidence mp-checkpoint/green/HANDOFF.json and
  root-verification.json; archive SHA256
  7f346690dcdddfc0d6e30e5b2c6b7f625435030f4a83bea1785e0a7eec69d4cc.
  WF-25's finite shared-primitive/history/policy/AD and measured-cost contract
  is accepted. Its small-point result is not a scaling or RSS-improvement claim;
  WF-32's method-level size/performance gate remains open.

### 2026-10-23 — WF-08/29: right-preconditioned primal HF corrections

- The unchanged aniline state diagnostic reached polishing after the entrance
  fix, but its first unpreconditioned GMRES needed84 actions. Read the installed
  JAX0.10.0 and PySCF2.13.0 sources before choosing a numerical change. JAX's
  left-preconditioner stopping norm differs from the physical residual, and
  info0 alone does not certify convergence. Retain the existing shifted operator
  A=H+sigma*I; solve A(D^-1*y)=-g and recover step=D^-1*y with M=None.
- A shared frozen-Fock gap helper supplies RHF4*gap, unrestricted2*occupation-
  difference*gap and common-orbital spin sums in the existing pair order.
  Stability reuses that helper. Only HF roks/uks with more than32 angles enable
  the new primal callback. Physical arrays remain live arguments; the cached
  callback identity contains static topology/functions only. Numerical D is
  stopped for AD and sign-floored at1e-8; the physical Hessian and original
  shift sequence, tolerances, true-residual/finite/descent/step checks, three
  refinements and implicit-backward equations remain unchanged. Preflight adds
  the callback graph and four real angle vectors under the original budget.
- Preserve the actual test history:11 expected missing-interface RED failures;
  first GREEN14 passes/2 cost-test failures. The latter incorrectly treated the
  old tiny-RHS GMRES as an exact oracle despite its absolute-epsilon cutoff.
  Keep both old physical guards, require the new solution's strict analytic
  trajectory and residual, and rerun only those two cases: both pass. The48D
  actions drop129/211 to12, with new physical residuals below2.58e-16; this is
  a controlled action-count result, not molecular wall-time speedup.
- Final acceptance comprises25 unique CPU cases and one CUDA case. Native
  water/6-31G and NH/6-31G spin2 actually exercise40/54 angles, full/direct
  state, density response and geometry gradient/HVP/FD with128 MB planning.
  The two geometry cases are additionally rerun with explicit finite checks.
  Three original stability nodes cover the helper's existing callers. The
  CUDA water MP2 energy/full gradient/directional HVP passes the CPU oracle,
  all finite/device checks and the real40-angle callback spy on physical GPU0.
  Its2615940 KiB complete host peak is not a VRAM measurement.
- The same14-atom/119-AO aniline input, all25 occupied/94 virtual orbitals,
  zero screening, eight direct workers and60000 MB budget now completes the
  state-plus-MP2-forward diagnostic. HF energy is-285.7279603824555 Eh,
  orbital gradient norm2.7521621007544088e-12, and correlation energy
  -0.9546311957491811 Eh. All three refinements are accepted with25 GMRES
  actions each and true relative residuals3.70646e-12/3.69774e-12/3.33905e-12.
  The325.220868 s child includes seed and MP2 work; measured SCF lowering,
  compile-only and first execution are13.130359/10.938037/249.377475 s.
  The earlier300 s run timed out after the same three solves; the final run
  changed only its cap to600 s and returned valid state/integrals/MP2 flags.
- Root verifies nine archives/228 payloads, original exits/XML, source/input
  maps, final source files, actual GPU UUID, unit counts and physical residuals;
  independent source/spec review passes. Evidence
  scf-orbital-preconditioner/root-verification.json and per-group HANDOFF files;
  final forward archive SHA256
  a45c621b82c96726078e83350bfe79f024ffc943d3edf3f72834d54ccabda79c.
  This is a primal efficiency prerequisite. Full aniline geometry gradient,
  matched warm/RSS comparison and increasing-size acceptance remain WF-08/32;
  no additional parent/child is closed by this change.

### 2026-10-23 — WF-08: completed fixed-aniline direct/full gradient pair

- At dde6f9a, the fixed14-atom/119-AO Cartesian6-31G* aniline MP2 pair completes
  sequentially on104081/n006, CPU0-15/JAX0.10/float64. Preserve all25 occupied/
  94 virtual orbitals, zero screening,60000 MB planning, original32 GiB XLA
  guard, input67e2 and the frozen PySCF2.13 oracle. Actual child/srun/oracle/pair
  exits are0; both modes finish first and one warm value/gradient execution.
- Both pass the original absolute energy3e-9 Eh and per-component gradient
  3e-7 Eh/bohr tolerances. Direct/full energy differs by2.2737367544323206e-13 Eh;
  the maximum difference among42 gradient components is3.545367558560608e-12
  Eh/bohr. Each mode's maximum PySCF gradient error is about4.5533e-8 Eh/bohr.
- Direct/full first times are603.337557/178.270864 s, warm1 times
  605.247343/178.587012 s, and complete child peaks5963260/23529724 KiB.
  XLA temporary estimates are3020471992/22467944224 B. Direct uses25.34% of
  the full process peak but takes3.389x the warm time on this point. This is
  a measured memory/time tradeoff; it does not justify a universal faster
  default or establish scaling. The explicit full path remains available.
- The full first finite gate and timing exist, but a separate first snapshot
  was not captured; do not claim full first/final bitwise equality. Final NPZ
  and accuracy comparisons are complete, and the direct artifacts remain intact.
  Independent stdlib review verifies60/46 full/direct archive payloads,1072
  committed source files, identical1774-entry runtime maps, eight unchanged
  direct-reference files and nonoverlapping execution. Evidence
  wf08-aniline-full-gradient-dde6f9a/root-pair-verification.json;
  full archive SHA25610f93ba3e737a6ab2abc8263864a514c580cd1a0e6dadb9728dade239db1f6f6.
  WF-08 remains open for the finite R/U size family and remaining joint-budget
  boundaries; no parent or child is added or closed by this single-point pair.

### 2026-10-23 — WF-08: finite CPU MP2 measurement gate accepted

- Complete five additional fixed full/direct Cartesian 6-31G* gradient pairs
  on 104443/n006, CPU 96-111, JAX 0.10, float64, without GPU use. Numerical
  sources remain `dde6f9a`; frozen HEAD `134efc2` differs only in documentation.
  Keep all orbitals, original tolerances, zero screening, eight direct workers,
  60000 MB planning and the 32 GiB compiler guard. The restricted size sequence
  is water/ethane/aniline at 19/42/119 AO; the unrestricted points are H3 doublet,
  NH triplet and O2 triplet at 6/17/30 AO. These are fixed geometries, not new
  optimizations or orbital-stability certifications.
- All ten new numerical children and their srun controls exit 0; all five
  allocation-side comparison steps exit 0 with 29 checks per pair. Every point
  has finite energy/full gradient and one first plus one warm execution. The
  maximum full/direct differences are 5.68435e-14 Eh and 2.95445e-13 Eh/bohr,
  within the unchanged absolute 3e-9 Eh / 3e-7 Eh/bohr tolerances, rtol=0.
  No new PySCF oracle was computed for these five points; aniline retains its
  independent frozen oracle. No first/final bitwise-identity claim is made.

  | Point | Full/direct warm time (s) | Full/direct complete child RSS (KiB) |
  | --- | ---: | ---: |
  | Water | 0.097220 / 0.665399 | 1304432 / 1395408 |
  | Ethane | 2.027117 / 12.008879 | 1444224 / 1736840 |
  | H3 doublet | 0.005206 / 0.033660 | 1314860 / 1380780 |
  | NH triplet | 0.075015 / 0.459878 | 1473944 / 1542720 |
  | O2 triplet | 0.385470 / 2.387316 | 1519752 / 1685472 |
  | Aniline (previous allocation) | 178.587012 / 605.247343 | 23529724 / 5963260 |

- Small systems pay additional direct-mode time and process memory. Aniline's
  measured peak falls by about 3.95x, meeting the original >=2x target, with a
  3.389x warm-time cost. Direct remains explicit opt-in; the full reference path
  stays available. The result is a finite size-family tradeoff, not an
  asymptotic scaling fit, universal speedup or repeated-sample timing estimate.
  Compilation, first/warm/full-child times, internal RSS and compiler temporary
  estimates are stored separately in `size-family/completed-family-summary.json`.
- Known-array accounting reuses the verified channel/T2/solver ledger
  `delta-v100-20261022/mp2-graph-budget-root-verification.json`, joint preparation
  and caller reservations in `delta-mp-callers-r1/review-p2/HANDOFF.json`, and
  `scf-ownership/phase-ledger.json` / `root-verification.json`. WF-04/07 retain
  the no-full-ERI/cotangent structural gates. Independent review accepts this
  evidence mapping; arbitrary outer-AD ownership and compiler/allocator hard
  bounds remain with WF-02/05/06/29/30/32, as specified by the original plan.
- Root verifies all five downloaded archive hashes/bytes, saved energy/gradient
  differences and sequential execution using only local standard-library code.
  Remote runtime manifests are unchanged; concurrent local implementation work
  was not uploaded. Evidence: `size-family/root-family-verification.json` and
  each point's download verification, plus the previous aniline pair review.
  **Accept WF-08 only.** Parent acceptance becomes 16/35; children remain 28/37.

### 2026-10-23 — WF-13: direct packed-coordinate UCC energy

- Share one 21-line energy contraction between UCC and Lambda. With the existing
  norm-preserving coordinates x2=2*t2, contract each independent double against
  `(g_ijab-g_jiab-g_ijba+g_jiba)/8`. Promote the four gathered values before
  subtraction. This preserves derivatives with respect to raw nonsymmetric
  integrals and mixed dtypes; assuming physical ERI symmetry here would change
  that input chart. Singles and T1*T1 terms retain the original formula.
- The residual, amplitude coordinates, Lambda dual factor and residual-memory
  envelope are unchanged. This avoids dense T2 only in energy/energy-gradient
  evaluation; it does not turn the whole UCC residual into a block algorithm.
  The removed unused import is the only other production cleanup.
- Final validation on 104443/n006, CPU 112-127, no GPU, yields 48 unique passes,
  zero failures/errors/skips, actual child/srun exits 0. The 47-case group covers
  raw energy JVP/VJP/HVP, float32/float64/mixed, empty/frozen CCD/CCSD, Lambda,
  restricted limits and existing UCC tests. A separate LiH+/STO-3G relaxed CCSD
  gradient/HVP case passes the original finite-difference checks. Preserve the
  29 expected missing-helper RED failures and earlier GREEN as provenance.
- The fixed synthetic energy component uses occupied (8,7), virtual (80,79),
  seed 13157, raw nonantisymmetric ERIs, JAX 0.10/float64 and seven warm samples.
  Dense/packed warm medians are 28.210126/9.022251 ms, compiler temporaries
  144688584/53667336 B, and primary pre-oracle RSS 907548/814100 KiB. Energy
  differs by 2.42862e-17 Eh. The compiled dense path has four full-T2 scatters;
  the packed path has none. Final packed whole-child RSS is 946340 KiB because
  it subsequently evaluates the opposite-mode oracle; do not substitute that
  value for primary-phase RSS or infer a complete-method speedup.
- Root reviews the exact formula and source changes and verifies final source
  hashes, archive SHA and original final exits. Evidence:
  `wf13-packed-energy/HANDOFF.json`, `local-verification.json` and
  `implementation-review.json`; 138 indexed payloads, archive SHA256
  `2d39b6ccc00cc4cc9574f09729fbaa5f1d0ccd32c376dd5c93b20fcd77c3c621`.
  Snapshot `134efc2` plus the three CC files and narrow test excludes concurrent
  AO2MO/device changes. Accept this component only; WF-13 remains open.

### 2026-10-23 — WF-02c: shared target-capacity checks

- Consolidate six AO2MO device-selection sites in `_memory.array_device`.
  Preserve a known eager array device, including legacy callable accessors;
  otherwise inherit the concrete `jax.default_device`. Shared budget calls
  without an explicit device now inherit that context too. Accelerator strings
  such as `"gpu"` cannot silently fall through to host capacity: require a
  concrete device and reject invalid objects. Do not enumerate/init other GPUs,
  alter allocators or introduce a new budget parameter.
- Independent review caught and removed an intermediate context-first rule
  that could override a committed eager array's real device, plus a missing
  callable normalization at the shared boundary. Both failures were reproduced
  on the earlier implementation before their guards passed. Existing CPU
  behavior is retained. An outer `jit(device=...)` or captured CPU constants do
  not reliably reveal the executable's target to Python planners. The supported
  traced GPU contract uses concrete target placement and default-device context;
  capacity checks occur during planning, not each cached executable invocation.
- Final remote validation yields 80 targeted CPU passes, 64 native CPU passes
  and two real CUDA passes, with actual child/srun exits 0 and unchanged runtime
  manifests. The CPU native group skips one separate GPU opt-in batch case;
  the two CUDA cases run without skips. They cover native coefficient/coordinate
  HVP against a CPU oracle and shared traced DF AO2MO/tensor-sum planning. The
  latter first queries real device capacity, then injects 4500 B to exercise a
  deterministic small cap. This is not a peak-VRAM or whole-process RSS bound.
- Preserve the failed test history. An incomplete legacy SCF routing mock did
  not accept the already-existing caller reservation; update its interface and
  assert forwarded ownership explicitly. The Cartesian-d accuracy fixture's
  2 MB budget also fails on the immutable baseline: its minimum declared budget
  is about 2.195475 MB with native scratch included. Raise only that fixture's
  workspace to 3 MB, retaining geometry, basis, original numerical tolerances
  and separate insufficient-budget tests. No production threshold is relaxed.
- Evidence: `device-budget/green/HANDOFF.json`, final CPU/CUDA download checks,
  preserved RED records, and independent source review. Run on 104443/n006,
  CPU 288-303; CUDA uses only UUID GPU-7aa3e8aa-431b-da04-7ba3-5c37ae5479d0.
  Final CPU/CUDA manifests are `d5057bcc...` / `38e91c3b...`; native SO is
  unchanged. A subsequent three-line AO2MO docstring change is AST-equivalent
  and recorded separately. The generic captured-constant/outer-JIT target
  boundary remains open under WF-02c; no parent or child closes here.

### 2026-10-23 — WF-03/05: consolidate existing finite acceptance evidence

- Independent review maps the original requirements to already completed
  gates, without rerunning a separate migration or enlarging the TODO tree.
  WF-03a/b/c provide native slabs, asymmetric/permuted/frozen requests, shared
  half transformations and coefficient/coordinate response. Existing MP, CC,
  CI and FCI producer integrations establish consumer use. The missing ledger
  link for WF-03d is the actual matched conversion measurement in
  `rcc-prepared/rcc-prepared-profile1/comparisons.json`, not whole-gradient time.
- The earlier jy_gpu_1 CPU80-87 runs compare full MO conversion plus selection
  with shared selected preparation on fixed water frames. All six setup/full/
  selected process exits are 0. For 19/65 AO, full/selected warm times are
  21.017/17.731 ms and 0.477329/0.554637 s; compiler temporaries and observed RSS
  are separately recorded. The 65-AO selected conversion is 16.2% slower,
  with RSS 1314344/1251212 KiB. These are historical conversion measurements,
  not new Delta timings, a 2x saving or a universal speedup. Later validation
  changes preserve the numerical transform and its already verified responses.
- WF-05 combines its selected restricted OV blocks, dense/s4/DF/PySCF and
  with/without-T2 tests with the shared output/solver/caller reservations from
  `033657e`. Evidence remains `delta-mp-callers-r1/HANDOFF.json` and
  `review-p2/HANDOFF.json`; the original failures and targeted reruns remain
  visible. WF-08 now supplies the finite same-model method measurements.
  This satisfies the restricted integral/energy gate. It does not close the
  separate unrestricted cross-channel saved-AD owner question in WF-06b.
- Root rereads the conversion comparisons, actual process exits and previously
  verified integration records. **Accept WF-03d/WF-03 and WF-05b/WF-05.**
  Acceptance becomes 18/35 parents and 30/37 children. WF-02b's canonical
  eigensolve caller relay still awaits its running test's terminal evidence;
  WF-02c, WF-06b and the remaining shared/release gates stay open.

### 2026-10-24 — WF-02b/29c: canonical orbital-solve budget relay

- The explicit canonical MP branch in `CorrelatedEnergy` previously called the
  shared Hermitian solver without its caller's budget. Forward the original MB
  value and retain parent state/inputs/reference owners, current Fock transforms
  and the future rotated coefficient output. UHF beta additionally retains the
  completed alpha output and validity scalar. Preserve both original and
  canonical coefficient owners in subsequent correlation preparation.
- The physical matrix, dense all-root settings, tolerances, eigenpair validity
  guards and default noncanonical path are unchanged. No solver is added.
  Four actual RED failures identify missing propagation and too-small budgets
  reaching diagonalization; root and peer source/AST reviews pass after repair.
- The single GREEN run completes nine cases with zero failures/errors/skips,
  actual child/srun exits 0, 43.462071 s and 1550048 KiB whole-test-process peak.
  Six new R/U cases cover early rejection, caller/prior-spin ownership and
  separated canonical H2/6-31G plus asymmetric H3/STO-3G value/gradient parity;
  three existing workflow-budget cases also pass. No new HVP coverage is claimed.
- An SSH timeout hid the result temporarily; recovery retrieves the original
  controller 1937150 / srun 1937175 terminal files without resubmission. Root
  verifies the 25-payload archive, XML, original exits, all 1782 runtime entries
  and current source/test hashes. Evidence `canonical-eigh-budget/HANDOFF.json`
  and `green/download-verification.json`, archive SHA256
  `38fa256d46a145dbef568171aac9d1846bba5c4bddb186f1b2554eaa97ad23f9`.
  **Accept WF-02b.** Parents remain 18/35; children become 31/37. WF-02c and
  the known MP2 saved-response-state and remaining EOM phase gates stay open.

### 2026-10-24 — WF-18/22: genuine U quadruples and budgeted CI spin

- Close the final preparation-coverage gap with four spatial orbitals, U(2,2)
  and 36 determinants. The test explicitly applies a nonzero rank-4 column,
  compares selected action/diagonal with the original raw oracle, and rejects
  complete-MO requests or determinant-connection construction. Earlier
  three-electron tests labelled rank4 could not exercise actual quadruples.
  This is one new test, with no preparation algorithm change.
- CI `spin_square` previously built a complete two-body density for one scalar
  and missed the facade's budget. Supply the factorized alpha/beta overlap
  exchange to the existing compressed normal-order kernels, then use the shared
  budgeted sigma contraction. Forward the original MB budget and retained
  source/result owners. Preserve normalization, frozen electrons, distinct UHF
  frames, overlap guards and differentiable CI coefficients. Compressed kernels
  may still contain quartic-sized pieces; no quadratic-storage claim is made.
- Actual verification: rank4 1 pass; old-source spin RED 6 expected failures;
  new spin 6 passes; unchanged CID/spin/PySCF regressions 13 passes. All GREEN
  child/srun exits are 0, with zero errors/skips/timeouts. New gates include
  float32 coefficients/float64 overlap, no-full-RDM traps, budget rejection and
  facade ownership, finite joint coefficient/overlap gradient/HVP and two-step
  differences. Existing tests include UHF/ROHF, frozen and multiroot cases,
  known spin states, integral response and invalid/polarized limits.
- Run on 104443/n006, CPU112-127, 120 GiB, no GPU, JAX/JAXlib0.10.0 and
  PySCF2.13.0. The separately inspected upstream source was PySCF2.9.0; do not
  conflate those versions. Source/runtime maps are unchanged within each run.
  Root verifies current owned-file hashes, all 119 indexed payloads, XML and
  actual terminal exits. Evidence: `ci-spin-budget/HANDOFF.json` and
  `root-verification.json`. A metadata-only archive count error was corrected
  with original archives retained; no computation was rerun for that correction.
- Together with the accepted selected R/U/CID/frozen/ROHF guards and existing
  raw-chart derivative gates, **accept WF-18**. Parents become 19/35; children
  remain 31/37. The spin scalar slice advances WF-22; its CIS(D) complete nuclear
  response gate remains separate until that test's evidence is reviewed.

### 2026-10-24 — WF-22b/22: complete supported CIS(D) nuclear response

- Add one test composing the existing native moving RHF, canonical Hermitian
  eigenpair response, AO CIS eigenpair response and four selected correction
  blocks. No production implementation, solver, public API or theory is added.
  The observable is the current API's CIS(D) excitation energy Omega in Eh;
  it is not an absolute excited-state energy. All SCF, C, epsilon and CIS
  amplitude dependencies remain in the differentiated map with their guards.
- At fixed water/STO-3G, Cartesian basis, singlet, the excitation energy is
  0.4720525303722449 Eh and the nonzero correction -0.012531607810343764 Eh.
  The minimum orbital gap, CIS-root gap and denominator are respectively
  0.06187906347, 0.07169424118 and 1.43654198042 Eh. Point values match the
  existing CIS_D facade. All nine gradient and HVP components are finite and
  the translation sum passes; no stationary-orbital or eigenvector response
  has been omitted to make the derivative work.
- Two central-difference steps, 3e-4 and 1e-4 Bohr along a fixed normalized
  nuclear direction (seed749), give directional gradient errors
  1.55964e-9/2.11825e-10 Eh/bohr and maximum HVP errors
  1.74123e-8/1.85488e-9 Eh/bohr^2. Keep the declared finite-difference
  tolerances and validity checks. The two original tests retain triplet
  rejection, singular-denominator invalidation and disabled-amplitude-response
  derivative rejection; no new spin/cluster support is inferred.
- Geometry 1 pass and original guards 2 passes, actual child/srun exits 0,
  zero failures/errors/skips/timeouts, on 104443/n006 CPU112-127, no GPU.
  Geometry child time is 36.897883 s and peak 2046192 KiB, distinct from the
  256 MB method array budget. Numerical source is `7a767af`; the test-only
  snapshot's documentation head is `6ad8227`. Root verifies current test SHA,
  all 69 payloads, XML, runtime maps and terminal exits. Evidence:
  `cisd-geometry/HANDOFF.json` and `root-verification.json`; geometry archive
  SHA256 `5d8785fd581f4ab9d11959e2f71a1f2eeead649fa799a8a40b5ead5682a1c475`.
- Together with the verified CI/FCI relaxed-frame, RDM/observable and scalar
  spin gates, **accept WF-22b and WF-22** for the existing supported branches.
  Acceptance becomes 20/35 parents and 32/37 children. Larger-method and final
  GPU acceptance remain WF-30/32; this one small point does not close them.
