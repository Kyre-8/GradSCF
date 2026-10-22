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

Current acceptance snapshot (2026-10-22): six of the thirty-five parent tasks
are fully accepted (WF-00/01/07/09/19/34); twenty-seven of the thirty-seven existing
children are checked. These are acceptance counts, not a percentage estimate of
engineering effort. Keep all other parent checkboxes open. In particular,
successful component derivatives do not establish a whole-method memory bound.

| Existing scope | Accepted frontier | Remaining acceptance boundary |
| --- | --- | --- |
| WF-02/03/04/05/06/08 | Shared AO2MO, native block response, selected R/U MP2, fixed-aniline full-input gradient | Joint spin/T2/resolvent/AD ownership, direct-input performance, measured size family, traced target-device capacity |
| WF-10/11/12/13/14/15 | Restricted/spin virtual dressing actions; selected UHF facade, Lambda/RDM1 and nonzero streamed-(T) geometry; five-block CCS and direct physical-coordinate boundaries | Efficient bare-vvvv policy, packed residual lifetime, phase budgets and contracted two-body properties |
| WF-16/17/34 | Shared EOM primal planner, factored EA virtual actions, native prepared EE/IP/EA, streamed Dyson, retained-state accounting and original-source matched component measurements | Native phase/spectral-response workspaces and whole-method measurements |
| WF-18/19/20/21/22/27 | Selected CI/UCIS/CIS(D), ROHF guards, FCI active/caller budgets, spatial properties and native CI/FCI relaxed geometry; small full-RDM fast path retained | Explicit raw CI solver/correction budget propagation, complete phase/AD ownership, GPU and large-method measurements |
| WF-23/24/25/26/33 | Exact target bounds, packed canonical actions including actual sextuples, shared Taylor history budgets and selected pure-frame R/U MPn second response | Moving native R/U MPn, eager budget/source integration, selected standalone MP3 and method-level size/performance acceptance |
| WF-29 | Hermitian/non-Hermitian primal and common linear/nonlinear phase planners | Spectral response propagation, physical caller/closure ownership and whole-method reconciliation |
| WF-28/30/31/32 | Existing small oracles and component device checks remain usable | Evidence-based disk decision, final CPU/GPU audit, public examples/capability table and cross-family regression/scaling |

The damaged jy_gpu_1 stages remain paused. The user subsequently authorized
testing on jy_gpu_2 after reading its live manual: one V100, 64 logical CPUs,
256G memory, 48 hours. Allocation 103505 on n002 runs from 2026-10-22 11:53:40
to 2026-10-24 11:53:40 Asia/Shanghai. Actual CPU binding and separate CPU/CUDA
float64 JIT/AD environment probes pass; these are readiness checks, not method
regressions. Keep pending implementations separate from accepted evidence.
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
  - [ ] **WF-02b:** wire caller budgets/resident-state ownership into the MP/CC
    consumers as WF-05/09 migrate, rather than using the block default everywhere.
    Facade inheritance/overrides and forwarding are verified; joint ownership
    remains open. Historical checkpoint: WF-02b1.
  - [ ] **WF-02c:** complete the native block/capability contract with WF-03/04,
    including target-device capacity when JIT tracing hides concrete devices.
  - [x] **WF-02d:** establish the dedicated AO2MO domain package and migrate
    existing callers without redundant forwarding files; share byte-reservation
    validation in `_memory` without an integral-to-solver dependency or a second
    validator. Includes the verified historical WF-02e checkpoint.
- [ ] **WF-03 — Exact native partial AO2MO** (`in_progress`). Generate requested MO tiles without
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
  - [ ] **WF-03d:** integrate the shared transformations with the method
    consumers and measure conversion/scaling workspace and time. Reuse the
    corresponding WF-05/06/09/12/18/19 integration evidence; do not create a
    separate migration or benchmark campaign for the same consumer.
- [ ] **WF-04 — Block integral JVP/VJP and HVP** (`in_progress`). Add coefficient/coordinate
  derivatives and contracted reverse accumulation. Acceptance: tangent/cotangent
  duality, dense AD, coordinate finite difference and HVP; structural/runtime
  absence of full ERI cotangent; explicit unsupported basis/mixed-order guards.

  - [x] **WF-04a:** local slab cotangents and coordinate products through order
    two, reusing the existing geometry primitive family; CPU and bounded GPU
    bridge verification. Preserve the existing full-integral higher products.
- [ ] **WF-05 — Restricted MP2 integral/energy path.** Connect the shared provider
  to OVOV/OV-factor consumption, preserving with/without-T2 behavior and public
  conventions. Acceptance: PySCF and dense GradSCF energy/T2, no full AO/MO
  materialization, non-DF and existing DF tests; byte-accounted resident T2.

  - [x] **WF-05a:** native direct SCF-to-OVOV and pure orbital preparation,
    existing dense/s4/DF compatibility and construction preflight.
  - [ ] **WF-05b:** joint block/T2/solver ownership and measured method scaling.
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
- [ ] **WF-08 — First measured memory/scaling gate.** Fixed water/aniline input
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
- [ ] **WF-33 — Standalone canonical MP3.** Migrate its CC-integral preparation
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

- [ ] **WF-18 — Truncated-CI integral preparation.** Supply needed blocks through
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
  - [x] **WF-19b:** all reference representations and joint consumer budgets,\n    verified for the stated real common-spatial CPU/known-array contract.\n    Reuse representation/core-fold layers and the native relaxed closure;\n    CUDA geometry and large-CAS scaling remain WF-30/32.
- [ ] **WF-20 — CI action/cache budget.** Replace fixed creator/spectator/RHS
  batches with byte planning and bound numerical/RDM caches by bytes.
  Acceptance: direct action/matmat parity, Davidson forward/response, no all-pairs
  connection build; repeated-call cache release and blockwise JVP/VJP.

  - [x] **WF-20a** (`verified`): replace Python-expanded multi-RHS batches
    with the existing shared compact column loop; preserve empty/small batches
    and validate values, coefficient/integral response and measured warm-time/
    workspace tradeoff. Byte-selected widths and cache ownership remain open.
- [ ] **WF-21 — FCI Hamiltonian action tiles.** Tile gather -> contraction ->
  transpose scatter instead of full E_pq*C stacks, rematerializing reverse tiles.
  Acceptance: alpha/beta signs, rectangular spin sectors, multiple RHS, transpose
  action and HVP; extra workspace scales with tile size, while input/output CI
  vectors and iterative subspace remain explicitly charged.

  - [x] **WF-21a:** alpha/beta/RHS Hamiltonian tiles, unit AD parity and
    measured action/reverse-residual memory and warm-time tradeoff.
  - [x] **WF-21b:** public facade preparation/action budgets and tiled
    capacity checks.
  - [ ] **WF-21c:** joint solver state, operator workspace and AD budget.
    The prior WF-21b1/WF-21b2 labels remain only in their evidence records.
- [ ] **WF-22 — CI/FCI properties and geometry integration.** Stream bra/ket
  contractions and existing perturbative-CI corrections where applicable.
  Acceptance: energy/RDM identities and observable contractions, geometry
  gradient/HVP, root-cluster guards; necessary full-CI state size reported early.

  - [x] **WF-22a:** tiled FCI one-/two-RDM, transition and spin contractions,
    public budget propagation and measured first-reverse lifetime.
  - [ ] **WF-22b:** CI perturbative properties and relaxed geometry integration.

### Recursive MP

- [ ] **WF-23 — Exact target-rank restriction** (`in_progress`). Avoid residual target blocks
  that do not enter the requested Taylor/Wigner energy coefficient. Acceptance:
  per-order E2...E12 against current recursion, signs, open-shell/frozen cases;
  record which excitation ranks each case actually exercises (a two-electron
  high-order test cannot verify triples or quadruples contraction blocks);
  do not truncate required input waves or change the perturbation partition.

  - [x] **WF-23a:** apply the existing direct-path Taylor/Wigner target bound
    to canonical tensor actions before contraction. Retain every required
    wave coefficient; compare projected actions/response and complete energy
    series, including actual triples/quadruples/sextuples and frozen spins.
- [ ] **WF-24 — Packed target-tile contractions.** Keep increasing-index packed
  states and expand only contraction tiles. Acceptance: each excitation-block
  action against the current oracle, peak live buffers, JVP/VJP/HVP and costs.
- [ ] **WF-25 — Taylor history lifetime/checkpoints.** Share each W*Psi_j once,
  schedule waves/energies by last use and choose checkpoint/recompute policy.
  Acceptance: all Taylor coefficients and derivatives, with_coefficients True/
  False, no illegal removal of history required by later orders; measured time/
  memory tradeoff. Reuse `solvers/nonlinear/taylor.py`, do not add a second engine.

  - [x] **WF-25a** (`verified`): stable whole-recurrence compilation at the
    public series entry, with physical inputs and tolerance dtypes left live.
    Measure cold and warm costs separately; preserve old coefficient, raw-input
    and nested-response gates. This removes eager dispatch fragmentation, not
    combinatorial state storage or the subsequent history-lifetime work.
- [ ] **WF-26 — MPn geometry integration.** Connect blocks and lifetime policy to
  moving noncanonical R/U references. Acceptance: order-specific first/second
  responses and internal-orbital-degeneracy cases, justified capacity errors;
  mathematical divergence of an MP series must not be hidden by this optimization.

### Properties, storage, public integration and release gates

- [ ] **WF-27 — Explicit full-RDM export.** Provide budgeted block export or
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
    verified; spectral-response and remaining method caller ownership stay open.
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

Every numerical test runs remotely, never locally. The current 2026-10-22 user
instruction authorizes **jy_gpu_2 / Delta** via Slurm: one V100, 64 logical CPUs,
256G memory and 48 hours. Read the latest `lwcluster manual` before operating;
submit from `~/private/homefile/` to `home` with typed `gpu:V100:1`. Execute all
tests/builds inside allocation 103505 and its checked srun steps, not on the
login node. Use only its assigned GPU0; GPU6 remains prohibited. The observed
CPU set is 0-31,40-71. Four CPU lanes may use 0-15,16-31,40-55,56-71; a GPU
test consumes a released CPU lane rather than adding host threads beyond 64.
CPU srun steps explicitly use `--gres=none --mem=60G --immediate=15`; otherwise
they inherit the allocation's sole GPU and unnecessarily serialize. This
behavior was observed and the CPU-only metadata step verified before use.
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
  However compile3.360->75.454s and child peak918324->3011128KiB worsen.
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
