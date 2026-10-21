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

    - [x] **WF-02b1:** MP/CC/CI facades inherit the SCF budget, accept an
      explicit override, and forward the current method budget to shared MO
      adapters and requested blocks. Joint resident-state ownership is still open.
  - [ ] **WF-02c:** complete the native block/capability contract with WF-03/04,
    including target-device capacity when JIT tracing hides concrete devices.
  - [x] **WF-02d:** establish the dedicated AO2MO domain package and migrate
    existing callers without redundant forwarding files.
  - [x] **WF-02e:** keep static byte-reservation validation in `_memory`,
    shared by integral and solver consumers without an integral-to-solver
    dependency or duplicated validation code.
- [ ] **WF-03 — Exact native partial AO2MO** (`in_progress`). Generate requested MO tiles without
  full AO-pair or full MO ERI. Reuse shell traversal and half transformations.
  Acceptance: asymmetric coefficient sets, permutation symmetry, packed/dense
  oracle, frozen/core indexing, measured conversion time and tile workspace.

  - [x] **WF-03a:** native static shell-slice producer using the pinned shell
    fill driver, without a full AO tensor or a new integral engine.
  - [ ] **WF-03b:** bounded half transformations consuming slabs; no quadratic
    Python-unrolled graph or separate full integral traversal per MO pair.

    - [x] **WF-03b1:** dynamic shell endpoints with static padded storage and
      shared coordinate products through order two; compact differentiated loops.
    - [ ] **WF-03b2:** shared AO-to-MO half transformations, coefficient response,
      caller integration and measured conversion workspace/time.

      - [x] **WF-03b2a:** exact compact half-transform kernel and small-system
        forward/coefficient/coordinate response on CPU and GPU7.
      - [ ] **WF-03b2b:** consumer integration, multiple-request reuse and
        measured conversion/scaling workspace and timing.
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

### Coupled cluster, perturbative corrections and EOM

- [ ] **WF-09 — RCC integral preparation.** Request only required blocks; make
  orbital counts independent of `vvvv.shape`. Acceptance: frozen/core indexing,
  dense/DF/direct small equivalence, no transient full-MO conversion.

  - [x] **WF-09a:** remove the extra full active-space ERI copy in the
    existing MO-array path; derive virtual counts from the Fock dimension.
  - [ ] **WF-09b:** prepare requested AO-source blocks and share consumers.
- [ ] **WF-10 — RCC vvvv and intermediate actions.** Factor bare-vvvv contraction
  from dressing; tile tau/W consumers and release/recompute by lifetime.
  Acceptance: residual and energy term comparison, converged CCD/CCSD, JVP/VJP,
  Lambda, HVP and measured action workspace; no hidden dressed-vvvv cache.

  - [x] **WF-10a:** shared restricted virtual dressing action, with a static
    smaller-scratch choice for occupied-dominated spaces. Algebra/AD and action
    measurements include CC2's complete singles-pair construction. Retain
    factored inputs when expanding their outer product would cost more.
  - [ ] **WF-10b:** remove the bare-vvvv residency through shared AO-source
    actions, joint amplitude budgets and measured converged calculations.
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
  - [ ] **WF-12b:** AO-source selected blocks, byte ownership and scale evidence.
- [ ] **WF-13 — UCC residual/amplitude workspace.** Reduce repeated dense
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
- [ ] **WF-34 — EOM Dyson properties.** Contract orbital/root tiles without
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
- [ ] **WF-19 — FCI active-reference preparation.** Retain core folding and only
  necessary active ERIs; remove upstream full AO-pair materialization.
  Acceptance: h_eff/E_core/ERI equality, active/frozen mapping, coefficient and
  geometry derivatives. A dense nact^4 input can remain a budgeted small-CAS
  choice; it is not a full total-MO tensor.

  - [x] **WF-19a:** native/projected CAS ERIs and AO core folding, fixed-frame
    coefficient/geometry products through two and allocation preflights.
  - [ ] **WF-19b:** all reference representations and joint consumer budgets.
- [ ] **WF-20 — CI action/cache budget.** Replace fixed creator/spectator/RHS
  batches with byte planning and bound numerical/RDM caches by bytes.
  Acceptance: direct action/matmat parity, Davidson forward/response, no all-pairs
  connection build; repeated-call cache release and blockwise JVP/VJP.
- [ ] **WF-21 — FCI Hamiltonian action tiles.** Tile gather -> contraction ->
  transpose scatter instead of full E_pq*C stacks, rematerializing reverse tiles.
  Acceptance: alpha/beta signs, rectangular spin sectors, multiple RHS, transpose
  action and HVP; extra workspace scales with tile size, while input/output CI
  vectors and iterative subspace remain explicitly charged.

  - [x] **WF-21a:** alpha/beta/RHS Hamiltonian tiles, unit AD parity and
    measured action/reverse-residual memory and warm-time tradeoff.
  - [x] **WF-21b1:** public facade preparation/action budgets and tiled
    capacity checks.
  - [ ] **WF-21b2:** joint solver state, operator workspace and AD budget.
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

  - [ ] **WF-23a:** apply the existing direct-path Taylor/Wigner target bound
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

Every numerical test runs on **jy_gpu_1**, never locally. GPU6 is prohibited.
The latest 2026-10-21 instruction cancels the Slurm allocation workflow: run
bounded tests directly, with **at most 64 logical CPUs** for this project.
GPU7 is the preferred permitted GPU, subject to a live availability check;
GPU6 is never used. The earlier 96-CPU limit and 48-hour allocation proposal
are superseded. Keep individual tests bounded (normally at most two hours).
Prefer tiny CPU tests first; inspect live project jobs before using resources.
Large molecule
calculations are not resumed as a side effect of writing this plan. Obtain the
required authorization before new long campaigns or large dependencies.

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
3. Append exact evidence and remaining limits to section 8. Update the checkbox
   only when its acceptance is satisfied; otherwise record the narrower progress.
4. Re-read dependencies and the global forward/AD/device matrix. Record whether
   the plan remains valid or what changed and why. Do not silently remove scope.
5. Keep the diff focused and commit a reviewable change when verified. Do not
   merge or push the engineering branch without a user instruction.

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
