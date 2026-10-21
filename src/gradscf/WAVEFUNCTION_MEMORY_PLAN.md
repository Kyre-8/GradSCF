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
- `src/gradscf/integrals/{plan.py,molecular/ao2mo.py}`;
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
Introduce only the smallest shared internal post-HF interface actually needed by
the first consumers, preferably in `integrals/molecular/`; reuse `ao2mo.py` and
the native integral plan. Avoid separate MP/CI/CC implementations of AO2MO,
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
- [ ] **WF-03 — Exact native partial AO2MO** (`in_progress`). Generate requested MO tiles without
  full AO-pair or full MO ERI. Reuse shell traversal and half transformations.
  Acceptance: asymmetric coefficient sets, permutation symmetry, packed/dense
  oracle, frozen/core indexing, measured conversion time and tile workspace.

  - [x] **WF-03a:** native static shell-slice producer using the pinned shell
    fill driver, without a full AO tensor or a new integral engine.
  - [ ] **WF-03b:** bounded half transformations consuming slabs; no quadratic
    Python-unrolled graph or separate full integral traversal per MO pair.
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
- [ ] **WF-06 — Unrestricted and noncanonical MP2.** Supply aa/ab/bb channels
  without spin ERI, retaining the shared tensor-sum resolvent. Acceptance:
  closed/open-shell equality as appropriate, occupied/virtual rotation covariance,
  strict internal degeneracy, residual tolerance, JVP/VJP/HVP. Streaming the
  noncanonical inverse itself is a separate optimization if necessary-state
  storage still fits poorly; do not relabel an unchanged O(o²v²) solve as streamed.
- [ ] **WF-07 — CorrelatedEnergy MP2 geometry integration.** Remove its obligatory
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
- [ ] **WF-10 — RCC vvvv and intermediate actions.** Factor bare-vvvv contraction
  from dressing; tile tau/W consumers and release/recompute by lifetime.
  Acceptance: residual and energy term comparison, converged CCD/CCSD, JVP/VJP,
  Lambda, HVP and measured action workspace; no hidden dressed-vvvv cache.
- [ ] **WF-11 — Existing approximate CC/QCI branches.** Inspect separate CCS,
  CC2/linearized/QCI pathways and migrate applicable consumers. Acceptance:
  branch-specific energy/residual/derivative regressions; explicit exceptions
  rather than asserting that CCSD migration automatically covers every branch.
- [ ] **WF-33 — Standalone canonical MP3.** Migrate its CC-integral preparation
  and residual-JVP implementation separately from recursive MP. Acceptance:
  E2/E3, with/without-T2, frozen orbitals, JVP/VJP/HVP, series-order-three cross
  comparison, and byte-accounted simultaneous first/second doubles workspace.
- [ ] **WF-12 — UCC spatial-spin integral blocks.** Remove full `(2*nmo)^4`
  construction, map allowed aa/ab/bb blocks lazily. Acceptance: signs, factors,
  antisymmetry, unequal occupations, frozen orbitals, existing ROHF conversion
  and residual parity.
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
- [ ] **WF-20 — CI action/cache budget.** Replace fixed creator/spectator/RHS
  batches with byte planning and bound numerical/RDM caches by bytes.
  Acceptance: direct action/matmat parity, Davidson forward/response, no all-pairs
  connection build; repeated-call cache release and blockwise JVP/VJP.
- [ ] **WF-21 — FCI Hamiltonian action tiles.** Tile gather -> contraction ->
  transpose scatter instead of full E_pq*C stacks, rematerializing reverse tiles.
  Acceptance: alpha/beta signs, rectangular spin sectors, multiple RHS, transpose
  action and HVP; extra workspace scales with tile size, while input/output CI
  vectors and iterative subspace remain explicitly charged.
- [ ] **WF-22 — CI/FCI properties and geometry integration.** Stream bra/ket
  contractions and existing perturbative-CI corrections where applicable.
  Acceptance: energy/RDM identities and observable contractions, geometry
  gradient/HVP, root-cluster guards; necessary full-CI state size reported early.

### Recursive MP

- [ ] **WF-23 — Exact target-rank restriction.** Avoid residual target blocks
  that do not enter the requested Taylor/Wigner energy coefficient. Acceptance:
  per-order E2...E12 against current recursion, signs, open-shell/frozen cases;
  record which excitation ranks each case actually exercises (a two-electron
  high-order test cannot verify triples or quadruples contraction blocks);
  do not truncate required input waves or change the perturbation partition.
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
