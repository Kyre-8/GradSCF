# Shared numerical solvers

`gradscf.solvers` owns numerical forward algorithms, implicit derivatives and
their convergence checks. Electronic-structure modules construct Hamiltonian
actions, residuals, orbital maps and physical selection criteria. They do not
maintain copies of these solvers. Clients import the shared modules directly;
historical solver forwarding modules have been removed.

## Ownership

| Implementation | Shared owner | Clients |
| --- | --- | --- |
| Symmetric/Hermitian Davidson | `eigen/davidson.py` | TDA, CIS/CI, stability, periodic TDA |
| Unified rank-one / subspace Sylvester response | `eigen/response.py` | TDA, CI, isolated states and degenerate-subspace invariants |
| Shared Ritz solve and diagnostics | `eigen/primal.py` | All real Hermitian response targets |
| Real RPA Davidson and metric energy derivatives | `eigen/rpa.py` | Full TDHF/TDDFT, periodic Gamma response |
| Bounded complex dense RPA and metric energy derivatives | `eigen/rpa.py` | Small non-Gamma periodic response |
| Regularized full-spectrum and inverse-square-root derivatives | `eigen/spectral.py` | SCF orbital diagonalization and overlap response |
| Checked GMRES/direct solves and transpose response | `linear/` | Eigenvector response, SCF/GW implicit derivatives |
| Complex-to-real linear adapter | `linear/complex.py` | Exciton–vibration retarded response |
| Scalar-bordered Schur solve | `linear/schur.py` | scGW charge-constrained response |
| Fixed-point/root implicit differentiation | `nonlinear/` | SCF and scGW |
| DIIS and callback-based orbital minimization | `nonlinear/` | SCF families |

`tddft/eigensolvers.py`, `tddft/eigenvector_differentiation.py`, `scf/implicit.py`,
`scf/diis.py`, `scf/_orbital_solver.py`, and `ci/response.py` have been removed.
Architecture tests prevent their return and reject imports of those paths.
The shared layer has no dependency on CI, TDDFT, SCF, GW or PBC modules.

Import migration:

| Removed path | Canonical import |
| --- | --- |
| `gradscf.tddft.eigensolvers` | `gradscf.solvers.solve_hermitian` or `solve_rpa` (legacy complex RPA remains in `eigen.rpa`) |
| `gradscf.tddft.eigenvector_differentiation` | `solve_hermitian` with `EigenResponseConfig(target="eigenpairs")` |
| `gradscf.scf.implicit` | `gradscf.solvers.nonlinear` and `gradscf.solvers.linear` |
| `gradscf.scf.diis` | `gradscf.solvers.nonlinear.diis` |
| `gradscf.scf._orbital_solver` | `gradscf.solvers.nonlinear.minimize` |
| `gradscf.ci.response` | `gradscf.solvers.diagnostics` |
| `solve_spectral_projector` | `solve_hermitian` with a subspace response and optional `probes` |

`ImplicitFixedPointConfig` and `implicit_fixed_point_solution` are no longer
re-exported from `gradscf.scf`; import them from `gradscf.solvers.nonlinear`.
Use `EigenGradientMode` from `gradscf.solvers.eigen.response` for solver mode
annotations. TD-SCF tolerances and positive-frequency defaults remain in
`gradscf.tddft.defaults`, which defines method configuration, not solver aliases.

Physical construction remains in method modules: TD response matrices and
amplitude conventions, CI spaces, orbital parameterizations, scGW equations,
and the beta-dependent charge-resolution threshold. Calls to JAX's elementary
matrix operations inside physical expressions are not separate solver engines.

## Public matrix-free API

```python
import jax
import jax.numpy as jnp
from gradscf.solvers import (
    LinearOperator, EigenSolverConfig, LinearSolverConfig,
    solve_hermitian, solve_linear,
)

jax.config.update("jax_enable_x64", True)

def energy(matrix):
    op = LinearOperator(
        matrix.shape, matrix.dtype,
        matvec=lambda v: matrix @ v,
        matmat=lambda x: matrix @ x,
        diagonal=jnp.diag(matrix),
    )
    return solve_hermitian(op, config=EigenSolverConfig(nroots=1)).values[0]

a = jnp.array([[1.0, 0.1], [0.1, 2.0]])
gradient = jax.jit(jax.grad(energy))(a)
response = solve_linear(a, jnp.ones(2), config=LinearSolverConfig(rtol=1e-10))
```

Vectors have shape `(n,)`; block vectors have shape `(n, nvec)`. `matmat` is
optional: its default applies `matvec` over columns with `vmap`. A supplied
`transpose_matvec` must implement the algebraic transpose; otherwise `.T`
constructs the transpose through JAX. It is not a conjugate-transpose alias.
Construct `LinearOperator` inside a transformed function, or close over it;
the callable container itself is not a dynamic JIT argument. Captured numerical
parameters remain differentiable. With `method="direct"`, `solve_linear` accepts
RHS shape `(n,nrhs)` and solves all columns together with a shared bounded dense
matrix and checked implicit transpose response. GMRES also accepts block RHS;
its column loop stays inside the opaque implicit solve for transpose response.

`solve_complex(A,b,config=...)` accepts complex square matrices/operators and
vector or block RHS. It represents the problem by a real operator of size
`2*n`, and delegates forward/backward to the same checked `solve_linear`.
GMRES keeps this representation matrix-free; the direct `max_dense` bound
applies to `2*n`. The true residual norm, NaN/status failure policy, JVP/VJP
and higher response are retained. This does not change the real input contract
of `solve_linear`. No separate complex numerical algorithm is implemented.

The new `solve_hermitian` and `solve_linear` APIs accept real floating-point
data. Dense symmetric inputs are checked rather than silently repaired;
matrix-free callers assert symmetry. Existing low-level complex forward paths
remain available for periodic calculations, without widening the real API's
derivative contract. Dense materialization has a configurable size limit.
Davidson's legacy default subspace can reach the full dimension; specify
`max_subspace` to bound basis storage for larger problems.

Results are JAX-compatible named tuples, containing the solution/eigenpairs,
true residual norms, convergence flags and integer status (`0` success,
`1` not converged/invalid; Hermitian `2` means unresolved response prerequisites).
State diagnostics are per root; subspace validity is scalar. Raw diagnostics
include guard roots and are never AD targets. Iteration counts
are not reported because the inherited kernels do not expose reliable counts.
Linear `maxiter` counts GMRES restart cycles, not individual Krylov steps.

## Derivative contracts

- Linear systems use `custom_linear_solve`: both operator and RHS derivatives
  are retained, including transpose solves and higher derivatives. Zero and
  tiny RHS are handled by scaling inside the opaque numerical solve. Forward
  and transposed solves are independently checked using the true residual.
- `solve_hermitian` has one explicit `EigenResponseConfig`: `eigenvalues`
  differentiates isolated energies while returning stopped diagnostic vectors; `eigenpairs` also
  supplies isolated vector response; `subspace` returns the projector action
  on optional probes and the selected energy sum, including internal degeneracy.
  All targets share one forward solve and one implicit invariant-space response
  core (rank one for an individual state). Its first tangent is the projected
  Sylvester equation; the nonlinear local residual preserves nested response.
  Isolated energies and invariant cluster observables have small-matrix
  second/third/fourth-order tests, including exact internal degeneracy. This
  does not extend the derivative contract of RPA, non-Hermitian EOM, or every
  downstream electronic-structure method.
  Forward values/vectors and first energy derivatives do not invoke a response
  solve. A frame derivative or second energy derivative activates the implicit
  root and its configured linear-solver capacity/convergence checks.
- Subspace requests expose `projection` and `eigenvalue_sum`; their `values` and
  `vectors` are None. Stopped `raw_*` fields include guard diagnostics. Individual
  state requests check all internal and boundary gaps; subspace requests check
  only the external boundary. Unresolved prerequisites produce NaN derivatives.
  See [the derivation and boundary policy](DEGENERACY.md),
  [migration plan](UNIFIED_EIGEN.md), and
  [example](../../../examples/solvers/degenerate_subspace.py).
- Legacy `EigenSolverConfig` AD fields (also used by RPA) remain honored when
  the explicit `response` argument is omitted. Mixing nondefault legacy controls
  with an explicit response raises an error. No independent projector or
  isolated-vector AD entry point/forwarding module remains.
- Failed eigenpairs remain inspectable; their first derivatives are NaN. Failed
  linear or adjoint solves return NaNs. Residual predicates stay inside opaque
  linear solves so that the tangent map can still be transposed correctly.
- Legacy RPA energy response uses the indefinite metric. The real structured
  Davidson and complex dense paths retain eigenvalue-only response with stopped
  X/Y. RPA roots must be isolated, real-frequency and of positive metric norm.
- `solve_stable_rpa(A,B,config=EigenSolverConfig(method="dense",...))` adds a
  bounded real symmetric reference path with isolated X/Y response. It requires
  positive-definite A-B and A+B and uses the Cholesky-Hermitian reduction, not
  symmetrization of the doubled RPA matrix. Both the reduced and reconstructed
  physical residuals must pass. `RPAResult` reports values, column X/Y with
  `X.T@X-Y.T@Y=I`, residuals, convergence, stability and response validity, plus
  `min eig(A-B), min eig(A+B)`. Invalid stability returns NaN physical outputs.
  Internal/boundary degeneracy invalidates derivatives of the whole requested
  root set; a smaller isolated prefix can be requested. The default config uses
  vector response; explicitly supplied configs honor their gradient_mode.
  `eigenvalue_only` stops both X/Y, including their reconstruction dependence.
- `solve_rpa(A,B,config=EigenSolverConfig(method="davidson",...))` accepts real
  symmetric matrices or `LinearOperator` actions and diagonal estimates. It
  uses a paired Davidson basis, a projected H/J pencil, and a metric-constrained
  checked GMRES adjoint for isolated X/Y response. No full physical matrix is
  formed. Stability margins are lowest Ritz estimates with explicit residuals;
  `stability_certified=False` distinguishes them from dense full-spectrum checks.
  `response_valid` checks primal/gap eligibility; adjoint convergence is checked
  when AD is requested and failure produces NaN. Dense dispatch reuses
  `solve_stable_rpa` and enforces its dimension cap. See the
  [metric derivation and limitations](../bse/MATRIX_FREE_BSE.md).
- Always check `response_valid` and convergence before nested differentiation.
  Differentiating a structurally constant invalid tangent can yield zero at
  higher order; a finite higher derivative alone is not a validity certificate.
  This preserves the existing invalid first-response policy without inventing
  derivatives for cut clusters or nonconverged eigenstates.
- `regularized_eigh` preserves the existing SCF gap-broadening policy near
  degeneracy. `inverse_sqrt` differentiates the matrix function through a
  Sylvester equation at repeated positive eigenvalues. Existing SCF second,
  third and mixed derivative contracts are retained.
- `attach_root` and `implicit_fixed_point_solution` attach the residual-based
  derivative to a supplied converged solution. The callback minimizer retains
  its actual unrolled trajectory; domain-level policy chooses between implicit
  and unrolled modes. Regularization changes the response problem and remains
  explicit rather than a silent numerical fallback.
- `stationary_value(function, solution, parameters)` evaluates a variational
  scalar objective with its envelope derivative. The caller supplies a
  stationary coordinate chart: if `dE/dz=0`, then `dE/dp=partial E/partial p`.
  The state is still differentiable inside that partial derivative, so higher
  response includes `E_pz dz/dp`. Constraints depending on p must be in the
  callback; raw constrained AO orbitals/densities are not stationary coordinates.
  Checked implicit orbital SCF applies this rule only to the final total energy,
  keeping overlap transport/Pulay terms in the callback. It avoids a spurious
  first-energy adjoint driven by numerical stationarity noise near a zero mode.
  Optimizer callbacks, density/orbital response, explicit trajectories and
  unchecked implicit behavior are unchanged. Higher or nonvariational state
  observables still require successful, physically defined response solves.

TD-SCF signatures, thresholds, positive-root selection and tuple layouts are
preserved. Unified Hermitian guesses also explore disconnected invariant sectors
that purely diagonal legacy guesses could miss. CI and TDA-BSE keep unfiltered
lowest-root selection; TDA explicitly requests the spectral lower bound. Invalid symmetric/RPA
eigenvalue derivatives now follow the shared NaN policy. The generic public
Hermitian API does not filter negative eigenvalues: stability analysis and CI
must retain those roots.

## Validation and further work

Tests cover nonsymmetric linear systems and transpose derivatives; zero/tiny
RHS; JIT/vmap; isolated-root energies and vector response; unconverged states;
real/complex structured RPA energy derivatives; SCF higher-order response;
and preservation of CI, TD-SCF, DIIS and physical stability calculations.
Run with CPU and float64:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=1 \
python -m pytest -q tests/solvers tests/ci tests/test_tddft_eigensolvers.py \
  tests/test_scf_higher_order.py tests/test_scf_implicit_and_xc_energy.py
```

Biorthogonal non-Hermitian vector response and variable-metric subspace
response remain future work. Hermitian higher response assumes a fixed real
Euclidean metric, fixed selected rank and a resolved external spectral gap.
The present RPA implementation is not a generic non-Hermitian EOM solver.
GPU behavior has not been validated in this migration.

## Symmetric tensor-sum inverse

`solve_tensor_sum(factors, rhs)` solves the sum of symmetric matrix actions on
the corresponding tensor axes. Factor k must have shape
`(rhs.shape[k], rhs.shape[k])`. It returns the solution, true residual norm,
convergence and minimum absolute Cartesian eigenvalue sum. No full Kronecker
matrix is built. The small-factor spectral bases are numerical factorizations;
`custom_linear_solve` differentiates the live operator and RHS, including when
individual factors have repeated eigenvalues.

The full tensor-sum spectrum must be resolved above `denominator_tol`. This
generic solver does not infer physical symmetry sectors or apply a pseudoinverse.
Invalid factors or failed primal/tangent/adjoint residual checks produce invalid
solutions. Storage is proportional to the full RHS tensor and factor matrices;
callers remain responsible for their tensor allocation budget. CC's opt-in
semicanonical triples path supplies such a budget. See
[`cc/SEMICANONICAL.md`](../cc/SEMICANONICAL.md).

## Generic non-Hermitian dense and Davidson solvers

`solve_nonhermitian(A, config=NonHermitianSolverConfig(...))` accepts a real
square matrix or `LinearOperator` without symmetrization. `method="dense"`
(default) has `max_dense=256`; `method="davidson"` bounds the search space with
`max_space=40`, `maxiter=100`, `guard_roots=1`, `seed=0`. Supply an operator's
approximate diagonal for preconditioning, or omit it for residual expansion.
Default diagonal guesses receive independent small full-support perturbations;
`initial_vectors` preserves explicit guesses and adds seeded exploration.
If preconditioned corrections are all dependent, raw residuals expand the basis.
The transposed action uses `operator.T`, which can be derived by JAX.

Both methods return left/right vectors with `L.T @ R = I`, true residuals,
conditioning, spectral gaps and first-order energy response. Iteration/restart
counts, guard residuals, and `spectrum_complete` expose what was computed.
Davidson rejects convergence if guard roots fail, uses O(n*m+m²) storage and
never silently materializes a physical dense matrix or falls back to dense.
Its projected roots/gaps are **estimates**, not a global spectral ordering or
isolation certificate when `spectrum_complete=False`. `response_valid` is then
conditional on the selected branch being isolated outside the sampled space.
This matches the explicit partial-spectrum boundary used by iterative RPA.

Isolated real eigenvalue JVP/VJP uses `d omega = L.T (dA) R`. Numerical vectors
and raw spectra have stopped AD. Complex selected roots have NaN real outputs;
negative real roots remain selectable. Failed requested/guard residuals,
observed unresolved gaps or excessive conditioning invalidate derivatives for
the complete requested set. Vector/cluster and higher-order response are not
provided. The caller owns CC amplitude response and sector interpretation;
see [EOM-CCSD](../cc/eom/README.md).

Real clusters unresolved at matrix-scaled roundoff are represented by real
left/right SVD null-space bases. Their full cluster overlap is solved before
truncating requested roots, so a conjugate numerical representation cannot lose
rank merely by taking real parts. This repair occurs in the projected matrix for
Davidson and the bounded physical matrix for dense solving. It preserves the
original eigenvalues and all physical residual/conditioning checks; it is not
Hermitianization, a pseudoinverse, or a new degenerate-root AD policy.


## Independent scalar roots

`solvers.nonlinear.solve_scalar_roots(function, x0, config=ScalarRootConfig(...))`
provides bounded secant, AD-Newton and hybrid forward iterations. Each residual
component must depend only on its corresponding trial component; for coupled
systems use the general nonlinear interfaces. The result includes per-lane
convergence/iterations and batch evaluation counters. `scan=True` permits
explicit trajectory AD; an implicit physical root rule can wrap the default
while-loop solver, as GW does. Rejected nonfinite trials do not enter the
explicit derivative graph. See [QP usage and validation](../gw/QP_SOLVERS.md).
