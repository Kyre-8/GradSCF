# Differentiating isolated spectral subspaces

`solve_hermitian(..., response=EigenResponseConfig(target="subspace"))` supports first-order JVPs and VJPs of the projector
onto the lowest `nroots` states of a real symmetric operator, including exact
internal degeneracy. It returns its action on supplied probes and the sum of
the selected eigenvalues. It does not expose individual eigenvector derivatives.

## Mathematical scope and reference

Reference: M. F. Kasim, *Derivatives of partial eigendecomposition of a real
symmetric matrix for degenerate cases*, arXiv:2011.04366v1 (2020),
[paper](https://arxiv.org/abs/2011.04366), Sections 4–5.
The paper identifies compatibility conditions for finite eigenvector response
and projects out the full degenerate eigenspace. Its formulas do not make an
arbitrary labeled degenerate eigenvector a differentiable function of every
matrix perturbation. No source code was copied from an external implementation.

GradSCF's initial API instead targets an isolated spectral subspace. If X is
an orthonormal basis, P=XX^T is unchanged under X -> XU for orthogonal U.
Internal eigenvalue crossings and splitting perturbations can leave P smooth
when the selected spectrum remains separated from the excluded spectrum.

Differentiating AX=XH with H=X^TAX, and choosing the local frame condition
X^T dX=0, gives the complement-space Sylvester equation

```text
P = X X^T, Q = I - P
Q (A Z - Z H) = -Q (dA) X,  X^T Z = 0
dP = Z X^T + X Z^T
d tr(X^T A X) = tr(X^T (dA) X)
```

This derivation uses the whole selected subspace, even if it contains distinct
or nearly equal eigenvalues. It does not divide by internal eigenvalue gaps.
For a concrete matrix, parameters and perturbations must preserve symmetry;
for example, construct A=(B+B^T)/2 from unconstrained trainable parameters B.
The public operator interface continues to require symmetry from matrix-free
callers. The generic RPA and non-Hermitian EOM problems are outside this scope,
as is a variable metric M in AX=MX Lambda.

## Public interface

```python
import jax
import jax.numpy as jnp
from gradscf.solvers import (EigenSolverConfig, EigenResponseConfig,
                             LinearSolverConfig, solve_hermitian)

jax.config.update('jax_enable_x64', True)
a = jnp.diag(jnp.array([1., 1., 3., 5.]))
probe = jnp.array([.3, .4, .5, .2])
config = EigenSolverConfig(nroots=2)
response = EigenResponseConfig(target="subspace",
    linear_config=LinearSolverConfig(rtol=1e-11))
out = solve_hermitian(a, probes=probe, config=config, response=response)
assert out.response_valid
print(out.projection, out.eigenvalue_sum)
```

The first argument also accepts `LinearOperator`. Probes have shape `(n,)` or
`(n,m)` and may themselves depend on the differentiated parameters. The returned
`projection` has the same shape. Passing `jnp.eye(n)` explicitly obtains a
dense projector for small tests; a few probes avoid storing that matrix.
For example, an unweighted sum of transition strengths over a complete cluster
has the form d^T P d and can be evaluated from `projection`.

`EigenSolverConfig` supplies numerical forward controls. `EigenResponseConfig`
selects `eigenvalues`, `eigenpairs` or `subspace` and supplies gap tolerances and
`linear_config` for the shared checked Sylvester solve. `method='dense'` keeps
the same response rule with a bounded dense forward oracle. Omitting probes
in subspace mode computes the energy sum without an unnecessary response solve.

The old independent projector entry and isolated-vector AD wrappers are removed.
For callers shared with RPA, legacy EigenSolverConfig AD controls remain mapped
when `response` is omitted. Explicit response plus nondefault legacy AD controls
raises an error rather than silently choosing one. New Hermitian callers should
use the explicit response configuration.
The result contains:

- `projection`: P times the supplied probes, with first-order response.
- `eigenvalue_sum`: the trace over the selected subspace, with first-order response.
- `residual_norms`: stopped diagnostics for the requested roots.
- `raw_values`, `raw_vectors`, `raw_residual_norms`: stopped diagnostics including
  an available guard; `raw_present` identifies absent interval slots.
- `values` and `vectors`: None in subspace mode, preventing accidental use of
  individual degenerate eigenpair derivatives.
- `boundary_gap`: stopped lambda[k]-lambda[k-1], or infinity for the full space.
- `converged`: convergence and orthogonality of the required Ritz roots.
- `response_valid`: those primal checks plus a resolved boundary separation.
- `status`: 0 for a usable subspace, 1 for an invalid primal eigenspace,
  2 for an unresolved boundary.

`response_valid` cannot certify a future loss-dependent adjoint solve. Failed
forward or transposed linear solves invalidate derivatives via the shared
NaN policy. Diagnostic fields are not differentiation targets.

## Boundary selection and failure policy

For k < n, the forward solver obtains k+1 roots. The extra root is a guard for
the selected upper boundary, not an additional differentiated state. Its
convergence is required. With default `gap_atol=gap_rtol=1e-8`, response requires

```text
lambda[k] - lambda[k-1]
    > gap_atol + gap_rtol * max(abs(lambda[k]), abs(lambda[k-1]))
      + residual_norm[k] + residual_norm[k-1]
```

The new interface starts Davidson from lowest-diagonal guesses mixed with
seed-0 Gaussian full-support columns, then orthonormalizes the block. Its width
is at most `2*(k+1)` and is bounded by the configured basis capacity. Pure
diagonal guesses can otherwise converge with zero residual inside an incorrect
invariant sector, missing lower roots in another block. Callers can override
the starting block with `initial_vectors`; guesses are stopped numerical
controls, not physical differentiation inputs. All Hermitian response targets
now share these guesses, guard roots and forward diagnostics.

The residual margin accounts for unresolved Ritz values. A gap among computed
roots is conditional on the forward solver finding the intended lowest roots;
it is not a proof that an iterative eigensolver cannot miss an invariant sector.
As usual for partial eigensolves, physically appropriate starting spaces and
comparison tests matter.

If k cuts a degenerate cluster, the numerical primal is retained for inspection,
`response_valid` is false, and first derivatives are nonfinite. Increase `nroots`
until the complete cluster is included. A fixed count preserves JIT shapes;
there is no silent root-count expansion. For Davidson, an explicit
`max_subspace` must be at least `min(n,k+3)`: the guard root plus expansion room.
Internal small gaps are neither averaged nor broadened.
The tolerances classify boundary resolution; they do not regularize the
Sylvester equation. A small accepted external gap can still be ill-conditioned.

## Numerical ownership and derivative order

The common core lives in `solvers/eigen/response.py`; forward numerical Ritz
data are constructed once in `solvers/eigen/primal.py`. A rank-one frame gives
the isolated-vector differential, while a rank-k horizontal frame is only used
for basis-invariant observables. Both use the same implementation. Response uses
`solvers.linear.checked_linear_solve` on flattened n-by-k blocks, including a
checked transpose solve. No method-local eigensolver or GMRES copy is added.
The augmented operator is

```text
K(Z) = Q [A (QZ) - (QZ) H] + PZ.
```

The PZ term enforces the frame constraint in the extension outside Q-space;
it does not shift the physical eigenvalues or the complement-space response.
Projector actions require O(nk) frame storage rather than an explicit n-by-n
projector. Davidson basis and GMRES workspaces additionally depend on their
configured subspace/restart sizes. Set `max_subspace` for bounded Davidson
storage; its legacy default can reach the full matrix dimension.

At an invalid boundary, the numerical Ritz frame/projector remains inspectable
without a response solve. If its response is requested, an identity extension
keeps the auxiliary chart well defined while the physical validity guard
rejects differentiation. This is not a regularized physical response.

Shared Davidson restart retention and accepted correction counts are bounded
by actual buffer capacity. When preconditioning produces a direction parallel
to the existing basis, the forward algorithm uses the projected original
residual instead. The physical eigenproblem and derivative equations are unchanged.
The generic real API scales its orthogonalization cutoff with the requested
residual tolerance and diagonal magnitude to avoid premature rejection of small
corrections. Removed low-level AD wrappers are not retained as forwarding aliases.

### Nonlinear local chart for higher response

The numerical Ritz frame X is held fixed as a local coordinate anchor. Define
`P = X X.T`, `Q = I-P`, and write a nearby invariant frame as

```text
Y = X + QZ,          X.T Y = I.
```

Invariance requires `AY = YH`. Multiplying on the left by X.T gives
`H = X.T AY`, so a square residual for Z is

```text
R(Z,A) = Q [AY - Y (X.T AY)] + PZ = 0.
```

At the converged numerical reference, Z=0 supplies the root. The first
differential is the Sylvester equation above. Differentiating the full
nonlinear residual retains its state-dependent coefficients and quadratic
terms in subsequent response equations. Freezing X and H in a single linear
correction would omit these terms. The shared `attach_root` mechanism supplies
the nested derivatives; no forward eigensolver trajectory is differentiated.

Normalize the chart with its small Gram matrix: if `Y.T Y = L L.T`, use
`U = Y L^(-T)`. A rank-one column has a locally fixed sign. At rank k the frame
is private; only `U U.T` actions and the selected energy trace are physical AD
outputs. Neither construction divides by internal cluster gaps. The external
gap must remain resolved and the selected rank fixed.

The stationary energy differential is

```text
d lambda = u.T (dA) u,
d sum(lambda_cluster) = trace(U.T (dA) U).
```

Operator parameters are made explicit in this custom JVP. The frame is a
lazy differentiable coefficient: forward evaluation returns the existing Ritz
frame; differentiating it activates the local implicit chart. Consequently,
forward eigenvalues/vectors and a first energy derivative do not need a
linear response solve or its `max_dense` capacity. Second energy derivatives
and first vector/projector derivatives do. All active response solves retain
the shared true-residual checks. Isolated roots are batched with `vmap` rather
than placing `custom_root` in `lax.map`: the latter lost third/fourth-order
terms in the tested JAX 0.10 composition, covered by closed-form regressions.

Tests cover isolated and exactly degenerate invariant observables through
fourth order, mixed-parameter Hessians, normalization, and existing JVP/VJP
contracts. They do not establish higher response for RPA, EOM, a varying AO
metric, or an entire electronic-structure chain. Individual state observables
must request eigenpair response. State targets reject unresolved internal gaps
for the whole requested set; use the subspace energy sum for an invariant
aggregate instead.

Validity flags remain authoritative at every derivative order. At an invalid
boundary, first response is rejected. A further derivative of a structurally
constant invalid tangent can become zero under AD; that zero does not restore
validity. Check the primal/response flags and the intervening derivative
results before interpreting a higher-order observable.

See [the executable example](../../../examples/solvers/degenerate_subspace.py) and
[validation](VALIDATION.md) for the test coverage and numerical evidence.

## Explicit spectral intervals

The generic solve still returns the lowest roots without positive-energy
filtering. TDA requests `EigenSolverConfig(value_min=excitation_threshold)`
to preserve its existing positive-root convention. In this opt-in mode the
common numerical path selects roots above the cutoff and checks both the
upper selected boundary and the closest Ritz value to the cutoff on either
side. `lower_boundary_gap` and `lower_boundary_residual` expose that diagnostic.
An absent upper guard is accepted only when the numerical basis covers the
full spectrum; absent requested roots are never converged. Interval topology
stays fixed within a differentiated call; a root at the cutoff is not a
differentiable selection. With no lower bound, the lower gap is infinity.
