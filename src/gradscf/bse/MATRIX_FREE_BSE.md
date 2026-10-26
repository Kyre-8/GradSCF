# Matrix-free full BSE and metric response

The 2026-09-24 screening update removes full screened-factor caches and adds
optional matrix-free auxiliary screening. Current configuration and storage
boundaries are in [README.md](README.md#storage-and-solver-boundaries).

2026-09-23. Continuation from `210cfee`, for real closed-shell static BSE.
The BSE module constructs A/B actions; all eigensolving and linear response
remain owned by `gradscf.solvers`. Public use:

```python
response = bse.BSE(mygw, tda=False, solver="davidson", nroots=3,
                   max_space=40, max_cycle=100).run()
print(response.e)
print(response.oscillator_strength())
```

## Forward problem and projection

Let `H=[[A,B],[B,A]]`, `J=diag(I,-I)`. Solve
`H z=omega J z`, `z=[X;Y]`, `z.T J z=1`. H and J are applied, not materialized.
A bounded paired basis `Q=[[V,W],[W,V]]` spans both frequency branches.

Q is Euclidean orthonormal but generally not J-invariant once W is nonzero.
Projecting the nonsymmetric `R=J H` as `Q.T R Q` loses the relevant metric
structure. Instead form only the small pencil

```text
Hp = Q.T H Q,    Jp = Q.T J Q
Hp c = omega Jp c
Hp = L L.T
(L^-1 Jp L^-T) u = (1/omega) u
c = sqrt(omega) L^-T u
```

Stable projected H is positive definite; the inverse-frequency problem is real
symmetric. The reconstructed physical roots, metric signs and residuals are
checked independently. Failure never triggers an unbounded dense fallback.
Numerical projector/eigenvectors are stopped before attaching physical AD.

Initial bases contain low-diagonal unit vectors plus independent deterministic
random vectors. This preserves the exact diagonal limit while exposing modes
outside the canonical initial subspace. Corrections are reorthogonalized twice.
Partial correction blocks fill the available capacity before restart; writes
past capacity are rejected and only accepted columns count toward basis size.
For the R residual `[r_x,-r_y]`, the diagonal correction uses the same lower
sign; a relative sign error in the inherited preconditioner was corrected.

## Implicit metric response

At a converged isolated root, differentiation gives

```text
domega = z.T (dH) z
(H-omega J) dz = -(dH)z + (Jz) domega
(Jz).T dz = 0
```

The rank-one augmented operator
`K=H-omega J+(Jz)(Jz).T` enforces the last condition. Its numerical solution
uses the shared checked GMRES path, with diagonal preconditioning from
`diag(H)-omega diag(J)+(Jz)**2` and an explicitly matching transpose solve.
The primal correction is zero; the live-zero RHS carries the physical tangent.
This supplies first-order JVP/VJP without differentiating iterations.

Energy-only mode stops all X/Y response and invalidates optical-property
AD in the BSE layer. For vector-response mode, both the forward linear tangent
solve and its transposed solve must converge. Invalid derivative guards are
placed before and after invalid-state output masking so neither JVP nor VJP
silently becomes zero. No higher-order response is promised.

## Stability and spectral boundaries

Two independent Hermitian Davidson calculations estimate the lowest roots of
A-B and A+B. A positive estimate with a small residual is a numerical stability
screen, not a rigorous global spectral lower bound. Thus Davidson reports
`stability_certified=False` even when `stable=True`. `stability_margins` stores
the Ritz estimates and `stability_residual_norms` stores their residuals.
The bounded dense reference can certify positivity from the complete spectra.
Nonpositive or unconverged stability screening invalidates the physical result.

Requested roots plus one available excluded root must converge, have positive
metric norm, and satisfy the configured separation margin. An unresolved root
or gap invalidates the whole requested derivative set. `response_valid` records
these *primal prerequisites*, not the convergence of an adjoint that has not
yet been requested. Checked adjoint failures return NaN.

## Resource and validation contract

No full A, B, H or J matrix is allocated by the Davidson/adjoint path. Paired
bases cost O(ntrans*max_space); projected matrices are at most
(2*max_space)-square. GMRES uses bounded restart storage. The BSE kernel still
stores orbital factors and a dense auxiliary dielectric; matrix-free response
does not remove those upstream costs.

Validation covers the dense stable reference, saved actual QuAcK A/B kernels,
old TDDFT/TDHF calls, oscillator-strength finite differences, exact diagonal
limits, general SPD blocks with restart, failed convergence, unstable cases,
and exact degeneracies. Forward/reverse JAXPRs for a 300-dimensional operator
are recursively inspected for full physical square arrays. A separate runnable
300-dimensional diagonal-plus-rank-two example with nonzero B records CPU/dtype,
seed, compile-plus-first-gradient time and process peak RSS; this is a synthetic
solver smoke test, not a molecular or GPU scaling benchmark.

Measured results are in [VALIDATION.md](VALIDATION.md). Open-shell, complex,
periodic, dynamical kernels, full GW outer response and nuclear/basis derivatives
remain outside this increment. Large-scale convergence/performance requires
separate measurements; increasing iteration count alone does not guarantee
convergence with an overly small subspace.
