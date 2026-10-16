# Optional AD forward algorithms for QP roots

```python
from gradscf import gw

calculation = gw.GW(mf, qp_solver='newton').run()
calculation = gw.GW(mf, qp_solver='hybrid').run()
```

The default stays `qp_solver='secant'`. This option also applies to `UGW` and
restricted/unrestricted molecular CD array drivers. It selects the nonlinear
**inner QP solve**; it does not select the evGW/evGW0 outer iteration, and
alternative settings are rejected for outer-facade or linearized/evaluate-only
calls. The existing physical models and QP implicit backward are unchanged.

For a QP residual R(w), AD supplies R'(w)=1-d Re Sigma/dw. Newton proposes
`w_new = w - R/R'`; hybrid refreshes that slope every four batch iterations
(including the first), for unusable secant differences, or after two poor
residual reductions. Invalid slopes use the secant proposal. Proposed steps
are bounded by 0.05 Ha in the GW adapter. Newton residual-increasing steps and
nonfinite Newton/hybrid trials undergo up to four halvings, then a secant
fallback. A still-invalid fallback retains the last valid iterate. Both the
step and physical residual must satisfy tolerance; unsuccessful lanes return
their best residual iterate with a false convergence flag.

The numerical implementation is shared:
`solvers.nonlinear.solve_scalar_roots` and `ScalarRootConfig`. It assumes
independent real scalar equations, not a coupled vector residual. JVP with a
unit tangent supplies the diagonal slopes without a dense Jacobian. GW owns
the physical residual and its existing implicit response checks. The former
private GW secant loops are removed; the periodic eager caller now imports
the shared secant path with the same numerical settings.

Newton caches the slope at its current iterate. Final QP weights and implicit
backward still independently evaluate the derivative at the **returned root**;
an earlier slope is not substituted. The public low-level QP `diff_mode` also
accepts `explicit` for the static scan (`unrolled` remains a legacy alias).
For explicit trajectories, rejected-trial residuals are stopped and the
accepted residual/slope is recomputed. This prevents discarded domain errors
from introducing NaN gradients. It adds work in scan mode and is not included
in the while-loop forward timing below.

These are safeguarded local methods, not globally bracketed root finders.
No guarantee of pole-free steps or selection of the main satellite branch is
made. AD uses the same locally constant contour-pole masks as before. Failed
roots and singular local slopes remain invalid for implicit differentiation.

## CPU measurement, 2026-09-30

The benchmark prepares an actual native RHF/G0W0 context, then times only the
QP root solve with fixed G/W. Every system/method runs in a fresh process.
CPU float64, Python 3.12.2, JAX 0.8.1, macOS arm64, nw=100, eta=.001 Ha,
step and residual tolerances 1e-8 Ha, maximum 100 iterations. All orbitals are
included for H2/STO-3G, water/STO-3G and LiH/STO-3G; water/6-31G uses two
occupied and two virtual frontier levels. Geometries and every control are
stored in the [measurement JSON](../../../tests/gw/data/qp_solver_cpu.json).
Repeated times are the median of seven synchronized calls after compilation.

| System / basis | Method | Maximum iterations | Repeated QP solve (ms) | Compile + first solve (s) | Process peak RSS (MB) |
|---|---|---:|---:|---:|---:|
| h2_sto3g | secant | 3 | 0.060 | 0.167 | 505.1 |
| h2_sto3g | newton | 3 | 0.095 | 0.290 | 527.1 |
| h2_sto3g | hybrid | 3 | 0.090 | 0.276 | 522.6 |
| water_sto3g | secant | 8 | 5.978 | 0.218 | 546.7 |
| water_sto3g | newton | 7 | 10.613 | 0.347 | 581.7 |
| water_sto3g | hybrid | 7 | 7.855 | 0.328 | 569.3 |
| lih_sto3g | secant | 4 | 1.511 | 0.178 | 526.7 |
| lih_sto3g | newton | 4 | 2.791 | 0.306 | 553.7 |
| lih_sto3g | hybrid | 4 | 2.014 | 0.295 | 542.9 |
| water_631g | secant | 4 | 34.976 | 0.221 | 715.1 |
| water_631g | newton | 4 | 68.030 | 0.383 | 784.9 |
| water_631g | hybrid | 4 | 47.560 | 0.360 | 736.1 |

All 19 selected roots converge in every method. The largest difference from
secant is 5.11e-14 Ha. Newton reduces some water/STO-3G orbital iteration
counts, but is 1.59–1.95 times slower across these four cases; hybrid is
1.31–1.49 times slower. The measured derivative cost outweighs the saved
iterations here. These results support keeping secant as the default.

Peak RSS includes SCF/context preparation, JAX and compiler memory; it is not
isolated QP working memory. Compiled temporary byte counts, residuals, per-lane
iterations and batch callback/derivative counters are also recorded. Counters
measure whole-batch evaluations, not active-orbital FLOPs. This is neither a
full-GW speed comparison nor a GPU or large-molecule performance claim.

```sh
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
python tests/comparisons/compare_qp_solvers.py --output /tmp/qp-solvers.json
```

## Verification

The affected-area regression gave 61 passed in 97.34 s, including shared
scalar solvers, QP status/AD, molecular factor response, evGW, UGW, the seven
MolGW fixtures, and complex periodic compatibility. There were 13 existing
complex-to-real AD warnings. The full repository suite and GPU were not run.
Review caught a discarded nonfinite-trial gradient leak and an unprotected
hybrid secant domain step; both were fixed and tested. The shared sqrt test
now converges to 0.01 with the correct explicit derivative 0.2 for both
methods, agreeing with finite differences. No convergence thresholds were
relaxed to obtain passing results.

The native [QP example](../../../examples/gw/qp_solvers.py) compares methods
through the public facade. Its default QP tolerance differs from the stricter
benchmark tolerance above.

After adding the canonical `explicit` spelling and UGW/linearized-mode boundary
checks, `tests/gw/test_qp_methods.py` gave **7 passed in 18.38 s**, with four
complex-to-real AD warnings. This overlaps the 61-test run and is not additive.
The native water example was also run: all three paths printed identical QP
energies to eight decimal Ha places. Maximum residuals were 2.39e-10 (secant),
2.82e-14 (Newton), and 2.11e-10 Ha (hybrid) at the facade's default tolerance.

```sh
PYTHONPATH=src:tests/gw:tests/bse JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
python -m pytest -q --import-mode=importlib \
  tests/solvers/test_scalar_roots.py tests/gw/test_gw_qp.py \
  tests/gw/test_qp_methods.py tests/gw/test_gw_differentiability.py \
  tests/gw/test_gw_evgw.py tests/gw/test_gw_cd_ugw.py \
  tests/bse/test_molgw_fixture.py \
  tests/gw/pbc/test_kpoint_invariants.py::test_gamma_matches_single_k_with_complex_orbitals \
  tests/gw/pbc/test_kpoint_invariants.py::test_kpoint_driver_is_invariant_to_independent_band_phases
```


The timing table above was recorded at `ae24d0e`, before the subsequent
molecular static-W contour subtraction. It remains a historical measurement,
not a timing claim for later self-energy evaluation graphs. Re-run the same
benchmark to measure the current revision; the default root method is unchanged.

## Implicit JVP, branch continuation and root precision

The current implicit rule is a custom JVP; reverse mode is its transpose.
For external physical parameters theta, it holds the converged frequency
fixed in the partial residual differential and evaluates

```
f(omega, theta) = omega - epsilon_mf(theta) - delta_v(theta)
                 - Re Sigma(omega, theta),
domega = -partial_theta(f) dtheta / partial_omega(f).
```

The numerical seed has zero implicit tangent. The returned root remains live
when this rule is differentiated again, retaining mixed and higher response.
No root-iteration trajectory is differentiated in implicit mode. Failed roots
and nonfinite or singular frequency slopes retain the original rejection
policy, including under JIT. The shared scalar forward solver is unchanged.

Restricted `g0w0_cd_restricted` additionally accepts `qp_initial`, a real
full-MO spectrum, and `qp_tol`, the positive absolute root step/residual
tolerance in Hartree. The defaults preserve the mean-field-offset seed and
`1e-6` tolerance. `qp_initial` supplies only the requested entries to the
nonlinear solve; it does not alter the mean-field base or G/W spectra and is
rejected for linearized/evaluate-only calls. These controls currently belong
to the restricted array driver, not to the `GW` facade or the evGW outer loop.

For small geometry displacements, a previously identified root can initialize
both neighbors. This is conditional continuation, not automatic selection of
the largest-Z peak. Multiple roots can each have a small Dyson residual while
independent seeds select different branches. A derivative across that discrete
selection change is not the local implicit derivative of either root.

The residue `Z=1/(1-Sigma_omega)` is more sensitive to root precision than the
energy itself. On a smooth branch,

```
dZ/dtheta = Z^2 * (Sigma_omega,theta + Sigma_omega,omega * domega/dtheta).
```

Both partials include the supplied physical input dependence. In evGW the
full outer response is a coupled spectrum Jacobian; a local fixed-G/W Z is
not a replacement for that inverse Jacobian.

The [2026-10-16 response record](../../../reproducibility/ad_showcase/GW_BSE_response_gates/README.md)
contains the failed independent-seed water comparisons, continued branches,
source boundaries and focused tests. The complete small H2 geometry chain
and root controls passed 27 tests in 246.69 s after adding explicit precision.
This is not a whole-suite or large-system validation. Water residue checks
use `qp_tol=1e-11` without relaxing the observable-error gate.

## Optional shared DIIS for the outer spectrum

The restricted/unrestricted `evgw_cd_*` array drivers accept `diis_space=0`
and `initial_guess=None`. Their defaults retain the original numerical
iteration and MF starting spectrum. Positive `diis_space` uses the shared
`solve_nonlinear` implementation; it does not introduce a GW-local DIIS or
change the physical residual

```
F_p(E, theta) = E_p - epsilon_mf,p(theta) - delta_v,p(theta)
               - Re Sigma_p(E_p; G[E,theta], W[E,theta]).
```

For evGW0, W is W0 throughout this iteration but still depends on theta.
Both spin channels are packed into the same residual in the unrestricted
driver. The existing iteration limit, damping and true residual tolerance
apply. The change in the sum of QP levels is an additional scalar convergence
monitor; it is not a total-energy functional. Shared implicit response gives
`dE = -(partial_E F)^-1 partial_theta F dtheta`, with checked linear solves.
The iteration trajectory is not differentiated.

`initial_guess` supplies a full real spectrum (two spin rows for unrestricted
inputs). Only requested entries initialize the iteration; all other entries
stay at their original MF values. It does not replace the MF base or W0 input.
Its implicit tangent is zero. Local continuation is useful because DIIS may
converge to distinct roots at nearby geometries; convergence alone does not
establish branch identity or globally largest spectral weight.

These controls belong to the array drivers and the molecular response helper;
the eager `GW` facade has not acquired these options. The 2026-10-16 remote
outer-response group passed 35 tests in 413.23 s, including R/U DIIS response,
selected/frozen orbitals, numerical-seed tangents and previous outer-method
regressions. UGW tests cover its existing energies and screening outputs;
the unrestricted driver does not currently expose `qp_weight`.
