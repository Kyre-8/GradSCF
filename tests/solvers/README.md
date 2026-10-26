# Spectral response regression boundaries

`test_spectral_higher_response.py` checks isolated Hermitian eigenvalues through
fourth derivatives, invariant observables of complete degenerate clusters,
normalized rank-one eigenvectors, batched roots, and a mixed-parameter Hessian
using the shared GMRES response solver. Closed-form expressions or a separate
dense `jax.numpy.linalg.eigvalsh` calculation supply the numerical oracles.

The response-capacity regression uses a four-dimensional dense forward problem
with `LinearSolverConfig(method="direct", max_dense=2)`. Numerical eigenvalues
and their first stationary derivative do not require a response solve. The
nonzero second derivative does require that solve and must reject the configured
capacity. An independent probe on the pre-lazy implementation confirmed that
both the numerical forward and the first derivative instead raised
`ValueError: Direct solve exceeds max_dense; use GMRES`. This is probe evidence,
not a pytest red run of the subsequently added regression. The probed source
SHA-256 values were:

- `eigen/api.py`: `3034c5377fb057ab99b71b21ed14ee89e011bd31111fab892841b549571fc3d9`
- `eigen/response.py`: `59185a2eb0042223815a81c4c538f8ede0ecd4bb91a4e1ae4c21046b311889c6`

The deliberately non-Hermitian input checks `response_valid=False` and rejection
of the first derivative with NaN. Higher derivatives of an invalid response can
be structural zeros when AD eliminates inactive paths; a zero higher derivative
does not override the response-validity flag. These tests do not promise NaN at
every order for every invalid path, or establish higher-order response for
non-Hermitian RPA, complex matrices, or method-specific molecular workflows.

Numerical verification is performed on `jy_gpu_1`, CPU float64, two threads,
using `/home/yjiao/opt/miniconda3/envs/jax_scf/bin/python`. The focused post-fix
run on 2026-10-15 passed: **3 passed, 10 deselected in 16.63 s**. The command was
`python -m pytest -q tests/solvers/test_spectral_higher_response.py -k 'capacity or mixed_parameter or invalid'`.
The log is `spectral-review-green.log` in
`/home/yjiao/gradscf_checks/ad-showcase-spectral-20261015`. Source SHA-256:

- `eigen/api.py`: `bf2845bc4b5900b6516d0cf56ecdeea629fe189a0ba7c442491d0fa3ae04a936`
- `eigen/response.py`: `d9673a91c41b85b6c2a71ac5cb9ddd84087838642292b98c85ac2581f50f551f`
- `test_spectral_higher_response.py`: `12d4b0e37a13ba4c6515a4a40d92f54bffd7c05acb8c4437f53db10df2e0c6b9`

The subsequent four-file regression on the same source passed **62 tests in
356.02 s**: `test_spectral_higher_response.py`, `test_subspace.py`,
`test_unified_eigen.py`, and `test_shared.py`. Its remote log is
`spectral-lazy-regression.log` in the same snapshot. This includes the new
lazy-dispatch tests and the existing first-order/shared-solver contracts.

## Stationary scalar values

`test_stationary_value.py` exercises the envelope differential of a scalar
stationary objective, keeping implicit state response at higher orders. Four
closed-form tests cover first through fourth derivatives. A flat-direction
example verifies that a first energy derivative can exist while the state
derivative remains undefined; a nonconverged value rejects its first response.

The SCF integration applies this rule only to the final energy of a checked
implicit solution. Its unconstrained orbital chart retains the moving overlap
metric. It does not change the optimizer, state-response equations, explicit
differentiation, or the requested linear tolerance. Metric-only first through
third derivatives and a real OH pi-degenerate force are tested separately in
`tests/test_scf_stationary_value.py`.

On `jy_gpu_1`, snapshot `ad-showcase-stationary-20261015`, CPU float64:

| Tests | Result | Log |
| --- | --- | --- |
| Six scalar-value tests and four metric-only SCF tests | 10 passed, 59.19 s | `stationary-value-green.log` |
| OH/6-31G* pi-zero-mode force, PySCF and multistep FD | 1 passed, 56.32 s | `stationary-oh-regression.log` |
| Existing orbital gradients and native geometry tests | 12 passed, 286.75 s | `stationary-scf-regression.log` |
| DFT orbital gradients and full omegaB97X-D molecular response | 11 passed, 1047.16 s | `stationary-dft-wb-regression.log` |
| Moving-grid PBE HVP and mixed Raman derivative | 2 passed, 277.30 s | `stationary-pbe-higher.log` |

The initial six-test red run failed because the new public symbol did not yet
exist. The physical pre-fix evidence is a separate matched-source OH probe:
the orbital root norm was `1.15e-14`, but the nearly zero energy-adjoint RHS
`2.81e-14` and residual `3.00e-16` failed the unchanged `rtol=1e-10` check.
The new envelope rule produces a zero RHS and a finite force; it does not
relax that check or define a response for arbitrary pi-orbital orientations.

Validated source SHA-256 values:

- `nonlinear/root.py`: `8ae5e282594f7601a60f92a15a98c771c687c847c17a8de689e78c6c2bf84512`
- `scf/orbital_optimization.py`: `e782c7c2f5a3681b66ce591a03b7e43883edfb7c71510de85501c90e290e66d7`
- `test_stationary_value.py`: `e59877b967aa2e3b7cba4d8174242ac969fd4d23440b1dccc86e7861d08e9c58`
- `test_scf_stationary_value.py`, including OH: `6e04047b8e9ce34c9f2a5851cb967f755f287bb597ecf05e922180d69118a03f`

These focused results are not a complete repository regression or a merged
integration. In particular they do not certify arbitrary open-shell density,
correlated-energy or degenerate-state higher derivatives.

The subsequent combined-source check in `ad-showcase-integration-20261015`
passed **19 tests in 315.02 s**: `test_cis_response_density.py`,
`test_stationary_value.py`, `test_scf_stationary_value.py`, and
`test_molecular_unrestricted_response.py`. This checks the AO-potential input
extension together with the stationary-energy wrapper, including a relaxed
CIS density. The log is `integrated-stationary-potential.log`.

## Nested spectral JIT and nonlinear forward isolation

The subsequent Hermitian response integration adds an explicit matrix of
independent rank-one charts for isolated roots. One root attachment contains
the columns, while the existing checked linear solver still solves each
n-dimensional column block. This preserves each root's `max_dense` boundary.
The complete-cluster chart retains its original basis-invariant contract.
The derivative-validity guard is applied after closure conversion, so its
boolean input remains explicit and invalid VJPs still reject. Details and
the failed candidate evidence are in the
[nested-JIT verification record](../../reproducibility/ad_showcase/spectral_nested_jit/README.md).
The final four-file regression passed **67 tests in 396.31 s**. The complete
MP2 scalar and tensor mixed-Raman gate passed separately in **175.97 s**.

`test_nonlinear_higher_response.py` covers a physical root nested upstream of
the shared DIIS driver. Stopping only the returned numerical state was too
late: JAX still traced the DIIS trajectory during higher differentiation.
Closure conversion now exposes every numerical-forward input, and all such
inputs are stopped before running the loop. The original physical residual
is retained for implicit response. Four new and two existing nonlinear tests
passed in **15.41 s**. Full RCCSD/UCCSD geometry/HVP and scalar mixed-Raman
tests then passed **3 tests in 303.52 s**; the production tensor Raman path
passed another focused check in **226.33 s**.

Existing CC tests passed in fresh, separate CPU processes after integration:

| File | Result |
| --- | --- |
| `tests/cc/test_ground.py` | 35 passed, 395.50 s |
| `tests/cc/test_ucc.py` | 12 passed, 141.64 s |
| `tests/cc/test_qcisd.py` | 13 passed, 180.57 s |

The earlier combined process aborted during JAX compilation before producing
a complete verdict. Its log contains no explicit allocation or thread error;
the observed node `vm.max_map_count=65530` alone does not prove its cause.
The aborted run is not counted as a pass. No global JIT/cache policy or node
kernel setting was changed. The accepted per-file logs and resource records
are in `ad-showcase-integration-20261015` as `cc-*-regression.log` and
`cc-*-resources.log`.

Further application regressions on the integrated source passed separately:
CI/CIS(D) **32 tests in 156.74 s**, FCI response **7 tests in 112.65 s**, and
the final CIS density/one-body/transition/Stark property tests **15 tests in
81.40 s**. The last group uses the final shared `excited.py` helper, including
its physical wavefunction probes. These are application-specific response
checks, not a full repository test run.
