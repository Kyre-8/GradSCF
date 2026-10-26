# Executed validation — 2026-10-05

Worktree `feat/bse-ep-coupling`, based on `82f2c44`. Python 3.12, JAX 0.8.1,
macOS arm64, CPU float64. No GPU or full-repository test result is claimed.

- Baseline BSE TDA and GW real-axis EP tests: **20 passed, 29.61 s**.
- Initial combined BSE/solver/GW regression: **127 passed, 10 warnings,
  290.83 s**. This covers static TDA/full/optical/API/unrestricted/matrix-free
  BSE, shared block/GMRES/direct/eigen solver paths, GW Fan/DW/periodic/real-axis
  kernels and scGW EP response. Later boundary fixes are covered by the final
  focused suite below, rather than relabeling this run as a final full suite.
- Final focused run: **47 passed, 5 warnings, 62.36 s**, after passivity,
  mixed-precision DW, zero-Bose and literal-zero-vertex fixes:

```sh
PYTHONPATH=src JAX_PLATFORMS=cpu python -m pytest -q \
  tests/bse/test_exciton_phonon.py tests/bse/test_optical_spectrum.py \
  tests/solvers/test_complex_linear.py tests/gw/test_ep_compact_quadratic.py \
  tests/gw/test_ep_coupling.py
```

Warnings concern JAX's projection of complex cotangents onto real parameters.
New AD checks independently compare analytic derivatives with finite differences
or `jax.numpy.linalg.solve`, including mixed second response and internal
excitation degeneracy. No failed tests or skipped dependencies in these runs.

Independent review covered projection indices, statistical weights, complex
conjugation, causal/antiresonant response, the adapter's AD guard, complex solve
realification and all quadratic consumers. Identified boundaries were fixed
or explicitly scoped: non-passive low-population satellites are diagnosed,
compact/full quadratic layouts agree, and mixed optical precision contracts
DW using the promoted occupations. A scalar near-resonance precision reproducer
at eta=1e-10 Ha gives exactly the independent reference
`alpha_xx = 0.9844997040767137 + 1e10 i`.

The native H2/3-21G example ran successfully, with all QP levels and three TDA
states included. Its fixed-model coupling-scale AD and recomputed finite
difference differ by 6.70e-11 Mb. Source, JSON/NPZ/CSV and figures are in
`examples/bse/h2_ep_coupling.py` and
`reproducibility/gw_bse/h2_bse_phonons`. The example is explicitly a strong
coupling stress case, not a quantitative vibrational-spectrum benchmark.

## Multiphonon water extension

`tests/bse/test_vibronic_hamiltonian.py` adds displaced-oscillator Poisson
line/weight oracles, a two-mode progression, zero-coupling recovery, quadratic
upper-cutoff and cross-mode matrix elements, complex frame covariance,
zero-coupling second AD, full/block optical equivalence and a shared-resolvent
gradient/finite-difference comparison. All **10 tests passed (7.56 s)**.

The combined vibronic/Fan/optical/shared-complex run passed **37 tests,
2 ComplexWarnings, 42.57 s**:

```sh
PYTHONPATH=src JAX_PLATFORMS=cpu python -m pytest -q \
  tests/bse/test_vibronic_hamiltonian.py tests/bse/test_exciton_phonon.py \
  tests/bse/test_optical_spectrum.py tests/solvers/test_complex_linear.py
```

The new matrix builder contains no numerical solver. Water forward lines use
the shared Hermitian solver; the coupling-scale derivative uses the shared
complex resolvent at fixed finite cutoff, avoiding individual eigenvector AD.
The example checks spectral convergence at eta=10 and 3 meV separately. Its
JSON records the actual electronic-sector cutoffs and numerical changes;
finite-space convergence does not certify basis/electronic-space convergence
or reproduce dissociative water absorption. See the artifact README under
`reproducibility/gw_bse/water_bse_phonons` for the executed result and scope.

## Pre-merge verification

On 2026-10-05, the following focused BSE/solver/GW suite passed **66 tests,
5 ComplexWarnings, 92.11 s**, with no skips:

```sh
PYTHONPATH=src JAX_PLATFORMS=cpu python -m pytest -q \
  tests/bse/test_vibronic_hamiltonian.py tests/bse/test_exciton_phonon.py \
  tests/bse/test_tda.py tests/bse/test_optical_spectrum.py \
  tests/solvers/test_complex_linear.py tests/gw/test_ep_compact_quadratic.py \
  tests/gw/test_ep_coupling.py
```

This final run includes the compact/full GW quadratic compatibility paths.
The pre-merge diff and committed artifacts are scoped to this feature; no
full-repository or GPU validation is claimed.
