# OFDFT validation

Measured on 2026-09-27 in `feat/ofdft` (base `1fc4b02`).

## Environment

- Apple M4 Pro (arm64), JAX CPU backend, float64.
- Python 3.12.2, JAX 0.8.1, NumPy 2.3.4.
- Native molecular integral library reused from the existing repository build.
- DFTpy source pinned to `fbc47b4e1df3def1a2206e58c5a1a1a4d51db1d0` from the provided `dev` branch.

## Independent DFTpy comparison

`examples/ofdft/lda_exchange.py` provides a standalone pure-JAX Dirac LDA exchange energy. Correlation is explicitly absent. `examples/ofdft/compare_dftpy.py` supplies independent NumPy exchange energy/potential to DFTpy and uses DFTpy's own TF, vW, WT, Hartree and L-BFGS optimizer.

Both packages use a cubic 6 Bohr cell, N=8, identical odd FFT meshes, uniform neutralizing background and no ionic energy. The local potential is `0.15 cos(2 pi x/6) + 0.1 sin(2 pi y/6) + 0.05 cos(2 pi z/6)` in Hartree. This is a controlled smooth model, not a materials-accuracy benchmark.

Fixed-density tests use `n=(8/216)[1+0.2 cos(2 pi x/6)+0.15 sin(2 pi y/6)+0.1 cos(2 pi z/6)]`. JAX gradients are divided by cell-volume/grid-count to compare functional potentials. HVPs use a zero-integral smooth direction and DFTpy potential differences with step `1e-5`.

WT potential/HVP comparisons remove the constant (mean) component: DFTpy's analytic potential holds the reference average density fixed, whereas AD includes its derivative. Their fixed-N responses agree after removing the chemical-potential gauge. TF and vW are projected by the same convention.

Maximum absolute discrepancies across the three meshes:

| Term | Energy / Ha | Projected potential | Projected HVP |
|---|---:|---:|---:|
| tf | 2.220e-15 | 3.331e-16 | 1.016e-11 |
| vw | 4.163e-17 | 1.499e-15 | 2.912e-10 |
| wt_nonlocal | 4.163e-17 | 4.302e-16 | 2.401e-11 |

Separate self-consistent optimizations from the same uniform density:

| Mesh | KEDF | GradSCF energy / Ha | DFTpy energy / Ha | Absolute energy error / Ha | Relative density L2 error |
|---|---|---:|---:|---:|---:|
| 5³ | tfvw | 0.508678473721344 | 0.508678473721345 | 1.332e-15 | 1.205e-08 |
| 5³ | wt | 0.483656090559183 | 0.483656090559183 | 1.110e-16 | 4.142e-09 |
| 7³ | tfvw | 0.508678541233129 | 0.508678541233136 | 6.439e-15 | 2.503e-08 |
| 7³ | wt | 0.483655294189910 | 0.483655294189913 | 3.275e-15 | 1.375e-08 |
| 9³ | tfvw | 0.508678541225408 | 0.508678541225417 | 9.548e-15 | 2.987e-08 |
| 9³ | wt | 0.483655294150012 | 0.483655294150027 | 1.515e-14 | 4.478e-08 |

GradSCF stationarity tolerance: `1e-9`; DFTpy energy tolerance: `1e-13` over three checks. All GradSCF residuals were below `1e-9`. DFTpy final states were independently checked with GradSCF's constrained gradient: residuals ranged from `8.33e-8` to `1.85e-6`, below the comparison gate `2e-6`. Thus an energy-only DFTpy convergence flag is not being treated as equal stationarity precision. All electron-number errors were below `8e-15`.

Wall times including JAX compilation for the 5³, 7³ and 9³ comparisons were 14.27, 13.13 and 16.04 seconds in one process. These are reproduction timings, not performance rankings.

```bash
PYTHONPATH=src:/private/tmp/dftpy-ofdft-source/src JAX_PLATFORMS=cpu \
  python examples/ofdft/compare_dftpy.py

# For the mesh sweep, call run_comparison(mesh) for (5,5,5), (7,7,7), (9,9,9).
```

The local DFTpy import emitted a caught optional numexpr/NumPy ABI warning; DFTpy used its own existing config-parser fallback. Its functional and optimization source was not patched. The provided source version reports `0.0.0` without generated packaging metadata; the commit pin above identifies the reference.

## Other executed examples

- `molecular.py`: H2/6-31G*, grid level 1, TF+vW, XC omitted. E=0.05889014102429713 Ha, stationarity residual 5.92e-10; integrated AO electron number 2, numerical-grid number 2.000003263. Density probe derivative wrt vW weight -0.07886838837426632 vs central FD -0.07886838261167428 (step 1e-4).
- `periodic.py`: local-GTH H2 cell, 9³ grid, TF+vW and WT in both grid and periodic-Gaussian representations; all four forward calculations converged. The example contains the measured output table.
- `neural_kinetic.py`: eight actual Adam updates through implicit density response reduced a synthetic density loss from 1.2825003e-6 to 2.9235043e-8. This is a trainability demonstration, not a learned physical KEDF benchmark.

## Solver fixes exercised by this work

- Symmetric stationary starts require differentiable Newton refinements to retain explicit response. The uniform periodic response regression compares with implicit AD and finite differences.
- On JAX 0.8.1, incremental GMRES can underestimate its residual after an early Arnoldi exit. A diagonal 343-dimensional regression with eigenvalues 1...229 gives true relative residual 1.16e-8 despite requested 1e-10. The shared numerical solve now checks and retries the upstream batched path at the same tolerances; no OFDFT-specific Krylov algorithm was added. Both normal and 1e-10-scaled RHS and adjoint checks are covered.
- Gaussian shell AO indices must remain NumPy static metadata when building traced coordinates; a one-line constructor correction is covered by a molecular nuclear-gradient test.

## Limits

- No claim of original ATLAS/CASTEP numerical identity, finite-difference spatial stencils, OEPP generation, open-shell OFDFT, isolated-grid WT/WGC, arbitrary pseudopotential readers or distributed/GPU scaling. WGC99 second-order validation and finite-basis KS references are documented in `examples/ofdft/WGC_REPRODUCTION.md` and `WGC.md`.
- WT exact Kohn-anomaly kernel derivatives are rejected, and its fractional density powers use the documented 1e-18 floor. Smooth positive reference densities do not exercise the floor.
- The optional real jax-xc LDA/GGA runtime was unavailable on this host. The three integration cases are skipped; the hand-written Dirac comparison is separate and does not claim to validate that external package.
- The full repository suite and GPU execution were not run.

## Regression suite

After the WGC and KS-reference additions, the focused suite completed
with **97 passed, 3 skipped** (CPU float64), including WGC energy/potential,
implicit response, odd-electron KS occupations, and the analytic
Fourier-plane interpolation regression. Elapsed time: 156.89 seconds. The skipped cases require the
unavailable jax-xc runtime. Command:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu python -m pytest -q \
  tests/ofdft tests/solvers/test_gmres_accuracy.py tests/solvers/test_shared.py \
  tests/solvers/test_block_linear.py tests/solvers/test_nonlinear.py \
  tests/test_scf_autodiff.py tests/integrals/test_basis_parameters.py \
  tests/pbc/test_foundation.py tests/test_jax_xc_only_boundary.py
```

The ATLAS-style OEPP/PZ-LDA solid benchmark is recorded separately in
`examples/ofdft/ATLAS_REPRODUCTION.md` and `atlas_results.json`.
