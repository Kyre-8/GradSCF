# FCI implementation validation

Executed 2026-09-26 in the isolated `feat/fci` worktree, based on release
`13610c4`. CPU, macOS 26.6.2 arm64, Python 3.12.2, JAX 0.8.1, float64,
PySCF 2.9.0. No GPU or full-repository verification is claimed.

## Numerical and integration regressions

```sh
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  /opt/anaconda3/bin/python -m pytest -q tests/fci tests/ci --tb=short
```

Final result: **114 passed**, no skips, 127.27 seconds. Two PySCF warnings
report that its local build does not provide OpenMP; they are not failed
numerical checks. The separate public-boundary suite
`python -m pytest -q tests/test_reference_boundaries.py` also passed all 11 tests
in 1.27 seconds. Its removed-adapter check now resolves relative imports, so
legitimate `fci.reference`, `bse.reference`, and `scf.reference` modules are not
misclassified as the removed top-level `gradscf.reference` adapter.
The CI suite is included because its existing fermion-phase
helper now lives in FCI's string module and is shared rather than copied.

Coverage includes:

- Independent PySCF Hamiltonian actions/diagonals for vacuum, full occupancy,
  alpha-only, beta-only, odd-electron and equal-spin-population spaces.
- Normalized spin-resolved and summed 1/2-RDMs; unnormalized bilinear transition
  matrices, including all four spin-resolved 2-RDM transition blocks; trace and
  contraction identities, energy reconstruction and S²/multiplicity.
- H2 (0.74 Angstrom), LiH (1.6 Angstrom), and asymmetric H4
  (z=0,1.0,2.1,3.3 Angstrom), STO-3G/RHF, dense and Davidson roots against
  PySCF. Native H2 and odd-electron Li/ROHF facade integration also pass.
- Energy and density first-order JVP/VJP versus finite differences; energy
  derivatives versus 1/2-RDM contractions; internal orbital-rotation invariance;
  scalar core-energy response and an empty active space.
- Exact internal degeneracy: invalid numbered-root derivatives versus valid
  complete-subspace energy sums/projector probes, using the common solver.
- Coefficient-dependent objectives reject incomplete energy-only response;
  nonconvergence and nonphysical integral symmetries invalidate results.
- Frozen-core folding against an independent projected full-space FCI matrix,
  including noncontiguous/reordered orbitals. AO folding is checked separately
  with full, packed s4 and DF data; only active orbital columns are transformed.
- Space/workspace limits before transformations, mutable-input freshness,
  failed-rerun invalidation, and dense/Davidson warm-start protocol.
- Forward/reverse JAXPR inspection for a 225-determinant Davidson problem:
  no determinant-square physical matrix is generated.

Action/density comparisons typically use 2–3e-12 absolute tolerances. Molecular
energies use 1e-9 Hartree (native odd-electron facade 1e-8). Directional-response
checks use step 1e-4 and approximately 3e-7 derivative tolerance. These are
numerical verification tolerances, not a complete-basis or model error claim.

## Product-space scale and response measurements

```sh
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  /opt/anaconda3/bin/python tests/comparisons/compare_fci_pyscf.py
```

Synthetic physical real integrals, seed 20260926; symmetric random h1 plus a
separated diagonal spectrum, and ERIs from 12 symmetric factors. Alpha/beta
populations are equal. Shared Davidson: one requested root, boundary guard,
20-column subspace, 120-cycle limit, residual tolerance 1e-9 Hartree. The
parameter scales h1 at fixed ERIs; central finite difference step 1e-4.
Every case runs in a fresh process. Numbers below are observations, not general
performance guarantees for strongly correlated molecules or larger active spaces.

| Active electrons/orbitals | Determinants | Energy difference vs PySCF (Ha) | AD–FD difference | First compile+gradient (s) | Steady value+gradient (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8 / 8 | 4,900 | 1.78e-15 | 2.41e-11 | 1.289 | 0.0368 |
| 10 / 10 | 63,504 | 3.55e-15 | 4.12e-11 | 1.812 | 0.4809 |

The 10/10 case has 252 alpha strings and 252 beta strings. Peak process RSS
is 1,139,703,808 bytes, including compiler/runtime/reference overhead; compiled
XLA temporary storage is 721,280,912 bytes. Multiple arrays and solver storage
coexist, so the default 8,000,000-element contraction budget is not a peak-RSS
bound. A hypothetical dense 63,504-square Hamiltonian alone would require about
32.3 decimal GB in float64; no such matrix is built here.

Settings, energies, derivatives, backend versions and memory figures are stored
in `tests/fci/data/scaling_cpu.json`. PySCF supplies the independent reference;
GradSCF does not invoke its FCI runtime.

## Native runnable examples

With the same CPU environment:

- `examples/fci/ground_state.py`: H2/STO-3G total energies
  `[-1.13728383, -0.53077336]` Ha; ground-state S²=0.
- `examples/fci/active_space.py`: LiH/STO-3G, one doubly occupied core and
  four active orbitals, active electron sector (1,1), embedded energy
  `-7.863681624885594` Ha. Orbitals remain fixed.
- `examples/fci/response.py`: two-site Hubbard model at U=4 Ha and t=1,
  energy `-0.8284271247461893` Ha and dE/dt=`-1.4142135623730945`, matching
  the analytic derivative. This is a hopping-parameter derivative, not a force.
- `examples/fci/bond_stretch.py`: H2 at 0.74, 1.5, 3.0 Angstrom gives
  energies `-1.1372838344885006`, `-0.9981493534714099`, `-0.9336318445584979`
  Ha. At 3.0 Angstrom natural occupations are `[1.07488823,0.92511177]`.

Independent code review identified and resolved early capacity checking,
core-integral memory, spin-resolved transition layout, and dense `ci0` handling.
Remaining boundaries are documented in README.md: common real orbitals,
fixed-Ms sectors, explicit capacities, and first-order derivative contracts.
