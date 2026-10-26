# Explicit restricted SCF multistart

`RKS.multistart()` and `scf.run_restricted_multistart()` run a baseline and a
finite set of native orbital-rotation density starts. They select the lowest
**finite, converged candidate** and return all attempt summaries. Optional
`require_stable=True` also requires a positive internal stability check. This is an
opt-in host-side branch-selection workflow. `RKS.run()` is unchanged.

```python
from gradscf import gto, scf, cc

mf = scf.RHF(
    gto.M(atom="N 0 0 0; N 0 0 1.1", basis="sto-3g", unit="Angstrom"), conv_tol=1e-12, conv_tol_density=1e-10,
    conv_tol_grad=1e-9, max_cycle=150,
)
selection = mf.multistart(amplitudes=(0.05, 0.15, 0.4), seed=20260923)
for attempt in selection.attempts:
    print(attempt.amplitude, attempt.energy, attempt.converged, attempt.cycles)
if not selection.converged:
    raise RuntimeError("No finite converged SCF candidate")
mycc = cc.CCSD(selection.selected).run()
```

The input facade is cloned and is not run or mutated. Its existing settings are
preserved for every attempt, including tolerances, damping, integral backend,
basis and molecular geometry. Even an already solved source gets a fresh
baseline calculation, so modified controls cannot silently reuse a stale result.
Only `init_guess` changes for the rotated starts.

The baseline has amplitude 0 and uses the source's configured initial guess.
Additional amplitudes must be finite and positive; `amplitudes=()` requests the
baseline only. `orbital_rotation_guesses` supplies deterministic orthogonal
rotations of the complete baseline MO matrix. Initial densities are
`D = C_rot diag(mo_occ) C_rot.T`, preserving the metric and 0/2 occupations.
No reference-software density or orbitals are used.

`RestrictedMultistartResult` contains:

- `selected`: the selected independent RKS facade, or None if no eligible attempt was found;
- `selected_index`: the index in `attempts`, including the baseline at index 0;
- `attempts`: immutable `RestrictedSCFAttempt` summaries with amplitude, energy
  (Hartree), convergence, iteration count, seed and optional stability status;
- `seed` (the first seed, preserving existing single-seed usage), the complete
  `seeds` tuple and the derived boolean `converged`.

Nonconverged/invalid-energy candidates are retained in the summaries but cannot
win selection. Configuration/backend exceptions are propagated rather than
silently treated as ordinary convergence failures. Restart orbitals must be
finite, real and complete, with integer restricted 0/2 occupations.

By default this does not perform a stability analysis. Neither mode establishes
a global minimum.
Selection can switch branches discretely and is not a JIT/AD interface. Use the
existing SCF differentiation APIs on the selected branch separately. The native
N2/STO-3G example demonstrates escape from one observed high-energy branch; it
is not a guarantee for other geometries or strongly correlated systems.
Unrestricted/generalized/fractional-occupation multistart is outside this helper.
HF was validated here; other supported RKS functionals retain their existing
backend/dependency requirements.

See [the native example](../../../examples/cc/nitrogen_multistart.py) and
[the associated EOM validation](../cc/eom/ROBUSTNESS_VALIDATION.md).

The [extended EOM study](../cc/eom/STRESS_VALIDATION.md) supplies a concrete
counterexample to treating one finite search as exhaustive: stretched F2 at
2.2 Angstrom requires broader seeds/rotation amplitudes than the default set to
reach the known lower HF branch. Keep all attempts and interpret `selected` as
lowest among those tried, not a global-ground-state assertion.

## Multiple seeds and optional stability filtering

```python
selection = mf.multistart(amplitudes=(0.5, 1.0), seeds=(0, 1), require_stable=True)
```

The same baseline is calculated **once**, and each seed rotates that baseline
using the existing guess generator. Thus two seeds and two amplitudes create
five attempts, not two independent three-attempt workflows. Explicit seeds must
be unique and nonnegative; do not combine `seeds` with a nondefault `seed`.
Baseline attempt.seed is None. Changing seeds does not alter convergence targets.

With `require_stable=True`, each finite converged candidate calls the shared
[internal stability analysis](STABILITY.md). Candidates with False or unresolved
(None) stability are recorded but cannot be selected. `selection.converged`
means an eligible candidate was found; individual attempts still retain their
original SCF convergence flags. Default Davidson stability is a partial-spectrum
estimate, not a global certificate, and restricted analysis does not test
spin/complex symmetry breaking. Unsupported backend errors propagate explicitly.

Single-seed calls, empty additional-start lists, input immutability and ordinary
`run()` behavior are preserved. The implementation has one baseline, one candidate
summary/selection path and one nested seed/amplitude loop.

For directed descent rather than random exploration, `mf.stabilize()` reuses the
shared [negative-mode following engine](STABILITY.md). `require_stable=True`
continues to filter only restricted-internal stability by default; explicitly
request `mf.stability(channel="spin")` or `mf.stabilize(channel="spin")` when
spin symmetry breaking should be examined or allowed.
