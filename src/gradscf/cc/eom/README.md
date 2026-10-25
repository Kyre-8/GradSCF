# Closed-shell EOM-CCSD: EE, IP and EA

This first implementation supports real molecular restricted CCSD references,
**EE singlets and IP/EA doublets**. Numerical diagonalization belongs to
`gradscf.solvers.solve_nonhermitian`; this package builds the physical actions.
PySCF is used only for independent comparisons, not at runtime.

## Public API

```python
from gradscf import cc

mycc = cc.CCSD(mf).run()  # mf is a converged GradSCF restricted HF object
excited = cc.EOMEE(mycc, nroots=3).run()
ionized = mycc.EOMIP(nroots=3).run()
attached = mycc.EOMEA(nroots=3).run()
print(excited.e, ionized.e, attached.e)  # Hartree, each shape (nroots,)
```

Convenience methods `mycc.eomee_ccsd_singlet()`, `mycc.ipccsd()` and
`mycc.eaccsd()` return `(energy, vector)` for one root or `(energies, vectors)`
for multiple roots, with vectors as rows in the latter case. Result-object
`right_vectors` and `left_vectors` always store roots in **columns**.
`amplitudes(root)` unpacks the right vector. There is intentionally no generic
all-spin EE dispatcher. Recomputing the source or changing frozen spaces,
amplitudes, or settings invalidates the eager snapshot; use `check_source()`
before reusing one after mutation.

| Sector | Operators retained | Energy convention | Packed dimension |
| --- | --- | --- | --- |
| EE singlet | 1p1h + 2p2h | E_k(N) - E_0(N) | ov + ov(ov+1)/2 |
| IP doublet | 1h + 2h1p | E_k(N-1) - E_0(N) | o + o²v |
| EA doublet | 1p + 2p1h | E_k(N+1) - E_0(N) | v + ov² |

Here o/v count active spatial orbitals. The conventional electron affinity is
**minus the EA eigenvalue**. Negative roots are retained. Frozen occupied and
virtual orbitals use the existing ground-state active-space definition.

## Equations and coordinates

For T = T1 + T2, the similarity-transformed Hamiltonian is
Hbar = exp(-T) H exp(T). The ground amplitudes obey F(T;h,g)=0. EE uses the
Jacobian A = dF/dT of the existing physical CCSD residual, which is equivalent
to the projected connected commutator at a converged reference. IP/EA use the
appropriate particle-number projections of Hbar - E_CC. Charged-sector
contractions are adapted from PySCF 2.9.0; see [source attribution](../NOTICE.md).

The EE independent doubles coordinates inherit `cc.AmplitudeSpace`: the
symmetric (ia,jb) matrix has sqrt(2)-weighted off-diagonal entries. PySCF's
unweighted packing differs by a similarity transformation. Tests unpack and
repack every probe and action before comparing. IP/EA use flattened spatial
r1/r2 arrays, without imposing an extra doubles antisymmetry.

The eigenproblem is non-Hermitian:

```
A R_k = omega_k R_k
A.T L_k = omega_k L_k
L.T R = I
```

Left vectors are duals in these same packed coordinates. They are neither
ordinary right-vector transposes nor the ground-state Lambda amplitudes.
They alone do not define transition moments or oscillator strengths.
Right vectors have unit Euclidean norm; the left norm then estimates the
simple eigenvalue condition number. Both eigen-equation residuals and the
biorthogonality error are reported.

## First-order automatic differentiation

For an isolated real root with L_k.T R_k = 1,

```
d omega_k = L_k.T (dA) R_k
```

The numerical eigenvectors are stopped when attaching this first-order energy
rule. To include the change in the converged CC amplitudes, solve CC **inside**
the differentiated function:

```python
import jax
from gradscf import cc

cc_cfg = cc.CCConfig(conv_tol=1e-12, residual_tol=1e-11)
eom_cfg = cc.EOMConfig(sector="ip", nroots=1)

def energy(t):
    ht = h + t * perturbation
    ground = cc.run_cc(ht, eri, nocc=nocc, config=cc_cfg)
    result = cc.run_eom(ht, eri, ground, nocc=nocc,
                        config=eom_cfg, cc_config=cc_cfg)
    return result.energies[0]

value, derivative = jax.jit(jax.value_and_grad(energy))(0.0)
```

Here the derivative is with respect to the explicit scalar perturbation t in
the fixed MO frame. `run_cc` supplies implicit T response. Holding a previously
computed ground result constant gives only a fixed-T partial derivative.
This is not a nuclear-coordinate gradient or differentiation through HF.

By default this interface supports **first-order energies**. Left/right vectors
and the raw spectrum are forward diagnostics with stopped AD; do not
differentiate them or take a second derivative of the default energy rule.
The optional isolated-pair response below retains live left/right vectors.
Complex selected roots produce NaN real outputs and preserve the raw complex
spectrum. Unconverged ground states/eigenpairs, unresolved gaps and excessive
conditioning invalidate energy derivatives (NaN), rather than changing the
method with a pseudoinverse. The response flag applies conservatively to the
whole requested root set. Dense solving checks the full computed spectrum.
Davidson checks the projected spectrum and converged guard roots, subject to
the spectral-completeness limitation below. A separate invariant-subspace mode
provides the complete-cluster observables described next.

## Complete-cluster response

All three sectors accept the shared `EigenResponseConfig(target="subspace")`.
For example, `cc.EOMEE(mycc, nroots=2, response=response).run(probes=q)` stores
a conditional `EOMSubspaceResult` with `projection`, `eigenvalue_sum`,
`response_valid`, `boundary_gap` and stopped `raw_result`. The functional
`run_eom`/`run_eom_from_integrals` entries use
`config=cc.EOMConfig(..., response=response), probes=q` inside a differentiated
function. Probe rows use the packed EOM amplitude
coordinate space; the trace sums sector energies relative to the CC ground
state. Add k times the live ground energy if a sum of absolute energies is
required.

Internal crossings are allowed for a complete real semisimple cluster with
a valid external boundary. The shared solver uses coupled left/right block
roots and the live overlap inverse, not independent sorted-root derivatives.
Its checked direct dimension is n*k+k*k; GMRES is also supported. Individual
amplitudes and Dyson orbitals require an eigenpair result and explicitly reject
this invariant result. Differentiated state properties require isolated-pair
response; legacy degenerate forward properties keep their existing contract. The eigenpair defaults/legacy tuple are intact.
See the [mathematics and numerical guards](../../solvers/DEGENERACY.md#non-hermitian-invariant-subspaces).

## Isolated charged-state Dyson properties

```python
from gradscf.solvers import LinearSolverConfig

ionized = mycc.EOMIP(
    nroots=2, vector_response=True,
    response_config=LinearSolverConfig(method="direct", max_dense=256),
).run()
dyson = ionized.dyson()  # solves the existing ground-state Lambda equations
print(dyson.weights)    # signed, one-spin biorthogonal pole weights
```

The functional form is `cc.dyson_orbitals(ground, lambda_result, eom_result)`.
Its left/right amplitude columns use the active occupied-then-virtual MO frame.
The sector is read from the EOM result; an explicit mismatching sector is
rejected even if EE/IP/EA spaces happen to have the same dimension. Sector and
`vector_response` are static pytree metadata. Existing NamedTuple access and
`_replace` remain available.

For IP, the endpoint contractions are
`D_left[p,k]=<Phi|(1+Lambda) a_bar_p^dagger R_k|Phi>` and
`D_right[p,k]=<Phi|L_k a_bar_p|Phi>`; EA uses the corresponding creation and
annihilation endpoints. Weights are `sum_p D_left[p,k]*D_right[p,k]`, rather
than either individual squared norm. The two amplitudes transform reciprocally
under a change of EOM normalization, leaving residues and weights invariant.
A spin-summed closed-shell spectrum includes a factor of two. Negative residues
are retained, and the convergence flag does not assert spectral positivity.

`vector_response=True` attaches the local right equations
`[A r-omega*r, (r.T@r-1)/2]=0`, followed by the left equations
`(A.T-omega*I)l+r*(r.T@l-1)=0`. Both reuse the shared root and checked linear
solvers. Their capacity is `n+1` and `n` per root, respectively. The default
forward diagonalization and first-energy rule remain unchanged. Converged
simple real roots, sufficient spectral separation and acceptable conditioning
are required; no pseudoinverse defines a response at a defective/degenerate root.

Forward-only EOM results can produce forward Dyson properties, but attempting
their derivatives is deliberately invalid. Full physical derivatives require
T, Lambda, both EOM vectors, and any moving reference/integral inputs inside
the differentiated function. The [H2 example](../../../../examples/cc/h2_dyson_response.py)
and [independent FCI comparison](../../../../examples/cc/compare_h2_dyson_fci.py)
demonstrate the complete first bond response in the two-orbital STO-3G limit.
See its [measured results and spectrum](../../../../reproducibility/ad_showcase/X14_h2_eom_dyson/README.md).
Its exact-FCI limit does not extend to EA-CCSD in a larger H2 orbital space.

## Capacity and current limits

The default `solver="dense"` materializes a bounded reference matrix:
`max_dense=256`, `block_size=16`. Select the shared iterative implementation
explicitly for larger EOM spaces:

```python
ee = cc.EOMEE(mycc, nroots=2, solver="davidson", max_space=40,
              max_cycle=150, guard_roots=1, seed=0).run()
print(ee.e, ee.result.iterations, ee.result.restarts)
print(ee.result.spectrum_complete, ee.result.spectral_gaps)
```

Davidson stores a real orthonormal search basis, expands with both right and
left residuals and restarts with both Ritz spaces. A stalled diagonal
preconditioner falls back to raw residual expansion. EOM provides orbital-energy
difference preconditioners and the physical action; its transpose is supplied
by JAX. No physical EOM matrix is assembled, including during energy AD.
Numerical storage is O(n*m + m²), m <= `max_space`. There is no dense fallback.
For a truncated search space, `max_space` must be at least
`4*(nroots+guard_roots)+2` to retain complex directions and expand after restart.
The extra guard roots must also converge before `converged` becomes true.

**Finite Ritz sampling does not certify the full spectral order or isolation.**
`spectrum_complete` is true only when the search basis spans the physical space
(or for the dense reference). Otherwise `spectral_gaps` and root ordering are
estimates. `response_valid=True` means that requested/guard residuals, observed
gaps and conditioning passed; it is conditional on the selected branch really
being isolated in the full spectrum. A small residual cannot exclude an unseen
invariant sector. Default diagonal guesses carry small independent full-support perturbations
to avoid immediate termination inside an exact invariant sector; independent
seeded vectors provide additional exploration. Explicit `initial_vectors` in
the shared solver are preserved. These choices improve exploration but are
not a proof. Validate sensitive crossings/near-degeneracies with larger spaces,
more guard roots/independent starts, and a dense oracle when feasible.

`raw_eigenvalues` holds the full dense spectrum or the final projected spectrum;
inactive padded entries are NaN. `subspace_dimension`, `iterations`, `restarts`
and `guard_residual_norms` describe the actual numerical solve. Guard residuals
are the maximum of unit-right and unit-left residual norms.

`max_intermediate_elements=20_000_000` still limits the input ERI element count
before EOM intermediates; it does not estimate peak memory or bound the earlier
CC calculation. Removing the EOM matrix bottleneck does not make integrals or
CC intermediates low-memory algorithms.

UHF/ROHF, triplet/SF, higher-rank EOM methods, complex orbitals, complex-cluster
response, transition properties beyond the charged-state Dyson
quantities above, and GPU performance remain outside this implementation.
The molecular example validates a complete first bond derivative, not a general
higher-order EOM molecular response facility.

Run [the dense native spectrum example](../../../../examples/cc/eom_ccsd.py),
[the response example](../../../../examples/cc/eom_response.py) and
[the iterative 6-31G example](../../../../examples/cc/eom_iterative.py) on CPU.
See [initial validation](VALIDATION.md) and
[iterative validation](ITERATIVE_VALIDATION.md), and the
[eight-molecule study](MOLECULAR_VALIDATION.md). The observed CO rank loss was
subsequently repaired in the shared numerical Ritz extraction: a roundoff-level
real eigenvalue cluster receives a full real left/right null-space basis and
cluster dual before root truncation. Physical residuals still decide convergence;
defective clusters are not regularized into successful eigenpairs. This repairs
forward bases only and does not add derivatives of degenerate roots or vectors.

For N2's higher-energy native HF branch, the opt-in
[restricted multistart API](../../scf/MULTISTART.md) records all attempts and
selects the lowest finite converged candidate. Single-run SCF defaults remain
unchanged. See [the robustness followup](ROBUSTNESS_VALIDATION.md) and
[extended molecular stress study](STRESS_VALIDATION.md). The latter retains
initial reference-convergence failures and documents SCF precision sensitivity
and incomplete finite multistart coverage in stretched F2.

## Method references

- J. F. Stanton and R. J. Bartlett, *The equation of motion coupled-cluster
  method. A systematic biorthogonal approach to molecular excitation energies,
  transition probabilities, and excited state properties*, J. Chem. Phys.
  **98**, 7029–7039 (1993), [doi:10.1063/1.464746](https://doi.org/10.1063/1.464746).
- M. Nooijen and R. J. Bartlett, *Equation of motion coupled cluster method for
  electron attachment*, J. Chem. Phys. **102**, 3629–3647 (1995),
  [doi:10.1063/1.468592](https://doi.org/10.1063/1.468592).
- M. Nooijen and J. G. Snijders, *Second order many-body perturbation
  approximations to the coupled cluster Green's function*, J. Chem. Phys.
  **102**, 1681–1688 (1995), [doi:10.1063/1.468900](https://doi.org/10.1063/1.468900).
  The upstream IP implementation cites its Eqs. (8)–(9); the contractions here
  use the full converged CCSD amplitudes, not an MP2 truncation.

EE/EA bibliographic metadata checked against publisher-deposited Crossref
records; IP metadata checked against the authors' university record.
Scientific attribution is separate from the Apache source-code attribution.

## Layered precision reports

```python
report = eom.diagnostics(scf_gradient_tol=1e-11,
                         cc_residual_tol=1e-11, eom_residual_tol=1e-9)
print(report.scf.gradient_norm, report.scf.stationary)
print(report.cc.residual_norm, report.cc_ok)
print(report.eom.residual_norms, report.eom_ok, report.all_passed)
```

This host-side report references the existing CC and EOM result objects and
uses the fresh stored [SCF diagnostic](../../scf/STABILITY.md); it neither reruns
solvers nor duplicates CC equations. CC uses its independent packed-coordinate
infinity residual norm. EOM checks right, normalized-left and guard residuals,
together with its existing forward convergence flags. Separate derivative and
spectral-completeness flags remain available on `report.eom`.

Default targets are the current solver settings. A stricter report target only
checks the stored result: it does not change configuration or silently relax a
threshold. `all_passed` is None when an explicit MO Hamiltonian has no attached
SCF state; otherwise it combines the stationarity and residual checks. Passing
these checks is neither an energy-error bound nor a ground-state/stability or AD
certificate. See [the precision validation](PRECISION_VALIDATION.md).
