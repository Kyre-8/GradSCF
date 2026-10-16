# A03 extension — complete-cluster VJP at higher dimension

On 2026-10-16, `jy_gpu_1` CPU float64 passed the controlled n=256/1024/4096
projector-VJP checks. A separate ordinary-`eigh` comparison directly shows the
backward distinction: for the **same smooth complete-projector loss**, ordinary
JAX `eigh` AD returned four NaNs at both n=32/256 exact-degeneracy points, while
the shared implicit VJP stayed finite and agreed with the analytic derivative.
The forward loss remained accurate to 2.22e-16/4.72e-16. This is an observed
numerical limitation of ordinary `eigh` AD in JAX 0.10.0, not a mathematical
non-differentiability of this invariant loss or a claim that dense solvers are
intrinsically non-differentiable. Dense forward with the shared implicit rule
is a successful correctness control.

The five small-model regressions passed in 19.64 s. Both forward methods passed
four internal-width points at all three dimensions; n=256/1024 additionally
passed the external-gap scan and explicitly rejected its unresolved 1e-10
boundary. Each completed report includes JVP/VJP, rotated-initial-frame and
cut-cluster checks. Independent multistep NumPy projector FD is bounded to
n<=256; larger cases use the independent known-spectrum analytic derivative.
All numerical execution and plot generation occurred on `jy_gpu_1`.

| Dimension | Dense + implicit exact-cluster VJP error | Davidson + implicit error |
|---:|---:|---:|
| 256 | 5.00e-14 | 1.31e-13 |
| 1024 | 3.87e-14 | 3.64e-14 |
| 4096 | 1.50e-14 | 1.55e-14 |

The table fixes external gap 0.7 and internal width zero. At external gap 0.001,
the largest n=256/1024 analytic discrepancy was 4.40e-11. Shrinking only the
internal gap did not introduce the analogous conditioning loss. These are
dimensionless model derivatives, not molecular force units.

At n=256 the nonzero exact-cluster VJP is
(-0.02830344, -0.02475005, 0.00454958, -0.00703223). Its smoke run's best of three
directional FD steps had maximum observable derivative error 6.50e-12.
See the [response figure](degenerate_vjp_response.png) and
[process measurements](degenerate_vjp_performance.png); PDF versions are saved
alongside them. The figure scripts only plot complete, passing records and
explicitly distinguish valid responses from expected boundary rejection.

The numerical snapshot is
`/home/yjiao/gradscf_checks/a03-degenerate-vjp-20261016-agent`.
The first source-Python SHA256 was
`d0e419a50cc96ebbcaebe8f491879343011f3b019c65582344741d587590d01e`;
each JSON also records the executed script hash, solver settings, process ID,
start time, environment and individual gate diagnostics.

The executable entry points are
[the structured model](../../../examples/solvers/degenerate_vjp.py),
[the ordinary-AD comparison](../../../examples/solvers/compare_degenerate_eigh_ad.py),
[native benzene](../../../examples/solvers/degenerate_benzene_vjp.py), and
[independent PySCF benzene](../../../examples/solvers/compare_degenerate_benzene_vjp.py).
Actual executed scripts are copied under `artifacts/ad_showcase/provenance/A03_degenerate_vjp/executed`; raw logs and RSS budgets
are under `verification/`. The narrow regression is
[`test_degenerate_vjp_model.py`](../../../tests/solvers/test_degenerate_vjp_model.py).

The public examples now accept ordinary Python parameters, for example:

```python
from examples.solvers.degenerate_vjp import run

run(dimension=256, method="davidson", sweep="all",
    output="artifacts/ad_showcase/A03_replay/n256_davidson.json")
```

Run each solver/dimension in its own bounded remote process. The original
executed CLI versions are preserved under `artifacts/ad_showcase/provenance/A03_degenerate_vjp/executed/cli_before_function_api`,
and the frozen numerical snapshots remain unchanged. The manifest at
`artifacts/ad_showcase/development/public_example_api/migration_manifest.json` records
unchanged numerical-helper and computation-body ASTs for this interface
conversion. A new interface replay does not replace the original benchmark
source and resource records. Current jobs use the shared CPU 16–31 pool with
OMP/OpenBLAS/MKL/NUMEXPR threads set to 2; the measurements below predate that
policy and must not be interpreted as measurements under the new allocation.

## Ordinary `eigh` AD on the same invariant loss

[`plain_eigh_comparison.json`](plain_eigh_comparison.json) records all four
gradient components, numerical internal gaps, forward errors and shared-rule
controls. All eight nonzero-splitting ordinary-AD cases remained finite and
passed the declared componentwise tolerance; the errors grew as follows:

| Internal spectral width | Ordinary max VJP error, n=32 | Ordinary max VJP error, n=256 |
|---:|---:|---:|
| 1e-1 | 2.10e-16 | 3.33e-16 |
| 1e-3 | 2.34e-14 | 1.23e-14 |
| 1e-6 | 2.60e-11 | 2.62e-11 |
| 1e-9 | 1.84e-8 | 1.34e-8 |
| 0 | four NaNs | four NaNs |

The shared implicit path passed all ten cases with maximum error 7.92e-14.
At both exact points the measured smallest internal eigenvalue gap was zero.
The [backward accuracy figure](plain_eigh_backward_accuracy.png) shows finite
errors and the exact-point NaNs in separate panels/status strips; a PDF is
available alongside it.
This comparison has no timing ratio: it measures derivative accuracy and
finite/nonfinite status for the same mathematical observable. Cutting the
cluster or assigning a sorted crossing-root derivative is a separate genuinely
undefined-boundary demonstration, already tested by the shared example.

The demonstration uses a reproducible real symmetric matrix-free
operator with a complete rank-four cluster. Its internal splitting and external
separation are independent controls. A thin orthogonal transformation mixes the
cluster with the entire bath, while four independent symmetric perturbations
couple every bath coordinate to the cluster. The main observable is a weighted
three-probe transition tensor, `L = sum(W * (B.T @ P @ B))`; the cluster energy
sum is auxiliary because its first derivative needs no adjoint solve.

The shared `solve_hermitian` subspace rule supplies both JVP and VJP. Dense and
Davidson forward methods use the same GMRES response settings. A separately
derived known-spectrum projector differential provides the analytic reference;
small dense NumPy projector differences provide an independent numerical check.
No labeled eigenvector derivative is assigned at exact degeneracy. External-gap
closure and requests that cut the cluster remain invalid cases.

The model and production observables must not materialize a full orthogonal
matrix, projector, or parameter Jacobian. Bounded dense-forward and NumPy
references necessarily materialize the small matrix and are identified as such.
Each solver/dimension should run in a separate process when comparing peak RSS.
Record first-call and warm retained-pullback times separately and synchronize
all JAX arrays. Forward speed is not the main claim.

The existing public spectral result reports Ritz and guard residuals, not the
actual loss-dependent adjoint residual. The shared checked linear solve validates
that residual internally, but its auxiliary diagnostic is not currently exposed
through the spectral result. Unavailable adjoint residuals, iterations and runtime
operator counts must remain explicitly unavailable; `response_valid` is only a
primal/boundary check. Diagnostic instrumentation must be separated from timing.

## Molecular gate

The regular-hexagon benzene RHF/CIS forward gate has completed with GradSCF native
SCF/integrals: Cartesian 6-31G*, 102 AOs, 21 occupied and 81 virtual orbitals,
1701 singles. The lowest bright in-plane doublet is zero-based roots 2:4, with
excitation energy 0.3126652954213 Eh, numerical splitting 1.55e-15 Eh and minimum
external gap 0.03062931118 Eh. Its transition tensor has
Txx≈Tyy≈5.339818359 e² bohr². Native RHF converged in 19 cycles to
-230.701828845233 Eh; the occupied-virtual Fock maximum is 6.04e-14 Eh.
The native fixed-reference VJP and the full independent PySCF three-step
projector-FD comparison have now also passed. For
`L=Txx-Tyy+0.5Txy`, the native two-parameter VJP is
`[-1.944683023291, -0.486170757440] e² bohr²/Eh`.
JVP–VJP duality error is 3.58e-13. The independent maximum VJP discrepancies
at source steps 2.4982e-4, 7.4945e-5 and 2.4982e-5 Eh are
7.48e-7, 6.75e-8 and 7.91e-9 e² bohr²/Eh, respectively. Both the tensor JVP
and the weighted-loss VJP pass the independent gates.

Independent SCF/cluster-energy/tensor differences are 2.81e-12 Eh,
9.62e-14 Eh and 2.60e-10 e² bohr². The main native workflow took 137.25 s,
including setup, compilation, VJP and JVP; whole-process peak RSS was 12.37 GiB.
The independent reference/FD workflow took 11.79 s. These measure different
workflows and are not a speed ratio. See [native data](benzene_gradscf.json),
[independent data](benzene_pyscf.json), and the
[molecular response figure](benzene_subspace_vjp.png), with a PDF alongside.

The native source is a separate snapshot,
`/home/yjiao/gradscf_checks/a03-benzene-vjp-20261016-agent`.
All 84 native source files and the shared library match the accepted X14/G08
85-entry [manifest](verification/benzene/native_source_sha256.txt), with all
85 checks preserved in [the verification log](../../../artifacts/ad_showcase/provenance/A03_degenerate_vjp/verification/benzene/native_source_check.log).
The actual library SHA256 is
`8176ca97f9758ba3810b727c5a2e90572ac4f8f205e4a16e0d7a7177221b1829`.
The main script imports no PySCF. The separate comparison rebuilds its own SCF,
AO integrals and operator using the recorded raw basis definition; it does not
share orbitals or AO integral arrays with the main calculation.

The fixed-reference source is `V_a=(S W_a+W_a S)/2`, where W contains per-atom
cos(2φ)/sin(2φ) weights on all its AOs. This inversion-even Mulliken-style AO
one-body model breaks the in-plane symmetry. It is not a quadrupole or local
real-space field. Parameters have units Eh; tensor derivatives have units
e² bohr²/Eh. The main loss is Txx−Tyy+0.5Txy. Fixed-reference response and fully
relaxed geometry response remain separate gates. No SCF stability or complete
geometry-derivative result is claimed.

## Controlled process measurements

Both methods use the same shared implicit GMRES response, tolerance 1e-10,
restart 40 and maximum 100 restart cycles. Forward Davidson has an explicit
64-vector capacity. The table uses the first case (internal width 0.1,
external gap 0.7) of each separate process, before FD/gauge validation:

| n | Method | Warm retained pullback, median of 3 / s | RSS high-water after first pullback / MiB |
|---:|---|---:|---:|
| 256 | dense | 0.002749 | 983.0 |
| 256 | Davidson | 0.002768 | 1126.1 |
| 1024 | dense | 0.011143 | 1019.6 |
| 1024 | Davidson | 0.009689 | 1135.5 |
| 4096 | dense | 0.070534 | 1845.6 |
| 4096 | Davidson | 0.051114 | 1151.7 |

RSS includes the forward/preparation/import high-water mark; it is not exclusive
backward live memory. The complete-process maxima are larger because later
validation and compilation are included. Timing is a three-repeat observation
on a shared CPU node, not an exclusive-core benchmark or a general claim that
Davidson is faster. These dimensions belong to the declared structured
diagonal-plus-low-rank model family; no molecular scaling exponent is inferred.

## Execution and resource boundary

The first launch incorrectly bounded virtual address space with `RLIMIT_AS`.
JAX initialized 394 threads under the unchanged default CPU affinity; it had
31.0 GiB virtual size but only 605 MiB resident memory, then failed to create a
CPU-codegen thread. This was a launch-policy error before any solver calculation,
not an observed eigensolver failure. `artifacts/ad_showcase/provenance/A03_degenerate_vjp/verification/address_space_probe.log` and
`artifacts/ad_showcase/provenance/A03_degenerate_vjp/verification/compiler_probe.log` retain the evidence. No affinity was changed.

The corrected launcher reuses
`artifacts/ad_showcase/development/p01_p02/k8_rss_budget.py` as `rss_budget.py`, polls the
scientific child's actual Linux `VmHWM` every 0.25 s, and terminates it if it
exceeds 32 GiB. A separate `timeout 1800` bounds each job. `artifacts/ad_showcase/provenance/A03_degenerate_vjp/run_budget.py` records
the exact child command, environment and resource verdict. The successful
five-test run used 1.72 GiB peak RSS. A passing resource record does not replace
the numerical gates in the corresponding JSON.
