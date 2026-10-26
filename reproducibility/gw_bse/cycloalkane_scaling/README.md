# Cycloalkane GW + BSE scaling

The [2026-10-02 serial scaling study](scaling_20261002.md) is archived
with [machine-readable results](scaling_20261002.json) and a
[timing figure](scaling_20261002.png). It uses the fixed geometries in
`geometries.json`, STO-3G/Weigend, CPU float64, synchronized GradSCF repeats,
and independent fresh-directory MolGW runs. This is a pre-review timing
snapshot: subsequent fixes retain the original finite-eta real-axis response
and restore checked cached-Cholesky solves. Its times are not measurements of
the final merged implementation. Both codes solve the full dense
singlet BSE spectrum; GW and BSE timings are reported separately.

The [resolvent reuse microbenchmark](resolvent_reuse.md) separately compares
old and new implementations of the *same* resolvent equation on C3H6. Its
speed ratios are controlled implementation comparisons, not a fitted
large-system complexity law.

## Historical artifacts and corrections

`benchmark.json`, `scaling.png`, and `scaling.pdf` are historical artifacts,
not the current validated comparison. Their original protocol mixed first
and later JAX calls without actual warm repeats. It also accepted MolGW
process completion without checking BSE convergence. The 2026-10-02 audit
found that the ten-root MolGW Davidson calculation on C3H6 exhausts its
subspace with a residual near 0.21; its timing must not be presented as a
converged ten-root solution.

The earlier interrupted C6H12 runs do not establish an out-of-memory cause.
Their quadrature settings differed between attempts, so they must not be
combined into a single controlled timing. The older qsGW+BSE omission is a
limitation of that benchmark adapter, not evidence that MolGW cannot perform
qsGW+BSE. No qsGW timings are included in the current three-method study.

The current implementation reuses spectral factors for forward resolvent
solves and differentiates the original matrix equation implicitly. The old
tradeoff between fast spectral evaluation and degeneracy-safe derivatives
is not required; see the resolvent report for the tests and mathematical scope.
