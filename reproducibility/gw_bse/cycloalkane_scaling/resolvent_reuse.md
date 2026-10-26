# Checked resolvent factorization reuse: final merge recheck

This serial C3H6/STO-3G/Weigend-RI benchmark compares a per-frequency dense
transition-space reference with the final shared shifted solver. Both use
exactly the same finite-eta CD response, including its asymmetric 2i*eta
real-axis convention. The imaginary-axis model is real; the retarded model
uses d=Delta-i*eta and z=omega+i*eta. Cached numerical eigenvectors are not
differentiated: JVP/VJP follow the original matrix equation. An inaccurate
complex eigenbasis falls back to a dense solve.

Inputs: 21 MOs, 255 auxiliary functions, 108 transitions; all G/W windows;
64 imaginary-axis points; eta=1e-5 Ha; outer tolerance 1e-7 Ha; maximum 40
cycles; damping 0.3. CPU float64 on Apple M4 Pro, Python 3.12.2 / JAX 0.8.1.
OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, and VECLIB_MAXIMUM_THREADS are 1;
JAX runtime pools retain defaults. No other chemistry job ran concurrently.

RHF and BSE are excluded. Results are synchronized. Each path/method is run
three times in one process; samples 2 and 3 define the warm median. First
calls are not independent fresh-process cold-start measurements. The JSON
records exact samples and source hashes.

| Method | Dense samples / s | Reused samples / s | Warm dense / s | Warm reused / s | Speed ratio |
|:---|:---|:---|---:|---:|---:|
| evgw0 | 16.200, 12.841, 12.576 | 1.884, 0.458, 0.445 | 12.709 | 0.452 | 28.14 |
| evgw | 13.331, 13.295, 13.320 | 0.698, 0.724, 0.719 | 13.307 | 0.721 | 18.45 |

| Method | Maximum QP difference / Ha | Maximum Dyson residual / Ha |
|:---|---:|---:|
| evgw0 | 1.7764e-15 | 4.0081e-08 |
| evgw | 1.7764e-15 | 9.6807e-08 |

The implementation reuses W0 and both factorizations within an evGW0 call,
solves only the required MO-pair residue columns, and uses a fixed JIT entry
point for shifted solves. Stopping the numerical factors does not stop the
physical matrix response. G0W0 retains its default auxiliary-space CD path;
evGW outer fixed-point differentiation remains outside the supported scope.

The merge review also routed cached static Cholesky solves through the shared
checked solver so failed primal/transpose residual checks invalidate the
solution and its derivatives. See [merge validation](../MERGE_VALIDATION.md)
for test commands, results, warnings, and the excluded stress test.

```sh
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python tests/comparisons/benchmark_resolvent_reuse.py
```

## Earlier timing snapshots

`resolvent_reuse_pre_review.json` preserves the earlier symmetric-broadening
prototype measurement. `scaling_20261002.json` and its plots likewise predate
the finite-eta and Cholesky-guard corrections. Those are historical snapshots,
not final-code scaling. This recheck establishes only the C3H6 implementation
comparison; it does not remeasure the larger cycloalkanes or BSE timings.
