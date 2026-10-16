# Hermitian response across nested JIT boundaries

This repair targets the shared Hermitian response used by canonical-orbital
observables. A 4×4 isolated-eigenvector observable exposed a third-derivative
failure when its second derivative was wrapped in an inner JIT. The same
composition occurred in the MP2 mixed geometry/field response. Numerical
validation uses the separate `jy_gpu_1` snapshot
`/home/yjiao/gradscf_checks/ad-showcase-spectral-jit-20261015`; it does not modify
the frozen A01–A07 or concurrent integration snapshots.

## Diagnosed boundaries

The minimal observable is `sum_i w_i (probe.T @ v_i)**2` for a real symmetric
`A(r,F)=A0+r B+F C+0.2*r*F B`. Both one and four selected isolated roots are
tested, with and without the inner JIT. Before the repair the two inner-JIT
cases failed with a broadcast dimension error; the two others passed. A separate
diagnostic using unbatched columns agreed with dense JAX eigenvectors to about
`2e-17` in the mixed third derivative. This isolates the composition of an
outer root batch and `custom_root`, rather than a physical degeneracy.

A second diagnostic found that a dynamically computed boolean validity flag
captured inside the operator also caused a tracer-constant failure. The installed
JAX 0.10.0 `closure_convert` implementation only hoists potentially perturbed
constants; its `_maybe_perturbed` explicitly excludes `float0` tangent values,
including boolean flags. The production code continues using public JAX APIs;
no private JAX implementation is imported or patched.

## Resulting implementation

Module-level custom JVP rules receive numerical frames and validity explicitly.
The isolated-root path attaches a single `(n,k)` implicit root, with independent
rank-one charts for each column:

```
P_i z_i = x_i (x_i.T z_i),  Q_i = I - P_i
y_i = x_i + Q_i z_i
residual_i = Q_i [A y_i - y_i (x_i.T A y_i)] + P_i z_i
```

Each resulting column is normalized independently. This retains rotations
between selected isolated eigenvectors; it does not replace their observables
with an arbitrary rank-k frame. The complete-degenerate-subspace chart remains
unchanged. A custom tangent dispatcher calls the existing checked linear solver
on each independent n-dimensional block. It does not assemble an `(n*k)`-square
direct system, and `max_dense` remains a per-root bound.

The operator is closure-converted before the validity derivative guard is
applied inside the frame JVP. This placement matters: simply removing the inner
guard made six old invalid/unconverged vector/projector VJPs return zero instead
of rejecting the derivative. The final form retains those guards as well as
the existing returned-observable flags and checks. Structural higher zeros do
not override `response_valid`.

The lazy energy rule is retained. Numerical eigenvalues and their first
stationary derivative do not require response capacity; their second derivative
does. New tests enforce this for one and four roots, and enforce `max_dense=4`
for a four-root 4×4 chart. No tolerance, Ritz solver, shared linear solver or
user energy expression is changed.

## Verification record

| Run | Observed result |
| --- | --- |
| Initial 1/4-root × inner-JIT matrix | 2 failed, 2 passed, 15.17 s |
| Guard omitted: original four-file regression | 6 failed, 61 passed, 349.12 s; all failures were invalid vector/projector derivative contracts |
| Final guard placement: focused nested-JIT, capacity and invalid cases | 12 passed, 27 deselected, 75.61 s |
| Final four-file regression | 67 passed, 396.31 s |
| Final MP2 molecular mixed response | 1 passed, 5 deselected, 175.97 s |

Both final commands exited zero. The four-file group comprises `test_spectral_higher_response.py`,
`test_subspace.py`, `test_unified_eigen.py`, and `test_shared.py`, retaining the
previous 62-test coverage plus five new parameter cases. An initial invocation
of this group did not run because three test files were absent from the cloned
snapshot; those test inputs were copied before the recorded numerical runs.

The molecular gate uses H2/RHF-MP2, Cartesian 6-31G, bond length 1.5 bohr, zero
field, and a unit bond-length direction. It compares the inner-jitted scalar
`-d³E/(dR dFz²)` with the full polarizability tensor JVP, checks tensor symmetry,
and checks displaced polarizabilities at `3e-4` and `1e-4` bohr. This establishes
that tested MP2 response chain, not every correlated method or molecular Raman
calculation.

Runtime: c20 CPU float64, Python 3.11.15, JAX 0.10.0, two OpenMP/OpenBLAS
threads, persistent compilation cache disabled. The exact source, helper/test
and native-library hashes are archived in [source_manifest.json](source_manifest.json).
The source tree hash is `4879dbbf932aa068c3dcb108f20640dce1a67a7f6ddc8d9716ca040b4d19fb88`;
post-run verification confirmed it and every listed file unchanged. Final logs:
[67-test regression](../../../artifacts/ad_showcase/provenance/spectral_nested_jit/spectral-final-guard-regression.log),
[MP2 mixed response](../../../artifacts/ad_showcase/provenance/spectral_nested_jit/spectral-final-guard-mp2.log), and
[verification manifest](verification_manifest.json). Run in the
snapshot with `PYTHONPATH=src`, `CUDA_VISIBLE_DEVICES=`, `JAX_PLATFORMS=cpu`,
`JAX_ENABLE_X64=1`, `JAX_ENABLE_COMPILATION_CACHE=false`, `OMP_NUM_THREADS=2`,
and `OPENBLAS_NUM_THREADS=2`:

```bash
PYTHON=/home/yjiao/opt/miniconda3/envs/jax_scf/bin/python
timeout 1800 "$PYTHON" -m pytest -q \
  tests/solvers/test_spectral_higher_response.py tests/solvers/test_subspace.py \
  tests/solvers/test_unified_eigen.py tests/solvers/test_shared.py
timeout 1800 "$PYTHON" -m pytest -q tests/test_molecular_correlated_response.py \
  -k 'mp2 and mixed_raman'
```

The column action temporarily embeds a vector in an `(n,k)` block before applying
the operator. General operators are not guaranteed to eliminate unused columns,
so these small-system tests establish neither large-k scaling nor performance
parity with the old batching layout. There is no full repository regression or
merge/commit claim in this record.
