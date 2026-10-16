# Nested-JIT static BSE response

The narrow BSE operator-closure fix passed 15 new response tests, 9 existing
numerical cases, and the original molecular reproducer on 2026-10-16.
All numerical work used `jy_gpu_1`, CPU float64, JAX 0.10.0. No shared solver,
GW equation, physical derivative or invalid-result guard was removed.

## Failure and scope of the change

The original composition `jit(JVP(jit(optical)))` failed with
`No constant handler for type: DynamicJaxprTracer`. A 3-MO/2-auxiliary array
model isolates the error without SCF: ordinary action differentiation and
materializing the same TDA matrix both pass, while attaching shared spectral
response to the BSE LinearOperator fails under nested JIT. The canonical-frame,
scalar QP and standalone screening controls also pass.

The [closure report](closure_before.json) shows two non-inexact traced values
retained as constants by `jax.closure_convert`: static virtual indices and
boolean screening validity. Six physical floating leaves are lifted correctly.
Independent 2x2 toys reproduce each source; host indices and a floating validity
gate reconstructed inside the action resolve their respective failures.

Only [bse/kernel.py](../../../../../src/gradscf/bse/kernel.py) changes in core:
orbital indices remain host metadata, and internal immutable screening copies
capture validity as floating 0/1. Each checkpointed action reconstructs boolean
validity before calling the unchanged screening routines. Public screening
states still expose boolean validity. Both physical-input and true-residual
checks remain active in forward and backward calculations.

## Verification

The [new regression](../../../../../tests/bse/test_nested_jit.py) first failed
on the unchanged source with the exact error ([red log](../../../../../artifacts/ad_showcase/provenance/GW_BSE_response_gates/validation/nested_jit/red.log)). The final
[15-pass run](../../../../../artifacts/ad_showcase/provenance/GW_BSE_response_gates/validation/nested_jit/green.log) checks raw versus nested JVP/VJP and finite differences:

- TDA dense/Davidson, eigenpair/energy-only response, direct/GMRES screening;
- full Davidson BSE with direct/GMRES screening;
- spin-conserving TDA/full BSE, reusing the restricted blocks;
- invalid gaps, asymmetric factors and failed screening solves, whose nested
  tangents and cotangents must remain nonfinite.

The run took 747.44 s inside pytest, 756.06 s total, with peak RSS 3,168,360 KiB.
These are test-process timings, including multiple cold compilations, not a
per-molecule or solver performance comparison.

Existing action/transpose, bounded-storage, roundoff and spin-oracle tests
provide 9 further numerical passes. The first command had 5 passes and 4 setup
failures because its isolated snapshot omitted the existing QuAcK fixture
([original log](../../../../../artifacts/ad_showcase/provenance/GW_BSE_response_gates/validation/nested_jit/existing.log)). After copying that unchanged fixture, only
those 4 cases reran and passed ([follow-up](../../../../../artifacts/ad_showcase/provenance/GW_BSE_response_gates/validation/nested_jit/existing-fixture.log)). The original
command is correctly retained with exit 1.

The final molecular run executes the saved 30-line probe for the
[matched water example](../../../X15_water_matched_vertices/README.md): G05 RHF
minimum, Cartesian 6-31G*, all 19 G/W/QP orbitals, 3 isolated TDA states, one
mass-normalized mode. All 22 nested derivatives are finite and match the
previously validated raw-JVP/outer-JIT array **exactly** in this run
([comparison](molecular_probe.json)). Wall time 45.43 s; peak RSS 1,946,168 KiB.
The earlier multistep geometry finite differences were not rerun or relabelled.

This validates the named first-order compositions and physical examples. It
does not establish general high-order GW/BSE response, new degenerate-root
rules, arbitrary global QP branch selection or complete cross-module integration.

## Provenance

Remote snapshot:
`/home/yjiao/gradscf_checks/ad-showcase-bse-nested-audit-20261016`.
Its source was copied from the accepted matched-water snapshot (tree SHA256
`94008e9b36cba7a8dd61645abce3b4da83e7cf7965bb62304edda8ad8e35ddd0`), then
only the BSE kernel was replaced. Exact final source/test hashes are in
[source_sha256.txt](source_sha256.txt). Native library SHA256 is
`8176ca97f9758ba3810b727c5a2e90572ac4f8f205e4a16e0d7a7177221b1829`.

Commands use `PYTHONPATH=src`, `JAX_PLATFORMS=cpu`, `JAX_ENABLE_X64=1`,
`JAX_ENABLE_COMPILATION_CACHE=false`, empty `CUDA_VISIBLE_DEVICES`, and
OMP/OpenBLAS thread counts 2. The molecular probe also adds `examples/bse` to
the Python path. CPU affinity, global JIT behavior and cache policy were not
modified. The source-only fix, tests and evidence remain uncommitted.
