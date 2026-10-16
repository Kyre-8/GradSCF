# P05: analytic infrared and nonanalytic-limit kernels

The [shared vibrational functions](../../../src/gradscf/pbc/vibrational.py)
now expose unweighted Gamma nonanalytic force constants, mode effective
charges and the retarded ionic dielectric tensor. The
[formula and conventions](../../../src/gradscf/pbc/INFRARED.md) specify the
Born tensor axes, atomic units, transverse optical modes and physical limits.
No electronic solver, eigensolver or line-shape implementation is duplicated.

On `jy_gpu_1`, 29 new infrared tests and 24 existing vibrational tests passed:

```sh
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
JAX_ENABLE_COMPILATION_CACHE=false OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=src timeout 600 python -m pytest -q \
    tests/pbc/test_infrared.py tests/pbc/test_vibrational.py
```

The actual final log reports **53 passed in 20.54 s**, process wall time
22.32 s and peak RSS 1,422,036 KiB (1.36 GiB), on an Intel Xeon Platinum
8457C CPU with JAX 0.10.0 and float64. These are small-array correctness
checks, not electronic-structure or scaling benchmarks. The recorded thread
settings do not restrict every JAX worker pool.

Independent checks cover:

- A two-atom isotropic oscillator's LO–TO splitting, three dark translations
  and exact static Lyddane–Sachs–Teller ratio.
- Macroscopic displacement energy, anisotropic nonsymmetric Born tensors,
  Cartesian rotation covariance and invariance to the direction's magnitude
  and sign. A deliberately nonneutral tensor retains its raw defect.
- Complete-mode tensor weight, complex gauge transformations within a
  degenerate optical group, and an independent Cartesian resolvent.
- An independent infinite-frequency quadrature of the oscillator-weight sum.
- JVP/VJP adjoint consistency and three-step finite differences in Born
  charges, epsilon-infinity, volume, direction, masses and mode frequencies.
- Eager/JIT/JVP/VJP rejection of zero direction, nonpositive/asymmetric
  dielectric tensors, nonpositive volume/mass and unstable mode energies.

The initial missing-API red test is retained. The first complete run had
40 passes and one overly strict analytic-zero assertion: a cancellation was
3.47e-18 versus an absolute tolerance of 1e-18. The final test uses an explicit
floating-point roundoff bound scaled by the charge norm. Production code and
physical convergence gates were unchanged; invalid-input AD checks were then
added. Both runs and their exits remain in [validation](validation/).

The [source/runtime manifest](source_manifest.json) records the frozen snapshot,
versions, environment and hashes. [Executed source and tests](../../../artifacts/ad_showcase/provenance/P05_infrared_kernels/executed)
preserve the exact final files. The test does not exercise native AO integrals,
SCF, material convergence or finite-q interpolation.

These functions perform no acoustic or charge-neutrality projection. Their
Gamma correction is not a substitute for long-range dipolar subtraction and
restoration in polar IFC interpolation. The matched-grid LiH IR calculation
is a separate example and requires its own source, convergence and field gates.
