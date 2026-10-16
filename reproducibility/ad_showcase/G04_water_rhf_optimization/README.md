# G04: water RHF geometry optimization

Measured on `jy_gpu_1` (c20), Intel Xeon Platinum 8457C, CPU float64,
2026-10-15. The calculation uses Cartesian 6-31G*, coordinates in bohr and
energies in Hartree. Native integrals and GradSCF RHF supply all energies and
stationary AD gradients. SciPy BFGS optimizes nine Cartesian coordinates.

Run `PYTHONPATH=src python examples/geometry/water_rhf_optimization.py` from
the repository root. The initial geometry is the nonstationary G01 geometry.
SCF tolerances are energy `1e-12`, density `1e-10` and orbital gradient `1e-9`.
The nuclear convergence condition is max absolute gradient below `1e-7` Eh/bohr.

- BFGS terminated successfully after 7 iterations and 8 energy/gradient calls.
- Final energy: **-76.01074650846029 Eh**.
- Maximum absolute gradient: **4.2898e-8 Eh/bohr**.
- Both O-H bonds: **1.79017327069 bohr**; H-O-H angle: **105.499743979 degrees**.
- Measured end-to-end optimization time: **4.384 s**, including first-use
  compilation within this process. This is not a separately timed backward.

[gradscf.json](gradscf.json) retains every evaluated geometry, energy and
gradient norm, including line-search evaluations. [G05](../G05_water_rhf_hessian/README.md)
checks the optimized point's Hessian, establishes three positive internal modes,
and compares the analytic Hessian and final gradient with independent PySCF.
The optimization alone does not establish minimum stability.

Runtime: Python 3.11.15, JAX 0.10.0, NumPy 2.4.4; PySCF reference 2.13.0.
Environment: `CUDA_VISIBLE_DEVICES=`, `JAX_PLATFORMS=cpu`, `JAX_ENABLE_X64=1`,
`JAX_ENABLE_COMPILATION_CACHE=false`, `OMP_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=2`.
Remote snapshot: `/home/yjiao/gradscf_checks/ad-showcase-20261015`, base `8f8c93d`
plus the example and workflow additions. All calculations and plotting were
remote; no local numerical test was used.
