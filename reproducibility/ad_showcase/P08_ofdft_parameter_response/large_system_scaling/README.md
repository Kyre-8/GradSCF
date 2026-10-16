# P08 extension: Al WGC accuracy and size dependence

This material example complements the small analytic P08 model. It uses fcc
Al, three valence electrons per atom, the bundled local OEPP, PZ-LDA, and the
second-order WGC99 kinetic functional (gamma 2.7). The volume is
18.435 Angstrom cubed per atom, as in the existing ATLAS-style examples.
It is a fixed-ion calculation: no forces, stresses, or nuclear-coordinate
derivatives are claimed here.

The ordinary entry points are
[large_system_response.py](../../../../examples/ofdft/large_system_response.py),
[large_reference.py](../../../../examples/ofdft/large_reference.py),
[large_ks_reference.py](../../../../examples/ofdft/large_ks_reference.py), and
[large_plot.py](../../../../examples/ofdft/large_plot.py).
Their settings are Python variables; there is no command-line interface.

## Accepted results on 2026-10-16

The current accepted GPU6 family contains 1 and 27 atoms. The frozen Python
source SHA256 is
`124faf9f7813a8a18dccc2f698c3a02ac27ad1af00b0ac47e6dbba105c8cdf22`.
The two-size acceptance record is preserved in
[`verification_manifest_1_27.json`](verification_manifest_1_27.json).
The 125- and 343-atom tasks have not yet been run; no scaling exponent is
inferred from these two points.

| Al atoms | FFT grid | Euler RMS / Ha | Energy difference / Ha atom⁻¹ | Density relative L2 difference |
| ---: | :---: | ---: | ---: | ---: |
| 1 | 17³ | 9.34e-10 | 2.04e-14 | 7.15e-8 |
| 27 | 51³ | 1.87e-14 | 1.32e-15 | 6.24e-8 |

Differences in this table compare the same discrete WGC model against the
independent DFTpy optimizer with the NumPy WGC adapter. They do not measure
the KEDF error relative to Kohn–Sham DFT or experiment. The 27-atom ground
density agrees with primitive-cell replication to relative L2 8.84e-11,
while its density JVP has a nonprimitive component of relative L2 0.2225
under a one-primitive-cell translation. The response test therefore contains
a resolved supercell response signal.

| Al atoms | Forward / s | Full JVP / s | Full VJP / s | Full HVP / s | Peak RSS / GiB | Sampled GPU peak / GiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4.712 | 8.395 | 7.746 | 13.615 | 4.985 | 2.802 |
| 27 | 9.606 | 17.731 | 18.510 | 32.836 | 6.359 | 3.628 |

Times are medians of three synchronized warm executions; each full derivative
operation includes the forward relaxation. Memory covers the complete fresh
process. Both resource records contain no CPU-affinity violation, foreign
GPU6 compute PID or sampling error. See [`scaling.csv`](scaling.csv),
[`scaling.png`](scaling.png) and the complete finite-difference step figure
[`response_fd_steps.png`](response_fd_steps.png).

An independent primitive-cell grid check from 17³ to 23³ gives an energy
increment of 1.11e-7 Ha, full-Fourier density and density-JVP relative L2
increments of 6.69e-5 and 8.47e-5, and a relative directional-curvature
increment of 1.07e-7. The 23³ native/reference energy difference is
1.60e-14 Ha and density difference is 9.95e-8. These are two-grid increments,
not rigorous error bounds relative to a continuum limit; all settings and
gates are retained in [`grid_refinement.json`](grid_refinement.json).

The separate finite-basis KS sequence reaches k=12 with an estimated
zero-smearing energy of -2.0871512653 Ha/atom. Its k=10→12 increment is
0.353 mHa/atom, larger than the 0.241 mHa/atom WGC–KS difference at k=12.
This sequence therefore does not isolate a converged KEDF physical error.
The complete k sequence is retained in
[`comparison_summary.json`](comparison_summary.json).

## Comparable systems and response directions

An odd repetition factor s gives s cubed atoms and a (17s) cubed FFT grid.
The real-space voxel is therefore exactly fixed. The allowed Fourier modes
change with the cell; this is fixed real-space resolution, not an assertion
that every finite-cell reciprocal cutoff is identical. Every native forward
calculation starts from a uniform amplitude.

Two zero-mean, unit-RMS potential probes use seed 1729. They combine a fixed
physical wavelength with a wavelength spanning the complete supercell.
The path direction is (0.6, 0.8), with unit Euclidean norm; the path parameter
has units Hartree. The density JVP must be finite, conserve the integrated
electron number, and have a nonzero RMS signal. The reverse observable is
the density-weighted mean of probe 1 plus 0.3 times probe 2, divided by N.
The energy HVP includes the converged density response. Full density Jacobians
and grid-sized Hessians are not formed.

## Optimization scale and the physical convergence criterion

The initial extensive-energy implementation exposed a size-dependent
optimization failure at 27 atoms. Its failed cold start, the separate
TF+vW-initialized attempt, independent DFTpy checks, and the E/N single-factor
diagnosis are retained as pilot evidence. A very negative unconverged energy
is rejected, not reported as a lower-energy solution.
The cold-start collapse and the TF+vW-initialized branch stalling outside a
fixed raw-gradient refinement threshold are two distinct observations.
Objective scaling changes step magnitudes, line-search rounding and refinement
activation together; the experiment does not uniquely attribute the collapse
to one threshold, establish a globally bounded WGC functional, or find its
global minimum. Independent DFTpy, densities and physical residuals support
the selected physical stationary branch.

The production fix normalizes the numerical sphere optimization by the
dynamic electron count. The original energy, raw stationarity norm and public
forward tolerance retain their original units. Implicit response uses the
equivalent physical-chemical-potential chart described below. The default
sphere-solver scale of one retains its
original behavior. Nonpositive or nonfinite scales return a rejected result
with nonfinite energy/residual under JIT. The final example uses one ordinary
`run_ofdft` call; it does not perform an extra optimization in the example.

Forward normalization exposed a separate response-conditioning problem: on
the same accepted Al27 state, the raw KKT JVP and adjoint equations missed the
declared linear tolerance. The equivalent coordinate and row transformation
`S=diag(1,...,1,N)`, `D=diag(1/N,...,1/N,1)`, `B=D A S` passed it, including
the recovered residuals of the original physical equations. The production
implicit state is now `(x,mu)`, with residual `(grad(E)/N-2*mu*x, x.T*x-1)`.
The moving chart keeps N as a differentiated input. Fixed-reference DFPT uses
`(grad(E)-mu*grad(count))/N_ref` and `(count-N)/N_ref`; raw reference validity
is checked before normalization. Its `Problem`/`Response` norms use these
normalized equations. No shared root or Krylov solver was changed.

For a normalized grid amplitude x, the energy part of the KKT residual is
r_x = dE/dx - 2 eta x. The physical Euler RMS is

```
||r_x||/(2N) = sqrt(integral rho (v_eff - mu)^2 / N), in Hartree.
```

The dimensionless sphere constraint is reported separately. The norm of a
vector containing both terms is not labelled a Hartree-valued error.
The native target is 1e-9 Hartree Euler RMS, with the raw solver tolerance
set to 2 N times that target. The independent reference uses a 5e-6 Hartree
Euler gate and 1e-14 Hartree/atom energy tolerance; these stopping criteria
are deliberately reported separately. Actual residuals accompany every
accepted state. Stationarity does not certify a global minimum or WGC
reliability outside the selected density regime.

## Independent accuracy checks

DFTpy is pinned to fbc47b4e1df3def1a2206e58c5a1a1a4d51db1d0. It provides its
own optimizer, TF, vW, Hartree, PZ-LDA, OEPP and Ewald evaluation. Its dev
functional collection has no native WGC: `wgc_dftpy.py` is an independent
NumPy WGC adapter, using direct-eta ODE integration and an analytic potential.
No GradSCF optimized density, energy, or derivative enters that optimization.

The independent objective first passes a fixed-density energy/potential
directional-derivative check. A failed DFTpy line search may restart its own
LBFGS history at most twice; all attempt codes, residuals and energies remain
in the record. Only accepted stationary states enter comparisons. Native
and independent code agreement uses energy per atom, weighted relative
density L2, potential assembly, ionic energy, electron number and residuals.

Native central differences use 1e-3, 3e-4 and 1e-4 Hartree steps; independent
DFTpy differences use 3e-3, 1e-3 and 3e-4 Hartree. All step errors are retained;
each independent perturbed state also records residuals and optimizer attempts.
The plots show every independent FD step. The best declared step can pass a gate, but the
smallest step is not assumed to be best and a nonmonotone sequence is not
hidden. The JVP/VJP dot-product identity is an internal consistency check,
separate from independent numerical validation. The actual implicit-adjoint
linear residual is unavailable from the public result and is recorded as
unavailable; finite accepted derivatives do not supply that missing number.
The energy curvature is measured along the declared mixed-potential direction
only. Its sign is a local diagnostic, not a full constrained-Hessian stability
test. For the linear potential parameterization, the stationary gradient is
`dE/dtheta = probes.T @ (weights * density)`. Contracting each independent
DFTpy density finite difference with these weights and probes also checks both
components of the energy HVP. `hvp_vector_validation.json` and the complete
`response_fd_all_steps.csv` retain every step. The independent VJP comparison
is the declared directional projection of the density-observable gradient;
it is not a full density Jacobian comparison.

The Gaussian PySCF KRKS comparison is a separate physical-model assessment.
It holds the local OEPP, PZ-LDA, cell and quadrature fixed, uses basis level 3
(58 functions/atom), 0.005 Hartree smearing and an overlap cutoff of 1e-7,
and varies k sampling. A k refinement is an increment, not a combined
basis/k/smearing error bar. WGC-versus-KS differences are not all attributed
to the KEDF, and same-model DFTpy agreement is not chemical accuracy.

## Timing, memory and limits

Each size runs in a fresh process on jy_gpu_1/c20. The final timing family uses
RTX 5090 GPU6 (UUID `GPU-1e6a853f-9000-6569-7e61-0fc9291c49b2`), float64,
CPU48-63 and process-local disabled preallocation;
independent DFTpy and KS references use CPU. Compilation, first execution
and three synchronized warm executions are separated. Full forward, JVP,
VJP and HVP costs include their forward relaxation. VJP preparation
synchronizes both the primal and captured pullback PyTree; pullback-only
timing is also retained and is never substituted for full reverse cost.
The forward counter is the library's L-BFGS step counter; line-search
evaluations and up to three local refinements are not separately counted.
Host CPU validation/reference jobs and other users can overlap compilation.
The CPU set is an eligibility limit, not a reservation against other users;
host load averages are recorded. Times are observations on a shared host.

The project-wide budget is 64 logical CPUs, split into disjoint pools by the
coordinator; this line uses 16 logical CPUs, 48-63. Every launcher, timeout,
monitor and calculation inherits that set. OMP, OpenBLAS, MKL and NUMEXPR
thread variables are all two. JAX's GPU client initially rebound three helper
threads by NUMA node despite taskset. The installed binaries were checked to
use hwloc, and the process-local `HWLOC_THISSYSTEM=0` policy passed a CUDA,
complex128 FFT and float64 gradient/HVP probe with per-TID sampling and no
affinity mutation. This also disables hwloc NUMA memory-binding hooks; it is
not the default NUMA policy and is held fixed for the final timing family.
The [official hwloc binding implementation](https://raw.githubusercontent.com/open-mpi/hwloc/master/hwloc/bind.c)
describes the no-op binding hooks for a topology marked nonlocal. The
[TSL NUMA implementation](https://raw.githubusercontent.com/openxla/xla/main/xla/tsl/platform/numa_hwloc.cc)
provides the corresponding thread and memory binding calls.

The monitor records all observed direct-child TIDs grouped by CPU set, plus
GPU UUID, utilization and compute PIDs at one-second intervals. It never
rewrites affinity. A CPU-set violation or another process on GPU6 rejects the
run and stops only this monitor's own compute child; later jobs in the batch
do not start. GPU6 must have no compute PID at admission. These observations
and the explicit runtime policy are not a claim of cgroup enforcement.

Input preparation includes the RECPOT structure factor and Ewald energy.
The current structure factor forms a grid-point by atom array. Ewald forms
an atom by atom by image array. Both can impose quadratic costs under fixed
density expansion. The FFT functional applications and matrix-free response
solves do not establish near-linear end-to-end scaling; iteration counts,
compilation and these input arrays all matter.

The existing RSS watchdog is reused, without an address-space limit. CPU
memory is a process high-water mark. GPU memory is total used board memory
sampled each second on GPU6, including context and
allocator reservation; it is not an exact live-array peak. The memory curves
cover the complete calculation process, not isolated per-operation peaks.
The initial gate
is 32 GiB RSS and 1800 seconds, with later individual tasks bounded by two
hours. Failed or resource-limited sizes are retained and excluded from fits.
No CPU/GPU speed ranking is inferred from the different native and reference
stopping criteria.
Earlier GPU0 records are historical. A separate user's gmx process was later
confirmed on GPU0, so those timings must not be described as physically
exclusive. CPU-policy-transition and taskset-only runs are also archived and
excluded from the uniform GPU6 performance curves; their passed numerical
checks remain evidence for their recorded inputs and source.
The legacy `*_gpu_samples.json` files use the earlier GPU0 sampler. Current
GPU6 timing acceptance uses `*_resources.json` joined to the per-TID and GPU
records in `*_runtime_samples.jsonl`; the legacy files do not supply the
accepted GPU6 memory values.

## Reproduction

All numerical runs and figure generation use the isolated jy_gpu_1 snapshot
`/home/yjiao/gradscf_checks/ad-showcase-ofdft-scaling-20261016`. The runtime is
Python 3.11.15, JAX/jaxlib/CUDA12 plugin 0.10.0, NumPy 2.4.4, SciPy 1.17.1,
ASE 3.29.0 and PySCF 2.13.0; `runtime_packages.json` records the checked set.
`hardware_and_reference.txt` records the Xeon Platinum 8457C host, RTX 5090
cards and driver, and the pinned DFTpy archive hash. Each accepted native JSON
has a matching full Python source manifest and environment.

Edit `REPEAT` in the two ordinary calculation scripts to the same odd repeat.
Use a fresh process per size. For native calculations, set
`CUDA_VISIBLE_DEVICES=6`, `JAX_PLATFORMS=cuda`, `JAX_ENABLE_X64=1`,
`XLA_PYTHON_CLIENT_PREALLOCATE=false`, `HWLOC_THISSYSTEM=0`,
`OMP_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=2`, `MKL_NUM_THREADS=2`,
`NUMEXPR_NUM_THREADS=2`, and `PYTHONPATH=src`. Wrap the complete launcher with
`taskset -c 48-63`. For the independent reference,
use an empty `CUDA_VISIBLE_DEVICES`, `JAX_PLATFORMS=cpu`, and
`PYTHONPATH=src:examples/ofdft:dftpy_pinned/src` with the same thread limits.
Run the scripts from the snapshot root. Resource JSON records preserve the
actual child command; raw logs preserve optimizer iterations and failed
attempts. The shared RSS monitor enforces the memory bound, with an external
timeout for each job. This snapshot does not modify global caches, system
configuration, other users' processes, or the installed JAX binaries.

After the accepted native and independent runs, `large_plot.py` writes the
performance and all-step finite-difference figures. The companion
`artifacts/ad_showcase/development/ofdft_scaling/validate_results.py` checks source hashes,
native/reference joins, particle counts, finite arrays, reference agreement,
the primitive-density replication invariant and the full HVP vector check,
then writes `verification_manifest.json`. All numerical postprocessing is
performed on jy_gpu_1; local checks are static source review and visual QA.

## Diagnosis and regression evidence

The pre-fix records in `history/forward_scaled_raw_kkt/` remain historical;
they do not enter the final timing curves. In the same-state Al27 diagnostic,
raw JVP/adjoint relative residuals were 1.67e-5 and 2.14e-5. The equivalent
normalized solves recovered residuals of the original physical equations of
6.05e-12 and 4.73e-12. These exposed diagnostic solves are separate from the
opaque public AD invocation; they are not relabelled as its residual telemetry.

The public Al27 preflight then produced finite JVP, VJP and HVP values,
relative electron-number response 4.94e-12 and adjoint-identity error 5.16e-13.
The mixed-potential energy curvature was -5.9994279937 Ha^-1/atom. Its four
first executions took 12.06, 23.89, 22.61 and 42.19 seconds, with compilation
recorded separately. This single-call preflight is not a warm timing sample.

On jy_gpu_1, the 14 scale/chart regressions passed, including dynamic N,
volume, a Gaussian overlap tangent, first/reverse/second response and DFPT
Taylor coefficients. The complete 28-test native OFDFT/DFPT group passed,
including molecular geometry, periodic force/strain and atomic force-constant
finite differences, without skips or deselections. The native library was
checked against 84 source hashes plus its own SHA256 before use. These
Gaussian and existing periodic geometry regressions do not validate geometry
derivatives through the NumPy material-input builder used by this benchmark.
The 16 WGC, original P08 and large-example cases also pass. One initial
failure was a missing libKEDF reference file in the remote fixture; its
original log remains, and the supplied repository fixture passed on retry.
The strengthened nonzero-Euler unit check was also rerun successfully.
`regression_verification.json` reconciles 58 unique selected tests and the
native ABI audit; this is not a claim about the complete repository suite.
