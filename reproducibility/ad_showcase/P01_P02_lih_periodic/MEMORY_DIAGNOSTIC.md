# Periodic implicit-SCF trajectory staging

This is a controlled implementation diagnostic for P01/P02, separate from the
physical basis, FFT and k-point convergence study. All numerical work ran on
`jy_gpu_1` (c20), CPU float64, with persistent compilation caching disabled.
No solver tolerance, iteration limit, seed or physical setting changed between
the two initial diagnostic processes.

The case is displaced two-atom rocksalt LiH, PBE/GTH-Pade/GTH-SZV, 3 Cartesian
AOs, 4 explicit electrons, a full 2×2×2 k mesh and a 31³ FFT mesh. SCF uses at
most 150 iterations and converges after 12. The initial public-facade force and
stress calculation reached 79,770,620 KiB peak RSS, motivating this diagnostic
before any larger grid or basis run.

## Same callback, different compilation boundary

The private probe reconstructs the same public periodic inputs and calls the
same SCF array solver and `energy_and_derivatives`. A fresh process runs either
that callback eagerly or `jax.jit(callback).lower(...).compile()`. Both first
perform the same forward SCF and compare its density with the archived state.
Both derivatives pass the original facade energy, force and stress gates.

| Boundary | Preliminary forward / s | Derivative phase / s | Peak RSS through derivative / KiB |
| --- | ---: | ---: | ---: |
| Eager array callback | 26.1165 | 84.8364 | 79,423,680 |
| Whole callback JIT | 27.4013 | 38.1534 | 4,956,680 |

The JIT derivative phase includes lowering and compilation. RSS is the process
high-water mark through the derivative phase, including preliminary SCF and
compilation; it is **not** derivative-only memory. These are diagnostic timings,
not a portable performance benchmark. The compiler separately estimates
1,374,030,304 temporary bytes for the whole-JIT executable; this estimate is not
process RSS.

The frozen Python source tree is
`c792320a88f24c695049469ac317d26953efc0888cd3236c44cb0fb393220abc`.
The probe SHA256 is
`c244c8080aeedd9222e6e6f4c20cd7d802608c925b80e9d0a198f387f822a209`.
The original snapshot remains at
`/home/yjiao/gradscf_checks/ad-showcase-p01-p02-20261016`.

## Dataflow evidence

The original k-point SCF scan returns `ys=None` and carries 11 state entries:
current density, energy, Fock, orbitals, orbital energies, two small DIIS
buffers and scalar control state. Its primal Jaxpr has no stacked outputs.
The differentiated Jaxpr adds 914 stacked outputs totaling **76,251,796,950
bytes**, all marked `DropVar`: their values are not consumed by the final
derivative. Four of the largest arrays each contain 5,147,884,800 bytes, with
shapes equivalent to `(150, 8, 29791, 3, 3)` in complex128.

The physical arguments remain differentiable while the numerical forward scan
is linearized. Its output density is stopped only later, when the implicit
residual rule is attached. Thus eager AD can stage iteration-by-grid reverse
residuals before discovering that the forward trajectory is unused. Whole-JIT
compilation can eliminate these dead outputs. The paired RSS measurements,
primal/derivative scan structure and compiler memory estimate support this
explanation; they are not an allocation-by-allocation profiler trace.

The Gamma solver differs slightly: it reuses the molecular scan, which already
returns seven small diagnostic histories. The regression therefore forbids
**additional AD trajectory arrays**, preserving these existing primal outputs.
On the original source, the corrected narrow regression reports 2 expected
implicit failures and 2 explicit passes in 53.46 s: Gamma adds 924 arrays and
k-point adds 885 for its smaller 7³ test case. The first test draft incorrectly
assumed Gamma also had no primal histories; that 3-failure draft is retained
separately and is not the accepted regression boundary.

## Narrow repair and validation state

The proposed repair stops physical arguments and the seed **before the
implicit numerical iteration**. The final implicit residual, energy and MO
reconstruction still use live inputs, including the overlap metric. Explicit
SCF retains its original differentiable trajectory. No global JIT/cache policy,
solver algorithm or physical convergence threshold changes.

In the independent `ad-showcase-p01-p02-memoryfix-20261016` snapshot, the six
narrow checks passed in 84.14 s: dataflow, explicit-path preservation, live final
orbital metric JVP, and a density-dependent observable's second derivative
against FD, each covering Gamma and k-point paths. The source tree hash is
`0de7c6c920f84563529fe04238cd8c92b625c3e05665bfe2fcccc278e0077470`.

The same fresh-process LiH **eager** callback then passed the archived E/F/stress
gates, with a 69.8712 s derivative phase and **5,860,376 KiB** peak RSS through
that phase. The length-150 SCF scan now has **zero additional stacked bytes**.
This intervention strengthens the dataflow diagnosis without changing the
numerical SCF or relying on whole-JIT dead-output elimination.

The existing periodic response regression batch passed **21 tests in 669.99 s**,
covering forces/strain, explicit and implicit bond derivatives, thermal free
energy, external functional parameters, full-state response after time-reversal
reduction, and periodic DFPT. Its first invocation exited 4 because the isolated
snapshot lacked the selected old test files; no tests were collected. That
infrastructure record is retained, and the explicitly named files were copied
before rerunning. It is not a scientific failure. Physical convergence work
resumes one case at a time with measured peak-memory checks before increasing
the next FFT, basis or k dimension.
