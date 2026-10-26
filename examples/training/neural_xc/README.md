# Neural XC implicit training memory test

`h2_implicit_memory.py` trains the existing `neural_xc` residual mixing
network on five H2 dissociation points, with fixed GradSCF FCI total-energy
labels. It reuses the shared molecular loss and Adam step underlying the older
H2 ground-state training workflow, now in `tests/comparisons/`. Geometry and
integrals are fixed; the differentiated variables are network parameters.

Defaults: H2/6-31G*, grid level 0, two 64-unit residual blocks, canonical
features, the default B3LYP semilocal component basis, HF channel off,
no PT2, seed 0, float64 parameters and optimizer moments, three Adam
updates at 1e-4. Loss is the mean of MSE plus MAE of total energies in Hartree.
This is a training/AD smoke test, not a converged dissociation-curve fit.
The component basis does not make this a B3LYP calculation: its coefficients
are trainable and the HF channel is absent. The existing HF `nograd` mode
updates density-dependent HF features while stopping their AD derivatives.
It therefore does not represent the full derivative checked by reconverging
the loss at perturbed weights. That approximation is excluded from this strict
gradient test; the production functional's behavior is unchanged.

## Run

This directory contains the complete example:

```text
neural_xc/
├── h2_implicit_memory.py  # H2 references, shared training API and phase measurements
├── gpu_memory.py          # PID-based NVML sampling and synchronized timing
├── run_memory.sbatch      # Delta GPU allocation and reproducibility records
└── README.md              # Instructions, measurement definitions and validated results
```

Run the commands below from the repository root. The Python example uses
editable settings and environment overrides, without an argparse CLI.

Use the existing GradSCF, jax-xc and nvidia-ml-py environment on one GPU:

```bash
PYTHONPATH=src JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
python examples/training/neural_xc/h2_implicit_memory.py
```

On Delta (`jy_gpu_2`), read the current `lwcluster` manual first. Submit from a
checkout under `~/private/homefile/` using the supplied `home` partition job:

```bash
sbatch examples/training/neural_xc/run_memory.sbatch
```

The script uses the existing `~/private/homefile/conda-envs/jax_scf` environment,
one typed V100 allocation, four CPU cores, 16 GiB host memory, and 20 minutes.
Check live availability/budget before submitting. If the checkout's native
integral library needs rebuilding, set `NEURAL_XC_BUILD_NATIVE=1` when submitting;
the build runs within that allocation using an independent Linux build directory.
`NEURAL_XC_JAX_CACHE_DIR` can reuse an existing compilation cache on the same
node after a timeout; the launcher records the actual cache path. Such a run's
first call includes cache loading rather than necessarily a fresh compilation.

Settings can be edited in the example or overridden with `NEURAL_XC_BASIS`,
`NEURAL_XC_GRID_LEVEL`, `NEURAL_XC_WIDTH`, `NEURAL_XC_DEPTH`, `NEURAL_XC_STEPS`, and
`NEURAL_XC_MEMORY_OUTPUT`. Each configuration should use a fresh process for
comparable residency. A CPU smoke test requires `JAX_PLATFORMS=cpu` and
`NEURAL_XC_REQUIRE_GPU=0`; its GPU-memory fields are null.

The production module's default network dimensions can be tested with
the same script:

```bash
NEURAL_XC_WIDTH=192 NEURAL_XC_DEPTH=4 \
  sbatch examples/training/neural_xc/run_memory.sbatch
```

## What is measured

| Phase | Work |
| --- | --- |
| `forward_*` | Converged SCF and loss, without retaining a pullback |
| `vjp_forward_*` | The same loss, retaining backward residuals |
| `backward_*` | Apply the implicit VJP at fixed parameters, without rerunning forward |
| `train_*` | Shared loss/gradient and an accepted Adam update |

The first call includes tracing, compilation or cache loading, and execution; the warm call uses
the compiled path. Phase timings synchronize output arrays before stopping.
Molecular data and VJP residual PyTrees are explicit JIT arguments, avoiding
large pullback arrays embedded as compilation constants.
The initial optimizer step counter is a device array, avoiding the dispatch
specialization change from a Python integer after the first update.
Data, parameters and optimizer moments are all explicitly committed to the
same device before measuring, so the first update does not change input placement.
The first pullback is released before measuring the next VJP forward, and all
saved residual arrays are synchronized. This avoids an artificial overlap of
two tapes. VJP-forward timing includes Python linearization overhead as well
as the device execution.
Native integrals, HF, FCI and grids are prepared explicitly on CPU and the
training PyTrees are transferred to the GPU. This is required by the native
CPU FFI, not an automatic training fallback. Preparation, network/optimizer
initialization and finite-difference validation are outside these
phase measurements; their persistent arrays remain resident.

NVML samples **this process's** GPU residency every 10 ms, independent of CUDA
device-number remapping. The recorded peak is a sampled lower bound, including
CUDA contexts, library workspaces and JAX allocator pools. The increase over
phase-start residency is not an exact temporary-buffer allocation. Fast spikes
may be missed. Executables and allocator pools from earlier phases remain in
the process, so phase peaks must not be subtracted to infer backward-only cost.

JAX `memory_stats()` is also recorded before and after each phase; its
`peak_bytes_in_use` is a cumulative allocator high-water mark, not a resettable
phase peak. Compiler buffer estimates are not substituted for measured residency.

The example rejects unconverged SCF, nonfinite **raw** VJP gradients, failed
directional finite differences, rejected optimizer updates and a nondecreasing
final loss. The direction is a normalized all-ones parameter tree; epsilon is
1e-4, with rtol 2e-3 and atol 2e-6. This one direction checks the full backward
path but is not an exhaustive gradient proof.

`summary.json` contains configuration, energies, gradient check, measurements
and training history; `params.msgpack` stores final weights. The Slurm launcher
also preserves package versions, GPU information and source/native-library
SHA256 manifests. Only `status: complete`, a successful process exit and valid
outputs together establish completion.

## Validated small-network run (2026-10-10)

Delta n003, one Tesla V100-SXM2-32GB, JAX 0.10.0, Python 3.11.15,
float64, job 95289 (`COMPLETED`, exit 0). The two-64-unit network has 9,604
parameters. Every H2 geometry has four AOs and 1,240 grid points. The run used
a pre-existing node-local JAX compilation cache; first-call times include
tracing and cache loading and are not cold compilation benchmarks.

| Phase | Warm synchronized time / s | Sampled process peak / MiB |
| --- | ---: | ---: |
| Value-only forward | 0.0699 | 418 |
| Forward retaining one VJP | 0.3638 | 1356 |
| Implicit backward | 0.0573 | 1362 |
| Complete Adam step, update 2 | 0.1147 | 1884 |
| Complete Adam step, update 3 | 0.1112 | 1884 |

All three updates were accepted. Loss decreased from 0.11599956006168284 to
0.1133567616675481. The normalized parameter-direction derivative was
-0.025103061842103695 (AD) versus -0.025103062011311295 (finite difference),
an absolute difference of 1.6921e-10. The core numerical configuration is
recorded in `summary.json`; SCF energy/density tolerances were 1e-10/1e-8,
and the unregularized implicit solve used tolerance 1e-9 and max_iter 40.

Two measurement issues were removed before this run: changing the initial
state's device commitment caused an extra JIT specialization after update 1;
retaining the first pullback during the next VJP forward kept two tapes alive.
Releasing the old pullback reduced that warm-forward sampled peak from
1868 to 1356 MiB in this case, with unchanged energies and derivatives.

A prior HF `nograd` trial had finite raw gradients but failed the full
reconverged-loss finite-difference check by about 1.24% in the tested direction.
Its stopped density derivatives have a different scope from that check;
it is not included among the validated full-response results above.

Local raw records and checkpoints are under
`artifacts/neural-xc-memory-20261010/` (ignored by Git). The remote checkout is
`~/private/homefile/neural-xc-memory-20261010/`, with per-job records in
`results/<job-id>/`. The 95289 source hash predates only the subsequent addition
of the configurable depth; its dimensions were fixed to two layers.

## Default network dimensions (2026-10-10)

Job 95295 used `NEURAL_XC_WIDTH=192 NEURAL_XC_DEPTH=4` with all molecular,
loss, precision and solver settings unchanged. It completed in 15 min 59 s
with exit 0 on the same V100 node, using a new compilation-cache directory.
Before the example-directory reorganization, the example and memory-helper
hashes were verified against its recorded source hashes. The relocation changes
file paths and documentation; the numerical and sampling implementations are
preserved. Historical manifests retain their original paths and hashes.

| Quantity | 2 x 64 | 4 x 192 |
| --- | ---: | ---: |
| Parameters | 9604 | 152836 |
| Warm value-only forward / s | 0.0699 | 0.0970 |
| Warm VJP forward / s | 0.3638 | 0.5258 |
| Warm implicit backward / s | 0.0573 | 0.1145 |
| Warm Adam steps / s | 0.1147, 0.1112 | 0.2294, 0.2222 |
| Sampled process peak, VJP forward / MiB | 1356 | 2870 |
| Sampled process peak, backward / MiB | 1362 | 4924 |
| Sampled process peak, Adam / MiB | 1884 | 4940 |
| JAX live buffers after VJP forward / MiB | 361.10 | 1538.02 |
| JAX cumulative allocator peak through backward / MiB | 395.73 | 2978.49 |
| Directional derivative absolute error | 1.69e-10 | 1.15e-10 |
| Accepted updates | 3/3 | 3/3 |

The default-sized network's loss decreased from 0.11598847435442008 to
0.1079289771492325. All five SCFs converged at every recorded training step;
cycle counts were 11, 11, 4, 5 and 11. The directional derivative was
-0.01662511318814785 versus finite difference -0.016625113303178618.
First calls took 131.27 / 179.13 / 195.07 / 419.25 s for value-only forward,
VJP forward, backward and the complete training step, respectively.

These are sequential phases within one process per configuration, not isolated
phase allocations. The small run reused a disk compilation cache; the large
run started with a fresh one. CUDA library/workspace residency can therefore
differ beyond the network's own buffers. The VJP-end JAX live-buffer increase
(361 to 1538 MiB) is a more specific observation of retained arrays than a
difference between whole-process peaks. Neither run is a general memory bound
for larger basis sets, grids or other training objectives.

Default-network records are in the local artifact folder's `default-network/`
subdirectory and in remote `results/95295/`.
