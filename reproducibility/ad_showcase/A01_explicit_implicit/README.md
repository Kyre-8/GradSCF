# A01 — Explicit and implicit derivatives of the same molecular observables

All eight HF/PBE × energy/dipole × explicit/implicit cases passed on `jy_gpu_1`
(`c20`, CPU float64). Each case used a fresh spawned process, the same seed
orbitals within its functional, identical convergence controls, and a nonzero
geometry/field point. Every case took **9 orbital iterations**; the largest
stationarity norm was `2.23e-16`. Equal initial orbitals and iteration counts do
not assert that every intermediate numerical trajectory is bitwise identical.

[Timing figure](phase_timing.png) · [validation and memory](validation_and_memory.png)
· [timing CSV](timing_summary.csv) · [native results](gradscf.json)
· [independent PySCF results](pyscf.json) · [frozen source](source_manifest.json)

## Physical definition

H2 uses Cartesian 6-31G, charge/spin zero, and seed coordinates
`[(0,0,-0.7),(0,0,0.7)]` bohr. With dimensionless `theta=(0.25,0.4)`,

```
R(theta) = Rseed + theta[0] * [(0,0,-0.05),(0,0,0.05)] bohr
F(theta) = theta[1] * (0,0,0.005) atomic electric-field units
```

The measured point therefore has bond length 1.425 bohr and `Fz=0.002 au`.
The scalar observables are `E/Eh` and `mu_z/(e bohr)` at a fixed origin `(0,0,0)`.
Energy includes electronic and nuclear interactions with the field; the
independent reference explicitly adds `-F dot sum_A Z_A R_A` to PySCF's SCF
energy. Dipoles are evaluated from the relaxed density and nuclear charges.
PBE uses grid level 0, a deliberately small comparison example rather than a
grid-converged chemistry prediction.

Both modes use `MolecularEnergy` and the existing orbital optimizer. The seed
SCF tolerances are energy `1e-12 Eh`, density `1e-11`, and gradient `1e-10`, with
150 SCF cycles. The shared orbital solve has gradient tolerance `1e-10` and
maximum 300 iterations. Implicit response uses tolerance `1e-10`, maximum 100
iterations and restart 40. The seed coefficient hashes, final densities,
energies, derivatives, stationarity, positive iteration counts, and nonzero
energy decrease are checked before mode parity is accepted.

| Check | Largest absolute discrepancy |
| --- | ---: |
| Explicit versus implicit energy / Eh | 4.45e-16 |
| Explicit versus implicit dipole / (e bohr) | 2.44e-17 |
| Explicit versus implicit derivative with respect to theta | 1.26e-16 |
| Native central FD at h=0.002, all observables and modes | 8.10e-10 |
| Independent PySCF central FD at h=0.002 | 8.10e-10 |
| Independent energy / Eh | 1.38e-13 |
| Independent dipole / (e bohr) | 6.85e-12 |

The HF point has `E=-1.1264351106504806 Eh`, `mu_z=0.0133434851787394 e bohr`;
PBE has `E=-1.161036735390585 Eh`, `mu_z=0.0130863713413914 e bohr`.
Native and independent central differences retain all steps `0.02, 0.007,
0.002` in theta, and native displaced states must be stationary. The additional
identity `-dE/dtheta_field / 0.005 = mu_z` verifies the field convention.

## What the timings measure

Each process first constructs its seed SCF. It then measures, separately:

1. The ordinary jitted scalar forward evaluation, first call and five repeats.
2. `jax.vjp(forward, theta)`, which evaluates a primal and prepares a pullback,
   first call and five repeats.
3. Calls to the actual retained pullback with cotangent one, first call and
   five repeats. These are direct measurements, not differences of wall times.

Every timed result is synchronized with `jax.block_until_ready`. “First” means
the first call of that transformed phase **after seed SCF and preceding phases**;
it does not mean a completely unwarmed runtime. Persistent compilation caching
is disabled and its actual JAX setting is recorded and asserted. These choices
follow the asynchronous-dispatch and compilation distinctions in the
[JAX benchmarking guide](https://docs.jax.dev/en/latest/benchmarking.html).
The primal/pullback split follows the documented
[`jax.vjp` contract](https://docs.jax.dev/en/latest/_autosummary/jax.vjp.html).

| Method | Observable | Mode | Forward median / ms | VJP prep median / ms | Pullback median / ms | Timed process peak RSS / MiB |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| HF | energy | explicit | 3.868 | 7.330 | 25.069 | 1442.5 |
| HF | energy | implicit | 3.515 | 5.964 | 4.170 | 1309.9 |
| HF | dipole | explicit | 3.677 | 10.081 | 33.906 | 1441.8 |
| HF | dipole | implicit | 3.440 | 6.866 | 3.212 | 1314.4 |
| PBE | energy | explicit | 45.787 | 72.898 | 3823.974 | 3905.6 |
| PBE | energy | implicit | 51.654 | 59.612 | 34.049 | 2622.5 |
| PBE | dipole | explicit | 51.101 | 71.091 | 3931.944 | 3858.7 |
| PBE | dipole | implicit | 50.437 | 68.541 | 35.063 | 2609.9 |

The CSV also retains first-call times; the case JSON files retain every repeat
and cumulative RSS after each phase. RSS includes imports, seed SCF and compiler
and runtime allocations. It is process high-water memory, **not backward-only
allocated memory**. Scientific validation occurs after the timed RSS window;
its later peak is reported separately. The figure uses median and min/max
whiskers, not statistical confidence intervals. These shared-node observations
do not establish asymptotic scaling, a universal speedup, or performance of
later source revisions.

## Reproduction and provenance

Executed in `/home/yjiao/gradscf_checks/ad-showcase-a01-a02-20261015`, cloned from
the frozen MolecularEnergy v7 response source. The Python source-tree hash is
`ca37f375d0e4cc673d1ca2e5b50f189b9047ad8bce56e00f7f397156fac3e43f`;
individual source/native-library hashes are in `source_manifest.json`. No later
stationary-value energy optimization is present in these measurements. The
machine used Intel Xeon Platinum 8457C CPUs, Python 3.11.15, JAX 0.10.0,
NumPy 2.4.4, PySCF 2.13.0, jax-xc 0.0.9, and SciPy 1.17.1.

Run these ordinary scripts in that snapshot with the following environment;
the scripts have no CLI options. Numerical validation and plotting were remote.

```bash
export CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
export JAX_ENABLE_COMPILATION_CACHE=false OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH=src
PYTHON=/home/yjiao/opt/miniconda3/envs/jax_scf/bin/python
timeout 7200 "$PYTHON" examples/response/explicit_implicit_response.py
timeout 900 "$PYTHON" examples/response/compare_explicit_implicit_pyscf.py
timeout 300 "$PYTHON" examples/response/plot_explicit_implicit_response.py
```

All three commands exited zero (`A01_native.log`, `A01_pyscf.log`,
`A01_plot.log` in the remote snapshot). Each child has a 1800-second bound;
the independent comparison took 1.783 s. Saved scientific JSON is unchanged
after measurement. Trailing measured-output comments were added to local
scripts afterward; the execution hashes identify the remote executed versions.
