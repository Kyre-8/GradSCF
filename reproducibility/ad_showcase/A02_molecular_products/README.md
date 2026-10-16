# A02 — Molecular JVP, VJP and energy HVP

Water RHF/Cartesian 6-31G* passed the native product, small dense-reference,
adjoint, multistep finite-difference and independent PySCF checks on `jy_gpu_1`
(`c20`, CPU float64). The observable contains total energy and three components
of the relaxed molecular dipole; every response includes the converged SCF and
moving integral/basis chain through `MolecularEnergy`.

[Response figure](molecular_products.png) · [native results](gradscf.json)
· [independent PySCF results](pyscf.json) · [Jacobian CSV](jacobian_theta.csv)
· [Hessian CSV](hessian_theta.csv) · [frozen source](source_manifest.json)

## Parameters, observables and contractions

The base coordinates are `O=(0,0,0)`, `H1=(0,1.43,1.11)`,
`H2=(0,-1.43,1.11)` bohr. The 12 dimensionless parameters have explicit scales:

```
R(theta) = R0 + 0.05 * reshape(theta[:9], (3,3)) bohr
F(theta) = 0.002 * theta[9:] atomic electric-field units
theta = [0.12,-0.08,0.05,-0.18,0.09,0.11,0.06,-0.01,-0.16,0.35,-0.20,0.25]
O(theta) = [E/Eh, mu_x/(e bohr), mu_y/(e bohr), mu_z/(e bohr)]
```

The origin is fixed at `(0,0,0)`. At this point the field is
`(0.0007,-0.0004,0.0005) au`, energy is `-76.01089796546204 Eh`, and the dipole
is `(-0.005166191480835,-0.001006161680377,0.877057491795415) e bohr`.
The orbital solve took 142 iterations with gradient norm `1.24e-14` or smaller.
The seed and response tolerances match A01: seed energy `1e-12 Eh`, density
`1e-11`, gradient `1e-10`, 150 SCF cycles; orbital tolerance `1e-10`, maximum
300; implicit tolerance `1e-10`, maximum 100, restart 40.

For `J=dO/dtheta` and `H=d²(E/Eh)/dtheta²`, the example evaluates `Jv`, `Jᵀw`
and `Hv` directly with JAX JVP/VJP/gradient transformations. The direction `v`
is a normalized NumPy seed-641 vector; `w` is normalized `(0.7,-0.2,0.3,0.4)`.
Exact vectors are archived. Dense 4×12 `J` and 12×12 `H` are constructed only
as small validation references. This avoids explicitly constructing these
dense observable tensors for the product calls; it makes no claim that every
internal SCF response operation is matrix-free.

**The plotted Hessian is with respect to scaled theta, not an unscaled
Cartesian Hessian.** Its field blocks also carry the 0.002 field scale. The
chosen geometry is not asserted to be a nuclear minimum, and no vibrational
frequencies or stability conclusion are inferred from this matrix.

| Identity or dense comparison | Maximum absolute error |
| --- | ---: |
| JVP versus Jv | 4.05e-15 |
| VJP versus Jᵀw | 1.22e-14 |
| HVP versus Hv | 1.39e-15 |
| w·(Jv) versus v·(Jᵀw) | 4.34e-16 |
| H versus Hᵀ | 2.51e-16 |

The script also checks `-dE/dtheta_field / 0.002 = mu` and the zero net
translation gradient of this neutral molecule. Native displaced endpoints
must remain stationary.

## Independent and finite-difference checks

The separate PySCF reference rebuilds and relaxes each molecular point. It
computes dipoles directly from the density and includes the nuclear field
term `-F dot sum_A Z_A R_A` in total energy. Its baseline energy differs by
`9.95e-14 Eh`, and its largest dipole difference is `4.66e-12 e bohr`.

| Dimensionless h | Native observable FD versus JVP | Native gradient FD versus HVP | Independent dense J FD error |
| ---: | ---: | ---: | ---: |
| 0.01 | 2.34e-10 | 3.94e-11 | 1.81e-9 |
| 0.003 | 2.96e-11 | 3.29e-12 | 2.52e-10 |
| 0.001 | 1.28e-12 | 2.68e-12 | 5.58e-10 |

The independent HVP uses four **energies**, with no GradSCF gradients or
Hessian and no PySCF AD, for each coordinate unit vector `ei`:

```
(Hv)i ≈ [E(theta+h(v+ei)) - E(theta+h(v-ei))
          - E(theta-h(v-ei)) + E(theta-h(v+ei))] / (4 h²)
```

| Dimensionless h | Independent four-energy HVP maximum error |
| ---: | ---: |
| 0.02 | 8.05e-10 |
| 0.01 | 1.10e-9 |
| 0.005 | 5.77e-9 |

The smallest steps show an error increase rather than monotonic improvement,
consistent with finite precision and converged-SCF energy cancellation. All
steps are retained; the independent HVP acceptance is `atol=2e-6` and
`rtol=2e-5` in the scaled coordinates. These measurements establish the stated
first derivatives and energy second derivatives, not third nuclear derivatives.

## Execution and provenance

The isolated snapshot is
`/home/yjiao/gradscf_checks/ad-showcase-a01-a02-20261015`. It uses the same frozen
MolecularEnergy v7 source as A01, Python source-tree SHA256
`ca37f375d0e4cc673d1ca2e5b50f189b9047ad8bce56e00f7f397156fac3e43f`.
The source and native binary hashes are archived. No later stationary-value
energy optimization was merged into this snapshot. Hardware/software: Intel
Xeon Platinum 8457C, Python 3.11.15, JAX 0.10.0, NumPy 2.4.4, PySCF 2.13.0,
jax-xc 0.0.9, SciPy 1.17.1; persistent compilation cache disabled.

```bash
export CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
export JAX_ENABLE_COMPILATION_CACHE=false OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH=src
PYTHON=/home/yjiao/opt/miniconda3/envs/jax_scf/bin/python
timeout 7200 "$PYTHON" examples/response/molecular_ad_products.py
timeout 1800 "$PYTHON" examples/response/compare_molecular_ad_products_pyscf.py
timeout 300 "$PYTHON" examples/response/plot_molecular_ad_products.py
```

All commands exited zero; remote logs are `A02_native.log`, `A02_pyscf.log` and
`A02_plot.log`. The independent calculation took 44.477 s. Recorded first-call
times were JVP 25.130 s, VJP preparation 16.948 s, pullback 5.465 s, and HVP
19.950 s; the dense validation matrices took 38.472 s and 40.228 s. These phases
share one process and prior setup, so they are not isolated cold comparisons.
Peak RSS was 2030.7 MiB after products and 2452.8 MiB after validation, both
cumulative. This is a small validation example, not a scaling benchmark.

Scientific JSON remains unchanged after measurement. Trailing measured-output
comments were added to local scripts afterward; execution hashes identify the
unchanged remote versions used for the scientific run.
