# P03 — Raw LiH Gamma force constants

**Gamma gate complete.** The full raw FFT31 Hessian agrees with independent,
reconverged PySCF gradient differences at two displacement sizes. Joint
cell/atom rotation and same-source direct/GMRES controls also pass. Refinement
through FFT71 passes all four original raw-matrix/acoustic convergence gates,
without ASR projection. The unsuccessful FFT51 and FFT61 refinements remain
in the archive with their failed gates and exit codes.

This capsule develops the Gamma part of P03. It does not yet provide a
validated finite-q dispersion, crystal IR, LO–TO correction or material
stability conclusion.

## Fixed model and units

The geometry is exactly the P02 finite-model scalar minimum,
`a = 7.780032276700287 bohr`, with rocksalt FCC row lattice
`A = (a/2) [[0,1,1],[1,0,1],[1,1,0]]` and fractional Li/H coordinates
`(0,0,0)` and `(1/2,1/2,1/2)`. The calculation uses PBE, Cartesian GTH-SZV,
GTH-Pade potentials with Li q3/H q1, **four explicit electrons**, a full k2³
mesh, zero temperature and fixed integer occupations. FFT31³ is the initial
grid. The lattice and atom positions stay fixed during the FFT refinement;
they are not reoptimized between grids.

The sampled gap must exceed `1e-3 Eh`, the SCF occupied/virtual Fock residual
must be below `2e-8`, and the AO overlap must meet the same conditioning gate
as P01/P02. The existing SCF controls are `conv_tol=1e-12`,
`conv_tol_density=1e-11`, `conv_tol_grad=1e-10`, at most 150 cycles and damping
0.1. All calculations use float64 on `jy_gpu_1`; no local numerical tests run.

Masses are the declared atomic isotope masses, `7.0160034366 u` for ⁷Li and
`1.00782503223 u` for ¹H. They are inputs to mass weighting, not parameters
of the electronic SCF. Sources: [NIST lithium](https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl?ele=Li)
and [NIST hydrogen](https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl?ele=H).

## Existing response path

The example calls `dfpt.atomic_problem` and `Phonons.kernel(asr=False)`.
There is no new physical or numerical solver. In the existing stationary chart,

`R(x, R_nuc) = 0`,

the response to a Cartesian displacement `u` solves

`R_x x^(1) = -R_R u`.

Differentiating the existing Pulay-correct stationary force then gives

`Phi u = -(F_x x^(1) + F_R u)`.

For this model the real Hermitian density chart has 144 variables. The shared
direct linear solver is explicitly bounded by `max_dense=256`, with
`rtol=1e-10` and `atol=1e-12`. Six unit Cartesian displacements produce the raw
6×6 Gamma force-constant matrix in Eh/bohr². The raw matrix, its Hermiticity
defect, the force sum, the torque and `Phi @ T` are retained, where `T` contains
the three uniform Cartesian translations.

The dynamical matrix is `D = M^(-1/2) Phi M^(-1/2)`, with masses converted
from u to electron masses. Mode vectors are orthonormal in mass-weighted
coordinates. The correct translation vectors in this space are `sqrt(M) T`;
they are not plain equal-amplitude vectors for unequal Li/H masses. Frequencies
are signed `sign(lambda) sqrt(abs(lambda))` in Hartree and then converted to
cm⁻¹. Negative values remain visible. Acoustic and optical groups in the grid
check are identified by overlap with the mass-weighted translation subspace,
not by assuming that the lowest three sorted frequencies are acoustic.

At fixed lattice, rotating the atoms alone is not a periodic rigid-motion
symmetry. The separate rotation check rotates Cartesian atoms, row lattice,
absolute k points and the fractional FFT grid together. It tests energy
invariance and covariance of one generic raw force-constant action. It does
not impose a molecular rotation sum rule on the Gamma atomic block.

## Numerical evidence

The one-column probe passed in 73.32 s with 11.61 GiB peak RSS. The full raw
Gamma calculation passed in 118.34 s with 13.34 GiB peak RSS. Both ran below
a 32 GiB process budget and a 30-minute timeout. Times include reference SCF
and compilation, and are not warmed derivative-only timings.

The raw FFT31 frequencies are approximately

`74.3443, 74.3574, 74.3574, 441.6007, 441.6007, 441.6240 cm^-1`.

The raw ASR Frobenius residual is `3.035375e-3 Eh/bohr²`. This is a substantial
translation defect despite essentially zero atomic forces at the symmetric
minimum. First-derivative stationarity therefore does not establish convergence
of the acoustic curvature. Independent same-grid gradient FD reproduces this
large translation defect; subsequent refinement tests its grid dependence.

The joint rotation check passed in 67.10 s with 11.46 GiB peak RSS. Its energy
difference is zero at printed float64 precision, and the largest rotated-action
error is `2.49e-15 Eh/bohr²`. The direction mixes both atoms and all three
Cartesian axes. This check does not repair the raw translation defect.

## Independent reference and fixed acceptance gates

The independent script imports PySCF only. It rebuilds the actual stored basis
and pseudopotential parameters, uses uniform grids and FFTDF with the same
mesh, preserves the ordered fractional k points, and reconverges SCF at every
displacement. It differences **analytic nuclear gradients**, not total energies,
at `h=3e-4` and `1e-4 bohr`. The 12 column/step pairs retain endpoint energies,
electronic residuals, gaps and timings. A displaced cold/warm calculation checks
the reference seed. Checkpoints permit a bounded job to resume completed pairs.

The same-grid gates are `2e-6 Eh` for energy, `3e-6 Eh/bohr` for the reference
gradient, `5e-6 Eh/bohr²` for each full raw FD matrix, and `5e-6 Eh/bohr²` for
the two-step matrix change. These do not require the finite-grid raw ASR defect
to vanish; that is tested separately by refinement.

All 12 column/step pairs and the displaced cold/warm check passed: 26 independent
SCF/gradient evaluations took 4309.89 s with 0.482 GiB peak process RSS. The
energy error is `5.38e-8 Eh`; the reference gradient error is
`1.81e-14 Eh/bohr`. The maximum raw Hessian errors are `1.9430e-6` and
`1.9645e-6 Eh/bohr²` at the larger and smaller steps, respectively. The two
FD matrices differ by only `2.15e-8 Eh/bohr²`. Their raw ASR norms are
`3.03875e-3` and `3.03879e-3 Eh/bohr²`, close to the native `3.03538e-3`.
The cold/warm displaced energy difference is `1.78e-15 Eh` and relative
density difference `1.19e-12`. No independent PySCF Hessian at FFT71 is claimed.

The following FFT refinement gates were fixed **before** FFT41/51 execution:

| Quantity | Required limit |
| --- | ---: |
| Largest component of `Phi51 - Phi41` | 2e-6 Eh/bohr² |
| Largest optical-frequency increment, FFT41 → FFT51 | 1 cm⁻¹ |
| FFT51 raw `norm(Phi @ T)` | 1e-6 Eh/bohr² |
| FFT51 largest absolute acoustic frequency, without ASR | 1 cm⁻¹ |

The observed FFT41 → FFT51 force-constant increment is `1.015397e-4 Eh/bohr²`
and the optical-frequency increment is `0.0286863 cm⁻¹`. FFT51 has raw ASR
`1.196317e-5 Eh/bohr²` and maximum raw acoustic frequency `4.758888 cm⁻¹`.
Only the optical increment passes. The failed report is saved as
`grid_convergence.json` and its gate job has exit 1. The rendering job separately
has exit 0; a successfully generated figure does not make the numerical gates
pass. The [grid figure](lih_gamma_grid_convergence.png) explicitly marks the
three failures and keeps the signed acoustic values visible.

| FFT mesh / solver | Raw acoustic frequencies / cm⁻¹ | Raw ASR / Eh bohr⁻² | Peak RSS / GiB |
| --- | --- | ---: | ---: |
| 31³ / direct | +74.3443, +74.3574, +74.3574 | 3.035375e-3 | 13.34 |
| 41³ / direct | +18.6791, +18.6791, +18.6791 | 1.867722e-4 | 27.11 |
| 51³ / direct | +4.7589, +4.7589, +4.7589 | 1.196317e-5 | 45.34 |
| 51³ / GMRES | +4.7589, +4.7589, +4.7589 | 1.196317e-5 | 14.66 |
| 61³ / GMRES | +1.1510, +1.1510, +1.1510 | 7.218558e-7 | 21.60 |
| 71³ / GMRES | +0.2313, +0.2313, +0.2313 | 3.596612e-8 | 29.81 |

FFT41 and FFT51 use separate fresh processes, at most 48 and 64 GiB RSS,
respectively, after checking node memory. An exceeded resource limit or a failed
numerical gate is retained as a failure. Neither thresholds nor raw modes are
silently modified. This capsule contains Gamma calculations only.

Matching basis, pseudopotential, k points and FFT dimensions does not make the
two packages' one-electron quadratures identical. The GradSCF periodic builder
forms overlap and kinetic matrices from a finite-G Fourier sum; the installed
PySCF k-point overlap calls Gaussian lattice integrals. This is a confirmed
implementation distinction, not a decomposition of the observed Hessian
discrepancy. The native/reference difference stays near `1.94e-6 Eh/bohr²`
at both FD steps, so it is not described as an O(h²) FD-truncation trend.
The FD step change and the code-to-code discrepancy are separate quantities.

The direct/GMRES control changes only the shared solver method in fresh
processes at FFT31. Both use `rtol=1e-10`, `atol=1e-12`, `restart=40`,
`maxiter=100` and `max_dense=256`; restart is inactive for the direct solve.
The methods must agree within `1e-8 Eh/bohr²` in the full raw matrix,
`1e-4 cm⁻¹` in signed frequencies and `1e-8 Eh/bohr²` in raw ASR. Each method
must pass its actual linear-residual checks. The new direct result must also
reproduce the original FFT31 matrix within `1e-10 Eh/bohr²`. Production
solvers and physical formulas are unchanged.

This control passed: the largest raw-matrix difference is
`4.73e-15 Eh/bohr²`, the largest signed-frequency difference is
`9.23e-11 cm⁻¹`, and the new direct result exactly reproduces the original
FFT31 matrix. The maximum true linear residual is `4.88e-17` for direct and
`8.53e-13` for GMRES; both satisfy their existing solver acceptance criteria.

| Grid / solver | Total cold wall time / s | Peak process RSS / GiB |
| --- | ---: | ---: |
| FFT31 / direct, paired source | 115.22 | 13.27 |
| FFT31 / GMRES, paired source | 120.19 | 6.72 |
| FFT51 / direct, archived source | 260.96 | 45.34 |
| FFT51 / GMRES bridge | 144.12 | 14.66 |
| FFT61 / GMRES | 176.91 | 21.60 |
| FFT71 / GMRES | 216.47 | 29.81 |

These are fresh-process observations including SCF and compilation on a shared
node. OpenMP/BLAS settings of two do not cap all JAX threads. They establish a
measured memory reduction for these cases, not a general timing or scaling law.
The direct solve forms the 144-column response Jacobian; shared GMRES uses its
action without materializing that matrix. No replacement solver was written.

At FFT51 the bridge differs from direct by `9.38e-16 Eh/bohr²` in the full
matrix and `3.19e-10 cm⁻¹` in signed frequencies. The production source and
physical settings match. Its example adds an explicit solver argument and
metadata; the old direct example's actual configuration is verified from its
saved source hash and `LINEAR` constant. The prior FFT31 control checks that
this parameterization preserves the original direct calculation.

The adjacent-grid comparisons use GMRES throughout, with identical response
settings. FFT51 → FFT61 still failed: the matrix increment was
`6.4834e-6 Eh/bohr²` and the largest acoustic frequency `1.15097 cm⁻¹`.
`grid_extension_to61.json` retains those failures. The next refinement passed:

| FFT61 → FFT71 quantity | Observed | Unchanged gate |
| --- | ---: | ---: |
| Largest raw force-constant increment | 4.02835e-7 Eh/bohr² | 2e-6 Eh/bohr² |
| Largest optical-frequency increment | 1.75260e-4 cm⁻¹ | 1 cm⁻¹ |
| FFT71 raw ASR Frobenius norm | 3.59661e-8 Eh/bohr² | 1e-6 Eh/bohr² |
| FFT71 largest absolute acoustic frequency | 0.231253 cm⁻¹ | 1 cm⁻¹ |

The final optical triplet is approximately `441.4103 cm⁻¹`. The true maximum
linear residual is `1.81e-13`. The [final convergence figure](lih_gamma_grid_extension_to71.png)
and `grid_extension_to71.json` use these measured values. FFT71 ran under the
64 GiB / 3600 s process limit. The lattice remains the FFT31 P02 minimum;
neither its optimization nor the Gamma Hessian's k/basis convergence was
repeated at FFT71. This result is a validated finite-model Gamma calculation,
not a complete material phonon spectrum.

## Scripts and provenance

- [Native Gamma workflow](../../../examples/pbc/lih_gamma_phonons.py).
- [Independent gradient FD](../../../examples/pbc/compare_lih_gamma_phonons.py).
- [Joint rotation covariance](../../../examples/pbc/lih_gamma_rotation.py).
- [Predeclared FFT gates](../../../examples/pbc/compare_lih_gamma_grids.py).
- [Same-source direct/GMRES checks](../../../examples/pbc/compare_lih_gamma_solvers.py).
- [Saved-data figures](../../../examples/pbc/plot_lih_gamma_phonons.py).

The original FFT31 scientific snapshot is
`/home/yjiao/gradscf_checks/ad-showcase-lih-gamma-20261016`, cloned from the
validated P01/P02 implicit-memory repair. Its frozen Python source hash is
`0de7c6c920f84563529fe04238cd8c92b625c3e05665bfe2fcccc278e0077470`;
the runtime contains the earlier native-integral library, not the unrelated
G08 higher-order build. This calculation uses the periodic Fourier/FFT builders.
The exact first-run example is retained in `artifacts/ad_showcase/provenance/P03_lih_gamma_phonons/verification/source_fft31`.
The current example adds explicit FFT/workspace arguments for subsequent
fixed-geometry checks and a shared linear-solver configuration; the original
remote snapshot is not overwritten. Direct FFT41/51 records are in
`ad-showcase-lih-gamma-grids-20261016`; the same-source solver comparison is in
`ad-showcase-lih-gamma-linear-20261016`, both under the same remote checks root.

`verification/` retains the exact executed example versions and job commands,
logs, exits and resource reports for all three snapshots. Python source,
native-library and runtime fingerprints are recorded in the three
`source_manifest_*.json` files. `archive_validation.json` checks eight raw
matrix records, independent references, input hashes, successful job exits and
the preserved numerical failures. `plot_validation.json` identifies the final
FFT71 convergence report; image rendering itself is not a scientific gate.
The environment was Python 3.11.15, JAX/JAXlib 0.10.0, NumPy 2.4.4,
SciPy 1.17.1 and PySCF 2.13.0 on an Intel Xeon Platinum 8457C host.

The public examples include measured trailing output comments added after the
runs. `verification/public_example_comments.json` verifies that these additions
leave each script's parsed Python code unchanged; saved fingerprints refer to
the executed copies, not to later editorial comments.

For example, after the paired-solver gate, the final native calculation was:

```python
from dataclasses import replace
from examples.pbc.lih_gamma_phonons import LINEAR, SETTINGS, run

run('pbe_szv_k2_fft71_gmres', dict(SETTINGS, mesh=[71, 71, 71]),
    max_grid=360000, max_pair_elements=210000000,
    linear=replace(LINEAR, method='gmres'))
```

`compare_lih_gamma_grids.run((51, 61, 71), method='gmres',
output_name='grid_extension_to71.json')` checks the last increment against the
original four limits. `plot_lih_gamma_phonons.main(grid_report=
'grid_extension_to71.json')` renders the saved-data figures. The independent
FFT31 FD and joint rotation controls are separate scripts. No production
solver or force formula was changed for this Gamma capsule.
