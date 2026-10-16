# P01/P02 — Periodic LiH forces, stress and a cubic lattice minimum

The finite-model force/stress and cubic-lattice workflows are complete, including
independent PySCF checks, separate FFT/k sensitivity series, and a conditioned
larger-basis comparison. The result is an implementation and differentiation
example. It does not establish a basis-complete or experimentally accurate LiH
equilibrium structure. In particular, the basis comparison changes the sign of
the displaced H force, and the cubic minimum belongs to the specified SZV/k2³
model.

The public entry points are
[LiH force/stress](../../../examples/pbc/lih_forces_stress.py),
[independent PySCF checks](../../../examples/pbc/compare_lih_forces_stress.py),
[cubic lattice optimization](../../../examples/pbc/lih_lattice_optimization.py),
[independent lattice minimum](../../../examples/pbc/compare_lih_lattice_pyscf.py),
and [plotting](../../../examples/pbc/plot_lih_forces_stress.py). These are ordinary
Python scripts with explicit settings and no argument parser. They use the
existing periodic SCF, input builder and geometry derivative APIs.

The five public scripts now end with measured-output comments. Their numerical
ASTs are unchanged; before/after hashes and matching execution copies are
recorded in [the comment-only audit](verification/public_example_comments.json).
The final plotting source also matches the original `plot_validation.json`
execution hash exactly. No calculation was rerun for these comments.

## Model and units

The two-atom rocksalt primitive cell has row lattice
`A = (a/2) * [[0,1,1], [1,0,1], [1,1,0]]` and fractional Li/H sites
`(0,0,0)` and `(1/2,1/2,1/2)`. The volume is `a³/4`. The force example uses
`a=7.6 bohr` and displaces H by `(0.025,-0.015,0.010) bohr` so its atomic force
is nonzero. The lattice optimization restores the symmetric fractional sites.

GTH-Pade has actual pseudopotential charges **Li q3 and H q1** in this snapshot:
there are **four explicit electrons**, not two inferred from chemical valence.
The primary model pairs PBE with these declared potentials. The HF comparison
uses the same potential, orbital basis and sampling. All orbitals are Cartesian
Gaussians; GTH-SZV has three AOs in this cell. These settings define a numerical
example, not a recommended converged material calculation.

Full uniform k meshes have equal weights. The calculations use zero electronic
temperature, fixed integer spin populations `(2,2)`, and a positive sampled
indirect-gap gate of `1e-3 Eh`. The numerical AO metric must satisfy
`min(eig(S)) > 1e-8 * max(1,max(eig(S)))`. The raw force sum and stress tensor are
saved; no acoustic projection or force correction is applied.

HF uses `exxdiv='ewald'` consistently in both codes. Pure PBE has no exact
exchange contribution. The G=0 handling and Ewald exchange correction belong to
the declared finite-sampling model; see the
[PySCF periodic SCF conventions](https://pyscf.org/user/pbc/scf.html).

## Derivative convention and independent checks

For Cartesian coordinates `R` and row lattice `A`, a symmetric strain obeys
`R(e)=R@(I+e).T` and `A(e)=A@(I+e).T`. Fractional atom positions and fractional
k vectors stay fixed under strain. GradSCF differentiates the rebuilt periodic
inputs and the converged SCF state. The stress is tensile-positive:

`sigma = sym((dE/dR).T @ R + (dE/dA).T @ A) / V`,

with forces `F=-dE/dR` and pressure `p=-Tr(sigma)/3`. Energies are in Eh, forces
in Eh/bohr, and stress/pressure in Eh/bohr³. FFT topology and the local Ewald
image setup stay fixed for a derivative. A separate cell is built for every
finite-displacement/finite-strain reference and every EOS geometry.

Independent PySCF calculations match the actual basis/potential parameters,
Cartesian convention, uniform FFT grid and the **ordered, unwrapped fractional
k vectors**, rather than just matching mesh labels. They reconverge all SCF
degrees of freedom at displaced/strained geometries. One displaced cold/warm
comparison checks seed dependence on this branch.

The public PySCF PBE stress is also checked. Its overlap and kinetic strain
terms use an internal `1e-5` finite difference, so this is not a purely analytic
stress oracle. Its raw asymmetric tensor is retained before taking the symmetric
part. The separate multi-step energy FD is the additional reference. See the
[PySCF stress implementation](https://pyscf.org/_modules/pyscf/pbc/grad/krks_stress.html).
For HF, six symmetric Voigt strain directions reconstruct the stress from
independently reconverged energies; off-diagonal directions carry the factor of
two appropriate to the full symmetric tensor.

For the initial SZV / k2³ / FFT31³ force example, the independent errors are
`2.1383e-8 Eh` in energy, `2.6699e-9 Eh/bohr` in the largest force component and
`2.6882e-9 Eh/bohr³` in the largest symmetric stress component. Two FD steps,
`3e-4` and `1e-4`, cover an internal displacement, translation, isotropic strain,
deviatoric strain and shear. The largest smaller-step strain contraction error
is `9.021e-7 Eh`, below the predeclared `5e-6 Eh` gate. These are implementation
checks at matched requested settings; separate grid/basis/k differences remain
physical discretization questions. At the finer SZV FFT41³ and FFT51³ settings,
the largest independent force errors decrease to `1.16e-13` and
`5.94e-15 Eh/bohr`, respectively; the largest symmetric stress errors are
`1.98e-12` and `3.02e-12 Eh/bohr³`.

The complete PBE multi-step FD record belongs to the original `c792...` source
snapshot and is retained in `verification/original_results/`. After the memory
repair, the same E/F/stress and independent-code comparison were repeated, but
the entire PBE FD sequence was not repeated. The repaired source additionally
passed the six new and 21 existing response regressions below. These evidence
sets are kept separate in the archive.

The same Γ/SZV/FFT31³ HF model passes the independent energy/force/full-stress
checks with maximum errors `2.7349e-8 Eh`, `3.9579e-10 Eh/bohr`, and
`3.0708e-9 Eh/bohr³`. Its 16 FD records comprise two coordinate directions and
six Voigt strains at two steps each. The largest smaller-step directional
error is `3.3700e-7 Eh`, within the original `5e-6 Eh` strain gate. The displaced
cold/warm reference agrees to `1.78e-15 Eh` and relative density `1.15e-10`.

## Discretization and basis sensitivity

Only one numerical axis changes in each series. These checks use the displaced
cell at `a=7.6 bohr`; they are separate from the cubic optimization below.

| Final increment | Absolute ΔE / Eh | max absolute ΔF / Eh bohr⁻¹ | max absolute Δσ / Eh bohr⁻³ |
| --- | ---: | ---: | ---: |
| SZV, k2³: FFT41³ → FFT51³ | 6.2566e-8 | 4.0603e-8 | 3.0327e-9 |
| SZV, FFT31³: k6³ → k8³ | 2.7507e-5 | 1.8090e-6 | 6.7091e-7 |

The declared last-increment gates are `(1e-7, 1e-7, 1e-7)` for the FFT series
and `(1e-4, 2e-6, 2e-6)` for the k series in these respective units. Both pass.
Odd and even k meshes are retained rather than selectively removing oscillating
points. Independent PySCF E/F/stress checks cover Γ/k2³/k3³/k4³ and FFT31³/41³/51³
at their explicitly matched settings. The k5³/k6³/k8³ points are native
discretization checks only. No joint fine-FFT/fine-k limit, full Brillouin-zone
gap certification, or denser-than-k8 calculation is claimed.

The raw norm of the net atomic force decreases from `1.7798e-6` to `9.3225e-8`
to `7.3523e-9 Eh/bohr` along FFT31³/41³/51³ at k2³. These values remain in the
figures without an acoustic projection. They expose the residual translation
error of the finite integration grid.

For the basis comparison, both SZV and compact `gth-dzvp-molopt-sr` use k2³ and
FFT41³. SR-DZVP has 11 Cartesian AOs and passes the same overlap/gap gates.
Its energy is `−8.04222151795099 Eh`, compared with SZV
`−8.007234076140234 Eh`. The displaced H x force changes from
`−2.6786566e-4` to `+1.2091984e-3 Eh/bohr`. This large basis sensitivity and sign
change are retained explicitly. The positive electronic gap and successful
SCF do not establish mechanical stability. No full phonon or elastic matrix
was computed, and this two-basis comparison is not a basis-convergence sequence.
The separate SR FFT31³ → FFT41³ sensitivity is saved in `plot_validation.json`.
Its increments are `1.9082e-6 Eh`, `8.8573e-7 Eh/bohr`, and
`5.9122e-8 Eh/bohr³`; this is a sensitivity check rather than a claim that the
SR-DZVP FFT limit is reached. The independent SR/FFT41³ E/F/stress comparison
passes with maximum errors `2.22e-12 Eh`, `3.62e-12 Eh/bohr`, and
`3.62e-12 Eh/bohr³`. It uses the public PySCF property calculation, without a
second full multi-step FD sequence, and took 1424.03 s on the declared host.

## Cubic optimization

Seven independent volume points first established an actual energy bracket at
`a=7.6,7.9,8.2 bohr`, with positive sampled gaps. The native workflow repeats
that bracket and checks opposite signs of the endpoint slopes before using
SciPy `brentq` on

`dE/da = V * Tr(sigma) / a`.

Each scalar point rebuilds the cell and uses a fresh process, bounding retained
compilation state. The minimum is accepted only with a small slope, lower energy
than both `a±0.005 bohr` points, and agreement of energy- and stress-derived
positive curvature. This optimizes one cubic parameter with fixed fractional
sites, not cell shape or internal coordinates. It does not test the full elastic
or phonon stability. The reported scalar bulk modulus follows
`B=(a² E''(a)-2a E'(a))/(9V)`.

The native SZV / FFT31³ / k2³ minimum is `a=7.780032276700287 bohr`; its complete
record is `lattice_optimization.json`. Its slope is
`1.9317e-10 Eh/bohr`; the energy- and stress-derived curvatures are
`0.0227883351` and `0.0227883728 Eh/bohr²`. The positive cubic bulk modulus is
`0.00130181322 Eh/bohr³` for this scalar model.

Independent PySCF bounded minimization of cold-reconverged energies gives
`a=7.780045423140039 bohr`, differing by `1.31464e-5 bohr`. At the native minimum,
the independent common-geometry energy error is `5.3799e-8 Eh`, the curvature
error is `1.1206e-6 Eh/bohr²`, and the smaller-step scalar derivative error is
`2.6191e-7 Eh/bohr`. All pass the original gates recorded in
`pyscf_lattice_optimization.json`. This finite k2³ minimum remains separate from
a k-converged material prediction; the fixed-a stress convergence figures do
not establish convergence of the optimized lattice parameter.

## Basis and memory boundaries

Legacy GTH-DZVP was **rejected before geometry AD**. Its sampled gap is positive
(`0.10358 Eh`), but the raw AO metric has relative minimum eigenvalue
`6.04e-10` and worst per-k condition number `1.88e8`. The smallest-overlap mode
is dominated by the diffuse Li 3s AO at a sampled zone-edge point. The numerical
SCF reaches a small occupied/virtual Fock residual but cannot satisfy the stated
AO-density tolerance in 150 cycles. This is an acceptance limit in the selected
AO representation; it is not evidence of a metallic material. The failed logs
and unaccepted arrays are retained. Screening existing compact candidates with
the same metric gate accepted SR-DZVP: its relative overlap minimum is
`5.43e-6`, sampled gap `0.07886 Eh`, and SCF converges in 13 cycles.

The first small-cell force run exposed unused reverse-AD trajectory staging in
the implicit periodic SCF. The narrow repair preserves the physical residual,
final overlap response and explicit path. Six new and 21 existing tests passed;
same-case eager peak RSS changed from 79,423,680 to 5,860,376 KiB. Detailed paired
dataflow, timing and failure records are in
[the memory diagnostic](MEMORY_DIAGNOSTIC.md). Memory through the preliminary
SCF and derivative phase is a process high-water mark, not derivative-only RSS.

All numerical work uses `jy_gpu_1` / c20, CPU float64, two OpenMP/OpenBLAS threads
and disabled persistent compilation caching. Those environment settings do not
limit every JAX thread pool; these are shared-node process timings, not exclusive
two-core performance measurements. Every task is bounded below two hours. The
original source snapshot and its memory evidence remain frozen;
physical extensions use the repaired source hash
`0de7c6c920f84563529fe04238cd8c92b625c3e05665bfe2fcccc278e0077470`.

| Native displaced-cell case | Wall time / s | Process peak RSS / GiB |
| --- | ---: | ---: |
| SZV / k2³ / FFT31³, repaired facade | 98.42 | 5.49 |
| SZV / k2³ / FFT51³ | 100.19 | 13.69 |
| SZV / k6³ / FFT31³ | 139.23 | 47.76 |
| SZV / k8³ / FFT31³ | 216.78 | 107.86 |
| SR-DZVP / k2³ / FFT31³ | 102.51 | 10.28 |
| SR-DZVP / k2³ / FFT41³ | 106.18 | 19.19 |

These fresh-process times include preliminary SCF, reconvergence and AD; they
are not warmed derivative-only timings. The final k8 job stayed below its
120 GiB budget. SR31 and SR41 stayed below their 64 and 32 GiB budgets. The first
k8 scientific calculation completed, but its monitor failed while reading an
exiting process whose `/proc` status no longer contained `VmHWM`. The repaired
monitor passed normal-exit and over-budget toy checks, then the same k8 model
was repeated with exit 0. The first result and wrapper failure remain archived.
An initial toy-budget assertion also used GiB where MiB was intended; that
incorrect fixture and the corrected 32 MiB checks are separately retained.

## Figures and provenance

- `lih_convergence.png` / `.pdf`: E, H force and pressure along separate FFT,
  k and basis axes, with independent-code points where checked.
- `lih_convergence_increments.png` / `.pdf`: successive increments over all force
  components and the full stress tensor
  on log scales, including the declared final-increment gates.
- `lih_forces.png` / `.pdf`: matched Γ HF/PBE force components and raw net-force
  sensitivity to the FFT mesh.
- `lih_equation_of_state.png` / `.pdf`: native and independent finite-model
  energies and AD pressure around the scalar cubic minimum. Smooth curves are
  Hermite guides through saved native energies and their AD slopes; calculated
  points remain visible. No interpolated point enters a numerical gate.
- `convergence.csv` and `plot_validation.json`: plotted numbers, input hashes,
  accepted final increments, and the independent/native-only coverage split.
- `source_manifest.json`: frozen scientific source and runtime fingerprints.
  `verification/baseline_jobs/` and `verification/repaired_jobs/` retain bounded
  commands and exit records; excluded large runtime files remain on the named
  validation host. Failed candidate or wrapper jobs are not counted as passes.
- `archive_validation.json`: 12 accepted native displaced-cell records, nine
  independent property comparisons, nine native EOS points, and their source,
  input-file and final-job checks. These record counts are not pytest counts.

The CPU is Intel Xeon Platinum 8457C; this snapshot uses Python 3.11.15,
JAX/JAXlib 0.10.0 and PySCF 2.13.0. The saved source family predates later
unrelated showcase changes; its measurements should not be presented as a
whole-worktree integration test.
