# Parent BODIPY: fixed-subspace Boys localization

The [calculation](bodipy_boys.py), [animation](animate_bodipy_boys.py), and
[independent PySCF comparison](../../tests/comparisons/compare_bodipy_boys.py)
are separate ordinary Python scripts, without command-line argument parsers.

## Structure and reference calculation

- Unsubstituted parent BODIPY: C9H7BF2N2, neutral singlet, 98 electrons.
- Identity: [PubChem CID 25058173](https://pubchem.ncbi.nlm.nih.gov/compound/25058173),
  InChIKey `GUHHEAYOTAJBPT-UHFFFAOYSA-N`.
- The explicit coordinates in the calculation are a fixed demonstration
  geometry, generated with RDKit 2025.03.3 (ETKDGv3 seed 11, followed by
  converged UFF minimization). They are not an HF-optimized geometry.
  Connectivity was checked against the PubChem 2D SDF; no PubChem 3D structure
  was available from the queried record.
- GradSCF native shell-direct RHF, Cartesian cc-pVDZ, float64 on CPU.
  There are 245 AOs, 49 occupied orbitals, and 1176 independent real rotations.
- All occupied orbitals, including core orbitals, participate in localization.

This example uses exact shell-direct J/K with the default zero screening
threshold. It does not use density fitting and does not allocate a stored ERI
tensor. For comparison, the *data payload alone* for 245 AOs would be 28.82 GB
in uncompressed s1, 7.26 GB in s4, or 3.63 GB in s8 (decimal GB, float64).
These storage sizes are not peak process-memory estimates.

## Workflow

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 python examples/localization/bodipy_boys.py
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 python tests/comparisons/compare_bodipy_boys.py
PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 python examples/localization/animate_bodipy_boys.py
python examples/localization/combine_bodipy_boys.py
python examples/localization/animate_bodipy_boys_rotation.py
```

Build GradSCF's native integrals before the first command. The animation needs
Matplotlib >= 3.7, Pillow and a scikit-image build compatible with the installed
NumPy. PySCF is used only in the independent comparison.

RHF is solved once. Boys optimization thereafter consumes the fixed occupied
orbitals and three position-integral matrices. JAX supplies rotation gradients
and HVPs. L-BFGS is followed by Newton-CG in a recentered rotation chart.
When the large absolute score prevents further energy-based refinement from
resolving the gradient, the final stage minimizes `0.5 * ||g||^2`, whose
gradient is `H @ g`, using the AD HVP. That stage stops as soon as the physical
local rotation-gradient norm is below `1e-7`.

The calculation writes `artifacts/bodipy_boys/trajectory.npz`. Rendering and
comparison read this fixed snapshot; neither reruns SCF or AD localization.
The movie has one initial frame and one frame per accepted optimizer update.
Trial line-search evaluations are not frames. Each page retains fixed orbital
labels, camera and isovalue; no temporal interpolation or frame subsampling is
used. The phase colors denote the sign of the real orbital amplitude.

The [montage script](combine_bodipy_boys.py) combines the seven page movies into
`artifacts/bodipy_boys/bodipy_boys_all.gif`, a synchronized 7-by-7 overview of all
49 occupied orbitals. It preserves every frame and its duration, uses a common
molecular scale, and retains one shared convergence plot. Initial and final
PNG previews are saved alongside the movie. This step only assembles existing
images; no electronic-structure calculation or isosurface rendering is repeated.

The [rotation animation](animate_bodipy_boys_rotation.py) also reads this snapshot
and the overview movie. It writes `bodipy_boys_rotation.gif` and
`bodipy_boys_all_with_rotation.gif` in the same artifact directory. The heatmap
shows the cumulative matrix `U` in `C_current = C_HF @ U`: rows label original
HF occupied orbitals, and columns label the current orbitals shown in the movie.
Its signed color scale is fixed to `[-1, 1]`. Orbital signs and labels are never
rematched between frames. The displayed step change is the Frobenius norm of
`U[k] - U[k-1]`, not a gradient or a generator. Frame zero includes the selected
small random initial rotation, so its matrix is close to, but not exactly, the
identity. Both animations preserve the overview's frame durations.

## Verified example

CPU, JAX 0.8.1, float64, fixed coordinates as listed in the script:

| Quantity | Value |
| --- | ---: |
| RHF total energy / Hartree | -677.535694988289 |
| Initial canonical-orbital Boys score / Bohr^2 | 546.452289644513 |
| Final Boys score / Bohr^2 | 1137.126348861702 |
| Final local gradient norm | 9.964e-8 |
| S-orthogonality error, Frobenius norm | 4.61e-14 |
| RHF density change, Frobenius norm | 1.02e-14 |
| AD score minus best independent PySCF score / Bohr^2 | -4.55e-13 |
| Saved frames | 142 |

The trajectory consists of one initial state, 100 L-BFGS updates, 18 Newton-CG
updates and 23 stationarity-refinement updates. The independent PySCF comparison
uses the same GradSCF HF orbitals and its own initial rotations; it does not
initialize from the AD-localized answer or run PySCF SCF. Its default initial
guess reached a different local minimum, so both default and multistart scores
are printed. Matching accounts for column permutations and signs.

The calculation checks the analytic local-chart gradient, a finite-difference
directional derivative in nonzero global coordinates, and an HVP against finite
differences of AD gradients. A local analytic orbital-gradient formula must not
be substituted for the derivative of nonzero global exponential coordinates.
No derivative through HF, nuclear-coordinate response, or global-optimum
guarantee is implied by this example.
