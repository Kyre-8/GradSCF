# A06 — Boys gradient and HVP in a fixed occupied subspace

The [archive wrapper](../../../examples/localization/archive_ethylene_boys.py)
runs the existing [ethylene example](../../../examples/localization/ethylene_boys.py)
and [independent PySCF comparison](../../../tests/comparisons/compare_ethylene_boys.py)
without replacing either implementation. The complete current run passed on
`jy_gpu_1` in 21.38 s. [Figure](boys_localization.png), [JSON](results.json),
[raw log](../../../artifacts/ad_showcase/provenance/A06_boys_localization/A06_native_pyscf.log), and [runtime/commands](../A03_A07_RUNTIME.md)
are available.

Native RHF/Cartesian cc-pVDZ supplies ethylene's eight occupied orbitals,
including both carbon cores. Geometry and this occupied space are then fixed.
The 28 dimensionless real rotation parameters define an antisymmetric K and
`U=exp(K)`. With `r_x` the occupied position matrices, the minimized loss is
`-sum_i,x [(Uᵀ r_x U)_ii]²`, in bohr². Position integrals and orbital centers
are in bohr. AD acts on the rotations; it does not include SCF or nuclear
derivatives in this localization stage.

Eight starts are compared before a local Newton-CG refinement using the AD
HVP. The canonical symmetry point has a nearly zero score and may be
stationary without being a localization minimum. The selected final score is
`25.574296815574 bohr²`, with local gradient norm `2.14e-9 bohr²`. Independent
PySCF starts from canonical/atomic and random orbitals, not from the final AD
orbitals. Its best score differs by `6.40e-14 bohr²`; all matched occupied
overlaps are one to printed precision. A default PySCF start alone reaches a
different score (`20.181293653848 bohr²`), which explains the use of multistart.
These observations do not establish a unique orbital gauge or global optimum.

| Check | Absolute discrepancy |
| --- | ---: |
| Nonzero-global-coordinate directional gradient versus FD / bohr² | 4.56e-10 |
| HVP versus gradient FD, max component / bohr² | 8.95e-10 |
| Analytic local-chart gradient versus AD / bohr² | 1.78e-15 |
| S-orthogonality Frobenius norm | 6.51e-15 |
| AO density change Frobenius norm | 1.37e-15 |

The FD step is `1e-5` in dimensionless rotation parameters. The local analytic
formula is checked in a fresh incremental chart and is not incorrectly compared
with derivatives of a nonzero global exponential chart. Orthogonality and
density invariance follow the occupied-space rotation and are verified directly.

Existing [ethylene/BODIPY animation scripts](../../../examples/localization/README.md)
remain available. They render these fixed-subspace orbital rotations; this
showcase adds a compact numerical plot and archives the current validation,
without making animation rendering a dependency of derivative checks.
