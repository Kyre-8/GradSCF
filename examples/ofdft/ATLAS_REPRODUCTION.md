# ATLAS-style TF + lambda vW solid reproduction

[View the figure gallery](FIGURES.md) — PNG/PDF exports and plotting instructions.

## Scope

This reproduces the **TF+lambda*vW cross-code solid calculation described in section 3.2** of [arXiv:1507.07373](https://arxiv.org/abs/1507.07373), for fcc Al and hcp Mg with lambda=1, 1/5, 1/9. It is not a reproduction of the WGC Table 1 energies, equilibrium volumes or bulk moduli. The subsequent **WGC–KS figure comparison** is documented separately in [WGC_REPRODUCTION.md](WGC_REPRODUCTION.md). ATLAS finite-difference stencils and the paper's random Mg/alloy structures remain outside these examples.

Both implementations are run independently with the same cell, OEPP file, grid and PZ-LDA. No GradSCF density, potential, derivative or energy is passed to the DFTpy optimizer. DFTpy uses its own TF, vW, Hartree, PZ-LDA, local potential, Ewald and L-BFGS routines. GradSCF uses a pure JAX PZ-LDA energy callback, tested against PySCF/libxc for both energy and potential. The dependency on PySCF is only in that optional unit test.

## Structural assumptions and inputs

- fcc Al: one-atom primitive cell, 18.435 Angstrom^3/atom, conventional lattice parameter 4.193413713305653 Angstrom.
- hcp Mg: two-atom hexagonal cell, 22.225 Angstrom^3/atom, a=3.1558686739233854 Angstrom, c=5.153511964230722 Angstrom. **Ideal c/a=sqrt(8/3) is an assumption**, because Table 1 does not supply c/a.
- The volumes are the Table 1 OF reference volumes. They are held fixed, not reoptimized for each semilocal KEDF.
- Valence electrons: Al=3, Mg=2. The density is unpolarized; OFDFT does not need integer KS orbital occupations.
- Author-supplied OEPP assets come from DFTpy dev commit `fbc47b4e1df3def1a2206e58c5a1a1a4d51db1d0`, `examples/DATA/Al_lda.oe01.recpot` and `Mg_lda.oe01.recpot`. SHA256 checksums and source paths are in `data/manifest.json`, with the upstream license in `data/LICENSE.DFTpy`.
- The Al file header explicitly lists radius 2.2 Bohr, consistent with the paper. Exact binary identity with the original 2015 calculation inputs cannot be established; Mg's bundled header does not expose every generation setting. These limitations prevent calling this an exact reproduction of unpublished original input files.
- RECPOT values are converted from eV Angstrom^3 to Ha Bohr^3 and inverse-Angstrom to inverse-Bohr. Cubic interpolation excludes the special finite G=0 core correction, which is retained explicitly. A common odd FFT mesh removes Nyquist ambiguity. Ewald uses valence charges and a neutral background convention.

## Scripts

- `atlas_gradscf.py`: short PySCF-example-style GradSCF calculation, plain settings at the top, no CLI.
- `atlas_dftpy.py`: separate DFTpy calculation using its own native PZ-LDA implementation.
- `atlas_inputs.py`: shared physical structures and file paths; independent eager GradSCF RECPOT input assembly and pure JAX PZ-LDA callback.
- `atlas_compare.py`: paired calculations, electron-number/potential/Ewald checks and mesh convergence; writes `atlas_results.json`.

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu python examples/ofdft/atlas_gradscf.py
PYTHONPATH=src:/path/to/DFTpy/src JAX_PLATFORMS=cpu python examples/ofdft/atlas_dftpy.py
PYTHONPATH=src:/path/to/DFTpy/src JAX_PLATFORMS=cpu python examples/ofdft/atlas_compare.py
```

ASE and SciPy are needed for these standalone examples. The GradSCF calculation does not import DFTpy. DFTpy imports shared structure/file helpers but performs no GradSCF calculation. jax-xc is not required for this explicitly selected PZ-LDA callback.

## Measured results

2026-09-27, Apple M4 Pro/arm64, CPU, float64, Python 3.12.2, JAX 0.8.1, NumPy 2.3.4. GradSCF stationarity tolerance: 1e-8. DFTpy energy tolerance: 1e-12 over three checks, with an independent constrained-residual gate of 5e-5. Different convergence criteria mean the reported runtimes are not a performance ranking.

At the 0.18 Angstrom target spacing (Al 17^3, Mg 19x19x29):

| System | lambda | GradSCF / Ha per atom | DFTpy / Ha per atom | Absolute difference / Ha per atom |
|---|---:|---:|---:|---:|
| Al | 1 | -2.072264566661 | -2.072264566660 | 3.157e-13 |
| Al | 1/5 | -2.152798390663 | -2.152798390663 | 2.278e-13 |
| Al | 1/9 | -2.187611287981 | -2.187611287981 | 2.327e-13 |
| Mg | 1 | -0.899530058755 | -0.899530058755 | 3.287e-13 |
| Mg | 1/5 | -0.933515515461 | -0.933515515461 | 1.245e-13 |
| Mg | 1/9 | -0.948322546023 | -0.948322546023 | 1.177e-13 |

Across all 18 paired calculations:

- Maximum energy discrepancy: 1.317e-12 Ha/atom.
- Maximum relative density L2 discrepancy: 8.085e-07.
- Maximum local-potential discrepancy: 1.110e-15 Ha.
- Maximum ionic Ewald discrepancy: 7.550e-15 Ha per cell.
- Maximum GradSCF constrained residual: 9.624e-09; all runs converged.
- Maximum independently evaluated DFTpy constrained residual: 2.133e-05.

## Mesh check

Target spacings are upper bounds along each primitive lattice vector; the actual mesh sizes are odd ceilings. The two codes agree on each discretization, which alone does not establish grid convergence. Al was therefore refined beyond 0.18 Angstrom.

| System | lambda | Last refinement | Energy change / meV per atom |
|---|---:|---|---:|
| Al | 1 | 0.14 -> 0.1 Angstrom | 0.000075 |
| Al | 1/5 | 0.14 -> 0.1 Angstrom | 0.000157 |
| Al | 1/9 | 0.14 -> 0.1 Angstrom | 0.000133 |
| Mg | 1 | 0.24 -> 0.18 Angstrom | 0.000124 |
| Mg | 1/5 | 0.24 -> 0.18 Angstrom | 0.009770 |
| Mg | 1/9 | 0.24 -> 0.18 Angstrom | 0.032611 |

The last refinement changes are all below 0.1 meV/atom. This is a successive-mesh check, not a rigorous complete-grid error bound. In particular, the earlier Al 0.24 -> 0.18 Angstrom changes reached 1.61 meV/atom for lambda=1/9, so that coarse comparison was not used alone as evidence of convergence.

Full lattice vectors, fractional coordinates, meshes, energies, densities' relative errors, electron counts, residuals and per-run wall times are retained in `atlas_results.json`. Example-output comments are included at the ends of the two standalone scripts.
