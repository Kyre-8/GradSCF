# WGC OFDFT versus Kohn–Sham density

This extension changes the comparison to the physical method pair in Fig. 4
of [the ATLAS paper](https://arxiv.org/abs/1507.07373): WGC OFDFT versus KS-DFT.
The OF side uses GradSCF WGC99 (second-order Taylor, gamma=2.7); the KS side
uses an independent PySCF Gaussian KRKS calculation. Both use the same local
OEPP, cell, electron number and PZ-LDA. **The KS data are not CASTEP data.**

![WGC versus Gaussian KS](figures/atlas_density_contours_wgc_ks.png)

[Vector PDF](figures/atlas_density_contours_wgc_ks.pdf)

## What is and is not reproduced

The four sections are Al (001), Al (011), Mg (0001), and Mg (01-bar-1-0).
Coordinates are in Angstrom; labels denote 100 times the density in
electron/Bohr³. Red solid lines show WGC, blue dotted lines show KS. Identical
contour levels are applied to both densities; Mg levels are spaced by 0.05,
Al by 0.2. Labels retain the contour level precision.
The original periodic Fourier series is evaluated on 256² points per section;
no smoothing filter, fit to the paper image, or synthetic density is used.

Cells and OEPP provenance are listed in [ATLAS_REPRODUCTION.md](ATLAS_REPRODUCTION.md).
The Al cell has a=4.1934137133 Å; Mg uses a=3.1558686739 Å and the assumed ideal
c/a=sqrt(8/3). Both volumes come from the paper's Table 1 OF rows, rather than
an independently established Fig. 4 input deck. The original figure's precise
cell, Mg c/a, pseudopotential file identity, Taylor order, and CASTEP input
files have not been established. We therefore do **not** claim exact Fig. 4
contours, equilibrium volumes, bulk moduli or original CASTEP parity.

As a useful check, Al WGC gives -56.800828 eV/atom at 0.18 Å, consistent with
the rounded Table 1 value -56.801. Mg gives -24.650459 eV/atom, whereas the
paper reports -24.577. The approximately 0.07346 eV/atom Mg discrepancy remains
unresolved; it is much larger than this implementation's cross-code/grid
errors. The two-code agreement establishes the shared implemented functional
and inputs, not their identity with all historical inputs. We do not tune
the density or geometry to conceal this difference.

## WGC verification

`atlas_wgc_gradscf.py` is the short standalone GradSCF example;
`atlas_wgc_dftpy.py` is the separate DFTpy example. DFTpy dev has no native
WGC. `wgc_dftpy.py` supplies an **independent NumPy WGC nonlocal adapter** with
direct-eta ODE integration and an analytic potential. DFTpy supplies its own
TF, full vW, Hartree, PZ-LDA, OEPP, Ewald and optimizer. No GradSCF density or
JAX derivative is passed into that minimization.

The GradSCF kernel instead uses a log-eta quintic table and AD. Equations,
parameter conventions, domain limits, backward mechanism, and BSD attribution
are in [WGC.md](../../src/gradscf/ofdft/WGC.md). The unit tests include Lindhard
response, energy/potential checks against the independent adapter and a legacy
libKEDF fixture, plus finite-difference tests of implicit density response.

Measured on Apple M4 Pro, CPU float64, Python 3.12.2, JAX 0.8.1, NumPy 2.3.4.
DFTpy source: `fbc47b4e1df3def1a2206e58c5a1a1a4d51db1d0`.
GradSCF constrained tolerance: 1e-8. DFTpy energy tolerance: 1e-12 over three
checks, followed by an independently evaluated constrained-residual gate 5e-5.

| System | Grid | GradSCF / Ha per atom | Absolute code difference / Ha per atom | Density relative L2 |
|---|---|---:|---:|---:|
| Al | 17³ | -2.087391936818 | 4.20e-13 | 4.44e-7 |
| Al | 23³ | -2.087392047529 | 1.15e-12 | 5.81e-7 |
| Mg | 19×19×29 | -0.905887662700 | 6.38e-13 | 1.37e-6 |
| Mg | 23×23×37 | -0.905887663820 | 4.31e-13 | 7.49e-7 |

For 0.18 → 0.14 Å target spacing, the energy changes are 0.003013 meV/atom
(Al) and 0.0000305 meV/atom (Mg). These are successive-mesh observations, not
rigorous error bounds. GradSCF times including compilation were 26.16, 36.70,
33.93 and 50.85 seconds; different stopping rules preclude a speed ranking.
Complete values, electron numbers, constrained residuals and timings are in
`atlas_wgc_results.json`. `atlas_wgc_density_volumes.npz` contains paired 0.18 Å
densities and the lattice vectors used to draw the figure.

## Independent KS reference and uncertainty

PySCF 2.9.0 KRKS uses the same tabulated local OEPP plus analytic Gaussian
kinetic integrals, FFT Hartree, and LibXC `LDA_X,LDA_C_PZ`. No GTH or nonlocal
projectors are substituted. The Ewald constant is shared and independently
checked against PySCF. Explicit Fermi occupations enforce exactly 3 electrons
per Al cell and 4 per Mg cell, including Gamma-only odd-electron tests.

The core-free even-tempered Gaussian basis is fully specified in
`atlas_ks_pyscf.py`; level 3 has 58 functions/atom. Overlap eigenvalues below
1e-7 are removed. The final stored calculation uses Al 10³ and Mg 8³ k points,
sigma=0.005 Ha, a 0.18 Å target grid, SCF energy tolerance 1e-9 Ha and gradient
tolerance 1e-6. Its integrated electron numbers are within 2e-12 of their
targets. CPU times for these final calculations were 284.39 and 612.16 seconds.

| Density comparison | Al relative L2 | Mg relative L2 |
|---|---:|---:|
| WGC versus final KS | 0.70874% | 0.92440% |
| KS last k refinement (Al 8→10; Mg 6→8) | 0.57103% | 0.25119% |
| KS prior k refinement (Al 6→8; Mg 4→6) | 0.44440% | 1.47596% |
| sigma 0.005→0.0025 Ha, level 3, k6 | 0.02047% | 0.26599% |
| Basis refinement at k4, sigma=0.01 Ha (Al 3→4; Mg 2→3) | 0.32349% | 0.20445% |

These refinements are separate tests at the stated settings, not a combined
error bound. Al's k convergence is not monotonic. The final KS states satisfy
their SCF criteria but are **not certified complete-basis/zero-smearing/k-limit
references**. In particular, the WGC–KS discrepancy cannot all be attributed
to the KEDF when KS discretization changes remain comparable.

`atlas_ks_pyscf_results.json` describes the two plotted references;
`atlas_ks_pyscf_densities.npz` holds their densities. The convergence history,
including basis/k/smearing settings, electron counts and elapsed time, is in
`atlas_ks_convergence.json`. To repeat an individual history row, call
`calculate(symbol, spacing, kmesh, basis_level, sigma)` with that row's values.

## Run and plot

All examples use editable Python settings, without a CLI:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu python examples/ofdft/atlas_wgc_gradscf.py
PYTHONPATH=src:/path/to/DFTpy/src JAX_PLATFORMS=cpu \
  python examples/ofdft/atlas_wgc_dftpy.py
PYTHONPATH=src:/path/to/DFTpy/src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  python examples/ofdft/atlas_wgc_compare.py
OMP_NUM_THREADS=1 PYTHONPATH=src JAX_PLATFORMS=cpu \
  python examples/ofdft/atlas_ks_pyscf.py
MPLCONFIGDIR=/tmp/gradscf-mpl python examples/ofdft/plot_atlas_wgc.py
```

The KS main block repeats the recorded fine reference and may take about
15 minutes on this host. Plotting the stored arrays takes seconds and requires
only NumPy/Matplotlib. Reproducing the independent reference additionally needs
PySCF/ASE, or DFTpy for the OF cross-check. jax-xc is not used in these explicitly
selected pure-JAX PZ-LDA examples; its optional integration remains separate.
