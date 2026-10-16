# Moller--Plesset examples

[A04 showcase evidence](../../reproducibility/ad_showcase/A04_taylor_mp/README.md)
links the historical convergence report and figures, current small H6/water
Taylor-MP checks, and the retained resource failure of the new full MP30 attempt.
The current small checks do not relabel the historical MP30/atomic figures as
successful new remote runs.

Run scripts from the repository root with `PYTHONPATH=src`; numerical examples
enable JAX float64. Native calculations require a built CPU integral library.
PySCF is used only where an independent comparison is explicitly identified.

## High-order convergence and divergence

Start with [the reproduction report](CONVERGENCE.md), which includes settings,
measured results, literature sources, verification, and known discrepancies.

| Script | Calculation |
| --- | --- |
| [he2_convergence.py](he2_convergence.py) | Native He2(2+) RHF/UHF, RMP/UMP2--30, FCI, optimized barriers |
| [compare_he2_convergence.py](compare_he2_convergence.py) | Independent PySCF HF/FCI and RS coefficient comparison |
| [ne_f_convergence.py](ne_f_convergence.py) | Native Ne/F- HF/MP2/MP3 plus complete external MP2--25/FCI benchmark |
| [plot_convergence.py](plot_convergence.py) | Plot computed curves, literature comparisons and error measures |

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu python examples/mp/he2_convergence.py
PYTHONPATH=src JAX_PLATFORMS=cpu python examples/mp/compare_he2_convergence.py
PYTHONPATH=src JAX_PLATFORMS=cpu python examples/mp/ne_f_convergence.py
python examples/mp/plot_convergence.py
```

Selected figures are included in the repository, so they can be viewed without
rerunning the calculations. PNG previews, vector SVGs and PDFs are linked from
the report. New runs write their raw data and figures to
`artifacts/mp_convergence/`. The scripts include measured output comments and
have no command-line parser.

The atomic high-order curves use an external PySCF symmetry action with a
NumPy recurrence. They are not native GradSCF high-order or AD results.

## Other examples

| Topic | Entry point |
| --- | --- |
| Molecular MP interfaces | [molecular.py](molecular.py) |
| Parameter gradients | [implicit_gradient.py](implicit_gradient.py) |
| Residual/Taylor MP series | [taylor_series.py](taylor_series.py) |
| Water comparison | [compare_water_pyscf.py](compare_water_pyscf.py) |
| Multiple molecular references | [compare_molecules_pyscf.py](compare_molecules_pyscf.py), [validation report](MOLECULAR_VALIDATION.md) |
| Shared Taylor-node timings | [benchmark_taylor_graph.py](benchmark_taylor_graph.py) |
| Excitation-block timings | [benchmark_excitation_blocks.py](benchmark_excitation_blocks.py) |
