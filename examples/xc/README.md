# Exchange-correlation functionals

Run the examples from the repository root with `PYTHONPATH=src`. They use
GradSCF SCF, native molecular integrals, and JAX float64; they contain no CLI.

| Example | Purpose |
|---|---|
| [standard_recipes.py](standard_recipes.py) | Inspect complete recipes and run an explicitly assembled PBE0 functional. |
| [custom_energy.py](custom_energy.py) | Supply an external input schema and handwritten energy; obtain its potential and kernel by AD. |
| [density_matrix_mlp.py](../training/density_matrix_mlp.py) | Use an external four-layer density-matrix network through the same `custom_xc` port. |
| [external_functional.py](../training/external_functional.py) | Train an external model in fixed-density, explicit and implicit modes. |

`dft.custom_xc(spec)` returns a traditional recipe. With `input_fn` and
`energy_fn`, it returns the existing `dft.Functional`; no additional model
class or training engine is introduced. The callback must return a real scalar
**total** XC energy in Hartree, including any quadrature or baseline it needs.
Periodic callbacks return energy per primitive cell.

The [XC library reference](../../src/gradscf/dft/xc/README.md) describes units,
exact exchange, local-backend availability and method boundaries.
