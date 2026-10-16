# Molecular response and AD products

The examples are ordinary Python scripts with editable settings, no command-line
interface. Numerical validation and figure generation run on `jy_gpu_1`.

| Example | Observable and derivative | Checked results |
| --- | --- | --- |
| [A01: explicit versus implicit](explicit_implicit_response.py) | H2 HF/PBE energy and dipole derivatives from the same seed and nonzero geometry/field point | [Parity, multistep PySCF FD, separate phase timings and process memory](../../reproducibility/ad_showcase/A01_explicit_implicit/README.md) |
| [A02: molecular products](molecular_ad_products.py) | Water RHF/6-31G* JVP, VJP and energy HVP in 12 scaled geometry/field parameters | [Dense references, adjoint identity and independent energy/dipole FD](../../reproducibility/ad_showcase/A02_molecular_products/README.md) |
| [R01/R02: water electric response](water_electric_response.py) | Total permanent dipole and static polarizability | [Field conventions and independent reference](../../reproducibility/ad_showcase/R01_R02_water_electric/README.md) |

`water_electric_response.py` computes the total permanent dipole and static
polarizability by differentiating a single relaxed energy E(R,F), including
the nuclear field interaction. `compare_water_electric_pyscf.py` is an
independent finite-field reference and `plot_water_electric.py` plots saved data.
Run the G04 RHF geometry optimization first; all numerical steps run on
`jy_gpu_1`. See [results and units](../../reproducibility/ad_showcase/R01_R02_water_electric/README.md).

The reusable `gradscf_tools.molecular_response.MolecularEnergy` adapter assembles
physical inputs and calls the existing orbital SCF and response solvers. Its
default mode is implicit; A01 also uses its explicit mode. Its derivative graph
includes the moving basis/overlap and,
for DFT, the moving quadrature. Each observable needs its own full-chain
validation; this RHF example does not establish every DFT higher derivative.
