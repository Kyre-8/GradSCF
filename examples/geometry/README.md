# Geometry derivatives

The [complete AD example plan](../AD_SHOWCASE_PLAN.md) includes gradients,
optimization, Hessians, vibrations and excited-state nuclear response.

## Water RHF gradient

Run these ordinary scripts in order from the repository root:

```sh
PYTHONPATH=src python examples/geometry/water_rhf_gradient.py
PYTHONPATH=src python examples/geometry/compare_water_rhf_pyscf.py
PYTHONPATH=src python examples/geometry/plot_water_rhf_gradient.py
```

The first script requires JAX and GradSCF's built native integral backend.
The independent comparison requires PySCF; plotting requires matplotlib.
Use CPU float64 and the thread controls recorded with the measured result.
Project validation is performed remotely on `jy_gpu_1`.

The water geometry is fixed and deliberately not optimized. Both programs use
**Cartesian 6-31G***, including six Cartesian d functions on oxygen. Energies
are in Hartree, coordinates in Bohr and gradients in Hartree/Bohr.

The differentiated map is

```text
nuclear coordinates
    -> moving basis centers and native AO integrals S, h, ERI
    -> stationary restricted orbital solution C(R)
    -> total RHF energy E(R)
    -> jax.value_and_grad -> nuclear gradient
```

For occupied-virtual orbital coordinates kappa, the converged state satisfies
g(kappa, R) = dE/dkappa = 0. Implicit response solves

    (dg/dkappa) (dkappa/dR) = -dg/dR.

GradSCF's shared solver supplies the backward rule; the example does not
implement another response solver. Equal alpha/beta occupations in the shared
spatial-orbital solver produce RHF here. The initial RHF orbitals are a seed,
and are reorthonormalized and optimized for every displaced geometry.

At stationarity, the orbital variation in the first energy derivative cancels
in the appropriate constrained formulation. The derivative of the AO metric
still matters: moving the nuclei also moves their basis functions. This is
why varying only nuclear attraction or freezing the overlap is insufficient.
This first-derivative check does not establish that a second or mixed third
derivative through the whole calculation is supported.

Validation consists of central differences along all nine Cartesian
coordinates at three step sizes, an independent PySCF analytic gradient, net
force and torque. The scripts stop on nonconvergence/nonfinite derivatives or
failed tolerances. Result files and plots are written to
[`reproducibility/ad_showcase/G01_water_rhf_gradient`](../../reproducibility/ad_showcase/G01_water_rhf_gradient/).

## Moving-grid PBE gradient

Run `water_pbe_gradient.py`, `compare_water_pbe_pyscf.py`, then
`plot_water_pbe_gradient.py`. All steps belong on `jy_gpu_1`. The comparison
matches the level-3 quadrature and enables PySCF's full grid response; a separate
level-5 result measures quadrature error. See the [PBE results](../../reproducibility/ad_showcase/G01_water_pbe_gradient/README.md).

## Optimization, Hessian and isotope shifts

```sh
PYTHONPATH=src python examples/geometry/water_rhf_optimization.py
PYTHONPATH=src python examples/geometry/water_rhf_hessian.py
python examples/geometry/compare_water_hessian_pyscf.py
python examples/geometry/plot_water_modes.py
```

The Hessian uses native second integrals and full implicit SCF response.
Normal-mode analysis retains raw rigid-motion diagnostics before projection.
H2O and D2O share the electronic Hessian and use different isotope masses.
The plot and mode animation read only saved results. See [G04](../../reproducibility/ad_showcase/G04_water_rhf_optimization/README.md)
and [G05/V04](../../reproducibility/ad_showcase/G05_water_rhf_hessian/README.md).

## Open-shell forces

`open_shell_forces.py`, `compare_open_shell_forces_pyscf.py`, and
`plot_open_shell_forces.py` demonstrate OH/NO doublet UHF, UKS-PBE, UMP2 and
UCCSD forces. Complete small orbital Hessians and spin diagnostics describe
the SCF/reference determinant; correlated forces include shared amplitude
and orbital response. The independent comparison reconverges the same real
pi branch; post-HF gradients use original PySCF assembly with a verified SciPy
augmented physical-pi-gauge response oracle. See [G03 results and limits](../../reproducibility/ad_showcase/G03_open_shell_forces/README.md),
including the two native source families, full response residuals, finite-grid
negative curvature and rejected finer-grid/default-response references.
