# Small-system Maxwell and mixed electric response checks

This is a verification of a shared full implicit root, not a material Raman
spectrum or a new public spectroscopy facade. The joint-input kernel in
[electric.py](../../../src/gradscf/dfpt/electric.py) accepts
`(periodic_inputs, functional_parameters, electric_field)` in one fixed
occupied-frame chart with one fixed Berry-phase anchor. Geometry changes the
existing integral inputs. The original reference is certified separately
from the new target residual; an old density need not already solve a
displaced geometry before the solver is allowed to converge it.

The implementation reuses `solve_nonlinear` and its implicit root derivative.
The existing fixed-geometry field wrapper delegates to this kernel. No SCF,
linear solve, canonical eigenvector derivative or frozen-density force
expression is introduced. Taylor mode remains disabled for this model.

For the converged electric enthalpy `F(R,E)=E_KS-V E.P`, we check

```
dF/dE = -V P
d²F/dE_alpha dE_beta = -V chi_alpha,beta
d force_kappa,beta / dE_alpha = V dP_alpha/dR_kappa,beta = Z*_kappa,alpha,beta
```

Here force is **minus the derivative of the complete root energy** with
respect to Cartesian position. This includes the moving overlap, AO links,
ionic polarization and projector response. The zero-field atomic force
adapter is used only as an independent check at zero field; its zero-field
Fock expression is not substituted into finite-field forces. In particular,
`stationary_value` on raw constrained densities would omit necessary
constraint responses and is not used.

The mixed observable is a directional derivative `dchi/dt` along a supplied
geometry or model path. At fixed volume, a normalized mode direction
`dR_kappa,beta/dQ = e_kappa,beta/sqrt(M_kappa)` would give
`d epsilon_inf/dQ = 4*pi*dchi/dQ` and `d alpha_cell/dQ = V*dchi/dQ`.
The present H2 direction is one Cartesian displacement, not a normalized
normal mode; no mode intensity or Raman normalization is inferred from it.

## Measured gates

All tests ran on `jy_gpu_1` (hostname `c20`) using CPU float64, JAX 0.10.0,
Python 3.11.15, OMP/BLAS threads set to two, and no affinity change. These are
process timings on a shared node, not exclusive-core benchmarks.

| Check | Outcome | Test time | Peak RSS |
| --- | --- | ---: | ---: |
| Missing joint-input API | 2 expected failures before implementation | 1.20 s | — |
| Analytic variable-hopping dimer | 1 passed | 156.55 s | 2,543,620 KiB |
| Periodic H2 geometry/field/mixed response | 1 passed | 311.38 s | 4,538,980 KiB |
| Existing fixed-field wrapper regression | 1 passed | 44.34 s | see resource record |

The dimer changes the hopping `t` in `[[a,t],[t,-a]]`. Its exact valence
occupation `(1+a/sqrt(a²+t²))/2` gives independent closed-form polarization,
susceptibility and mixed derivatives. The mixed result is
`dchi_xx/dt = 0.01518423088709614`, with maximum analytic tensor error
`1.04e-17`. Maxwell and both enthalpy identities pass, together with two
reconverged-root finite differences at `3e-4` and `1e-4`. This synthetic
parameter is not a nuclear coordinate.

The physical H2 test uses RHF, the default GTH-SZV/GTH-Pade model, a cubic
`6 bohr` cell, FFT9³ and the full `(3,1,1)` k mesh. Coordinates in bohr are
`H (0.4,0.5,0.6)` and `H (1.8,0.8,0.9)`. The direction moves the second atom
along x at fixed cell. The actual force-field derivative is
`dF_Hx/dE_x = -0.10333382950068501 e` and the susceptibility derivative is
`dchi_xx/dR_Hx = 0.03560916661968866` per bohr.
The off-center, unoptimized FFT9 cell is deliberately coarse. These values
are derivative checks, not converged isolated-molecule effective charges or
susceptibilities.

All zero-field force, energy-field, energy-curvature and Maxwell comparisons
pass. For the mixed derivative, each displaced zero-field state is rebuilt
and cold-converged by the ordinary SCF, then its own electric response is
calculated in a fresh chart. Central differences at `3e-4` and `1e-4 bohr`
give errors `5.6990e-10` and `6.2734e-11`, respectively. Two separately
reconverged finite-field force differences also agree with the Born/Maxwell
value. This distinguishes input differentiation and complete implicit
response from a derivative evaluated at an unrelaxed reference density.

The test source is [test_electric_mixed.py](../../../tests/dfpt/test_electric_mixed.py).
The [manifest](source_manifest.json) and [verification records](verification/)
retain actual exits, commands and source hashes. This source stage adds the
small joint-input root wrapper; the FFT31 and FFT71 material records use
their separately frozen, earlier field source. No material mixed derivative,
finite-temperature/metallic Berry response, lattice derivative, or jet
support is established by these two small-system checks.
