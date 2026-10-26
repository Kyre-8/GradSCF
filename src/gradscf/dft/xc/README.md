# Exchange-correlation library

`gradscf.dft.xc` owns traditional functional definitions and their local JAX
energy graphs. Molecular and periodic SCF and response consumers assemble
their own AO/Bloch operators from these shared definitions. External formulas
and neural models enter through `gradscf.dft.custom_xc` and the existing
`Functional` energy interface.

## Public entry points

```python
from gradscf import dft
from gradscf.dft import xc

recipe = xc.parse_recipe('CAM-B3LYP')
metadata = xc.info('CAM-B3LYP')
local_backend_available = xc.available('CAM-B3LYP')
traditional = dft.custom_xc('.25*HF + .75*gga_x_pbe + gga_c_pbe')
```

A recipe records local terms, full/SR/LR exact exchange, omega, literature
references and any dispersion, nonlocal correlation or PT2 contribution.
`info` reports these requirements separately from local factory availability.
`available()` lists installed local channels for standard recipes. Neither
registration nor factory presence certifies every method or derivative order.

`eval_xc_energy_density`, `point_energy`, `grid_energy`,
`grid_value_and_grad`, `grid_hessian` and `eval_xc_response_tensor` evaluate
the **semilocal contribution**. They do not add HF exchange, dispersion, VV10
or perturbative correlation. Complete SCF consumers check extra requirements
and reject a recipe when a required physical term is unavailable.

## Standard names and conventions

The bundled catalog contains reference metadata exported from PySCF 2.13.0 /
Libxc 7.0.0 on 2026-10-19. It includes these families:

| Family | Representative registered names |
|---|---|
| LDA | `slater`, `svwn`, `lda,pw`, `lda,pz` |
| GGA | `pbe`, `pbesol`, `revpbe`, `rpbe`, `pw91`, `blyp`, `bp86`, `olyp`, `opbe`, HCTH variants |
| Tau meta-GGA | `tpss`, `revtpss`, `scan`, `rscan`, `r2scan`, `m06l`, `m11l`, `mn12l`, `mn15l`, MS variants |
| Global hybrids | `pbe0`, `b3lyp`, `b3lyp5`, `b3pw91`, `bhandhlyp`, B97 variants, `tpssh`, `scan0`, M05/M06 variants, `mn15` |
| Range-separated hybrids | `lc_wpbe`, `lrc_wpbeh`, `lc_blyp`, `cam_b3lyp`, `wb97`, `wb97x`, `hse03`, `hse06`, `mn12sx` |
| Recipes with additional terms | `b97d`, `wb97xd`, `wb97x_d3`, `b97m_v`, `wb97x_v`, `wb97m_v`, `scan_rvv10`, `b2plyp` |
| Laplacian-dependent metadata | `scanl`, `r2scanl`; not implemented by the rho/gradient/tau evaluator |

`xc.list_functional_recipes()` gives the full list, including aliases added to
the snapshot. Names are case-insensitive; common hyphen/underscore variants
are normalized. Signed expressions and scientific-notation coefficients are
accepted. A raw hybrid local component without a known complete recipe is
rejected as a complete functional because its HF coefficient is not inferred.

`lda` retains the existing Slater + PW convention. `b3lyp` retains the
VWN-RPA convention; `b3lyp5` explicitly selects the VWN5 variant. Programs
can use different defaults for the same short name, so comparisons must name
the actual components. The original `wb97xd` recipe includes its CHG dispersion
term in the molecular path; it is distinct from D3 and VV10 variants.

All quantities are in atomic units. The kinetic-energy density is
`tau_sigma = 1/2 sum_i occupation_i |grad(phi_i_sigma)|^2`. The local scalar
graph uses spin fields in this order:

| Kind | Point variables |
|---|---|
| LDA | `rho_a, rho_b` |
| GGA | `rho_a, rho_b, grad_a_x, grad_a_y, grad_a_z, grad_b_x, grad_b_y, grad_b_z` |
| MGGA | GGA fields followed by `tau_a, tau_b` |

The mixed invariant `sigma_ab = grad_a dot grad_b` is retained. Restricted
response is a linear equal-spin projection of the same graph. The MGGA AO
potential and kernel include the tau terms; a rho/gradient-only projection
would not represent the stated functional.

## Exact exchange and operator ownership

A complete recipe specifies

\[
E_{xc}=E_{xc}^{\rm local}+a_f E_x^{\rm HF}[1/r]
+a_s E_x^{\rm HF}[\operatorname{erfc}(\omega r)/r]
+a_l E_x^{\rm HF}[\operatorname{erf}(\omega r)/r]+E_{\rm extra}.
\]

For `RSH(omega, alpha, beta)`, the short-range HF fraction is `alpha+beta`
and the long-range fraction is `alpha`. Full-range, short-range and long-range
operators are distinct inputs to the method assembly. HSE03 also has different
HF and local screened-exchange omega values; its local component is not
assembled with the HSE06 parameters.

Omega is in inverse Bohr. AD of tunable runtime omega currently requires a
positive value. Fixed local omega=0 terms have a separate stable graph for
their energy and density/gradient derivatives; fully dynamic JIT derivatives
with respect to omega at zero are not a supported boundary.

Molecular methods use AO ERIs for the required operators. Periodic 3D methods
use screened Fourier Coulomb kernels and corresponding Ewald probe-charge
corrections, including finite G=0 limits. The SR and LR operators sum to the
full operator. FFT and k-point convergence still need to be established for
each material. Reduced-dimensional screened exchange and `vcut_sph` are
explicitly unsupported.

## Custom formulas and external neural networks

```python
functional = dft.custom_xc(
    input_fn=make_inputs,       # DensityInputs -> arbitrary input PyTree
    energy_fn=energy,           # (params, inputs) -> real scalar total Exc
    init_fn=initialize,         # optional (key, inputs) -> params
    name='my_xc',
)
mf = dft.RKS(mol, xc=functional, params=params).run()
```

This returns the existing `dft.Functional`, with no prescribed network
architecture. A handwritten formula, pure JAX network, Flax model or another
differentiable callback uses the same interface. Trainable parameters remain
explicit in `params`; avoid closing over mutable model weights. The callback
must return total XC energy in Hartree, or energy per primitive cell for PBC.
It includes any desired baseline and grid integration. A local energy per
particle or per volume must be integrated by the callback before returning.

`DensityInputs` evaluates quantities from the current SCF trial density:
spin and total AO density matrices, density, density gradients, tau and density
Laplacian; it also exposes AO values/derivatives, grid coordinates/weights,
overlap and atomic geometry. Molecular facades prepare missing AO second
derivatives when needed. Periodic inputs use their corresponding Bloch data
and normalized k weights. MO eigenvalues and frozen reference caches are not
provided as if they were functions of the current density.

For an arbitrary feature map `x(D)`, AD differentiates the entire energy
`Exc(params, x(D))`. The AO potential follows from

\[
\delta E_{xc}=\sum_\sigma\operatorname{Tr}(V_{xc,\sigma}\,\delta D_\sigma),
\qquad K_{xc}[\delta D]=\frac{d}{d\eta}V_{xc}[D+\eta\delta D]\big|_0.
\]

Periodic contractions include normalized k weights. Feature derivatives,
including second derivatives of nonlinear features, remain in this graph.
Grid points may be coupled: the custom response is the full AO Hessian
action, not a local-grid approximation. Nonfinite energy or derivatives are
reported; they are not converted to zero.

The existing training modes are unchanged:

| Mode | Density and derivative |
|---|---|
| `fixed_density` | Supplied density is fixed; differentiate the energy/model. |
| `explicit` | Solve self-consistently using the existing explicit derivative path. |
| `implicit` | Solve self-consistently; use the stationary/fixed-point adjoint for backward. |

Molecular RKS/UKS and adiabatic TD response bind the same model parameters.
Changing parameters invalidates a completed SCF; rerun it before response.
An existing TD object reads current SCF parameters when constructing its next
solver, unless `xc_params` was explicitly supplied as an override.

Periodic custom energy callbacks support the existing SCF and q=0 static
response paths. A nonzero-q extension requires the separate `Functional`
`q_energy_fn` contract: diagonal-k features alone do not specify finite-q
physics. A scalar callback does not infer a separate nonlocal HF operator,
arbitrary-k band Hamiltonian or additional hybrid fraction. Use a complete
traditional recipe when exact-exchange operators are required.

## Consumer scope

| Consumer | Shared XC integration and boundary |
|---|---|
| Molecular RKS/UKS and TDDFT/TDA | LDA, GGA, tau MGGA, global and range-separated exchange; complete recipe requirements checked. |
| Periodic RKS/UKS/KRKS/KUKS and response | Same local graph; 3D screened exchange; method-specific dimensional, sampling and response restrictions remain explicit. |
| GKS/ROKS | Existing supported families remain; new screened operators are rejected rather than silently reduced to a global coefficient. |
| OFDFT | Reuses local semilocal XC; no orbital-dependent HF/RSH contribution is inferred. |
| Molecular GW | Uses the actual converged RKS/UKS mean-field potential when removing the starting XC contribution; standard RSH references retain full metadata. |
| Molecular BSE | Uses the recorded GW QP/screening frame. Its static W kernel does not acquire a TDDFT `fxc` term. |

The restricted public GW facade connects existing G0W0, evGW/evGW0, qsGW and
Matsubara scGW drivers. This does not add unrestricted qsGW/scGW drivers.
Full-frequency Matsubara scGW has no inferred QP band energies: constructing
a static BSE reference requires validated analytic continuation and QP
coverage. Periodic GW/BSE algorithms are outside this XC-library change.
Screened-exchange nuclear geometry derivatives remain a separate unsupported
boundary in the DFPT geometry entry points.

OFDFT also rejects recipes requiring unimplemented dispersion, NLC or PT2.
Periodic GW, arbitrary-k GW queries, KS optical vertices and BerkeleyGW export
reject the newly recognized screened-hybrid sources until their corresponding
operators are implemented. Orbital stability for RKS/UKS rejects RSH and
external `Functional` sources rather than evaluating an incomplete curvature.
Molecular custom DFPT uses `DFPT(mf.to_reference(), functional=mf.xc,
params=mf.params)` to retain its full energy graph; the traditional facade
adapter rejects external models instead of using the HF initialization label.
These checks preserve the supported method scope during library expansion.

## Implementation and provenance

`types.py`, `registry.py` and `parser.py` own metadata and expressions;
`features.py` and `evaluate.py` own the shared local graph. `_jax_xc.py` and
`_compat.py` privately adapt upstream kernels. The former parallel
`dft/libxc_jax` implementation has been removed; consumers import `dft.xc`.

Runtime local formulas use [jax-xc](https://github.com/sail-sg/jax_xc), not
PySCF. `xc.jax_xc_backend_info()` distinguishes the Python distribution and
the active C parameter library: the validation environment has jax-xc 0.0.9
bound to Libxc 6.0.0, while the independent PySCF oracle uses Libxc 7.0.0.
The optional dependency starts at the tested jax-xc 0.0.9 release. The official
0.0.11 Linux wheel was inspected in isolation, but its import on this host
failed because its binary requires `GLIBC_2.34`. It was not installed into the
working environment and is not counted as a numerically validated runtime.
Use a binary or source build compatible with the target host.
Vector-parameter and compound-functional adapters preserve the actual
factory parameters. The private B97-family structure reader checks its
versioned ABI against the recorded
[Libxc 6.0.0 source](https://gitlab.com/libxc/libxc/-/raw/6.0.0/src/mgga_xc_b97mv.c).
Unsupported ABI layouts are rejected rather than guessed.

Stable E1 and scaled-erfc branches use analytic derivative rules and safe
branch inputs; mathematical references are the DLMF
[E1 series](https://dlmf.nist.gov/6.6.E2),
[E1 asymptotic expansion](https://dlmf.nist.gov/6.12.E1) and
[erfc asymptotic expansion](https://dlmf.nist.gov/7.12.E1).
The recipe snapshot follows the documented
[PySCF Libxc interface](https://pyscf.org/_modules/pyscf/dft/libxc.html).

WPBE uses a stable algebraic omega=0 limit and a private, WPBE-only correction
to the upstream tiny-gradient floor, which otherwise suppresses positive-SR
exchange curvature. The
[official Maple expression](https://gitlab.com/libxc/libxc/-/raw/6.0.0/maple/gga_exc/gga_x_wpbeh.mpl)
defines the independent small-gradient curvature oracle, checked against
80-digit evaluation of its energy terms. For the predefined zero-gradient
HSE domain, this analytic oracle tests the Hessian at the same tolerance as
other Hessian checks. Double-precision Libxc energy differences and VXC are
retained as diagnostics there: cancellation and the `s^4 log(s)` term make
their naive small-step Hessian extrapolation unreliable. Nonzero-gradient
points retain the independent Libxc energy/potential comparisons.

`tests/xc` separates recipe contracts, independent local Libxc comparisons,
density-domain checks, molecular E/F/TD comparisons, screened periodic
operator checks and custom-model tests. Point checks do not certify molecular
grid convergence; a finite gradient does not certify an optimizer or SCF has
converged. The runnable [XC examples](../../../../examples/xc/README.md) and
[training examples](../../../../examples/training/README.md) show public usage.

### Verification snapshot: 2026-10-20

Validation used JAX 0.10.0, float64 and CPU execution on `jy_gpu_1`, with
disjoint CPU affinities and no GPUs. Numerical oracles used PySCF 2.13.0 /
Libxc 7.0.0; runtime formulas used jax-xc 0.0.9 / Libxc 6.0.0.

| Focused verification | Result and scope |
|---|---|
| Registered local energies and potentials | 76 passed; 2 Laplacian-dependent recipes explicitly skipped. Local channels are not complete-method certification. |
| Local energy, potential and Hessian references | 19 passed across split groups, including independent omega=0 WPBE and PBE channels. |
| Existing Neural XC compatibility | 95 passed, including model presets, runtime and generic-module boundaries. |
| Final public contracts and support boundaries | 52 passed. |
| Molecular custom-XC review regressions | 17 passed: actual SCF/GW/BSE, response refresh, Traditional overrides and parameter binding. |
| Four restricted GW methods from four XC starting points | 26 passed including facade metadata; H2/STO-3G with small frequency grids tests integration, not spectral convergence. |
| Final HSE broad-domain checks | 2 passed, retaining fixed derivative tolerances and all raw difference diagnostics. |
| Periodic HSE06/CAM-B3LYP | 4 independent Gamma/2k SCF/TDA/TDDFT cases passed on a 31-cubed FFT mesh; 2 energy/Fock/q=0 consistency checks passed on a 7-cubed mesh. |
| LC-wPBE after the final WPBE correction | 4 molecular R/U SCF and full A/B reference checks passed. |
| Public examples and packaging | 4 examples passed; an isolated wheel contains the catalog and imports without PySCF or the retired implementation. |

Groups overlap and are not added into one suite count. Earlier timeouts and
failed derivative/oracle checks were retained during diagnosis. The passing
records above refer to completed focused checks; no full-repository or GPU
validation is claimed. The short external-network example reproduces the
existing loss reductions in fixed-density, explicit and implicit modes.
