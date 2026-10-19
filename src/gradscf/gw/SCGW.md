# Finite-temperature matrix scGW

`gradscf.gw.scgw_matsubara_restricted` solves real, spin-restricted molecular
scGW in a fixed orthonormal orbital basis. It iterates full frequency-dependent
Green's-function and self-energy matrices. The density is obtained from the
interacting Green's function; it is not reconstructed from quasiparticle pole
occupations. The finite-temperature imaginary-axis formulation follows the
matrix GW equations described by [Yeh et al., PRB 106, 235104 (2022)](https://arxiv.org/abs/2206.07660).

## Equations and conventions

All energies and frequencies are in Ha; beta and imaginary time are in Ha^-1.
G and d are per-spin quantities; D=2d is the spin-summed density. In the fixed
orthonormal basis supplied by the initial AO coefficients C0:

```text
F = h + J[D] - K[D]/2
G(iw) = [(iw + mu)I - F - Sigma_c(iw)]^-1
Pi_PQ(tau) = 2 Tr[B_P G(tau) B_Q G(-tau)]
W_c(i nu) = solve(I - Pi(i nu), Pi(i nu))
Sigma_c(tau) = -sum_PQ B_P G(tau) B_Q W_c,PQ(tau)
G(-tau) = -G(beta-tau)
```

Each iteration solves mu so that `Tr(D)=2*nocc`, using the density of the same
Dyson G. Both static F and dynamic Sigma are rebuilt from G. Mixing acts on F,
Sigma and its high-frequency moment; convergence checks the **unmixed** updates.

The reported internal energy is

```text
E = Tr(D h) + Tr(D J)/2 - Tr(D K)/4 + E_GM,dynamic + E_nuclear
E_GM,dynamic = -sum_nu Tr[Pi(i nu) W_c(i nu)] / (2 beta)
```

The bubble already contains the factor of two for spin. `correlation_energy`
stores this dynamical GM contribution, not `total_energy - HF_energy`.
`total_energy` is a finite-temperature internal energy, not a free energy.

## Grids and analytic reference subtraction

- `nw` counts positive fermionic frequencies: there are `2*nw` fermionic
  nodes and midpoint time samples. The bosonic grid includes zero and both
  Nyquist endpoints, with half weights on those endpoints.
- The current static F supplies an analytic reference Green's function. Its
  occupation and time dependence are added back exactly after subtracting it
  from the discrete G. The density therefore recovers the noninteracting limit
  even on a coarse grid.
- The reference bubble includes all orbital pairs. At zero bosonic frequency,
  equal-energy pairs use `f'(epsilon)=-beta*f*(1-f)`, including intraband and
  degenerate thermal terms.
- An analytic reference RPA interaction and reference G0 W_RPA self-energy
  supply the bosonic and fermionic tails. The remaining dressed corrections
  are transformed on the grid. These are subtraction/addition identities:
  they do not freeze the actual G or W to the reference spectrum.
- The self-energy moment `M` in `Sigma(iw) ~ M/(iw)` is mixed with Sigma and
  also used to improve G's third-order tail. Equal-mode limits in bosonic
  sums use relative frequency comparisons, so distinct soft modes are not
  conflated.
- GM energy uses the bosonic Pi Wc sum with the infinite reference sum added
  analytically. A plain midpoint time integral has much larger cusp errors.

## API and returned state

Use the runnable example below to construct inputs from the public molecular
facades. C0 must satisfy `C0.T @ S @ C0 = I`; AO hcore and DF factors are
projected into that basis.

```python
from gradscf.gw import scgw_matsubara_restricted

result = scgw_matsubara_restricted(
    mo_energy=mf_result.mo_energy,
    mo_coeff=mf_result.mo_coeff,
    nocc=nocc,
    df_factors=ao_df_factors,
    hcore_matrix=mf_result.hcore_matrix,
    nuclear_repulsion=mf_result.nuclear_repulsion,
    beta=80.0,
    nw=320,
    max_iter=300,
    mixing=0.5,
    tol=1e-7,
    particle_tol=1e-12,
)
```

Important result fields:

| Field | Meaning |
| --- | --- |
| `grid` | Actual tau, fermionic/bosonic frequency nodes, endpoint weights, beta |
| `mo_coeff` | Fixed basis C0 used by the matrices, not QP orbitals |
| `green_iw`, `self_energy_iw`, `fock_mo` | G and the F/Sigma used in its Dyson equation |
| `mapped_self_energy_iw`, `mapped_fock_mo` | Sigma[G] and F[D] rebuilt from that G |
| `density_mo`, `density_matrix` | Spin-summed density in the fixed MO basis and AO basis |
| `sigma_moment` | High-frequency moment accompanying the Dyson self-energy (Ha^2) |
| `fixed_point_residual` | Maximum unmixed F/Sigma update, including the moment scaled by the lowest frequency (Ha) |
| `particle_number_error` | Tr(D)-2*nocc, in electrons |
| `static_mo_energy`, `static_mo_coeff` | Eigenpairs of static F only |
| `mo_energy` | **None**: QP poles require separate analytic continuation |

The Dyson inputs and mapped quantities differ by at most the specified
fixed-point tolerance. The GM energy is evaluated as the GW functional of the
returned G. No unchecked last iterate is reported as converged.

`scgw_cd_restricted` is a deprecated compatibility name which emits a warning
and delegates to the Matsubara implementation. Its former pole approximation
has been replaced. A supplied `eta` raises an error: real-axis broadening has
no role in this imaginary-axis solver. The meaning of `nw` is now the count of
positive Matsubara frequencies, not a contour-deformation quadrature size.

## Reproduction and measured small-system checks

Run the commands below from the repository root. Environment used on
2026-09-20: JAX 0.8.1, CPU (`TFRT_CPU_0`), arm64,
float64/complex128. H2 at 0.74 Angstrom, Cartesian STO-3G, HF start;
`tol=1e-7 Ha`, `particle_tol=1e-12`, `mixing=0.5`, `max_iter=400`.

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  python examples/gw/scgw_matsubara_h2.py --beta 80 --nw 40 80 160 320 640 \
  --output /tmp/scgw_h2_beta80.json
```

The JSON records devices, precision, controls, energies, occupations,
dielectric eigenvalue checks, residuals, and elapsed time per calculation
(including any compilation for that grid shape).

| beta (Ha^-1) | nw | Total internal energy (Ha) | Iterations |
| ---: | ---: | ---: | ---: |
| 80 | 40 | -1.136599925211 | 45 |
| 80 | 80 | -1.136893744197 | 29 |
| 80 | 160 | -1.136967964766 | 28 |
| 80 | 320 | -1.136984740689 | 28 |
| 80 | 640 | -1.136988412428 | 28 |

The last refinement changes the energy by `3.67e-6 Ha`. This is a measured
grid difference, not an absolute error bound. Small iterative residuals alone
do not establish grid or basis accuracy.

To change temperature at fixed time resolution and frequency cutoff:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  python examples/gw/scgw_matsubara_h2.py --beta 40 --nw 160 --output /tmp/scgw_h2_beta40.json
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  python examples/gw/scgw_matsubara_h2.py --beta 160 --nw 640 --output /tmp/scgw_h2_beta160.json
```

Together with `(beta,nw)=(80,320)`, these produce energies
`-1.136984741030`, `-1.136984740689`, and `-1.136984740488 Ha`:
spread `5.42e-10 Ha`. Electron-number errors are below `1e-12`, density
eigenvalues remain in `[0,2]`, and the minimum eigenvalue of `I-Pi` is 1 to
roundoff for these H2 runs. These tests establish a low-temperature plateau
for this finite basis and grid resolution, not a general zero-temperature or
complete-basis result.

## Implicit response

Pass `differentiation=SCFDifferentiationConfig(mode="implicit", ...)` to opt
into the compiled primal loop and implicit first-order response. The default
path remains eager. Real symmetric Hamiltonian/DF parameters and their C0
projection retain both direct observable derivatives and self-consistent
state response. The initial orbital-energy spectrum is only a starting guess;
its implicit derivative is zero. beta, nw, nocc and solver controls are static.

```python
import jax
import jax.numpy as jnp
from gradscf.scf import SCFDifferentiationConfig
from gradscf.gw import scgw_matsubara_restricted

response = SCFDifferentiationConfig(
    mode="implicit", tolerance=1e-9, max_iter=20, restart=30,
    regularization=0.0, require_converged=True,
)

# inputs contains the AO data and static grid/forward-solver controls.
def loss(scale):
    result = scgw_matsubara_restricted(**{
        **inputs,
        "df_factors": inputs["df_factors"] * scale,
        "differentiation": response,
        "charge_response_tol": 1e-10,
    })
    return result.total_energy

value, derivative = jax.jit(jax.value_and_grad(loss))(jnp.array(1.0))
```

The real packed state contains symmetric F, the real/imaginary upper triangles
of positive-frequency Sigma, M/omega_min, and mu. Negative frequencies follow
by conjugate reflection. The residual includes the full F/Sigma/moment map
and `(Tr(D)-2*nocc)/beta`. For `R(x,theta)=0`, the response solves
`R_x dx = -R_theta dtheta`; it does not differentiate the mixing trajectory or
the final iteration alone. Observables are recomputed from the attached root.

The matrix-free linear solves reuse the shared SCF checked GMRES machinery.
The charge row is treated through a scalar Schur complement of the fixed-mu
matrix block. This handles mu response explicitly and checks both cancellation
and the resulting charge susceptibility. A stable grand-canonical
finite-temperature branch has `dN/dmu >= 0`; a negative susceptibility must
not pass merely because its magnitude is large. The tangent and adjoint
guards require the Schur complement to give a positive susceptibility above
`charge_response_tol`. They never substitute a fixed mu.
`charge_response_tol` is an absolute floor in electrons/Ha
(default `1e-10`), in addition to the relative linear-solve checks.

Failure policy:

- An unconverged primal, failed number bracket/bisection, or nonfinite forward
  update raises an error, also under JIT (wrapped by the JAX runtime).
- An unsuccessful tangent/adjoint solve, negative or unresolved charge Schur complement,
  or unsolvable fixed-mu block produces NaN response, following the SCF policy.
  In particular, low-temperature insulating charge response can be numerically
  indistinguishable from zero. A converged forward state does not imply a
  reliable canonical derivative.
- Only implicit mode with zero regularization and required primal convergence
  is accepted. The stopped forward solver must not be presented as an unrolled
  gradient. `converged` and `n_iter` are dynamic scalar leaves under JIT.

The Schur method conservatively requires the fixed-mu block to be solvable;
it can reject a case even if the complete coupled Jacobian is invertible.
Lowering `charge_response_tol` does not repair that conditioning or regularize
the equations. Always test the response against tighter solves and independent
finite differences when moving to a new regime.

### Reproducing a physical-parameter derivative

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  python examples/gw/scgw_implicit_response_h2.py \
  --beta 8 --nw 8 --output /tmp/scgw_h2_implicit_response.json
```

This scales all DF vertices as `B -> s*B`, so the bare interaction scales as
`s^2`; h and the nuclear energy remain fixed. It is not a nuclear-force test.
For H2/STO-3G at 0.74 Angstrom, CPU/JAX 0.8.1, float64/complex128, with
forward tolerance `1e-10 Ha`, particle tolerance `1e-12`, response tolerance
`1e-9`, and finite-difference step `1e-4`, the measured derivative at s=1 is:

| Method | dE/ds (Ha) |
| --- | ---: |
| JIT implicit VJP | 1.2251106625468804 |
| Independently reconverged central difference | 1.225110663635176 |

Absolute difference: `1.09e-9 Ha`. This validates the derivative of that
finite-temperature, finite-grid model, not a converged ground-state response.

## Validation boundaries

Tests cover analytic one-/two-level limits, a soft-mode regression, dressed-G
feedback, weak-coupling order, general real-matrix kernel gradients, orbital
and auxiliary basis rotations, energy-origin covariance, particle number,
Dyson/diagram closure, and independent nw/beta refinement.

First-order outer JVP/VJP tests include canonical noninteracting internal-energy
derivatives, interacting two-/three-orbital models, H2 DF response, Hamiltonian
diagonal/off-diagonal directions, density and mu response, a dense residual
Jacobian comparison, energy-origin covariance at beta=5 and 20, trajectory
independence, and explicit primal/adjoint/charge failure cases.

Thermal static-reference density and Green's functions use smooth matrix
functions with first Frechet response, rather than derivatives of individual
eigenvectors. At zero interaction, the correlation map vanishes exactly; its
first response therefore supports repeated static levels. The Hartree and
exchange map remains outside that branch and retains the physical quadratic
dependence on the DF vertices. Tests compare a three-level canonical density
response with its analytic formula and the finite-coupling quadratic energy
with the Hartree/exchange expression.

This zero-interaction branch does not supply fourth-order interaction
derivatives of the correlation diagrams. Interacting reference/RPA spectral
degeneracies remain unsupported; this is not a general degenerate scGW
response implementation. Higher derivatives and
complete nuclear-coordinate forces (including integral/basis derivatives) are
not validated. beta differentiation and unrolled outer AD are not implemented.
Complex/unrestricted/periodic scGW, analytic continuation, GPU operation and
large-basis performance are not validated here. Dense time-frequency transforms
and transition-space reference diagonalization make this implementation most
suitable for small-system correctness and differentiation work initially.
