# Differentiable orbital-free DFT

`gradscf.ofdft` minimizes the density functional

\[
E[n]=T_s[n]+E_H[n]+E_{xc}[n]+\int v_{ext}n+E_{ion},\qquad \int n=N.
\]

All quantities use atomic units. The first implementation supports real,
unpolarized densities, isolated all-electron Gaussian molecules, neutral 3D
periodic FFT grids, and periodic Gaussian amplitude expansions. It does not
construct occupied KS orbitals or require an HF/KS reference.

## Representations and API

```python
from gradscf import gto, ofdft

mol = gto.M(atom='H 0 0 0; H 0 0 .74', basis='6-31g*')
calculation = ofdft.OFDFT(mol, kinetic='tfvw', xc='pbe').run()
print(calculation.e_tot, calculation.result.residual_norm)
```

The eager facade uses the existing jax-xc backend for `xc='svwn'`, `xc='pbe'`
and other supported density-only LDA/GGA specifications. Hybrids and orbital
meta-GGAs are rejected. `xc=None` explicitly omits XC; it is useful for kinetic
and solver regression tests and is not a substitute physical XC approximation.
Missing jax-xc raises the existing dependency error without silently changing XC.

For JIT/AD, build inputs and use `run_ofdft(inputs, ...)`. Continuous inputs and
functional parameters are PyTrees. `gaussian_inputs(mol, coordinates=...)` also
accepts traced nuclear coordinates in Bohr. Native packed ERI production is
forward-only: traced geometry uses the native full-ERI geometry rule. This
path currently supports first nuclear derivatives, not nuclear Hessians.

A Gaussian density amplitude is `phi = ao @ coefficients`, with `n=phi**2` and
`coefficients.T @ overlap @ coefficients=N`. Its `outer(coefficients,coefficients)`
is used only to contract the density's external and Hartree energies; it is not
an occupied KS density matrix. `T_vW = coefficients.T @ kinetic @ coefficients`.
The standalone molecular builder accepts named Cartesian basis sets; advanced
callers can construct `OFDFTInputs` from their own AO/integral arrays.

For grid amplitudes, `sum(weights*phi**2)=N`. Periodic input construction:

```python
inputs = ofdft.periodic_inputs(lattice, (9, 9, 9), nelectron=2.,
                              external_potential=local_grid_potential,
                              nuclear_repulsion=ionic_energy)
result = ofdft.run_ofdft(inputs, config=ofdft.OFDFTConfig(xc='pbe'))
```

`lattice` is in Bohr and its rows are lattice vectors. Mesh shape is static,
three-dimensional, odd in each direction, with sizes >=3. Grid quantities are
flattened in NumPy/JAX C order. Derivatives and Coulomb convolutions use FFTs.
The Hartree `G=0` mode is zero (uniform compensating-background convention);
explicit-potential callers own consistent ionic energy and potential zero modes.

`inputs_from_cell(cell)` builds a neutral cell with a **local** GTH potential
and Ewald ionic energy. A GTH potential containing nonlocal projectors is
rejected; removing those projectors would not define the original physical
model. General OEPP/UPF/RECPOT import is not a module feature; the ATLAS examples
include a small eager RECPOT reader for the two pinned reference assets. Supply an
explicit local grid potential to use another potential provider.

`periodic_gaussian_inputs(grid_inputs, topology, parameters)` projects the grid
problem onto a Gamma-periodic Gaussian amplitude space. It reuses the same
Coulomb/KEDF grids without an AO-pair potential cache. Grid and Gaussian minima
need not coincide: the latter is a restricted variational space. Increase both
mesh and basis to check convergence and product-density aliasing.

## Kinetic functionals and neural adaptation

Implemented KEDFs:

- `tf`: Thomas-Fermi, `C_TF integral(n**(5/3))`.
- `vw`: von Weizsaecker, `1/2 integral(|grad(phi)|**2)`.
- `tfvw`: TF + vW, with explicit differentiable `kinetic_params={'tf':a,'vw':b}`.
- `wt`: standard TF + full vW + Wang-Teter, alpha=beta=5/6; periodic only.
- `wgc`: TF + full vW + second-order WGC99, alpha=(5+sqrt(5))/6,
  beta=(5-sqrt(5))/6, gamma=2.7; periodic only. Optional scalar
  `kinetic_params={'reference_density': rho_s}`. See [WGC.md](WGC.md) for the
  density-dependent kernel, cached dimensionless ODE table, AD and validation.

The TF amplitude path evaluates `abs(phi)**(10/3)` to keep its Hessian finite at
nodes, avoiding the spurious `0*inf` of differentiating `(phi**2)**(5/3)` twice.
The public density-only `thomas_fermi` derivative is defined for positive density.
WT evaluates powers with a declared `1e-18` density floor; the same energy is
differentiated for both the potential and Hessian. Its uniform reference density
is the current cell average, not a stale cached constant. An exact eta=1 Kohn
anomaly has a finite kernel value but undefined kernel derivatives; AD returns
nonfinite responses there, including higher orders. Away from that anomaly,
small/large wavevectors use analytic series. WGC has the same density-domain
and exact-anomaly limitations. The TF/vW family
is a numerical baseline rather than a generally accurate molecular KEDF.

KEDF callbacks have the contract `energy(params, features) -> scalar Hartree`.
`KineticFeatures` includes `rho`, `grad_rho`, weights, coordinates, vW energy,
phi and (for periodic problems) reciprocal vectors, volume and static mesh.
No separate neural potential or kernel needs to be supplied.

```python
functional = ofdft.NeuralKineticFunctional(energy_fn=network_correction)
params = {'network': model_parameters, 'baseline': {'tf': 1., 'vw': 1.}}

# network_correction(network_params, features) returns an integrated energy.
def loss(params):
    result = ofdft.run_ofdft(inputs, kinetic=functional,
                            kinetic_params=params, config=config)
    return ((result.density - target_density)**2 * inputs.weights).sum()

value, gradients = jax.jit(jax.value_and_grad(loss))(params)
```

The adapter follows neural XC's external init/apply parameter pattern. It accepts
Flax, Haiku or pure-JAX models through a callback and does not depend on a specific
network framework. Parameters must remain explicit inputs; do not capture traced
weights in the callback. Smooth activations are necessary for meaningful Hessian
response. The adapter does not enforce Pauli positivity, scaling, or other exact
constraints on an arbitrary network: these are model design choices.

Custom density-only neural XC may use `xc_energy_fn(params, features)` and
`xc_params`, with `config.xc=None`. The callback is a total XC energy, so it can
wrap an existing semilocal network; orbital HFX/PT2 channels are unavailable.

## Forward and backward

`ofdft/problem.py` defines the energy and constraint residual. Forward optimization
lives in `solvers/nonlinear/sphere.py`; no OFDFT-local optimizer or Krylov solver
is duplicated. It uses bounded JAX L-BFGS scans and line search.

We whiten the amplitude metric: `q=sqrt(N) L^(-T) x`, `L L.T=M`, `x.T x=1`.
For grid amplitudes, M is diagonal quadrature weights and no dense matrix is
formed. For Gaussian amplitudes, M is the overlap matrix. The KKT residual is

\[
F(x,\eta;\theta)=\left(\nabla_x E-2\eta x,\;x^T x-1\right),\quad \mu=\eta/N.
\]

`SCFDifferentiationConfig` and `attach_scf_backward` are shared with DFT. Implicit
response solves the constrained transposed Jacobian system with shared GMRES (a failed incremental true-residual check retries the upstream
batched path inside the opaque numerical solve);
it does not differentiate the optimization history or initial guess. Lattice,
volume, moving overlap, KEDF and XC parameters remain in the response graph.
Failure to converge the forward stationarity residual or adjoint solve produces
nonfinite derivatives rather than a misleading approximate response.

`mode='explicit'` differentiates the accepted finite iterates. Up to three local Newton
refinements (AD HVPs and the shared linear solver) polish a nearly converged
state. Unrolled mode also performs these refinements at an already stationary
initial density to retain its parameter response. Small refinement systems use
the shared direct solver; larger ones remain matrix-free. This is not a full
ATLAS truncated-Newton implementation. Refinement failures remain nonfinite.

Check `converged` and `residual_norm`, not just the optimizer's energy change.
Stationarity is not a global-minimum or stability certificate. As with existing
SCF response, a locally isolated, nonsingular constrained stationary point is
required. Any nonzero adjoint regularization biases the response.

## Validation and references

Tests cover analytic TF derivatives/HVPs, uniform WT/Lindhard response, density
nodes, the Kohn anomaly, number conservation, implicit/FD density gradients,
explicit symmetric-start response, neural parameter gradients, lattice/volume
response, molecular coordinate response, Gaussian/grid fixed-density parity,
local-potential boundaries, and facade/input validation. jax-xc integration tests
are optional when that dependency is unavailable. See `VALIDATION.md` for the
actual local run and limitations.

Examples are under `examples/ofdft/`, without CLIs. Molecular and periodic
examples explicitly disable XC to provide dependency-light solver demonstrations;
`semilocal_xc.py` exercises the real jax-xc path when available. The neural
example performs actual optimization of a synthetic density-matching loss.

- Mi et al., *ATLAS: A Real-Space Finite-Difference Implementation of Orbital-Free
  Density Functional Theory*, [arXiv:1507.07373](https://arxiv.org/abs/1507.07373).
  Relevant: Eqs. 10-14 (amplitude/constraint), 18-24 (periodic electrostatics),
  29-38 (constrained minimization). ATLAS uses finite-difference Laplacians and
  a truncated-Newton algorithm; our periodic spatial derivatives currently use
  FFTs and our HVPs use AD. The WGC Table 1 results and original finite-difference ATLAS calculations
  are not claimed reproduced; the separate TF+lambda*vW solid comparison is
  documented under `examples/ofdft/ATLAS_REPRODUCTION.md`. The subsequent
  WGC implementation and independent Gaussian KS figure comparison are in
  `examples/ofdft/WGC_REPRODUCTION.md`, including the unresolved Mg discrepancy.
- Wang and Teter, *Kinetic-energy functional of the electron density*,
  [Phys. Rev. B 45, 13196 (1992)](https://doi.org/10.1103/PhysRevB.45.13196).
- [DFTpy dev reference, fbc47b4e](https://github.com/Quantum-MultiScale/DFTpy/tree/fbc47b4e1df3def1a2206e58c5a1a1a4d51db1d0):
  KEDF decomposition, periodic FFT and constrained density optimization were
  inspected as references. DFTpy is not imported or vendored at runtime.

## Pure-JAX XC and external numerical comparison

`examples/ofdft/lda_exchange.py` implements Dirac LDA exchange directly in JAX,
without correlation or a jax-xc dependency. `compare_dftpy.py` is a separate
optional reference script: it compares TF, vW and WT energy/potential/HVP at
the same nonuniform density, then lets both packages optimize independently
with the same Dirac exchange and external potential. This comparison does not
substitute for the optional jax-xc integration tests. See the measured tables
and exact commands in `VALIDATION.md`.
