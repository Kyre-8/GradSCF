# Static density-functional perturbation theory

`gradscf.dfpt` constructs response equations from existing ground-state energy
and Fock builders. It does not rerun or differentiate through SCF iterations.
Linear solves, including their backward rules, belong to `gradscf.solvers`.
The package name uses the standard abbreviation **DFPT**.

## Public interface

Finite-q periodic electronic response is available through
`DFPT(mf, q=[qx,qy,qz]).response(potential=vq)`, with q in fractional reciprocal
coordinates. See [finite-q theory, conventions and validation](FINITE_Q.md).
This supports HF/semilocal KS and external `dft.Functional` models with an
explicit `q_energy_fn` on closed, equally weighted k meshes. Moving
overlap/density sources and a bounded commensurate phonon driver are described
in [atomic response](ATOMIC_RESPONSE.md); OFDFT phonons use its supercell KKT
path. The external callback supplies the real per-cell energy for D+Zq+Zq†;
AD derives the full, possibly nonlocal kernel, including the normalization at
self-inverse q. A diagonal-k energy alone does not define this extension. The
examples below use the electronic-potential q=0 interface.

```python
from gradscf import dfpt

problem = dfpt.DFPT(mf)                 # converged reference snapshot
first = problem.response(potential=v1)
assert first.converged
rho1, energy1 = first.density, first.energy

series = problem.series(potential=v1, order=2)
assert series.converged
energy_second_derivative = 2 * series.energy[1]
```

`v1` is the derivative of the electronic external potential with respect to a
scalar perturbation. Units are atomic units: an AO potential matrix is in Eh
per perturbation unit; a real-space potential is in the same units. A molecular
electric field couples as$+F*r$ to the electronic Hamiltonian, so the electronic
polarizability is $-Tr(r*D1)$. A periodic AO potential is **not** a uniform field.

For a general perturbation, `problem.response(tangent)` takes a PyTree matching
`problem.inputs`, with each leaf replaced by its first derivative. Alternatively,
`problem.series(lambda t: inputs_at_t, order=n)` specifies a scalar path. The path
must pass through the reference inputs at zero and preserve shapes and tree
structure. Both explicit operator derivatives and induced density response enter
the returned observable derivatives. `None` leaves remain `None`.

The facade is an eager snapshot. Rebuild it after changing or rerunning the
source calculation. The array-level `ks_problem`, `ofdft_problem`, and `Problem`
interfaces permit custom differentiable functionals and parameter PyTrees.
An outer derivative requiring reference relaxation must supply a differentiable
reference construction; a stored numerical snapshot does not contain that
missing dependence. External molecular `dft.Functional` models can be used
directly, with the same explicit parameter PyTree used for SCF:

```python
def response_loss(params):
    state, info = scf_solver.run(reference, functional, params)
    problem = dfpt.DFPT(state, functional=functional, params=params,
                       converged=info.converged)
    result = problem.response(potential=v1)
    return jnp.mean((result.density - target_density_response)**2)

loss, gradient = jax.value_and_grad(response_loss)(params)
```

Here `scf_solver` is the existing `DifferentiableSCF`, configured for
`mode='self_consistent'` and `gradient_mode='explicit'` or `'implicit'`.
The adapter reuses `dft.derivatives`, the shared Coulomb builders, and shared
SCF energy/Fock algebra. Restricted references stay in the equal-spin subspace;
unrestricted references have independent spin response. `density` is spin-summed
and `values['density_spin']` contains both spin channels. Orbitals, stored density,
and integer occupation metadata must agree. The reference is checked against
the supplied functional and parameter values.

The existing XC Trainer also accepts `Sample(reference, response=(v1, D1))`
and `trainer.loss={'response': {'mse':1., 'mae':0.1}}`. It constructs DFPT on
the self-consistent state already used by the loss, without a second SCF run.
Response supervision requires `explicit` or `implicit`; fixed-density energy,
potential, and kernel training remain available. A nonstationary fixed density
is not a valid screened DFPT reference. Failed SCF/response calculations produce
nonfinite loss and reject the update through the existing optimizer guard.
Trainer response labels currently cover real molecular energy functionals,
without separately supplied hybrid exchange. The same `Sample`/`Trainer`
interface also supports periodic energy and density supervision, including
unrestricted and thermal references; periodic response labels remain
unsupported. See `examples/training/periodic_xc.py` and the
[Trainer contract](../model/neural_xc/training/README.md#periodic-references).

Periodic external models use `periodic_problem(..., functional=..., params=...)`
for q=0 and `finite_q_problem(..., functional=..., params=...)` for nonzero q,
or `DFPT(mf, q=q)` after `pbc.dft.KRKS(..., functional=..., params=...).run()`.
Construct both the SCF reference and the array factory inside a custom loss
to retain model-parameter and relaxed-reference AD. No local approximation is
applied to the external kernel. The finite-q tests independently check a
nonlocal energy Hessian and a GGA-type screened-response parameter gradient;
see [the contract and tested boundaries](FINITE_Q.md#external-energy-functionals)
and `examples/dfpt/03_external_finite_q.py`.

## Stationary response and Taylor recurrence

For a converged state `x0`, input parameters `a0`, and residual

$$
R(x_0,a_0)=0,\qquad J_0=\partial_x R(x_0,a_0),
$$

implicit differentiation gives

$$
J_0x^{(1)}=-\partial_aR\,a^{(1)}.
$$

JAX supplies the Jacobian action and right-hand side by JVPs. Shared GMRES is
the default; `LinearSolverConfig(method='direct')` is a small-system reference
option. No dense Jacobian is formed in the GMRES path. After solving, the code
checks the actual linear residual, and differentiates each observable `O(x,a)`
jointly along `(x1,a1)`.

For higher orders, write **normalized** coefficients

$$
x(t)=x_0+\sum_{n\geq1}x_nt^n,\qquad
x_n=\frac{1}{n!}\frac{d^nx}{dt^n}(0).
$$

Substitute the known lower orders into the residual, leaving `x_n=0`. With
`b_n=[t^n]R(x0+sum_{j<n}x_j*t^j,a(t))`, the next coefficient is

$$
J_0x_n=-b_n.
$$

`solvers.nonlinear.root_taylor` computes `b_n` with JAX `jet`, reuses the same
base Jacobian action, and calls the shared linear solver at every order. This
is the reusable numerical recurrence, independent of MP or DFPT physics. It
does not replace an existing MP implementation or introduce CI/CC methods.
The first axis of a `series` result contains orders **1 through n**; it excludes
the reference value. Observable series use the same normalization.

## KS density chart

For each spin/k block, let `C` be a complete reference MO basis, `C†S0C=I`.
Use a Hermitian projector coordinate `Q` and occupation factor `g`:

$$
D=gCQC^\dagger,\qquad s=C^\dagger SC,\qquad f=C^\dagger F[D]C.
$$

The density must satisfy idempotency in its physical metric and stationarity:

$$
QsQ-Q=0,\qquad fQs-sQf=0.
$$

To avoid redundant constraints, the residual contains the occupied-occupied
and virtual-virtual blocks of the first equation, plus the virtual-occupied
block of the second. Independent Hermitian entries are packed as real numbers.
Real symmetric charts have `n(n+1)/2` coordinates per block; complex charts have
`n^2`. The chart becomes complex if the orbitals, overlap, or Fock operator are
complex. Real molecular facade perturbations are explicitly restricted to real
symmetric matrices.

This construction avoids differentiating individual eigenvectors at degeneracy
inside the occupied or virtual space. It requires a separated occupied/virtual
boundary and an isolated, nonsingular screened response problem. The base check
rejects a closed occupied/virtual gap. A positive orbital gap and small linear
residual do **not** certify physical stability or prove Jacobian invertibility.

`fock(densities, inputs)`, `energy(densities, inputs)`, and `overlap(inputs)` are
the only physical callbacks needed by `ks_problem`. Differentiating the Fock
builder includes Coulomb, exchange, and the XC response kernel. External neural
XC energy callbacks can use the existing `dft.derivatives` machinery.

Changing the AO overlap enters both equations and the electron-number
observable. A nuclear displacement also changes the Hamiltonian, ERIs, AO/grid
data, quadrature, and nuclear energy. Supplying only an hcore derivative is not
a complete nuclear response or force constant.

## OFDFT constrained response

Reuse the existing OFDFT energy `E(q,a)`, where `q` is the density amplitude or
its Gaussian coefficients. With a fixed reference whitening map `q=B0*x`, define
the physical electron count `N(x,a)` from the perturbed grid weights or overlap.
The state stores physical chemical potential `mu`; with fixed reference `N0`,
the square residual is

$$
R=\begin{pmatrix}
(\nabla_xE-\mu\nabla_xN)/N_0\\
(N(x,a)-N_e(a))/N_0
\end{pmatrix}.
$$

The density is `rho=|q(r)|^2`. The fixed numerical chart avoids differentiating
a new Cholesky factor at every Taylor order; the physical metric and particle
number remain differentiable. AD includes the kinetic-functional Hessian,
Hartree kernel, and XC kernel. Existing kinetic and XC callbacks and their
parameter PyTrees are reused without a separate functional implementation.
Reference validity first checks the original raw stationarity components
before division by `N0`. `Problem.residual`, response residual norms, and linear
solver tolerances use the normalized equations above; `OFDFTResult.residual_norm`
and its forward tolerance keep their original raw units. Neither full residual
is a purely energy-unit quantity because the number constraint is dimensionless.

## Implemented scope and limits

| Input | Static first response | Taylor series |
| --- | --- | --- |
| Molecular RHF/RKS, real closed shell | AO potential and supplied input tangents | Requires jet-supported builders |
| Periodic Gamma/k-point KS, fixed integer occupations | q=0, Hermitian AO potential and input tangents | Gamma HF potential path tested |
| Periodic insulating full equal-weight k mesh | Berry Born charges and macroscopic static electric response via `dfpt.electric` | Disabled; separate nested-AD Maxwell/mixed tests cover a dimer and small H2 only |
| Periodic HF/KS on a closed equal-weight k mesh | Finite-q electronic potential; integer or Fermi occupations | Not supported in a single q sector |
| Periodic external `dft.Functional` | q=0; finite-q electronic potential requires `q_energy_fn` | Not supported in the finite-q sector |
| Molecular Gaussian OFDFT | Potential and input tangents | Potential path tested |
| Periodic grid OFDFT | Potential, particle number, volume/input tangents | Potential path tested, including external PBE |
| Periodic Gaussian OFDFT | Potential/input interface; potential tested | Backend-dependent, not separately benchmarked |

Molecular facade support currently covers RHF and RKS with HF/LDA/GGA/global
hybrid builders. Custom spin-resolved energy/Fock functions use `ks_problem`;
there is no built-in molecular UKS facade adapter in this first version.
External `dft.Functional` states from unrestricted DifferentiableSCF are supported.
Periodic facade
validation reuses the existing periodic response restrictions.

First-order response needs JVP support. Higher orders additionally need JAX
`jet` rules for **every active primitive** in the path. Native integral calls,
packed exchange scans, moving-lattice inverses, or other unsupported primitives
can therefore prevent a Taylor calculation even when first response works.
There is no silent finite-difference fallback or replacement of a missing
derivative by zero. The potential shortcuts keep unrelated geometry and metric
leaves constant to avoid spurious active Taylor paths.

Finite-temperature periodic SCF now has a separate q=0 first-response adapter
using the Fermi-density fixed-point residual, rather than the integer projector
chart. It includes occupation and chemical-potential response at fixed spin
populations. Thermal Taylor series and higher derivatives at degeneracy are not
provided; see [periodic foundations](../pbc/README.md).

Atomic/Pulay response and small commensurate reference phonons are available
through the separate [geometry adapter](ATOMIC_RESPONSE.md). Scalable native
primitive-cell phonons, dynamical response and a turnkey nuclear-Hessian driver are not
provided here. No all-order geometry capability is implied. The KS chart stores
full density matrices, so matrix-free linear algebra alone does not make this
a large plane-wave production implementation.

`from gradscf.dfpt.electric import born_problem, electric_problem` supplies
fixed-cell zero-temperature Berry response for gapped integer occupations on
a complete equal-weight k mesh. The first observes `dP/dR`; multiply by cell
volume to obtain Born charges. The second observes `chi=dP/dE`, with
`epsilon_inf=I+4*pi*chi`, from the electric enthalpy `E_KS-V E.P`. A Hermitian
AO-potential perturbation is not a replacement for this macroscopic field.
The checked reference and fixed local phase branch reuse the shared projector
residual, occupied-frame construction and root/linear solvers. See the
separate [LiH Born](../../../reproducibility/ad_showcase/P04_lih_born/README.md),
[electric-field](../../../reproducibility/ad_showcase/P04_lih_dielectric/README.md)
and [small mixed-response](../../../reproducibility/ad_showcase/P04_electric_mixed/README.md)
records. Material convergence, general mixed response and Taylor-jet support
are not implied; no density/charge/acoustic projection is applied.

Invalid reference states and failed linear solves return `converged=False` and
NaN response coefficients. Shape errors and unsupported facade/q choices raise
explicit errors. Always inspect convergence before using results.

## Validation and examples

Run `PYTHONPATH=src JAX_PLATFORMS=cpu python -m pytest -q tests/dfpt
tests/solvers/test_taylor.py` from the repository root (as one command).
Tests use CPU float64. They cover analytic screening, occupied degeneracy,
complex Hermitian response, moving constraints, normalized Taylor coefficients,
outer forward/reverse AD, and independently perturbed SCF calculations.

`tests/dfpt/test_pbe.py` contains a handwritten, spin-polarized PBE energy
functional used only as a test oracle. Pointwise energies, first derivatives,
and Hessians are checked against PySCF/libxc, including zero-gradient points.
Its AO energy callback also tests GradSCF self-consistency and CPKS against
PySCF, and its grid form tests periodic OFDFT response and second-order energy.
Separate built-in SVWN tests require optional `jax_xc`; an absent dependency is
reported as a skip, not replaced by this test functional.

Runnable, non-CLI demonstrations are in `examples/dfpt/`. The four-layer
external-network response-training example is `examples/training/dfpt_response.py`.

Theory: [Baroni et al., Rev. Mod. Phys. 73, 515 (2001)](https://doi.org/10.1103/RevModPhys.73.515).
PBE: [Perdew, Burke, Ernzerhof, Phys. Rev. Lett. 77, 3865 (1996)](https://doi.org/10.1103/PhysRevLett.77.3865).
