# Exciton–vibration coupling

This submodule adds fixed-bath, low-population neutral-excitation response to
molecular BSE. It reuses the GW phonon data layout, Bose occupations and DW
contraction, and the shared linear solver. Electron and hole vertices enter
with opposite signs. No independent eigensolver, linear solver, SCF loop or
phonon generation lives here.

## API

```python
from gradscf import bse
from gradscf.bse import ep_coupling as ep

# result and space come from a converged restricted TDA-BSE calculation.
# mo_phonons is gradscf.gw.ep_coupling.PhononModel in the same MO frame.
model = ep.project_phonons(result, space, mo_phonons)
dipoles = bse.transition_dipoles(result, dipole_mo, space)
alpha = ep.polarizability(result.excitation_energies, model, dipoles, omega,
                         beta=1000., eta=.001)
sigma = ep.absorption_cross_section(result.excitation_energies, model,
                                    dipoles, omega, beta=1000., eta=.001,
                                    unit='Mb')
```

All energies, frequencies, eta, and vertices are in Hartree; beta is finite,
positive and in Ha^-1. Dipoles are in Bohr. A large finite beta provides the
zero-temperature limit, following the GW phonon API. The optical tensor is
in a0^3, cross sections in a0^2 or Mb. Cross sections use the existing BSE
polarization and unit conversion: default orientation average, or a normalized
real three-vector polarization.

Mixed-precision inputs are promoted before pole/resolvent arithmetic. Matrix
Fan skips an absorption solve only if its Bose occupation is exactly zero;
nonzero thermal weights are not thresholded.

| Function | Contract |
| --- | --- |
| `PhononModel` | Positive mode energies, Hermitian excitation vertices, optional quadratic operators; JAX pytree |
| `project_couplings` | Explicit orthonormal TDA arrays and MO vertices; no eigensolver implied |
| `project_phonons` | Checked restricted TDA BSE-result adapter; preserves amplitude-response guards |
| `vibronic_hamiltonian` | Finite vibrational-space Hamiltonian and vacuum Condon dipoles; JIT/AD-compatible assembly, no solver |
| `fan_retarded` | Full matrix neutral-excitation Fan self-energy, without Fermi occupations |
| `debye_waller` | Shared static quadratic-vertex contraction |
| `spectral_function` | Matrix spectral density per Ha, using the anti-Hermitian part of the resolvent |
| `polarizability` | Resonant plus antiresonant tensor for a scalar or vector frequency |
| `absorption_cross_section` | Passive positive-frequency absorption; rejects negative values beyond roundoff |

The response functions accept either a real vector of canonical excitation
energies or a full positive Hermitian excitation Hamiltonian. The matrix path
uses resolvents directly; it does not diagonalize H. Its response remains
well-defined at internal degeneracy. Vertices and dipoles must use the same
basis. The existing BSE facade continues to require real collinear data;
explicit complex excitation frames are supported by the new response functions.

## Projection and normalization

Let X[s,i,a] denote unit-spatial-norm TDA amplitudes. In a fixed reference
orbital frame, the one-body derivative of the electron–hole Hamiltonian is

    V[nu,ia,jb] = delta_ij g[nu,a,b] - delta_ab g[nu,j,i].
    G[nu,s,t]   = sum_ia,jb X[s,ia]* V[nu,ia,jb] X[t,jb].

There is no additional sqrt(2) spin factor in G. The closed-shell singlet
transition dipole already includes sqrt(2) in `bse.transition_dipoles`.
A common perturbation g=c I cancels exactly in the neutral-excitation space.
This differs from combining separate charged electron/hole linewidths.

The implementation contracts MO occupied/virtual blocks with the selected
amplitude columns. It does not construct (nmode,ntransition,ntransition).
An optional electronic-kernel derivative can be supplied as a small dense
array or a matrix-free callback:

```python
# Each call acts on only (ntransition,nstate) columns.
def kernel_derivative(mode, columns):
    return electronic_kernel_derivative_action(mode, columns)

model = ep.project_phonons(result, space, mo_phonons,
                          kernel_derivative=kernel_derivative,
                          kernel_quadratic=kernel_second_derivative_action)
```

`kernel_quadratic` is the diagonal-mode second operator derivative, with the
same callback signature. Missing callbacks explicitly freeze the electronic
BSE kernel. They are derivatives of the electronic kernel, not an extra
phonon-exchange interaction already represented by the exciton Fan model.
Current projection rejects nonzero full-BSE Y amplitudes and unrestricted
spin-channel tuples: their metric/spin treatment needs a separate extension.

Quadratic data use either compact (nmode,nstate,nstate) diagonal-mode vertices
or full (nmode,nmode,nstate,nstate) arrays. Full arrays are symmetric in the
mode indices; both layouts are Hermitian in excitation indices. With
Q_nu=b_nu+b_nu† and H_X(Q)=H_X+G Q+Lambda QQ/2,

    DW = sum_nu Lambda[nu,nu] (2 n_B(omega_nu)+1)/2.

Compact storage avoids allocating unused cross-mode blocks for the harmonic
DW average. Lambda must be the operator's second derivative relative to the
ground surface. An individual eigenvalue Hessian already contains linear
vertex-induced state mixing and must not be added as Lambda on top of Fan.

## Dynamic response

The one-phonon self-energy is

    Pi(z) = sum_nu G_nu [(n_nu+1)(z I-H_X-omega_nu)^-1
                        + n_nu (z I-H_X+omega_nu)^-1] G_nu†.
    R(z)  = [z I-H_X-DW-Pi(z)]^-1.
    A(z)  = -(R(z)-R(z)†)/(2 pi i),  z=omega+i eta.

The internal propagator uses bare H_X; no phonon Dyson iteration is performed.
Canonical vectors have an analytic pole-sum path. Matrix H_X and optical
resolvents use `gradscf.solvers.solve_complex`, which realifies the system
and reuses the existing real solver's primal/transpose checks and implicit
AD. `solver_config=LinearSolverConfig(method='gmres', ...)` is optional;
bounded direct solves are the default. `max_dense` applies to the realified
size 2*nstate. Failed solves raise, with no fallback or sign repair.

With d[s,alpha]=<s|r_alpha|0>, define M(z)=d† R(z) d. The full tensor is

    alpha(omega) = -M(omega+i eta) - M(-omega+i eta)*.

This restores the static BSE resonant/antiresonant formula exactly when
vertices vanish. It is invariant under unitary excitation-frame changes,
including bases within degenerate subspaces. Matrix inputs permit AD through
H, phonon energies, vertices, quadratic operators and dipoles without
canonical eigenvector derivatives. The result adapter still guards incomplete
or degenerate individual-amplitude AD; it does not upgrade the existing BSE
solver's derivative contract. Explicit fixed-basis arrays or a fixed
transition-space matrix provide the alternative for such calculations.

## Multiphonon reference

`vibronic_hamiltonian(H_X, model, dipoles, max_quanta=6)` returns the matrix
and transition moments in the product excitation/vibration basis. It retains
all occupation tuples with total quanta at most the static cutoff, and uses
the same G and Lambda conventions as Fan/DW:

    H_vib = H_X + sum_nu omega_nu b_nu† b_nu
            + sum_nu G_nu Q_nu + sum_nu,mu Lambda_nu,mu Q_nu Q_mu / 2.

Ground-surface zero-point energy is subtracted. The optical initial state is
the vibrational vacuum, with fixed Condon dipoles. This is a zero-temperature
finite-space reference, not a replacement for thermal Fan response. Quadratic
products retain intermediate states outside the cutoff before projection;
in particular `<n|Q²|n>=2n+1` holds at the upper boundary. A separate DW shift
must not be added to this matrix. Diagonalizing the linear model includes
multiphonon effects within that model; setting Lambda=None is not Fan+DW.

The module only assembles the dense matrix and moments. Use shared
`solve_hermitian` for forward lines or `solve_complex` for invariant resolvent
derivatives. A line-by-line diagonalization does not automatically authorize
degenerate eigenvector derivatives. Dimensions grow as
`nstate * binomial(max_quanta+nmode, nmode)`, so this is a small-molecule
reference path. Spectral convergence, electronic-state completeness and
infinite-model quadratic stability must be checked separately.

The executable `examples/bse/water_ep_coupling.py` compares three native water
modes, one-phonon Fan/DW, a linear multiphonon Hamiltonian and a diagonal
Franck-Condon oracle. Its C2v block reduction is checked against the supplied
vertices, and a unit regression checks full/block optical equivalence. The
example uses frozen GW corrections and electronic kernel; it is not a
quantitative dissociative absorption calculation.

## Approximation and population boundary

This is a number-conserving effective excitation Hamiltonian with a fixed
harmonic bath and negligible excitation population. Vacuum/multiple-excitation
intermediates, full thermal detailed balance, phonon feedback, periodic
momentum transfer and a complete conserving many-body BSE kernel are not
included in the Fan/DW response. Dipoles are fixed (Condon); Herzberg–Teller
effects are not generated. The separate finite-space Hamiltonian provides a
multiphonon comparison at zero temperature.

Even causal A>=0 does not ensure passive optical absorption throughout all
frequencies in this low-population approximation. If a phonon quantum exceeds
an excitation energy, narrow eta may resolve a negative-frequency thermal
satellite. The antiresonant term can then produce negative absorption. For
example E=.3, omega_ph=.4, G=.01 Ha, beta=40 Ha^-1 and eta=1e-12 Ha gives a
non-passive response at photon energy .1 Ha despite a positive bare gap.
`absorption_cross_section` explicitly rejects negative values beyond roundoff;
raw `polarizability` and `spectral_function` retain their diagnostic values.
The guard detects the supplied frequencies/polarization, not global stability.
Use this model for an appropriate low-population optical window, generally
with phonon energies well below electronic excitations. A smaller eta does
not fix missing thermal-population physics; eta is numerical resolution.

Use a pure-electronic GW–BSE baseline with this self-energy. Feeding already
EP-renormalized excitations and then adding the full EP correction again can
double-count contributions. No subtraction is inferred from `reference`.
All relevant bright and dark intermediate states must be retained; a short
list of bright roots is not a convergence certificate for Pi.

Theory reference: G. Antonius and S. G. Louie, *Theory of exciton-phonon
coupling*, Phys. Rev. B **105**, 085111 (2022),
https://doi.org/10.1103/PhysRevB.105.085111. The implementation above is the
stated finite effective-Hamiltonian approximation, not a claim to reproduce
all diagrams or material calculations from that paper.

## Example and validation

[Native H2/3-21G example](../../../../examples/bse/h2_ep_coupling.py) optimizes
the RHF bond, computes its analytic stretching mode and screened HF vertex,
obtains a finite-displacement quadratic operator, solves all G0W0 levels and
TDA-BSE states, and compares bare/Fan/Fan+DW absorption. The electronic GW
correction and BSE kernel are frozen under displacement. No PySCF or CLI is
used. Results are saved under `reproducibility/gw_bse/h2_bse_phonons`.

Tests cover independent bosonic resolvent sums, charge-cancellation, actual
BSE-kernel derivative projection, compact/full Lambda, zero-EP BSE optics,
complex frame covariance, gradients at degeneracy, direct/GMRES equivalence,
invalid root/solver/model handling and resolved non-passive satellites.

```sh
PYTHONPATH=src JAX_PLATFORMS=cpu python -m pytest -q \
  tests/bse/test_exciton_phonon.py tests/solvers/test_complex_linear.py \
  tests/gw/test_ep_compact_quadratic.py
```
