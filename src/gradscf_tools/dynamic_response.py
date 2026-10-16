"""Complete-space restricted dipole response using shared TD and linear solvers.

The molecule-facing adapter rebuilds and relaxes the same molecular inputs as
MolecularEnergy. No excitation matrix, eigensolver, or SCF algorithm is copied.
"""
from dataclasses import replace
from typing import NamedTuple

import jax
import jax.numpy as jnp

from gradscf.scf.molecules import QuadratureGrid, RestrictedMolecule
from gradscf.solvers import LinearOperator, LinearSolverConfig
from gradscf.solvers.diagnostics import require_converged_derivative
from gradscf.solvers.linear import solve_complex
from gradscf.tddft import gen_tdhf_vind
from gradscf.tddft._semilocal_response import SemilocalResponseFunctional

from .molecular_response import MolecularEnergy


class DynamicPolarizabilityResult(NamedTuple):
    polarizability: object
    residual_norm: object
    converged: object


def dynamic_polarizability(molecule, xc_functional, frequency=0., *, damping=0., config=None):
    """Retarded alpha in bohr^3; scalar frequency/damping in Hartree.

    A real closed-shell reference and the full occupied/virtual space are
    required. The caller owns input freshness, stationarity and stability.
    An explicit AO fock_matrix permits noncanonical occupied/virtual frames;
    without it, the existing canonical mo_energy contract applies.

    With L=[[A,B],[-B,-A]] and position integrals d_x, solve
    (L-(frequency+i*damping)I) Z_x = (d_x,-d_x), then
    alpha_xy=2*d_x.T*(Z_y,upper+Z_y,lower). At zero frequency/damping
    this is 4*d_x.T*(A+B)^-1*d_y. No excitation-root truncation is used.
    The shared complex solver checks primal and transpose residuals, with its
    direct capacity measured on the real representation (4*nocc*nvir).
    """
    c = jnp.asarray(molecule.mo_coeff)
    position = jnp.asarray(molecule.dipole_integrals)
    if c.ndim != 2 or jnp.iscomplexobj(c) or jnp.iscomplexobj(position):
        raise NotImplementedError("Dynamic restricted response requires real closed-shell arrays.")
    if position.shape != (3, c.shape[0], c.shape[0]):
        raise ValueError("dipole_integrals must have shape (3, nao, nao).")
    if molecule.nocc is None:
        raise ValueError("Supply static nocc for the complete restricted response space.")
    nocc = int(molecule.nocc)
    n = nocc*(c.shape[1]-nocc)
    if n <= 0:
        raise ValueError("Need nonempty occupied and virtual spaces.")
    frequency, damping = jnp.asarray(frequency), jnp.asarray(damping)
    if (frequency.ndim or damping.ndim or jnp.iscomplexobj(frequency)
            or jnp.iscomplexobj(damping)):
        raise ValueError("Frequency and damping must be real scalars.")
    valid = jnp.isfinite(frequency) & jnp.isfinite(damping) & (damping >= 0)
    if not isinstance(valid, jax.core.Tracer) and not bool(valid):
        raise ValueError("Frequency must be finite and damping finite and nonnegative.")
    shift = frequency+1j*damping
    dipole = jnp.einsum('pi,xpq,qa->xia', c[:, :nocc], position, c[:, nocc:]).reshape(3, n).T
    rows = gen_tdhf_vind(molecule, xc_functional)

    def real_action(value):
        return rows(value[None, :])[0] if value.ndim == 1 else rows(value.T).T

    def action(value):
        # Existing AO/XC kernels are real-linear. Preserve the imaginary part
        # explicitly instead of allowing their real dtype casts to discard it.
        return real_action(value.real)+1j*real_action(value.imag)-shift*value

    operator = LinearOperator((2*n, 2*n), jnp.result_type(c, shift), action, matmat=action)
    rhs = jnp.concatenate([dipole, -dipole], axis=0)
    if config is None:
        config = LinearSolverConfig(method='direct', rtol=1e-11, atol=1e-13, max_dense=2048)
    result = solve_complex(operator, rhs, config=config)
    alpha = 2*dipole.T@(result.solution[:n]+result.solution[n:])
    valid = valid & result.converged
    alpha = require_converged_derivative(jnp.where(valid, alpha, jnp.nan), valid)
    return DynamicPolarizabilityResult(alpha, result.residual_norm, valid)


class MolecularPolarizability:
    """Relaxed restricted TDHF/adiabatic TDDFT alpha(R, frequency+i*damping).

    R is in bohr and alpha in bohr^3. Every call reassembles native full ERIs,
    the moving DFT grid/AOs and fixed-origin dipoles, and relaxes the orbitals.
    The final AO Fock matrix enters the existing TD builder dynamically, so
    no canonical eigenvectors or diagonal-only orbital-energy approximation
    are introduced. Geometry derivatives include the SCF, AO/MO transition
    operator and response-kernel chains. Nuclear response is clamped: nuclei
    move only through the externally supplied geometry direction.

    Convergence of a driven linear system does not establish ground-state
    stability. The examples check stability and independent static limits.
    This helper implements real closed-shell electric singlet response only.
    """

    def __init__(self, mf, *, gradient_tolerance=1e-10, config=None):
        self.energy = MolecularEnergy(mf, gradient_tolerance=gradient_tolerance)
        if self.energy.unrestricted:
            raise NotImplementedError("MolecularPolarizability currently supports restricted closed shells only.")
        self.nocc = int(jnp.count_nonzero(self.energy.occupations[0]))
        self.functional = SemilocalResponseFunctional(self.energy.xc)
        self.config = config

    def reference(self, coordinates):
        """Return a fresh array reference and its stationary flag, inside AD."""
        state, inputs = self.energy.solve_with_inputs(coordinates)
        c = state.mo_coeff
        fock = state.fock_matrix[0]
        moving = replace(self.energy.parameters, centers=coordinates[self.energy.owners],
                         nuclear_coords=coordinates)
        position = self.energy.plan.evaluate('dipole', moving, origin=jnp.zeros(3))
        eri = inputs['eri']
        extra = {}
        if self.energy.rsh is not None:
            extra = dict(eri_lr=inputs['eri_lr'], dispersion_energy=inputs['dispersion_energy'],
                         range_separated_coefficients=self.energy.rsh.to_range_separated_hybrid_coefficients())
        reference = RestrictedMolecule(
            ao=inputs['ao'], ao_deriv1=inputs['ao_deriv1'],
            grid=QuadratureGrid(inputs['grid_weights']), dipole_integrals=position,
            rep_tensor=eri if eri.ndim == 4 else jnp.zeros(0, dtype=eri.dtype),
            eri_pair_matrix=eri if eri.ndim == 2 else None,
            mo_coeff=c, mo_occ=state.mo_occ[0]*2, mo_energy=jnp.diag(c.T@fock@c),
            rdm1=state.density_matrix, h1e=inputs['hcore'],
            nuclear_repulsion=inputs['nuclear_repulsion'], overlap_matrix=inputs['overlap'],
            nocc=self.nocc, fock_matrix=fock, **extra)
        return reference, state.stationary

    def __call__(self, coordinates, frequency=0., *, damping=0.):
        reference, stationary = self.reference(jnp.asarray(coordinates))
        result = dynamic_polarizability(reference, self.functional, frequency,
                                         damping=damping, config=self.config)
        valid = stationary & result.converged
        alpha = require_converged_derivative(jnp.where(valid, result.polarizability, jnp.nan), valid)
        return DynamicPolarizabilityResult(alpha, result.residual_norm, valid)


__all__ = ['DynamicPolarizabilityResult', 'dynamic_polarizability', 'MolecularPolarizability']
