"""Small CIS geometry workflows using the existing HF and CI solvers.

Nuclear gradients and total-energy Hessian-vector products use the shared
implicit response. State matching uses normalized transition-density overlaps,
not many-electron wavefunction overlaps.
"""
from dataclasses import dataclass, replace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from gradscf import ci
from gradscf.integrals.molecular.ao2mo import transform_integrals
from .molecular_ep import NativeRHF
from .molecular_response import MolecularEnergy


class CISOneBody(NamedTuple):
    ground: object
    unrelaxed: object
    attachment: object
    detachment: object
    transition: object


class CISNTOs(NamedTuple):
    weights: object
    holes: object
    particles: object
    degenerate_groups: tuple


def _singlet_cis_arrays(coefficients, amplitude, nocc):
    c, x = jnp.asarray(coefficients), jnp.asarray(amplitude)
    if (c.ndim != 2 or x.ndim != 2 or not 0 < nocc < c.shape[1]
            or x.shape != (nocc, c.shape[1]-nocc)):
        raise ValueError("CIS amplitudes must span the complete occupied/virtual MO blocks.")
    if jnp.iscomplexobj(c) or jnp.iscomplexobj(x):
        raise NotImplementedError("These singlet CIS one-body properties require real arrays.")
    return c, x


def singlet_cis_densities(coefficients, amplitude, *, nocc):
    """AO one-body fields of a normalized real singlet CIS wavefunction.

    The occupied/virtual MO columns must be metric-orthonormal. A nonzero
    amplitude is normalized as a state vector; callers separately check the
    solver's own normalization. Attachment and detachment each carry one
    electron. The unrelaxed density is D0+attachment-detachment. It is distinct
    from the total-energy derivative density returned by density_and_dipole.
    Transition[p,q]=<excited|a_q^dagger a_p|ground> has occupied-virtual block
    sqrt(2)*X; its metric Frobenius norm squared is two. No eigensolve is used.
    """
    c, x = _singlet_cis_arrays(coefficients, amplitude, nocc)
    norm = jnp.sum(x*x)
    valid = (norm > 0) & jnp.isfinite(norm) & jnp.all(jnp.isfinite(c))
    x = x/jnp.sqrt(jnp.where(valid, norm, 1.))
    occupied, virtual = c[:, :nocc], c[:, nocc:]
    ground = 2*occupied@occupied.T
    detachment = occupied@(x@x.T)@occupied.T
    attachment = virtual@(x.T@x)@virtual.T
    transition = jnp.sqrt(2.)*occupied@x@virtual.T
    values = (ground, ground+attachment-detachment, attachment, detachment, transition)
    return CISOneBody(*(jnp.where(valid, value, jnp.nan) for value in values))


def singlet_cis_ntos(coefficients, amplitude, *, nocc, degeneracy_tolerance=1e-8):
    """Host-side NTO analysis; individual degenerate SVD frames have no AD rule.

    Weights sum to one. Each pair reconstructs sqrt(2)*s*hole*particle.T in the
    transition matrix. Pair signs are arbitrary; degenerate groups represent
    subspaces, whose weighted density projectors are the invariant quantities.
    """
    c, x = _singlet_cis_arrays(coefficients, amplitude, nocc)
    c, x = np.asarray(c), np.asarray(x)
    norm = np.linalg.norm(x)
    if not np.isfinite(norm) or norm == 0 or not np.isfinite(c).all():
        raise ValueError("Supply finite coefficients and a nonzero CIS amplitude.")
    if not np.isfinite(degeneracy_tolerance) or degeneracy_tolerance < 0:
        raise ValueError("The NTO weight degeneracy tolerance must be nonnegative.")
    left, singular, right = np.linalg.svd(x/norm, full_matrices=False)
    holes, particles = c[:, :nocc]@left, c[:, nocc:]@right.T
    # A reproducible display sign, applied to both members of each pair,
    # preserves their product. It is not a physical orbital phase convention.
    signs = np.sign(holes[np.argmax(np.abs(holes), axis=0), np.arange(len(singular))])
    signs = np.where(signs == 0, 1., signs)
    holes, particles = holes*signs, particles*signs
    weights = singular**2
    groups, begin = [], 0
    for stop in range(1, len(weights)+1):
        if stop == len(weights) or abs(weights[stop]-weights[stop-1]) > degeneracy_tolerance:
            if stop-begin > 1:
                groups.append(tuple(range(begin, stop)))
            begin = stop
    return CISNTOs(weights, holes, particles, tuple(groups))


def transition_density_overlaps(reference_coefficients, reference_amplitude,
                                coefficients, amplitudes, cross_overlap, *, nocc):
    """Compare C_occ X C_vir.T in the cross-geometry AO overlap metric.

    The result is invariant under state signs and separate occupied/virtual
    rotations with consistently transformed amplitudes. Across geometries it
    is a tracking diagnostic, not the overlap of normalized CIS wavefunctions.
    """
    left, right = np.asarray(reference_coefficients), np.asarray(coefficients)
    previous, candidates = np.asarray(reference_amplitude), np.asarray(amplitudes)
    occupied = left[:, :nocc].T @ np.asarray(cross_overlap) @ right[:, :nocc]
    virtual = left[:, nocc:].T @ np.asarray(cross_overlap) @ right[:, nocc:]
    projected = np.einsum("ij,sjb,ab->sia", occupied, candidates, virtual)
    denominator = np.linalg.norm(previous)*np.linalg.norm(candidates, axis=(1, 2))
    if np.any(denominator <= 0) or not np.isfinite(denominator).all():
        raise ValueError("State amplitudes must have finite nonzero norms.")
    return np.abs(np.einsum("ia,sia->s", previous, projected))/denominator


def select_tracked_root(overlaps, energies, *, minimum_overlap=.85,
                        minimum_margin=.20, minimum_gap=1e-4):
    """Reject lost, ambiguous or nearly degenerate tracked states explicitly."""
    overlaps, energies = np.asarray(overlaps), np.asarray(energies)
    if overlaps.ndim != 1 or energies.shape != overlaps.shape or len(energies) < 2:
        raise ValueError("Tracking requires at least two candidate roots.")
    if not np.isfinite(overlaps).all() or not np.isfinite(energies).all():
        raise RuntimeError("Nonfinite state overlap or excitation energy.")
    order = np.argsort(overlaps)[::-1]
    root, runner_up = int(order[0]), float(overlaps[order[1]])
    best = float(overlaps[root])
    if best < minimum_overlap:
        raise RuntimeError(f"Tracked-state overlap {best:.6f} is below {minimum_overlap}.")
    if best-runner_up < minimum_margin:
        raise RuntimeError("Tracked-state match is ambiguous.")
    gap = float(np.min(np.abs(np.delete(energies, root)-energies[root])))
    if gap < minimum_gap:
        raise RuntimeError(f"Tracked-state gap {gap:.3e} Hartree is too small.")
    return dict(root=root, overlap=best, runner_up_overlap=runner_up,
                gap_hartree=gap, overlaps=overlaps.tolist())


class CISGeometryState(NamedTuple):
    ground_energy: object
    excitation_energies: object
    coefficients: object
    amplitudes: object
    scf_stationary: object
    ci_converged: object
    ci_residual_norms: object


@dataclass(frozen=True)
class CISAnchor:
    coordinates: object
    coefficients: object
    amplitude: object
    root: int


class CISGeometry:
    """Relaxed RHF -> noncanonical CIS energy and geometry response.

    MolecularEnergy supplies native moving inputs and implicit HF response.
    Shared AO-to-MO transformation and solve_cis own the electronic physics;
    no A matrix, SCF iteration or eigensolver is implemented here. NativeRHF
    supplies only its previously tested cross-geometry overlap operation.
    """

    def __init__(self, mf, *, nroots=6, gradient_tolerance=1e-10, response=None):
        if str(mf.xc).strip().lower() != "hf" or np.asarray(mf.mo_coeff).ndim != 2:
            raise NotImplementedError("CISGeometry currently requires an RHF reference.")
        self.energy = MolecularEnergy(mf, gradient_tolerance=gradient_tolerance)
        self.nocc = int(np.count_nonzero(np.asarray(mf.mo_occ)))
        self.nroots = nroots
        dimension = self.nocc*(mf.mo_coeff.shape[1]-self.nocc)
        if not 2 <= nroots <= dimension:
            raise ValueError("Choose at least two CIS roots within the excitation space.")
        self.config = ci.CIConfig(nroots=nroots, solver="dense", conv_tol=1e-10,
                                  gradient_mode="eigenvalue_only")
        self.response = response
        self.overlap_engine = NativeRHF(self.energy.spec, basis=self.energy.basis)
        self._state = jax.jit(self._evaluate)
        self._jacobian = jax.jit(jax.jacrev(self._values_with_state, argnums=0, has_aux=True))
        self._total_gradient = jax.jit(jax.value_and_grad(self._total_with_state, argnums=0, has_aux=True))
        self._total_first = jax.grad(self._total_with_state, argnums=0, has_aux=True)
        self._total_hvp = jax.jit(self._hessian_with_state)
        self._electric_first = jax.jit(jax.grad(self._total_with_state, argnums=(2, 3), has_aux=True))
        self._transition_values = jax.jit(self._transition_with_state)
        self._transition_jacobian = jax.jit(jax.jacrev(self._transition_with_state, argnums=0, has_aux=True))
        self._field_first = jax.grad(self._total_with_state, argnums=2, has_aux=True)
        self._field_gradient = jax.jit(self._field_first)
        self._field_hvp = jax.jit(self._field_hessian_with_state)

    def _evaluate(self, coordinates, field=None, potential=None):
        state, inputs = self.energy.solve_with_inputs(coordinates, field, potential=potential)
        coefficients = state.mo_coeff
        h1, eri = transform_integrals(inputs["hcore"], coefficients, eri=inputs["eri"])
        result = ci.solve_cis(h1, eri, nocc=self.nocc, config=self.config, response=self.response)
        valid = state.stationary & jnp.all(result.converged)
        return CISGeometryState(jnp.where(valid, state.total_energy, jnp.nan),
                                jnp.where(valid, result.excitation_energies, jnp.nan),
                                coefficients, result.amplitudes, state.stationary,
                                result.converged, result.residual_norms)

    def _values_with_state(self, coordinates, root, field=None, potential=None):
        state = self._evaluate(coordinates, field, potential)
        omega = state.excitation_energies[root]
        values = jnp.stack([omega, state.ground_energy+omega])
        return values, (values, state)

    def _total_with_state(self, coordinates, root, field=None, potential=None):
        state = self._evaluate(coordinates, field, potential)
        return state.ground_energy+state.excitation_energies[root], state

    def _hessian_with_state(self, coordinates, root, direction):
        _, product, state = jax.jvp(lambda r: self._total_first(r, root),
                                    (coordinates,), (direction,), has_aux=True)
        return product, state

    def _transition_with_state(self, coordinates, root):
        state = self._evaluate(coordinates)
        one_body = singlet_cis_densities(state.coefficients, state.amplitudes[root], nocc=self.nocc)
        moving = replace(self.energy.parameters, centers=coordinates[self.energy.owners],
                         nuclear_coords=coordinates)
        position = self.energy.plan.evaluate("dipole", moving, origin=jnp.zeros(3))
        moment = -jnp.einsum("pq,xqp->x", one_body.transition, position)
        strength = (2./3.)*state.excitation_energies[root]*jnp.sum(moment**2)
        values = jnp.concatenate((moment, strength[None]))
        return values, (values, state)

    def _field_hessian_with_state(self, coordinates, root, field, direction):
        _, product, state = jax.jvp(lambda f: self._field_first(coordinates, root, f),
                                    (field,), (direction,), has_aux=True)
        return product, state

    @staticmethod
    def _check(state):
        if not bool(state.scf_stationary) or not np.all(np.asarray(state.ci_converged)):
            raise RuntimeError("RHF stationarity or CIS eigenstate convergence failed.")
        if not np.isfinite(np.asarray(state.excitation_energies)).all():
            raise RuntimeError("CIS excitation energies are nonfinite.")

    def state(self, coordinates, *, field=None, potential=None):
        state = self._state(jnp.asarray(coordinates), field, potential)
        self._check(state)
        return state

    def transition_properties(self, coordinates, root):
        """Return [mu_x, mu_y, mu_z, f] and state, in length-gauge atomic units.

        Moments are electronic ground-to-CIS transitions. Orthogonal electronic
        states have no nuclear transition term. The state phase is arbitrary.
        """
        values, (_, state) = self._transition_values(jnp.asarray(coordinates), root)
        self._check(state)
        if not np.isfinite(np.asarray(values)).all():
            raise RuntimeError("CIS transition properties are nonfinite.")
        return np.asarray(values), state

    def transition_derivatives(self, coordinates, root):
        """Return transition properties, their Cartesian Jacobian, and state.

        Coordinates are Bohr. Relaxed HF orbitals, CIS amplitudes and moving AO
        dipole integrals all respond. An explicit eigenpair response is required
        so stopped amplitudes cannot silently give an incomplete derivative.
        """
        if self.response is None or self.response.target != "eigenpairs":
            raise ValueError("Transition derivatives require response.target='eigenpairs'.")
        jacobian, (values, state) = self._transition_jacobian(jnp.asarray(coordinates), root)
        self._check(state)
        if not np.isfinite(np.asarray(jacobian)).all() or not np.isfinite(np.asarray(values)).all():
            raise RuntimeError("CIS transition-property nuclear response failed.")
        return np.asarray(values), np.asarray(jacobian), state

    def derivatives(self, coordinates, root):
        """Return ([omega,E0+omega], their Cartesian gradients, state)."""
        gradients, (values, state) = self._jacobian(jnp.asarray(coordinates), root)
        self._check(state)
        if not np.isfinite(np.asarray(gradients)).all():
            raise RuntimeError("CIS first-order nuclear response failed.")
        return np.asarray(values), np.asarray(gradients), state

    def total_gradient(self, coordinates, root):
        """Return total excited energy, Cartesian gradient and state."""
        (energy, state), gradient = self._total_gradient(jnp.asarray(coordinates), root)
        self._check(state)
        if not np.isfinite(np.asarray(gradient)).all():
            raise RuntimeError("CIS first-order nuclear response failed.")
        return float(energy), np.asarray(gradient), state

    def total_hessian_vector(self, coordinates, root, direction):
        """Return H(R) v in Hartree/Bohr^2 for a dimensionless Cartesian v.

        This differentiates the existing total-energy gradient through native
        integrals, relaxed HF orbitals and the shared isolated-root CIS response.
        It requires a converged, isolated state; matching across geometries and
        checking full-Hessian symmetry/rigid residuals remain workflow gates.
        """
        coordinates, direction = np.asarray(coordinates), np.asarray(direction)
        if direction.shape != coordinates.shape or not np.isfinite(direction).all():
            raise ValueError("Supply a finite Cartesian direction matching the coordinates.")
        product, state = self._total_hvp(jnp.asarray(coordinates), root, jnp.asarray(direction))
        self._check(state)
        if not np.isfinite(np.asarray(product)).all():
            raise RuntimeError("CIS second-order nuclear response failed.")
        return np.asarray(product)

    def density_and_dipole(self, coordinates, root, *, field=None):
        """Return relaxed response density, total permanent dipole and state.

        D=d(E0+omega)/dV uses the symmetric AO-potential coordinate convention;
        mu=-d(E0+omega)/dF includes the nuclear field term. The density is an
        energy-derivative response object, distinct from the unrelaxed CIS
        wavefunction 1-RDM; positivity is not an acceptance condition.
        Coordinates are Bohr and the dipole is in atomic units (e Bohr).
        """
        coordinates = jnp.asarray(coordinates)
        field = jnp.zeros(3, dtype=coordinates.dtype) if field is None else jnp.asarray(field)
        n = self.energy.coefficients.shape[0]
        potential = jnp.zeros((n, n), dtype=coordinates.dtype)
        (field_gradient, density), state = self._electric_first(coordinates, root, field, potential)
        self._check(state)
        if not np.isfinite(np.asarray(density)).all() or not np.isfinite(np.asarray(field_gradient)).all():
            raise RuntimeError("CIS relaxed one-body response failed.")
        return np.asarray(density), -np.asarray(field_gradient), state

    def electric_response(self, coordinates, root, *, field=None):
        """Return total excited-state dipole, static polarizability and state.

        At fixed nuclei, mu=-d(E0+omega)/dF and alpha=-d²(E0+omega)/dF².
        The three columns use JVPs of the same total-energy field gradient.
        Nuclear field energy is retained. The selected branch must remain
        isolated; excited-state polarizabilities need not be positive.
        Atomic units are used: dipole e Bohr, field Eh/(e Bohr), alpha au.
        """
        coordinates = jnp.asarray(coordinates)
        field = jnp.zeros(3, dtype=coordinates.dtype) if field is None else jnp.asarray(field)
        gradient, state = self._field_gradient(coordinates, root, field)
        self._check(state)
        columns = []
        for direction in jnp.eye(3, dtype=coordinates.dtype):
            product, current = self._field_hvp(coordinates, root, field, direction)
            self._check(current)
            columns.append(np.asarray(product))
        dipole, alpha = -np.asarray(gradient), -np.column_stack(columns)
        if not np.isfinite(dipole).all() or not np.isfinite(alpha).all():
            raise RuntimeError("CIS total excited-state electric response failed.")
        return dipole, alpha, state

    @staticmethod
    def anchor(coordinates, state, root):
        return CISAnchor(np.array(coordinates), np.array(state.coefficients),
                         np.array(state.amplitudes[root]), int(root))

    def match(self, coordinates, state, anchor):
        cross = self.overlap_engine.cross_overlap(anchor.coordinates, coordinates)
        overlaps = transition_density_overlaps(anchor.coefficients, anchor.amplitude,
                                               state.coefficients, state.amplitudes,
                                               cross, nocc=self.nocc)
        return select_tracked_root(overlaps, state.excitation_energies)
