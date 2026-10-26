"""Native RHF analytic response ingredients for molecular electron phonons.

Energies are Hartree, coordinates are bohr, and stationary gradients are
Hartree/bohr. Integrals and their geometry derivatives use GradSCF's native
backend; no external SCF, integrals, or forces are used. Orbital transport
defines an effective RHF Hamiltonian in a fixed reference MO frame. Its
derivative includes self-consistent HF screening, not a GW self-energy vertex.
"""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from gradscf import integrals
from gradscf.integrals.basis import BasisParameters, BasisTopology
from gradscf.integrals.molecular.jk import exchange_matrix
from gradscf.scf.energy import XCContribution, restricted_energy
from gradscf.scf.rhf import nuclear_repulsion_energy
from gradscf.scf.rks import RKSConfig, run_rks_from_integrals_traceable


def align_fock(frame_overlap, energies, *, min_overlap=0.95):
    """Return polar-transported canonical Fock and smallest overlap singular value.

    ``frame_overlap`` is ``C_ref.T @ S_cross @ C_moving``. Its polar factor
    removes basis-subspace metric contraction while preserving maximum overlap.
    Sign, permutation, and degenerate-subspace canonical-orbital gauges cancel.
    The threshold guards against steps for which the two AO spaces differ too
    much for this local transport model.
    """
    overlap = np.asarray(frame_overlap, dtype=float)
    energies = np.asarray(energies, dtype=float)
    if overlap.ndim != 2 or overlap.shape != (energies.size, energies.size):
        raise ValueError("Frame overlap must be square and match orbital energies.")
    if energies.ndim != 1 or not np.all(np.isfinite(overlap)) or not np.all(np.isfinite(energies)):
        raise ValueError("Frame overlap and orbital energies must be finite arrays.")
    left, singular, right = np.linalg.svd(overlap, full_matrices=False)
    smallest = float(singular.min())
    if smallest < min_overlap:
        raise ValueError(f"Minimum orbital-frame overlap {smallest:.8f} is below {min_overlap}.")
    rotation = left @ right
    fock = (rotation * energies[None, :]) @ rotation.T
    return (fock + fock.T) * .5, smallest


class NativeRHF:
    """Reuse native integral plans and compiled SCF/gradient functions.

    The molecular shell ownership and all exponents/contractions remain fixed.
    Only atom-centered coordinates move. A gradient is the stationary RHF
    Pulay Lagrangian derivative with converged density and energy-weighted
    density held fixed. ``response`` differentiates this gradient using native
    second derivatives and the shared implicit orbital-stationarity response;
    it never differentiates the SCF loop or canonical eigenvector gauges.
    Float64 must be enabled before constructing this object.
    """

    def __init__(self, atom, basis="sto-3g", *, unit="Bohr", charge=0, config=None):
        if not jax.config.x64_enabled:
            raise ValueError("NativeRHF requires JAX float64 enabled.")
        self.topology, self.parameters = integrals.prepare_basis(
            atom, basis, unit=unit, charge=charge, spin=0)
        self.coordinates = np.asarray(self.parameters.nuclear_coords).copy()
        distance = np.linalg.norm(np.asarray(self.parameters.centers)[:, None, :]
                                  - self.coordinates[None, :, :], axis=-1)
        owners = np.argmin(distance, axis=1)
        if not np.all(distance[np.arange(len(owners)), owners] < 1e-12):
            raise ValueError("All shells must be centered on molecular atoms.")
        self.owners = jnp.asarray(owners)
        self.charges = jnp.asarray(self.topology.nuclear_charges)
        self.nelectron = sum(self.topology.nuclear_charges) - charge
        if self.nelectron <= 0 or self.nelectron % 2 or self.nelectron > 2 * self.topology.nao:
            raise ValueError("NativeRHF requires a valid positive closed-shell electron count.")
        self.config = config or RKSConfig(xc_spec="hf", max_cycle=100,
                                          conv_tol=1e-12, conv_tol_density=1e-10,
                                          conv_tol_grad=1e-9)
        if self.config.xc_spec.lower() != "hf" or self.config.jk_backend != "full":
            raise ValueError("NativeRHF requires xc_spec='hf' and jk_backend='full'.")
        self.plan = integrals.make_plan(self.topology, backend="native")
        self._integrals = jax.jit(self._evaluate_integrals)
        self._scf = jax.jit(self._run_scf)
        self._gradient = jax.jit(jax.grad(self._lagrangian, argnums=0))
        doubled = BasisTopology(
            self.topology.angular_momenta * 2, self.topology.primitive_counts * 2,
            self.topology.contraction_counts * 2, self.topology.nuclear_charges,
            self.topology.cart)
        self.cross_plan = integrals.make_plan(doubled, backend="native")
        self._cross_overlap = jax.jit(self._evaluate_cross_overlap)

    def _parameters_at(self, coordinates):
        return replace(self.parameters, nuclear_coords=coordinates,
                       centers=coordinates[self.owners])

    def _evaluate_integrals(self, coordinates):
        parameters = self._parameters_at(coordinates)
        overlap = self.plan.evaluate("overlap", parameters)
        hcore = (self.plan.evaluate("kinetic", parameters)
                 + self.plan.evaluate("nuclear", parameters))
        eri = self.plan.evaluate("eri", parameters)
        enuc = nuclear_repulsion_energy(coordinates, self.charges)
        return overlap, hcore, eri, enuc

    def _run_scf(self, overlap, hcore, eri, enuc, init_density):
        nao = self.topology.nao
        return run_rks_from_integrals_traceable(
            overlap=overlap, hcore=hcore, eri=eri, nelectron=self.nelectron,
            nuclear_repulsion=enuc, ao=jnp.zeros((0, nao)),
            ao_deriv1=jnp.zeros((3, 0, nao)), grid_weights=jnp.zeros(0),
            init_density=init_density, config=self.config)

    def _lagrangian(self, coordinates, density, weighted_density):
        overlap, hcore, eri, enuc = self._evaluate_integrals(coordinates)
        coulomb = jnp.einsum("pqrs,rs->pq", eri, density)
        exchange = exchange_matrix(eri, density)
        xc = XCContribution(energy=jnp.asarray(0.), potential=jnp.zeros_like(density),
                            exact_exchange_fraction=jnp.asarray(1.))
        return (restricted_energy(density, hcore, coulomb, exchange, xc,
                                  nuclear_repulsion=enuc)
                - jnp.einsum("pq,qp->", weighted_density, overlap))

    def _check_coordinates(self, coordinates):
        coordinates = np.asarray(coordinates, dtype=float)
        if coordinates.shape != self.coordinates.shape or not np.all(np.isfinite(coordinates)):
            raise ValueError("Coordinates must be a finite (natom, 3) array in bohr.")
        return jnp.asarray(coordinates)

    def integrals(self, coordinates):
        """Return native ``(overlap, hcore, eri, nuclear_repulsion)`` arrays."""
        return self._integrals(self._check_coordinates(coordinates))

    def evaluate(self, coordinates, init_density=None, *, gradient=True):
        """Return ``(RKSResult, gradient_or_None)``; raise on failed SCF convergence."""
        coordinates = self._check_coordinates(coordinates)
        result = self._scf(*self._integrals(coordinates), init_density)
        if not bool(result.converged):
            raise RuntimeError(f"Native RHF did not converge in {int(result.cycles)} cycles.")
        if not gradient:
            return result, None
        coeff = result.mo_coeff
        density = (coeff * result.mo_occ[None, :]) @ coeff.T
        weighted = (coeff * (result.mo_occ * result.mo_energy)[None, :]) @ coeff.T
        force_gradient = self._gradient(coordinates, density, weighted)
        return result, np.asarray(force_gradient)

    def _evaluate_cross_overlap(self, reference_coordinates, moving_coordinates):
        centers = jnp.concatenate([reference_coordinates[self.owners],
                                   moving_coordinates[self.owners]], axis=0)
        parameters = BasisParameters(self.parameters.exponents * 2,
                                     self.parameters.coefficients * 2,
                                     centers, reference_coordinates)
        overlap = self.cross_plan.evaluate("overlap", parameters)
        nao = self.topology.nao
        return overlap[:nao, nao:]

    def response(self, coordinates, reference, *, differentiation=None):
        """Build a reusable analytic ``direction -> (H @ direction, dF)``.

        ``reference`` must be a converged RHF state at ``coordinates``. The
        first output has coordinate shape and units Ha/bohr^2; the second is
        Ha/bohr in the reference MO frame, using the maximum-overlap transport
        of :meth:`transport`. Directions are dimensionless Cartesian vectors.
        Both include the full self-consistent density response. No displaced
        SCF or coordinate ERI Jacobian is formed. The supplied state is used as
        the stationary root without another minimization.

        The first call compiles the directional calculation; later calls reuse
        it. ``differentiation`` accepts an unregularized implicit
        SCFDifferentiationConfig; regularization would bias the exact response.
        Failed root/linear response checks raise rather than returning NaNs.
        """
        from gradscf.scf.autodiff import SCFDifferentiationConfig, attach_scf_backward
        from gradscf.scf.orbital_optimization import _orthonormalize, _problem_functions

        coordinates = self._check_coordinates(coordinates)
        if not bool(reference.converged):
            raise ValueError('Analytic response requires a converged RHF reference.')
        config = differentiation or SCFDifferentiationConfig()
        if config.mode != 'implicit' or not config.require_converged:
            raise ValueError('Analytic response requires implicit, convergence-checked differentiation.')
        if config.regularization != 0:
            raise ValueError('Analytic response requires zero regularization for the exact Hessian.')
        coeff = jnp.asarray(reference.mo_coeff)
        if jnp.iscomplexobj(coeff) or coeff.dtype != jnp.float64:
            raise ValueError('Reference MO coefficients must be real float64.')
        occ = np.asarray(reference.mo_occ)
        nao = self.topology.nao
        if (coeff.shape != (nao, nao) or occ.shape != (nao,)
                or not np.all(np.isfinite(coeff))
                or not np.all((occ == 0.) | (occ == 2.))
                or occ.sum() != self.nelectron):
            raise ValueError('Reference must contain complete, finite closed-shell RHF orbitals.')
        occupations = tuple(tuple(float(x) for x in occ/2) for _ in range(2))
        dimension, rotate, density_from, evaluate, energy = _problem_functions(
            'roks', 'hf', 'col', occupations)
        anchor = jax.lax.stop_gradient(coeff[None])
        origin = jnp.zeros(dimension, dtype=coeff.dtype)
        stationarity_tolerance = max(1e-8, 10*self.config.conv_tol_grad)

        def tangent_energy(angles, root_inputs):
            args, root_anchor = root_inputs
            return energy(angles, _orthonormalize(root_anchor, args['metric']), args)

        residual = jax.grad(tangent_energy)

        def state(r):
            overlap, hcore, eri, enuc = self._evaluate_integrals(r)
            args = dict(metric=overlap, hcore=hcore, eri=eri, nuclear_repulsion=enuc,
                        ao=jnp.zeros((0, nao)), ao_deriv1=jnp.zeros((3, 0, nao)),
                        grid_weights=jnp.zeros(0))
            base = _orthonormalize(anchor, overlap)
            # The local polar derivative below assumes M=I. A converged flag
            # alone cannot establish that the reference belongs to this metric.
            valid = jnp.max(jnp.abs(coeff.T @ overlap @ coeff-jnp.eye(nao))) <= 1e-8
            if dimension:
                stationary = jnp.linalg.norm(residual(origin, (args, anchor))) <= stationarity_tolerance
                valid = valid & stationary
                angles = attach_scf_backward((args, anchor), solution=origin,
                                             residual=residual, config=config,
                                             converged=valid)
                orbitals = rotate(angles, base)
            else:
                orbitals = base
            spin_density = density_from(orbitals)
            spin_density = jnp.where(valid, spin_density, jnp.nan)
            density = jnp.sum(spin_density, axis=0)
            _, spin_fock = evaluate(spin_density, args)
            fock = spin_fock[0]
            # Equivalent to C occ epsilon C.T at stationarity, with no
            # eigenvector derivatives inside equal-occupation/degenerate blocks.
            weighted = .5*density @ fock @ density
            frame_fock = base[0].T @ fock @ base[0]
            frame_overlap = coeff.T @ self._evaluate_cross_overlap(coordinates, r) @ base[0]
            return density, weighted, frame_fock, frame_overlap

        @jax.jit
        def directional(direction):
            (density, weighted, fock, _), (ddensity, dweighted, dfock, doverlap) = jax.jvp(
                state, (coordinates,), (direction,))
            valid = jnp.all(jnp.isfinite(jnp.stack([density, weighted, ddensity, dweighted])))
            # The native derivative contractions reject nonfinite tangent
            # buffers. Keep failed implicit solves out of that FFI call so the
            # public wrapper can report a response convergence error clearly.
            hvp = jax.lax.cond(valid, lambda: jax.jvp(
                self._gradient, (coordinates, density, weighted),
                (direction, ddensity, dweighted))[1],
                lambda: jnp.full_like(coordinates, jnp.nan))
            # At the reference M=I, d polar(M) = skew(dM). This avoids
            # differentiating an SVD with completely degenerate singular values.
            rotation = .5*(doverlap-doverlap.T)
            transported = dfock + rotation @ fock - fock @ rotation
            return hvp, .5*(transported+transported.T)

        def checked_response(direction):
            direction = self._check_coordinates(direction)
            hvp, fock = map(np.asarray, directional(direction))
            if not np.all(np.isfinite(hvp)) or not np.all(np.isfinite(fock)):
                raise RuntimeError('Native RHF analytic response failed stationarity or linear convergence checks.')
            return hvp, fock

        return checked_response

    def cross_overlap(self, reference_coordinates, moving_coordinates):
        """Native AO overlap with reference functions on the left."""
        return self._cross_overlap(self._check_coordinates(reference_coordinates),
                                   self._check_coordinates(moving_coordinates))

    def transport(self, result, reference_coeff, reference_coordinates, coordinates,
                  *, min_overlap=0.95):
        """Transport converged canonical Fock to the reference canonical MO frame."""
        cross = np.asarray(self.cross_overlap(reference_coordinates, coordinates))
        frame_overlap = np.asarray(reference_coeff).T @ cross @ np.asarray(result.mo_coeff)
        return align_fock(frame_overlap, result.mo_energy, min_overlap=min_overlap)


__all__ = ["NativeRHF", "align_fock"]
