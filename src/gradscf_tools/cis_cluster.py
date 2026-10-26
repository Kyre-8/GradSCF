"""Relaxed real RHF/CIS cluster observables using existing GradSCF response.

No labeled eigenvector derivative is assigned inside a degenerate cluster.
The production observable applies its projector to three moving dipole probes;
it does not form the full excitation-space projector.
"""
from dataclasses import replace
from numbers import Integral
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from gradscf.solvers import (EigenResponseConfig, EigenSolverConfig, LinearOperator,
                            LinearSolverConfig, solve_hermitian)
from gradscf.solvers.diagnostics import require_converged_derivative
from gradscf.tddft._semilocal_response import SemilocalResponseFunctional
from gradscf.tddft.response import build_restricted_tda_operator
from .dynamic_response import MolecularPolarizability


class CISClusterResult(NamedTuple):
    transition_tensor: object
    excitation_energy_sum: object
    scf_stationary: object
    converged: object
    response_valid: object
    boundary_gaps: object
    max_ritz_residual: object


class CISClusterResponse(NamedTuple):
    state: CISClusterResult
    gradient: object
    tensor_jvp: object
    jvp_vjp_error: object


def _require_response(state):
    if not bool(state.scf_stationary):
        raise RuntimeError("Relaxed RHF is not stationary")
    if not bool(state.converged):
        raise RuntimeError(f"CIS Ritz solve did not converge; maximum residual {float(state.max_ritz_residual):.3e}")
    if not bool(state.response_valid):
        raise ArithmeticError("CIS cluster boundary is not spectrally isolated or numerically resolved")
    if not np.isfinite(np.asarray(state.transition_tensor)).all():
        raise ArithmeticError("CIS transition probes or tensor are not finite")


def _operator_and_probes(reference):
    if reference.fock_matrix is None:
        raise ValueError("CIS cluster response requires the complete AO fock_matrix")
    coefficients = jnp.asarray(reference.mo_coeff)
    position = jnp.asarray(reference.dipole_integrals)
    nocc = reference.nocc
    if (coefficients.ndim != 2 or jnp.iscomplexobj(coefficients)
            or position.shape != (3, coefficients.shape[0], coefficients.shape[0])
            or not isinstance(nocc, Integral) or not 0 < nocc < coefficients.shape[1]):
        raise ValueError("Use a real closed-shell reference with occupied/virtual spaces")
    dimension = nocc*(coefficients.shape[1]-nocc)
    action, diagonal, _ = build_restricted_tda_operator(reference,
        SemilocalResponseFunctional("hf"), response_kernel_options={"two_electron_mode": "direct"})
    operator = LinearOperator((dimension, dimension), diagonal.dtype,
        lambda v: action(v[None, :])[0], diagonal=diagonal, matmat=lambda v: action(v.T).T)
    probes = -jnp.sqrt(2.)*jnp.einsum("pi,xpq,qa->iax", coefficients[:, :nocc],
        position, coefficients[:, nocc:]).reshape(dimension, 3)
    return operator, probes


def cluster_properties(reference, bounds, *, config=None, response_config=None, stationary=True):
    """Return D.T P D and the excitation-energy sum for a fixed [start,stop).

    ``reference`` must be a stationary RHF reference in a real, possibly
    noncanonical MO frame. Its Fock matrix, integrals, coefficients and dipoles
    are all live inputs. Only complete clusters separated from the exterior
    have defined response. The caller keeps bounds static during AD.
    Cumulative projector differences handle a cluster above lower states.
    """
    operator, probes = _operator_and_probes(reference)
    if (len(bounds) != 2 or any(not isinstance(x, Integral) or isinstance(x, bool) for x in bounds)
            or not 0 <= bounds[0] < bounds[1] <= operator.shape[0]):
        raise ValueError("bounds must satisfy 0 <= start < stop <= dimension")
    start, stop = map(int, bounds)
    config = config or EigenSolverConfig(method="dense", nroots=stop, atol=1e-10)
    policy = EigenResponseConfig(target="subspace", linear_config=response_config or
        LinearSolverConfig(method="gmres", rtol=1e-10, atol=1e-12, restart=40, maxiter=100))
    ranks = (stop, start) if start else (stop,)
    states = tuple(solve_hermitian(operator, config=replace(config, nroots=rank),
                                  response=policy, probes=probes) for rank in ranks)
    projected, energy = states[0].projection, states[0].eigenvalue_sum
    if start:
        projected, energy = projected-states[1].projection, energy-states[1].eigenvalue_sum
    converged = jnp.all(jnp.stack([state.converged for state in states]))
    valid = jnp.asarray(stationary) & jnp.all(jnp.stack([state.response_valid for state in states]))
    tensor = probes.T@projected
    # Keep primal diagnostics live. Replacing the entire invalid observable
    # by a constant NaN would disconnect its VJP and can silently yield zero.
    tensor, energy = (require_converged_derivative(value, valid) for value in (tensor, energy))
    return CISClusterResult(tensor, energy, jnp.asarray(stationary), converged, valid,
        jnp.stack([state.boundary_gap for state in states]),
        jnp.max(jnp.stack([jnp.max(state.raw_residual_norms) for state in states])))


class CISCluster:
    """A complete bright singlet CIS cluster with fully relaxed geometry response.

    Native integrals, AO metric transport and implicit HF are supplied by
    MolecularEnergy through the existing MolecularPolarizability reference
    adapter. The TDA HF operator and shared subspace rule supply all excited
    response. No canonical MO derivative, SCF iteration or solver is copied.
    Coordinates are bohr; D.T P D has units e² bohr². Nuclear dipoles have zero
    ground-to-excited matrix element and do not enter these transition probes.

    Call discover once, outside JIT/AD, before constructing derivative graphs.
    The resulting static cluster is kept at displaced geometries; an exterior
    gap closure invalidates response instead of relabeling individual roots.
    SCF stationarity alone does not certify ground-state stability.
    """
    def __init__(self, mf, *, method="davidson", max_subspace=32,
                 gradient_tolerance=1e-10, response_config=None):
        if str(mf.xc).lower() != "hf" or np.asarray(mf.mo_coeff).ndim != 2:
            raise NotImplementedError("CISCluster requires a real restricted HF reference")
        self.reference_model = MolecularPolarizability(mf, gradient_tolerance=gradient_tolerance)
        self.nocc = self.reference_model.nocc
        nmo = mf.mo_coeff.shape[1]
        self.dimension = self.nocc*(nmo-self.nocc)
        self.config = EigenSolverConfig(method=method, nroots=1, atol=1e-10,
            maxiter=150, max_subspace=min(max_subspace, self.dimension),
            max_dense=self.dimension, seed=11)
        self.response_config = response_config
        self.bounds = None

    def reference(self, coordinates, *, frame=None):
        reference, stationary = self.reference_model.reference(jnp.asarray(coordinates))
        if frame is not None:
            frame = np.asarray(frame)
            n = reference.mo_coeff.shape[1]
            if (frame.shape != (n, n) or np.iscomplexobj(frame)
                    or not np.allclose(frame.T@frame, np.eye(n), atol=1e-12, rtol=0)
                    or np.max(abs(frame[:self.nocc, self.nocc:]), initial=0.) > 1e-12
                    or np.max(abs(frame[self.nocc:, :self.nocc]), initial=0.) > 1e-12):
                raise ValueError("frame must be a fixed real block-orthogonal occupied/virtual rotation")
            coefficients = reference.mo_coeff@jnp.asarray(frame)
            reference = replace(reference, mo_coeff=coefficients,
                mo_energy=jnp.diag(coefficients.T@reference.fock_matrix@coefficients))
        return reference, stationary

    def discover(self, coordinates, *, rank=3, degeneracy_tolerance=1e-7, brightness_tolerance=1e-8):
        """Bounded dense forward discovery; no derivative or root assignment."""
        if self.bounds is not None:
            raise RuntimeError("Cluster already selected; construct a new CISCluster to select again")
        reference, stationary = jax.jit(self.reference)(jnp.asarray(coordinates))
        if not bool(stationary):
            raise RuntimeError("Relaxed HF is not stationary")
        operator, probes = _operator_and_probes(reference)
        state = solve_hermitian(operator, config=replace(self.config, method="dense",
            nroots=self.dimension, max_subspace=None),
            response=EigenResponseConfig(target="subspace"))
        if not bool(state.converged):
            raise RuntimeError("Complete dense CIS discovery did not converge")
        energies, vectors, dipole = map(np.asarray, (state.raw_values, state.raw_vectors, probes))
        cuts = np.r_[0, np.flatnonzero(np.diff(energies) > degeneracy_tolerance)+1, self.dimension]
        groups, selected = [], None
        for start, stop in zip(cuts[:-1], cuts[1:]):
            moments = vectors[:, start:stop].T@dipole
            tensor = moments.T@moments
            row = dict(start=int(start), stop=int(stop), rank=int(stop-start),
                energies_hartree=energies[start:stop].tolist(),
                splitting_hartree=float(np.ptp(energies[start:stop])),
                lower_gap_hartree=float(energies[start]-energies[start-1]) if start else None,
                upper_gap_hartree=float(energies[stop]-energies[stop-1]) if stop < self.dimension else None,
                transition_tensor=tensor.tolist(), brightness=float(np.trace(tensor)))
            groups.append(row)
            if selected is None and stop-start == rank and row["brightness"] > brightness_tolerance:
                selected = row
        if selected is None:
            raise RuntimeError("No complete bright cluster of the requested rank was found")
        if self.config.method == "davidson" and self.config.max_subspace < min(self.dimension, selected["stop"]+3):
            raise ValueError("max_subspace cannot hold the cumulative cluster and guard/expansion slots")
        self.bounds = (selected["start"], selected["stop"])
        return dict(selected=selected, clusters=groups, dimension=self.dimension,
                    scf_stationary=True, max_ritz_residual=float(jnp.max(state.raw_residual_norms)))

    def properties(self, coordinates, *, frame=None):
        if self.bounds is None:
            raise RuntimeError("Call discover outside JIT/AD before evaluating the fixed cluster")
        reference, stationary = self.reference(coordinates, frame=frame)
        return cluster_properties(reference, self.bounds, config=self.config,
                                  response_config=self.response_config, stationary=stationary)

    def geometry_response(self, coordinates, weights, direction, *, frame=None):
        """Checked host orchestration of a full VJP and one directional JVP.

        Call outside JIT/AD; use properties inside nested transformations.
        Primal certificates are checked before applying the pullback. A failed
        Ritz solve is distinct from an unresolved/undefined cluster boundary.
        Both returned derivatives must also be finite.
        """
        coordinates, weights, direction = map(jnp.asarray, (coordinates, weights, direction))
        if any(isinstance(value, jax.core.Tracer) for value in (coordinates, weights, direction)):
            raise TypeError("geometry_response is host orchestration; transform properties instead")
        if weights.shape != (3, 3) or direction.shape != coordinates.shape:
            raise ValueError("Use a (3,3) tensor cotangent and a full coordinate direction")
        def evaluate(r):
            result = self.properties(r, frame=frame)
            return result.transition_tensor, result
        evaluate = jax.jit(evaluate)
        _, pullback, state = jax.vjp(evaluate, coordinates, has_aux=True)
        _require_response(state)
        gradient = pullback(weights)[0]
        _, tangent, tangent_state = jax.jvp(evaluate, (coordinates,), (direction,), has_aux=True)
        _require_response(tangent_state)
        if not (np.isfinite(np.asarray(gradient)).all() and np.isfinite(np.asarray(tangent)).all()):
            raise ArithmeticError("CIS cluster geometry response is not finite; inspect the response solves")
        duality = jnp.abs(jnp.sum(weights*tangent)-jnp.sum(gradient*direction))
        return CISClusterResponse(state, gradient, tangent, duality)


__all__ = ["CISCluster", "CISClusterResult", "CISClusterResponse", "cluster_properties"]
