"""Small molecular GW/BSE geometry maps using GradSCF's existing methods."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from gradscf import bse, gw, scf
from gradscf.integrals.molecular.factorization import eri_pair_matrix_to_df_factors_traceable
from gradscf.integrals.molecular.jk import build_j_from_df
from gradscf.gw.g0w0 import _exchange_mo, _mo_factors
from gradscf.solvers import EigenResponseConfig, EigenSolverConfig, solve_hermitian
from .molecular_response import MolecularEnergy


class MolecularGW:
    """Real Aufbau RHF -> G0W0/evGW0/evGW -> static BSE, with moving native inputs.

    Coordinates are in bohr and energies in Hartree. All AO integrals, the
    metric, HF orbitals, QP roots and screening are rebuilt inside the map.
    Canonical MO roots must be isolated; orbital crossings are not assigned
    individual derivatives. Full AO ERIs and full-shape symmetric pair factors
    restrict this helper to small molecules. Factor eigenvalues must remain
    separated from the fixed truncation threshold.

    evGW0 holds W0 fixed only during its QP iteration. W0 still depends on the
    geometry. evGW updates both G and W. Both use the existing outer implicit
    response; neither differentiates the numerical fixed-point trajectory.
    ``max_cycle`` and ``damp`` control that trajectory; the outer residual
    tolerance remains 1e-10 Hartree for this response adapter.
    A QP residue is a local frequency derivative, not an EOM Dyson weight.
    """

    def __init__(self, mf, *, method="g0w0", nw=100, eta=1e-5, qp_solver="secant", qp_tol=1e-11,
                 factor_tolerance=1e-12, orbs=None, g_orbitals=None,
                 screening_occupied=None, screening_virtual=None, qp_initial=None,
                 max_cycle=100, damp=.3, diis_space=0):
        if str(mf.xc).lower() != "hf" or np.asarray(mf.mo_coeff).ndim != 2:
            raise NotImplementedError("This geometry adapter requires a real RHF reference.")
        if method not in {"g0w0", "evgw0", "evgw"}:
            raise ValueError("Choose g0w0, evgw0 or evgw.")
        if qp_solver not in {"secant", "newton", "hybrid"} or (method != "g0w0" and qp_solver != "secant"):
            raise ValueError("QP solver selects only the inner G0W0 root method.")
        if not np.isfinite(factor_tolerance) or factor_tolerance <= 0:
            raise ValueError("The factor truncation tolerance must be positive and finite.")
        self.reference = MolecularEnergy(mf, gradient_tolerance=1e-10)
        self.coordinates = self.reference.coordinates
        self.method, self.nw, self.eta = method, nw, eta
        self.qp_solver = qp_solver
        self.qp_initial = qp_initial
        self.qp_tol = qp_tol
        self.max_cycle, self.damp = max_cycle, damp
        self.diis_space = diis_space
        self.nmo = mf.mo_coeff.shape[1]
        self.nocc = int(np.count_nonzero(np.asarray(mf.mo_occ)))
        self.factor_tolerance = factor_tolerance
        self.pairs = np.tril_indices(mf.mo_coeff.shape[0])
        self.windows = dict(orbs=orbs, g_orbitals=g_orbitals,
                            screening_occupied=screening_occupied,
                            screening_virtual=screening_virtual)
        self.outer_response = scf.SCFDifferentiationConfig(tolerance=1e-10, max_iter=100, restart=40)

    def _prepare(self, coordinates):
        state, inputs = self.reference.solve_with_inputs(coordinates)
        coefficients = state.mo_coeff
        fock = state.fock_matrix.mean(axis=0)
        fock_mo = coefficients.T@fock@coefficients
        orbitals = solve_hermitian(.5*(fock_mo+fock_mo.T),
            config=EigenSolverConfig(method="dense", nroots=self.nmo, atol=1e-10),
            response=EigenResponseConfig(target="eigenpairs"))
        coefficients = coefficients@orbitals.vectors
        density = state.density_matrix.sum(axis=0)
        metric = inputs["overlap"]
        occupation = coefficients.T@metric@density@metric@coefficients
        expected = jnp.diag(jnp.concatenate([jnp.full(self.nocc, 2.), jnp.zeros(self.nmo-self.nocc)]))
        reference_valid = state.stationary & jnp.all(orbitals.response_valid)
        reference_valid &= jnp.max(jnp.abs(occupation-expected)) < 1e-8
        rows, columns = self.pairs
        pair = inputs["eri"][rows, columns][:, rows, columns]
        factors = eri_pair_matrix_to_df_factors_traceable(
            pair, nao=coefficients.shape[0], tol=self.factor_tolerance)
        common = dict(mo_energy=orbitals.values, mo_coeff=coefficients, nocc=self.nocc,
                      df_factors=factors, fock_matrix=fock, hcore_matrix=inputs["hcore"],
                      density_matrix=density, nw=self.nw,
                      eta=self.eta, **self.windows)
        return common, reference_valid

    def _solve(self, coordinates):
        common, reference_valid = self._prepare(coordinates)
        if self.method == "g0w0":
            result = gw.g0w0_cd_restricted(**common, resolvent_expansion=True,
                                          qp_solver=self.qp_solver, qp_initial=self.qp_initial,
                                          qp_tol=self.qp_tol)
        else:
            result = gw.evgw_cd_restricted(**common, update_w=self.method == "evgw",
                max_iter=self.max_cycle, tol=1e-10, damping=self.damp,
                diis_space=self.diis_space, initial_guess=self.qp_initial,
                differentiation=self.outer_response)
        # Preserve the GW driver's per-level diagnostics on numerical failure.
        # The result's masks certify QP convergence; its implicit rule rejects
        # derivatives of failed roots. An invalid HF frame invalidates all levels.
        result = replace(result, mo_energy=jnp.where(reference_valid, result.mo_energy, jnp.nan),
                          converged=result.converged & reference_valid,
                          converged_mask=result.converged_mask & reference_valid)
        mo_factors = _mo_factors(common["df_factors"], common["mo_coeff"])
        return result, mo_factors

    def pole_seeds(self, coordinates=None, *, max_dense=2048):
        """Host-only eta=0 largest-weight seeds for the full G0W0 orbital space.

        Native HF and the existing direct-RPA pole expansion provide the
        model; the shared Hermitian solver selects isolated spectral groups.
        Use the resulting seeds in the finite-eta CD solve and check its actual
        roots/residues. The seed selector itself is not differentiated and does
        not certify finite-eta global maximum weight or geometric continuity.
        Partial G/W windows are deliberately excluded from this small adapter.
        """
        from gradscf.gw.branches import pole_seeds
        from gradscf.gw.poles import rpa_pole_expansion

        coordinates = self.coordinates if coordinates is None else coordinates
        if any(isinstance(x, jax.core.Tracer) for x in jax.tree.leaves(coordinates)):
            raise TypeError("Molecular pole seeds are host-only; build them outside JIT and AD")
        if self.method != "g0w0":
            raise NotImplementedError("This seed adapter currently constructs G0W0 starting branches")
        allowed = dict(orbs=tuple(range(self.nmo)), g_orbitals=tuple(range(self.nmo)),
                       screening_occupied=tuple(range(self.nocc)),
                       screening_virtual=tuple(range(self.nocc, self.nmo)))
        if any(value is not None and tuple(value) != allowed[name] for name, value in self.windows.items()):
            raise NotImplementedError("Molecular pole seeds currently require the full G/W/QP windows")
        dimension = 1+self.nocc*(self.nmo-self.nocc)*self.nmo
        if dimension > max_dense:
            raise ValueError(f"Pole seed dimension {dimension} exceeds max_dense={max_dense}")
        inputs, valid = self._prepare(jnp.asarray(coordinates))
        if not bool(valid):
            raise RuntimeError("Converge an isolated canonical Aufbau HF reference before making seeds")
        c, factors = inputs["mo_coeff"], inputs["df_factors"]
        mo_factors = _mo_factors(factors, c)
        # Reuse the same exchange subtraction as the production CD driver.
        j_matrix = build_j_from_df(factors, inputs["density_matrix"])
        v_mf = c.T@(inputs["fock_matrix"]-inputs["hcore_matrix"]-j_matrix)@c
        delta_v = jnp.diag(_exchange_mo(mo_factors, self.nocc)-v_mf)
        poles, couplings = rpa_pole_expansion(inputs["mo_energy"],
            mo_factors[:, :self.nocc, self.nocc:], mo_factors)
        return pole_seeds(inputs["mo_energy"]+delta_v, inputs["mo_energy"], self.nocc,
                          poles, couplings, max_dense=max_dense)

    def __call__(self, coordinates):
        """Return a GWResult, including actual W spectrum, QP masks and residues."""
        return self._solve(coordinates)[0]

    def optical(self, coordinates, *, config=None, occupied=None, virtual=None):
        """Return (GW result, BSE result, moving MO position, excitation space).

        This is a complete dOmega/dR or optical-property geometry map, not a
        definition of a variational GW+BSE total excited-state energy.
        Full BSE and TDA have their separate shared response contracts.
        """
        result, factors = self._solve(coordinates)
        space = bse.make_bse_space(self.nmo, self.nocc, occupied=occupied, virtual=virtual)
        screening = bse.make_bse_space(self.nmo, self.nocc,
            occupied=result.screening_occupied, virtual=result.screening_virtual)
        response = bse.run_bse(result.mo_energy, result.screening_energy, factors, space,
            screening_space=screening, qp_computed_mask=result.qp_computed_mask,
            qp_converged_mask=result.converged_mask, config=config)
        moving = replace(self.reference.parameters, centers=coordinates[self.reference.owners],
                         nuclear_coords=coordinates)
        position = self.reference.plan.evaluate("dipole", moving, origin=jnp.zeros(3))
        dipole = jnp.einsum("xpq,pi,qj->xij", position, result.mo_coeff, result.mo_coeff)
        return result, response, dipole, space
