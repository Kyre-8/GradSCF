"""Relaxed molecular MP/CC energies using the shared integral and solver layers."""
import jax
import jax.numpy as jnp
import numpy as np

from gradscf import cc, mp
from gradscf._memory import resolve_memory_mb
from gradscf.ao2mo import transform_integrals, transform_unrestricted_integrals
from gradscf.mp.integrals import from_orbitals
from gradscf.mp.mp2 import evaluate_mp2
from gradscf.solvers import EigenResponseConfig, EigenSolverConfig, solve_hermitian
from .molecular_response import MolecularEnergy


class CorrelatedEnergy:
    """E(R,F,V) for real RHF/UHF-based MP or CCSD/(T), in atomic units.

    The HF state, AO integrals, MO transformation and correlation solution are
    all rebuilt inside the differentiated map. This adapter adds no solver.
    MP and CCSD use the transported occupied/virtual frame directly by default.
    MP's noncanonical Fock resolvent permits internal MO degeneracy. An explicit
    canonical MP configuration instead requires isolated canonical orbitals.
    CCSD(T) adds the bounded-panel general-reference correction in the same
    transported frame. Its JVP/VJP supports second-order response; third-order
    requests raise explicitly. Ordinary MP/CCSD response behavior is unchanged.
    MP2 transforms only occupied/virtual blocks. Explicit ``jk_backend='direct'``
    also removes full moving AO ERIs through the native shell/direct operators.
    Its noncanonical resolvent still owns O(o**2*v**2) state. CC and higher MP
    retain full MO integrals and remain small-system reference paths here.
    ``max_memory`` inherits the SCF source budget and plans MO transformation
    arrays; complete solver/AD process residency requires separate accounting.
    No frozen orbitals, density fitting or DFT reference is used here.

    Correlated dipoles and polarizabilities must be derivatives of this total
    energy. The reference's stationary dipole is only the HF observable.
    """

    def __init__(self, mf, method, *, config=None, gradient_tolerance=1e-10,
                 jk_backend="full", max_memory=None):
        self.method = method.lower()
        if self.method not in {"mp", "mp2", "ccsd", "ccsd(t)"} or str(mf.xc).lower() != "hf":
            raise ValueError("Use MP, CCSD or CCSD(T) with a real RHF/UHF reference.")
        self.max_memory = resolve_memory_mb(max_memory, mf)
        self.reference = MolecularEnergy(mf, gradient_tolerance=gradient_tolerance,
                                         jk_backend=jk_backend)
        self.coordinates = self.reference.coordinates
        occupations = np.asarray(mf.mo_occ)
        self.nocc = (tuple(int(np.count_nonzero(row)) for row in occupations)
                     if self.reference.unrestricted else int(np.count_nonzero(occupations)))
        self.nmo = mf.mo_coeff.shape[-1]
        if config is None:
            config = (mp.MPConfig(order=2, with_t2=False, orbital_basis="noncanonical")
                      if self.method in {"mp", "mp2"}
                      else cc.CCConfig(conv_tol=1e-11, residual_tol=1e-10))
        if self.method in {"mp", "mp2"}:
            if not isinstance(config, mp.MPConfig):
                raise ValueError("The MP adapter requires an MPConfig.")
            if self.method == "mp2" and config.order != 2:
                raise ValueError("The MP2 adapter requires an order-2 MPConfig; use method='mp' for higher orders.")
        elif not isinstance(config, cc.CCConfig) or config.method != "ccsd":
            raise ValueError("The CCSD adapter requires a CCSD configuration.")
        self.config = config
        self._selected_mp2 = (self.method in {"mp", "mp2"} and config.order == 2
                              and not config._use_series(self.reference.unrestricted))
        if jk_backend == "direct" and not self._selected_mp2:
            raise NotImplementedError("Direct correlated inputs require order-2 MP without a series expansion.")

    def __call__(self, coordinates, field=None, *, potential=None):
        state, inputs = self.reference.solve_with_inputs(coordinates, field, potential=potential)
        coefficients, valid = state.mo_coeff, state.stationary
        if self.method in {"mp", "mp2"} and self.config.orbital_basis == "canonical":
            def canonicalize(coeff, fock):
                fock_mo = coeff.T @ fock @ coeff
                orbitals = solve_hermitian(.5*(fock_mo+fock_mo.T),
                    config=EigenSolverConfig(method="dense", nroots=self.nmo, atol=1e-10),
                    response=EigenResponseConfig(target="eigenpairs"))
                return coeff @ orbitals.vectors, jnp.all(orbitals.response_valid)
            if self.reference.unrestricted:
                alpha, alpha_valid = canonicalize(coefficients[0], state.fock_matrix[0])
                beta, beta_valid = canonicalize(coefficients[1], state.fock_matrix[1])
                coefficients, valid = (alpha, beta), valid & alpha_valid & beta_valid
            else:
                coefficients, canonical_valid = canonicalize(coefficients, state.fock_matrix.mean(axis=0))
                valid &= canonical_valid
        if self._selected_mp2:
            cs = tuple(coefficients) if self.reference.unrestricted else (coefficients,)
            fs = tuple(state.fock_matrix) if self.reference.unrestricted else (state.fock_matrix.mean(axis=0),)
            focks = tuple(c.T @ f @ c for c, f in zip(cs, fs))
            data = from_orbitals(coefficients, focks, self.nocc, state.total_energy,
                eri=inputs["eri"], direct_basis=inputs.get("direct_basis"),
                frozen_gap_tol=self.config.denominator_tol, max_memory=self.max_memory)
            result = evaluate_mp2(data, self.config)
        else:
            transform = transform_unrestricted_integrals if self.reference.unrestricted else transform_integrals
            h1, eri = transform(inputs["hcore"], coefficients, eri=inputs["eri"],
                                max_memory=self.max_memory)
            solve = (mp.run_mp if self.method in {"mp", "mp2"} else
                     cc.run_ucc if self.reference.unrestricted else cc.run_cc)
            result = solve(h1, eri, nocc=self.nocc, nuclear_repulsion=inputs["nuclear_repulsion"],
                           config=self.config)
        valid &= result.valid if self.method in {"mp", "mp2"} else result.converged
        total = result.total_energy
        if self.method == "ccsd(t)":
            correction = cc.evaluate_triples(
                h1, eri, result, nocc=self.nocc, orbital_basis="semicanonical",
                algorithm="streamed", denominator_tol=self.config.denominator_tol,
                residual_tol=self.config.residual_tol)
            total += correction.energy
            valid &= correction.valid
        return jnp.where(valid, total, jnp.nan)

    def dipole(self, coordinates, field=None, *, potential=None):
        """Return -dE_total/dF, including reference and amplitude response."""
        field = jnp.zeros(3) if field is None else jnp.asarray(field)
        return -jax.grad(lambda applied: self(coordinates, applied, potential=potential))(field)
