"""Small-system MP2/CCSD energies with relaxed molecular input response."""
import jax
import jax.numpy as jnp
import numpy as np

from gradscf import cc, mp
from gradscf.integrals.molecular.ao2mo import transform_integrals, transform_unrestricted_integrals
from gradscf.solvers import EigenResponseConfig, EigenSolverConfig, solve_hermitian
from .molecular_response import MolecularEnergy


class CorrelatedEnergy:
    """E(R,F,V) for real RHF/UHF-based MP2 or CCSD, in atomic units.

    The HF state, AO integrals, MO transformation and correlation solution are
    all rebuilt inside the differentiated map. This adapter adds no solver.
    MP2 uses canonical orbitals and requires their isolated-root response;
    CCSD uses the transported noncanonical occupied/virtual frame directly.
    The full MO integrals and UCC spin integrals limit this to small systems.
    No frozen orbitals, density fitting or DFT reference is used here.

    Correlated dipoles and polarizabilities must be derivatives of this total
    energy. The reference's stationary dipole is only the HF observable.
    """

    def __init__(self, mf, method, *, config=None, gradient_tolerance=1e-10):
        self.method = method.lower()
        if self.method not in {"mp2", "ccsd"} or str(mf.xc).lower() != "hf":
            raise ValueError("Use MP2 or CCSD with a real RHF/UHF reference.")
        self.reference = MolecularEnergy(mf, gradient_tolerance=gradient_tolerance)
        self.coordinates = self.reference.coordinates
        occupations = np.asarray(mf.mo_occ)
        self.nocc = (tuple(int(np.count_nonzero(row)) for row in occupations)
                     if self.reference.unrestricted else int(np.count_nonzero(occupations)))
        self.nmo = mf.mo_coeff.shape[-1]
        if config is None:
            config = (mp.MPConfig(order=2, with_t2=False) if self.method == "mp2"
                      else cc.CCConfig(conv_tol=1e-11, residual_tol=1e-10))
        if self.method == "mp2":
            if not isinstance(config, mp.MPConfig) or config.order != 2:
                raise ValueError("The MP2 adapter requires an order-2 MPConfig.")
        elif not isinstance(config, cc.CCConfig) or config.method != "ccsd":
            raise ValueError("The CCSD adapter requires a CCSD configuration.")
        self.config = config

    def __call__(self, coordinates, field=None, *, potential=None):
        state, inputs = self.reference.solve_with_inputs(coordinates, field, potential=potential)
        coefficients, valid = state.mo_coeff, state.stationary
        if self.method == "mp2":
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
        transform = transform_unrestricted_integrals if self.reference.unrestricted else transform_integrals
        h1, eri = transform(inputs["hcore"], coefficients, eri=inputs["eri"])
        solve = (mp.run_mp if self.method == "mp2" else
                 cc.run_ucc if self.reference.unrestricted else cc.run_cc)
        result = solve(h1, eri, nocc=self.nocc, nuclear_repulsion=inputs["nuclear_repulsion"],
                       config=self.config)
        valid &= result.valid if self.method == "mp2" else result.converged
        return jnp.where(valid, result.total_energy, jnp.nan)

    def dipole(self, coordinates, field=None, *, potential=None):
        """Return -dE_total/dF, including reference and amplitude response."""
        field = jnp.zeros(3) if field is None else jnp.asarray(field)
        return -jax.grad(lambda applied: self(coordinates, applied, potential=potential))(field)
