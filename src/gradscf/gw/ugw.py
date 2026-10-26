"""PySCF-style unrestricted GW facade.

References
----------
- L. Hedin, Phys. Rev. 139, A796 (1965). DOI:10.1103/PhysRev.139.A796
- T. Zhu and G. K.-L. Chan, J. Chem. Theory Comput. 14, 4856 (2018).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import jax.numpy as jnp

from gradscf.integrals.molecular.factorization import eri_pair_matrix_to_df_factors
from ..scf.reference import as_reference, reference_state_signature, _array_signature
from .g0w0 import g0w0_cd_unrestricted, _mo_factors


class UGW:
    """Spin-unrestricted G0W0/evGW/evGW0 from HF or complete KS states.

    Parameters
    ----------
    mf:
        A converged :class:`gradscf.dft.UKS` facade object.  The
        unrestricted reference (``mf.reference``) supplies spin-resolved
        orbital energies/coefficients, densities and ``h1e``.
    nw, eta:
        Imaginary-grid size and broadening.
    """

    def __init__(self, mf, *, nw: int = 100, eta: float = 1e-3, qp_solver="secant",
                 method="g0w0", max_cycle=20, conv_tol=1e-6, damp=0.):
        if getattr(mf, "mo_energy", None) is None:
            raise RuntimeError("UGW requires a converged mean-field object; call mf.kernel() first.")
        self.qp_solver = qp_solver
        self.method, self.max_cycle, self.conv_tol, self.damp = method, max_cycle, conv_tol, damp
        self._scf = mf
        self.nw = int(nw)
        self.eta = float(eta)
        self.mo_energy = None
        self.mo_coeff = getattr(mf, "mo_coeff", None)
        self.mo_occ = getattr(mf, "mo_occ", None)
        self.converged = None
        self.result = None
        self._source_key = None
        self._ao_factors = None

    def _source_signature(self):
        mf = self._scf
        if not mf.converged or mf._cached_scf_key != mf._scf_signature():
            raise RuntimeError("SCF source changed or is unconverged; run SCF and GW again")
        reference = as_reference(mf, allow_range_separated=True)
        arrays = []
        for name in ("mo_energy", "mo_coeff", "mo_occ"):
            value = _array_signature(getattr(reference, name))
            if value != _array_signature(getattr(mf, name)):
                raise RuntimeError("SCF orbitals changed; run SCF and GW again")
            arrays.append(value)
        for name in ("rdm1", "h1e", "fock_matrix", "df_factors", "rep_tensor", "dipole_integrals"):
            value = getattr(reference, name, None)
            arrays.append(None if value is None else _array_signature(value))
        return (self.nw, self.eta, self.qp_solver, self.method, self.max_cycle,
                self.conv_tol, self.damp,
                reference_state_signature(mf, allow_range_separated=True), tuple(arrays))

    def state_signature(self):
        """Reject stale SCF/GW data before downstream optical calculations."""
        if self.result is None or self._source_key is None:
            raise RuntimeError("Run GW before requesting BSE inputs")
        if self._source_signature() != self._source_key:
            raise RuntimeError("GW source or settings changed; run GW again")
        if _array_signature(self.mo_occ) != _array_signature(self._scf.mo_occ):
            raise RuntimeError("GW occupations changed; run GW again")
        for name in ("mo_energy", "mo_coeff"):
            if _array_signature(getattr(self, name)) != _array_signature(getattr(self.result, name)):
                raise RuntimeError("GW orbitals changed; run GW again")
        arrays = (self.result.mo_energy, self.result.mo_coeff, self.result.screening_energy,
                  self.result.qp_computed_mask, self.result.converged_mask, self._ao_factors)
        return self._source_key, id(self.result), tuple(
            None if x is None else _array_signature(x) for x in arrays)

    def get_bse_inputs(self, *, max_aux=1024, max_factor_elements=20_000_000):
        """Spin-resolved MO data sharing one AO Coulomb auxiliary metric."""
        self.state_signature()
        c = self.result.mo_coeff
        naux, nmo = self._ao_factors.shape[0], c.shape[-1]
        if naux > max_aux or 2 * naux * nmo * nmo > max_factor_elements:
            raise ValueError("BSE factors exceed max_aux or max_factor_elements")
        ref = as_reference(self._scf, allow_range_separated=True)
        if any(x is None for x in (self.result.screening_energy,
                                   self.result.qp_computed_mask, self.result.converged_mask)):
            raise ValueError("GW result is missing screening or QP coverage metadata")
        return dict(qp_energy=self.result.mo_energy,
            screening_energy=self.result.screening_energy,
            mo_factors=jnp.stack([_mo_factors(self._ao_factors, coeff) for coeff in c]),
            mo_coeff=c, nocc=tuple(int(np.count_nonzero(o > 0)) for o in self.mo_occ),
            dipole_ao=getattr(ref, "dipole_integrals", None),
            qp_computed_mask=self.result.qp_computed_mask,
            qp_converged_mask=self.result.converged_mask,
            screening_occupied=None, screening_virtual=None)

    def _df_factors(self, reference):
        factors = getattr(reference, "df_factors", None)
        if factors is not None:
            return factors
        inputs = getattr(self._scf, "_scf_inputs", None)
        pair = getattr(inputs, "eri_pair_matrix", None) if inputs is not None else None
        if pair is not None:
            nao = int(np.asarray(self._scf.mo_coeff)[0].shape[0])
            return eri_pair_matrix_to_df_factors(pair, nao=nao, tol=1e-12)
        # Last resort: rebuild the packed ERI from the molecular spec (the
        # UKS facade does not cache integral inputs).
        from ..integrals import basis_from_pyscf_spec, eri_pair_matrix_packed

        mol = self._scf.mol
        basis = basis_from_pyscf_spec(
            mol.atom,
            basis=mol.basis,
            unit=mol.unit,
            charge=mol.charge,
            spin=mol.spin,
            cart=mol.cart,
        )
        pair = eri_pair_matrix_packed(basis)
        return eri_pair_matrix_to_df_factors(pair, nao=basis.nao, tol=1e-12)

    def kernel(self, orbs: Sequence[int] | None = None):
        self.result, self._source_key, self.converged = None, None, False
        if self.method not in {"g0w0", "evgw", "evgw0"}:
            raise ValueError("GW method must be g0w0, evgw or evgw0")
        if self.method != "g0w0" and self.qp_solver != "secant":
            raise ValueError("qp_solver selects the G0W0 root, not the evGW outer fixed point")
        mf = self._scf
        signature = self._source_signature()
        reference = as_reference(mf, allow_range_separated=True)
        if reference is None:
            raise RuntimeError(
                "Unrestricted GW expects the UKS unrestricted reference "
                "(mf.reference); run mf.kernel() first."
            )
        mo_energy = np.asarray(reference.mo_energy)
        mo_coeff = np.asarray(reference.mo_coeff)
        mo_occ = np.asarray(reference.mo_occ)
        if mo_energy.shape[0] != 2:
            raise RuntimeError("mf.reference does not look unrestricted (expected spin axis 0 of size 2).")
        nocc_a = int(np.count_nonzero(mo_occ[0] > 0.0))
        nocc_b = int(np.count_nonzero(mo_occ[1] > 0.0))
        rdm1 = np.asarray(reference.rdm1)
        df_factors = self._df_factors(reference)

        # F-h-J contains the whole starting potential, including semilocal,
        # tau-dependent and nonlocal hybrid/RSH exchange contributions.
        fock = getattr(reference, "fock_matrix", None)
        hfx = float(getattr(reference, "exact_exchange_fraction", 0.0) or 0.0)
        if fock is None and abs(hfx - 1.0) > 1e-12:
            raise ValueError("A DFT GW starting point requires actual spin Fock matrices.")
        from .evgw import evgw_cd_unrestricted
        driver = g0w0_cd_unrestricted if self.method == "g0w0" else evgw_cd_unrestricted
        controls = dict(qp_solver=self.qp_solver) if self.method == "g0w0" else dict(
            max_iter=self.max_cycle, tol=self.conv_tol, damping=self.damp,
            update_w=self.method == "evgw")
        res = driver(
            mo_energy=(jnp.asarray(mo_energy[0]), jnp.asarray(mo_energy[1])),
            mo_coeff=(jnp.asarray(mo_coeff[0]), jnp.asarray(mo_coeff[1])),
            nocc=(nocc_a, nocc_b),
            df_factors=df_factors,
            fock_matrix=None if fock is None else tuple(jnp.asarray(f) for f in fock),
            hcore_matrix=None if fock is None else jnp.asarray(reference.h1e),
            density_matrix=(jnp.asarray(rdm1[0]), jnp.asarray(rdm1[1])),
            nw=self.nw,
            eta=self.eta,
            orbs=orbs,
            **controls,
        )
        self.result = res
        self.converged = res.converged
        self.mo_energy = res.mo_energy
        self.mo_coeff = res.mo_coeff
        self.mo_occ = reference.mo_occ
        self._ao_factors = df_factors
        self._source_key = signature
        return self.mo_energy

    def run(self, orbs: Sequence[int] | None = None) -> "UGW":
        self.kernel(orbs)
        return self


__all__ = ["UGW"]
