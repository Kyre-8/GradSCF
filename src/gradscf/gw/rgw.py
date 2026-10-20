"""PySCF-style restricted GW facade.

Example
-------
>>> from gradscf import gto, scf
>>> from gradscf.gw import GW
>>> mol = gto.M(atom="O 0 0 0.117; H 0 0.755 -0.471; H 0 -0.755 -0.471",
...             basis="cc-pvdz")
>>> mf = scf.RHF(mol).run()
>>> gw = GW(mf).run()
>>> gw.mo_energy  # quasiparticle energies

References
----------
- L. Hedin, Phys. Rev. 139, A796 (1965). DOI:10.1103/PhysRev.139.A796
- T. Zhu and G. K.-L. Chan, J. Chem. Theory Comput. 14, 4856 (2018).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import jax.numpy as jnp
import numpy as np

from gradscf.integrals.molecular.factorization import eri_pair_matrix_to_df_factors
from .g0w0 import g0w0_cd_restricted, _mo_factors
from .evgw import evgw_cd_restricted
from ..scf.reference import reference_state_signature, _array_signature


class GW:
    """Molecular restricted G0W0, evGW, qsGW and Matsubara scGW on RHF/RKS.

    Parameters
    ----------
    mf:
        A converged :class:`gradscf.scf.RHF` or :class:`gradscf.dft.RKS`
        facade object (any ``xc``,
        including ``"hf"``).
    method:
        "g0w0" (default), "evgw" (update G and W), "evgw0" (fixed W0),
        "qsgw" (static Hermitian effective potential), or "scgw" (Matsubara
        Dyson self-consistency). This facade performs eager orchestration;
        differentiable array drivers expose their own response configuration.
    max_cycle, conv_tol, damp:
        Outer evGW/evGW0 iteration limit, Dyson residual tolerance (Ha), and
        damping. These do not alter the inner G0W0 secant controls.
    g_orbitals, screening_occupied, screening_virtual:
        Independent correlation-sum windows in original zero-based MO indices.
        None retains all allowed indices; empty windows suppress correlation.
        QP target indices are supplied separately to run(orbs=...).
    qp_solver:
        G0W0 forward root method: "secant" (default), "newton" or "hybrid".
        This is separate from the physical method and cannot select an evGW
        outer iteration algorithm. The implicit QP backward is shared.
    nw:
        Imaginary-axis quadrature size (default 100, PySCF convention).
    eta:
        Real-frequency W broadening (1e-3 Ha). The subtracted imaginary-axis
        Green denominator uses its analytic zero-broadening limit.
    beta, mixing:
        scGW inverse temperature (Ha^-1) and accepted fixed-point mixing.
        Matsubara scGW does not supply inferred QP energies for static BSE.
    """

    def __init__(self, mf, *, nw: int = 100, eta: float = 1e-3,
                 method: str = "g0w0", max_cycle: int = 20, conv_tol: float = 1e-6,
                 damp: float = 0.0, g_orbitals=None, screening_occupied=None,
                 screening_virtual=None, qp_solver="secant", beta=100., mixing=.5):
        if getattr(mf, "mo_energy", None) is None:
            raise RuntimeError("GW requires a converged mean-field object; call mf.kernel() first.")
        self.qp_solver = qp_solver
        self.method = str(method).lower()
        self.beta, self.mixing = float(beta), float(mixing)
        self.max_cycle, self.conv_tol, self.damp = max_cycle, conv_tol, damp
        self.g_orbitals = None if g_orbitals is None else tuple(g_orbitals)
        self.screening_occupied = None if screening_occupied is None else tuple(screening_occupied)
        self.screening_virtual = None if screening_virtual is None else tuple(screening_virtual)
        self._scf = mf
        self.nw = int(nw)
        self.eta = float(eta)
        self.mo_energy = None
        self.mo_coeff = mf.mo_coeff
        self.mo_occ = getattr(mf, "mo_occ", None)
        self.converged = None
        self.result = None
        self._source_key = None
        self._ao_factors = None
        self._dipole_ao = None

    def _source_signature(self):
        mf = self._scf
        result = getattr(mf, "scf_result", None)
        inputs = getattr(mf, "_scf_inputs", None)
        if result is None or inputs is None or not mf.converged:
            raise RuntimeError("Run and converge the restricted SCF before GW")
        if mf._cached_scf_key != mf._scf_signature():
            raise RuntimeError("SCF inputs changed; run SCF and GW again")
        arrays = [
            result.mo_energy,
            result.mo_coeff,
            result.mo_occ,
            result.fock_matrix,
            result.hcore_matrix,
            result.density_matrix,
            getattr(inputs, "df_factors", None),
            getattr(inputs, "eri_pair_matrix", None),
        ]
        for name in ("mo_energy", "mo_coeff", "mo_occ"):
            if _array_signature(getattr(mf, name)) != _array_signature(
                getattr(result, name)
            ):
                raise RuntimeError("SCF orbitals changed; run SCF and GW again")
        return (
            self.nw,
            self.eta,
            self.method, self.max_cycle, self.conv_tol, self.damp, self.qp_solver,
            self.beta, self.mixing,
            tuple(None if x is None else tuple(x) for x in
                  (self.g_orbitals, self.screening_occupied, self.screening_virtual)),
            reference_state_signature(mf, allow_range_separated=True),
            tuple(None if x is None else _array_signature(x) for x in arrays),
        )

    def state_signature(self):
        """Checked eager provenance for post-GW consumers."""
        if self.result is None or self._source_key is None:
            raise RuntimeError("Run GW before requesting its reference data")
        if self._source_signature() != self._source_key:
            raise RuntimeError("GW source or settings changed; run GW again")
        if self.method == "scgw":
            arrays = (self.result.mo_coeff, self.result.self_energy_iw,
                      self.result.fock_mo, self.result.density_matrix,
                      self.result.sigma_moment, self.result.chemical_potential)
            return self._source_key, id(self.result), tuple(_array_signature(x) for x in arrays)
        if _array_signature(self.mo_energy) != _array_signature(
            self.result.mo_energy
        ) or _array_signature(self.mo_coeff) != _array_signature(self.result.mo_coeff):
            raise RuntimeError("GW orbitals changed; run GW again")
        dipole = getattr(self._scf._scf_inputs, "dipole_integrals", None)
        if dipole is None:
            dipole = self._dipole_ao
        arrays = (
            self.result.mo_energy,
            self.result.mo_coeff,
            self.result.qp_computed_mask,
            self.result.converged_mask,
            self.result.screening_energy,
            self._ao_factors,
            dipole,
        )
        return (
            self._source_key,
            self.result.g_orbitals, self.result.screening_occupied,
            self.result.screening_virtual, self.result.method,
            id(self.result),
            tuple(None if x is None else _array_signature(x) for x in arrays),
        )

    def get_bse_inputs(self, *, max_aux=1024, max_factor_elements=20_000_000):
        """Validated fixed-frame GW screening snapshot; no BSE dependency."""
        if self.method == "scgw":
            raise NotImplementedError(
                "Molecular Matsubara scGW needs validated analytic continuation and QP "
                "coverage before a static BSE reference can be constructed.")
        if self.result is None or self._ao_factors is None:
            raise RuntimeError("Run GW before requesting BSE inputs")
        naux = self._ao_factors.shape[0]
        nmo = self.result.mo_coeff.shape[1]
        if naux > max_aux:
            raise ValueError("BSE screening exceeds max_aux before MO transformation")
        if naux * nmo * nmo > max_factor_elements:
            raise ValueError(
                "BSE factors exceed max_factor_elements before MO transformation"
            )
        self.state_signature()
        mf = self._scf
        if self.result.qp_computed_mask is None or self.result.converged_mask is None:
            raise ValueError("GW result does not contain QP coverage/convergence metadata")
        if self.result.screening_energy is None:
            raise ValueError("GW result does not contain a recorded screening spectrum")
        available = getattr(mf._scf_inputs, "dipole_integrals", None)
        if available is not None:
            self._dipole_ao = available
        elif self._dipole_ao is None:
            from ..scf.builders import complete_restricted_response_inputs

            completed = complete_restricted_response_inputs(
                mf._scf_inputs, mf._spec(), mf.mol.basis, cart=mf.mol.cart
            )
            self._dipole_ao = completed.dipole_integrals
        return dict(
            qp_energy=self.result.mo_energy,
            screening_energy=self.result.screening_energy,
            mo_factors=_mo_factors(self._ao_factors, self.result.mo_coeff),
            mo_coeff=self.result.mo_coeff,
            ao_factors=self._ao_factors,
            nocc=int(np.count_nonzero(np.asarray(mf.mo_occ))),
            dipole_ao=self._dipole_ao,
            qp_computed_mask=self.result.qp_computed_mask,
            qp_converged_mask=self.result.converged_mask,
            screening_occupied=self.result.screening_occupied,
            screening_virtual=self.result.screening_virtual,
        )

    def _df_factors(self):
        inputs = getattr(self._scf, "_scf_inputs", None)
        factors = getattr(inputs, "df_factors", None) if inputs is not None else None
        if factors is not None:
            return factors
        pair = getattr(inputs, "eri_pair_matrix", None) if inputs is not None else None
        if pair is None:
            eri = getattr(inputs, "eri", None) if inputs is not None else None
            if eri is not None:
                from ..integrals.molecular.factorization import eri_to_df_factors
                return eri_to_df_factors(eri, tol=1e-12)
            raise RuntimeError(
                "GW requires low-rank ERI factors or the packed ERI pair "
                "matrix on the mean-field object; neither was found."
            )
        nao = int(np.asarray(self._scf.mo_coeff).shape[0])
        return eri_pair_matrix_to_df_factors(pair, nao=nao, tol=1e-12)

    def kernel(self, orbs: Sequence[int] | None = None):
        self.result = None
        self.converged = False
        self._source_key = None
        if self.method not in {"g0w0", "evgw", "evgw0", "qsgw", "scgw"}:
            raise ValueError("GW method must be g0w0, evgw, evgw0, qsgw or scgw")
        if self.qp_solver not in {"secant", "newton", "hybrid"}:
            raise ValueError("qp_solver must be secant, newton or hybrid")
        if self.method != "g0w0" and self.qp_solver != "secant":
            raise ValueError("qp_solver selects the inner G0W0 root, not the evGW outer fixed point")
        mf = self._scf
        signature=self._source_signature()
        result = getattr(mf, "scf_result", None)
        if result is None:
            raise RuntimeError(
                "Restricted GW expects the RKS facade result (mf.scf_result); "
                "run mf.kernel() first."
            )
        mo_occ = np.asarray(result.mo_occ)
        nocc = int(np.count_nonzero(mo_occ > 0.0))
        if result.mo_coeff.ndim != 2 or np.iscomplexobj(result.mo_coeff):
            raise NotImplementedError("Restricted GW facade requires real closed-shell orbitals")
        if not np.array_equal(mo_occ,np.r_[np.full(nocc,2.),np.zeros(len(mo_occ)-nocc)]):
            raise ValueError("Restricted GW requires integer closed-shell occupations, occupied first")
        factors=self._df_factors()
        if self.method in {"qsgw", "scgw"}:
            if orbs is not None or any(x is not None for x in
                    (self.g_orbitals, self.screening_occupied, self.screening_virtual)):
                raise ValueError("qsGW/scGW require the complete orbital space.")
            common = dict(mo_energy=result.mo_energy, mo_coeff=result.mo_coeff,
                nocc=nocc, df_factors=factors, hcore_matrix=result.hcore_matrix,
                nw=self.nw, max_iter=self.max_cycle, tol=self.conv_tol)
            if self.method == "qsgw":
                from .qsgw import qsgw_cd_restricted
                res = qsgw_cd_restricted(**common, eta=self.eta, damping=self.damp)
                # The driver certifies the entire static-Hamiltonian spectrum.
                # Record that coverage explicitly for fixed-frame BSE consumers.
                nmo = res.mo_energy.shape[0]
                res = replace(res, qp_computed_mask=jnp.ones(nmo, dtype=bool),
                    converged_mask=jnp.full(nmo, res.converged, dtype=bool),
                    g_orbitals=tuple(range(nmo)),
                    screening_occupied=tuple(range(nocc)),
                    screening_virtual=tuple(range(nocc, nmo)))
            else:
                from .scgw import scgw_matsubara_restricted
                res = scgw_matsubara_restricted(**common, beta=self.beta,
                    mixing=self.mixing, nuclear_repulsion=result.nuclear_repulsion)
            self.result, self.converged = res, res.converged
            self.mo_energy, self.mo_coeff = res.mo_energy, res.mo_coeff
            self._ao_factors, self._source_key = factors, signature
            self._dipole_ao = None
            return self.mo_energy
        driver = g0w0_cd_restricted if self.method == "g0w0" else evgw_cd_restricted
        controls = dict(qp_solver=self.qp_solver) if self.method == "g0w0" else dict(
            max_iter=self.max_cycle, tol=self.conv_tol, damping=self.damp,
            update_w=self.method == "evgw")
        res = driver(
            mo_energy=result.mo_energy,
            mo_coeff=result.mo_coeff,
            nocc=nocc,
            df_factors=factors,
            fock_matrix=result.fock_matrix,
            hcore_matrix=result.hcore_matrix,
            density_matrix=result.density_matrix,
            nw=self.nw,
            eta=self.eta,
            orbs=orbs,
            g_orbitals=self.g_orbitals,
            screening_occupied=self.screening_occupied,
            screening_virtual=self.screening_virtual,
            **controls,
        )
        self.result = res
        self.converged = res.converged
        self.mo_energy = res.mo_energy
        self.mo_coeff = res.mo_coeff
        self._ao_factors = factors
        self._source_key = signature
        self._dipole_ao = None
        return self.mo_energy

    def run(self, orbs: Sequence[int] | None = None) -> "GW":
        self.kernel(orbs)
        return self


__all__ = ["GW"]
