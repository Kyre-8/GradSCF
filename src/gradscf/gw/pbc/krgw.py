"""Gamma-point periodic G0W0 with contour deformation (Stage 2, slice 1).

The driver reuses the spin/system-agnostic molecular kernels
(:mod:`gradscf.gw.g0w0`) with the plane-wave product basis of
:mod:`gradscf.gw.pbc.product_basis` (complex factors, ``conjugate=True``
paths).  Momentum transfer is trivial at Gamma (all k indices collapse),
so the quasiparticle equation is solved band by band exactly as in the
molecular case.

Consistency notes
-----------------
- The exact-exchange potential uses the same Ewald-corrected FFT exchange
  as the SCF (``get_jk(exxdiv='ewald')``), so ``v^x - v^mf = 0`` for an HF
  starting point, mirroring the molecular driver.
- The body screened interaction carries ``v(G=0) = 0``. With ``fc=True``,
  the q -> 0 head/wing corrections of the inverse dielectric matrix
  (PRB 83, 245122 (2011)) are added via ``gradscf.gw.pbc.q0``.

References
----------
- L. Hedin, Phys. Rev. 139, A796 (1965). DOI:10.1103/PhysRev.139.A796
- J. Deslippe et al., Comput. Phys. Commun. 183, 1269 (2012).
  DOI:10.1016/j.cpc.2011.12.023
- T. Zhu and G. K.-L. Chan, J. Chem. Theory Comput. 14, 4856 (2018)
  (contour-deformation self-energy convention).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import jax.numpy as jnp
from jax.lax import Precision
from jaxtyping import Array

from ...integrals.periodic.fft import get_jk
from ..evgw import _evgw_loop
from ..freq import scaled_legendre_grid
from ..g0w0 import _qp_loop, _requested_orbitals
from ..outer_response import require_occupation_gap
from ..polarizability import rho_response_iw
from ..screened import screened_w_imag_axis
from ..types import GWResult
from .product_basis import gamma_product_factors, _require_paired_lehmann
from .q0 import screened_w_imag_axis_head_wing, transition_moments_gamma


def _validate_gw_reference(mf):
    from ...dft.xc import hybrid_coeff, parse_recipe
    from ...pbc.scf import validate_reference

    validate_reference(mf)
    recipe = parse_recipe(mf.xc)
    recipe.require_complete_support()
    if recipe.exact_exchange.omega is not None:
        raise NotImplementedError('Periodic GW range-separated starting potentials are not implemented.')
    if mf.exxdiv != "ewald" and hybrid_coeff(mf.xc) != 0:
        raise NotImplementedError(
            "Periodic GW currently requires Ewald exchange for HF/hybrid references."
        )


def g0w0_cd_gamma(
    *,
    inputs,
    mo_energy: Array,
    mo_coeff: Array,
    nocc: int,
    fock_matrix: Array,
    hcore_matrix: Array,
    density_spin: Array,
    mesh: tuple[int, int, int],
    nw: int = 100,
    eta: float = 1e-3,
    orbs: Sequence[int] | None = None,
    diff_mode: str = "implicit",
    fc: bool = True,
    mo_energy_poles: Array | None = None,
    linearized: bool = False,
    evaluate_only: bool = False,
) -> GWResult:
    """Gamma-point periodic G0W0-CD (spin-restricted).

    Parameters
    ----------
    inputs:
        :class:`gradscf.integrals.periodic.fft.PeriodicInputs` (Gamma).
    mo_energy, mo_coeff:
        Gamma-point mean-field band energies ``(nmo,)`` and coefficients
        ``(nao, nmo)``.
    nocc:
        Occupied bands per spin channel (restricted: doubly occupied).
    fock_matrix, hcore_matrix:
        Converged Gamma-point Fock and core Hamiltonian (AO basis).
    density_spin:
        Converged spin densities, shape ``(2, nao, nao)``.
    mesh:
        FFT grid shape the inputs were built with (``cell.mesh``).
    nw, eta, orbs, diff_mode, linearized, evaluate_only:
        As in :func:`gradscf.gw.g0w0_cd_restricted`.

    Returns
    -------
    :class:`gradscf.gw.types.GWResult`

    Notes
    -----
    Folded contour screening requires paired OV/VO transition covariances.
    This is checked explicitly. Parameter derivatives are supported along
    paths preserving that condition; generic complex orbital AD is not.
    """
    mo_energy = jnp.asarray(mo_energy, dtype=jnp.float64)
    poles = (
        mo_energy
        if mo_energy_poles is None
        else jnp.asarray(mo_energy_poles, dtype=jnp.float64)
    )
    mo_coeff = jnp.asarray(mo_coeff, dtype=jnp.complex128)
    nocc = int(nocc)
    nmo = mo_energy.shape[0]
    if not 0 < nocc < nmo:
        raise ValueError('Restricted periodic GW requires occupied and virtual orbitals.')
    if poles.shape != mo_energy.shape:
        raise ValueError('G/W spectra must match the mean-field spectrum.')
    require_occupation_gap(poles, nocc, 'Gamma GW')
    orbs = _requested_orbitals(orbs, nmo)

    b = gamma_product_factors(inputs, mo_coeff, mesh=mesh)  # (nG, nmo, nmo) complex
    b_ov = b[:, :nocc, nocc:]
    pairs = np.array([(i, a) for i in range(nocc) for a in range(nocc, nmo)])
    differences = poles[pairs[:, 0]]-poles[pairs[:, 1]]
    _require_paired_lehmann(jnp.concatenate((b[:, pairs[:, 0], pairs[:, 1]],
        b[:, pairs[:, 1], pairs[:, 0]]), axis=1), jnp.r_[differences, -differences],
        jnp.r_[jnp.ones(len(pairs)), -jnp.ones(len(pairs))], label='Gamma GW')

    density_spin = jnp.asarray(density_spin)
    j_mat, k_ewald = get_jk(inputs, density_spin, exxdiv="ewald", with_k=True)
    k_ewald = k_ewald[0]  # restricted: K_alpha == K_beta
    coeff_t = mo_coeff.conj().T
    v_mf_mo = coeff_t @ (jnp.asarray(fock_matrix) - jnp.asarray(hcore_matrix) - j_mat) @ mo_coeff
    vk_mo = coeff_t @ k_ewald @ mo_coeff  # note: F = h + J - K, so v^x = -K
    delta_v = jnp.diag(-(vk_mo + v_mf_mo)).real
    # delta_v = v^x - v^mf = -K - (F - h - J) = -(K + v_mf_mo)

    ef = 0.5 * (jnp.max(poles[:nocc]) + jnp.min(poles[nocc:]))
    freqs, wts = scaled_legendre_grid(nw)

    if fc:
        # q -> 0 head/wing finite-size correction (PRB 83, 245122 (2011)).
        # The frozen orbital frame retains its MF k.p vertex; only the
        # independent-particle response denominators use updated G/W poles.
        qij, q_abs = transition_moments_gamma(inputs, mo_energy, mo_coeff, nocc)
        volume = float(jnp.asarray(inputs.weights[0])) * b.shape[0]
        wmn, del00, delP0 = screened_w_imag_axis_head_wing(
            b, b_ov, poles, qij, q_abs, volume, freqs
        )
        static, head0, wing0 = screened_w_imag_axis_head_wing(
            b, b_ov, poles, qij, q_abs, volume, jnp.zeros(1)
        )
        del_w_static = head0[0]+wing0[0]
        delP0 = delP0.T  # (nw, nmo) -> (nmo, nw)
        q0 = {"qij": qij, "q_abs": q_abs, "volume": volume, "nkpts": 1}
    else:
        def response_fn(omega):
            return rho_response_iw(omega, poles, b_ov, spin_factor=4.0, conjugate=True)

        wmn = screened_w_imag_axis(b, response_fn, freqs, conjugate=True)
        static = screened_w_imag_axis(b, response_fn, jnp.zeros(1), conjugate=True)
        del00 = delP0 = q0 = None
        del_w_static = None

    qp_energy, sigma_qp, converged_mask, converged, residual, weight, diagnostics = _qp_loop(
        mo_energy=poles,
        b_mn=b,
        channels=((poles, b_ov, 2.0),),
        wmn=wmn,
        freqs=freqs,
        wts=wts,
        ef=ef,
        eta=float(eta),
        delta_v=delta_v,
        nocc=nocc,
        orbs=orbs,
        diff_mode=diff_mode,
        conjugate=True,
        linearized=linearized,
        evaluate_only=evaluate_only,
        del00=del00,
        delP0=delP0,
        q0=q0,
        e_mf=mo_energy,
        wmn_static=static[0],
        del_w_static=del_w_static,
        return_diagnostics=True,
    )
    return GWResult(
        mo_energy=qp_energy,
        mo_coeff=mo_coeff,
        converged=converged,
        sigma_qp=sigma_qp,
        converged_mask=converged_mask,
        nw=int(nw),
        qp_residual=residual,
        qp_weight=weight,
        qp_computed_mask=jnp.zeros_like(qp_energy, dtype=bool).at[jnp.asarray(orbs)].set(not evaluate_only),
        screening_energy=poles,
        method='periodic_gamma_cd',
        **diagnostics,
    )


class KRGW:
    """Gamma-point periodic GW facade over ``gradscf.pbc.scf._SCF``.

    Parameters
    ----------
    mf:
        A converged Gamma-point periodic mean-field object
        (:class:`gradscf.pbc.scf.RHF` or ``dft.RKS``).
    nw, eta:
        Imaginary-grid size and broadening.
    fc:
        Include the q -> 0 head/wing finite-size correction (default True,
        matching PySCF ``KRGW(fc=True)``).
    """

    def __init__(self, mf, *, nw: int = 100, eta: float = 1e-3, fc: bool = True):
        if getattr(mf, "result", None) is None:
            raise RuntimeError("KRGW requires a converged periodic mean-field object; call mf.kernel() first.")
        self._scf = mf
        self.nw = int(nw)
        self.eta = float(eta)
        self.fc = bool(fc)
        self.mo_energy = None
        self.mo_coeff = None
        self.converged = None
        self.result = None

    def kernel(self, orbs: Sequence[int] | None = None):
        mf = self._scf
        _validate_gw_reference(mf)
        result = mf.result
        kpts = np.asarray(mf.kpts)
        if kpts.shape == (1, 3) and not np.any(kpts):
            mo_energy = jnp.asarray(result.mo_energy_spin[0, 0])
            mo_coeff = jnp.asarray(result.mo_coeff_spin[0, 0])
            occ = jnp.asarray(result.mo_occ_spin[0, 0])
            nocc = int(occ.sum())
            res = g0w0_cd_gamma(
                inputs=mf.inputs,
                mo_energy=mo_energy,
                mo_coeff=mo_coeff,
                nocc=nocc,
                fock_matrix=jnp.asarray(result.fock_spin[0, 0]),
                hcore_matrix=jnp.asarray(mf.inputs.hcore[0]),
                density_spin=jnp.asarray(result.density_spin[:, 0]),
                mesh=tuple(int(m) for m in mf.cell.mesh),
                nw=self.nw,
                eta=self.eta,
                orbs=orbs,
                fc=self.fc,
            )
        else:
            from .ksigma import g0w0_cd_kpoints

            kpts_frac = kpts @ np.asarray(mf.cell.lattice).T / (2.0 * np.pi)
            occ_k = jnp.asarray(result.mo_occ_spin[0])  # (nk, nmo)
            nocc = int(occ_k[0].sum())
            res = g0w0_cd_kpoints(
                inputs=mf.inputs,
                kpts_frac=kpts_frac,
                mo_energy_k=jnp.asarray(result.mo_energy_spin[0]),
                mo_coeff_k=jnp.asarray(result.mo_coeff_spin[0]),
                nocc=nocc,
                fock_k=jnp.asarray(result.fock_spin[0]),
                hcore_k=jnp.asarray(mf.inputs.hcore),
                density_spin=jnp.asarray(result.density_spin),
                mesh=tuple(int(m) for m in mf.cell.mesh),
                nw=self.nw,
                eta=self.eta,
                orbs=orbs,
                fc=self.fc,
            )
        self.result = res
        self.converged = res.converged
        self.mo_energy = res.mo_energy
        self.mo_coeff = res.mo_coeff
        return self.mo_energy

    def run(self, orbs: Sequence[int] | None = None) -> "KRGW":
        self.kernel(orbs)
        return self


__all__ = ["g0w0_cd_gamma", "evgw_cd_gamma", "KRGW"]


def evgw_cd_gamma(
    *,
    inputs,
    mo_energy: Array,
    mo_coeff: Array,
    nocc: int,
    fock_matrix: Array,
    hcore_matrix: Array,
    density_spin: Array,
    mesh: tuple[int, int, int],
    nw: int = 100,
    eta: float = 1e-3,
    orbs: Sequence[int] | None = None,
    diff_mode: str = "implicit",
    fc: bool = True,
    max_iter: int = 20,
    tol: float = 1e-6,
    damping: float = 0.0,
) -> GWResult:
    """Gamma-point periodic evGW (QP energies iterated to self-consistency).

    References: M. Shishkin and G. Kresse, Phys. Rev. B 75, 235102 (2007).
    DOI:10.1103/PhysRevB.75.235102
    """
    result = _evgw_loop(
        g0w0_cd_gamma,
        mo_energy,
        mo_energy=mo_energy,
        inputs=inputs,
        mo_coeff=mo_coeff,
        nocc=int(nocc),
        fock_matrix=fock_matrix,
        hcore_matrix=hcore_matrix,
        density_spin=density_spin,
        mesh=mesh,
        nw=int(nw),
        eta=float(eta),
        orbs=orbs,
        diff_mode=diff_mode,
        fc=bool(fc),
        max_iter=int(max_iter),
        tol=float(tol),
        damping=float(damping),
    )
    return replace(result, method='periodic_gamma_evgw')
