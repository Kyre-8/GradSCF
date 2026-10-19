"""Gamma-point periodic unrestricted G0W0 (Stage 2, KUGW slice).

Spin-resolved analogue of :mod:`gradscf.gw.pbc.krgw`: the RPA response is
summed over both spin channels (factor 2 each), the screened interaction
is shared, and the Green's function / exchange / Fermi estimate are
spin-resolved.  The q -> 0 head/wing correction is not yet wired for the
unrestricted path (fc is forced off and documented).

References
----------
- L. Hedin, Phys. Rev. 139, A796 (1965). DOI:10.1103/PhysRev.139.A796
- J. Deslippe et al., Comput. Phys. Commun. 183, 1269 (2012).
  DOI:10.1016/j.cpc.2011.12.023
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import jax.numpy as jnp
from jax.lax import Precision
from jaxtyping import Array

from ...integrals.periodic.fft import get_jk
from ..freq import scaled_legendre_grid
from ..g0w0 import _qp_loop, _requested_orbitals
from ..outer_response import require_occupation_gap
from ..polarizability import rho_response_iw
from ..screened import screened_w_imag_axis
from ..types import GWResult
from .product_basis import gamma_product_factors, _require_paired_lehmann
from .krgw import _validate_gw_reference


def g0w0_cd_gamma_unrestricted(
    *,
    inputs,
    mo_energy: tuple[Array, Array],
    mo_coeff: tuple[Array, Array],
    nocc: tuple[int, int],
    fock_matrix: tuple[Array, Array] | None,
    hcore_matrix: Array,
    density_spin: Array,
    mesh: tuple[int, int, int],
    nw: int = 100,
    eta: float = 1e-3,
    orbs: Sequence[int] | None = None,
    diff_mode: str = "implicit",
) -> GWResult:
    """Gamma-point periodic unrestricted G0W0-CD.

    For an HF starting point the exchange potentials are rebuilt from the
    Ewald-corrected FFT exchange (``fock_matrix=None``); otherwise the
    converged spin Fock matrices are required.
    The total spin-summed OV/VO covariance must support an even-W contour.
    Empty/full spin sectors retain their fixed occupations, with a contour
    Fermi level outside that sector's spectrum. AD requires a parameter path
    preserving the checked covariance pairing.
    """
    e_a, e_b = (jnp.asarray(e, dtype=jnp.float64) for e in mo_energy)
    c_a, c_b = (jnp.asarray(c, dtype=jnp.complex128) for c in mo_coeff)
    nocc_a, nocc_b = int(nocc[0]), int(nocc[1])
    if e_a.ndim != 1 or not e_a.size or e_b.shape != e_a.shape:
        raise ValueError('Spin spectra must share a nonempty (nmo,) shape.')
    nmo = e_a.shape[0]
    require_occupation_gap(jnp.stack((e_a, e_b)), (nocc_a, nocc_b), 'Gamma unrestricted GW')
    orbs = _requested_orbitals(orbs, nmo)

    b_a = gamma_product_factors(inputs, c_a, mesh=mesh)
    b_b = gamma_product_factors(inputs, c_b, mesh=mesh)
    b_ov_a = b_a[:, :nocc_a, nocc_a:]
    b_ov_b = b_b[:, :nocc_b, nocc_b:]
    vertices, gaps, signs = [], [], []
    for factors, energy, occupied in ((b_a, e_a, nocc_a), (b_b, e_b, nocc_b)):
        pairs = np.array([(i, a) for i in range(occupied) for a in range(occupied, nmo)],
                         dtype=int).reshape(-1, 2)
        differences = energy[pairs[:, 0]]-energy[pairs[:, 1]]
        vertices.extend((factors[:, pairs[:, 0], pairs[:, 1]], factors[:, pairs[:, 1], pairs[:, 0]]))
        gaps.extend((differences, -differences))
        signs.extend((jnp.ones(len(pairs)), -jnp.ones(len(pairs))))
    _require_paired_lehmann(jnp.concatenate(vertices, axis=1), jnp.concatenate(gaps),
                           jnp.concatenate(signs), label='Gamma unrestricted GW')

    density_spin = jnp.asarray(density_spin)
    j_mat, k_ewald = get_jk(inputs, density_spin, exxdiv="ewald", with_k=True)
    ca_t = c_a.conj().T
    cb_t = c_b.conj().T
    if fock_matrix is None:
        # HF start: v^mf_sigma = -K_sigma, so delta_v = 0 identically.
        delta_v_a = jnp.zeros(nmo)
        delta_v_b = jnp.zeros(nmo)
    else:
        vmf_a = ca_t @ (jnp.asarray(fock_matrix[0]) - jnp.asarray(hcore_matrix) - j_mat) @ c_a
        vmf_b = cb_t @ (jnp.asarray(fock_matrix[1]) - jnp.asarray(hcore_matrix) - j_mat) @ c_b
        delta_v_a = jnp.diag(-(ca_t @ k_ewald[0] @ c_a + vmf_a)).real
        delta_v_b = jnp.diag(-(cb_t @ k_ewald[1] @ c_b + vmf_b)).real

    def fermi(energy, occupied):
        if occupied == 0:
            return jnp.min(energy)-(1+jnp.ptp(energy))
        if occupied == nmo:
            return jnp.max(energy)+(1+jnp.ptp(energy))
        return .5*(jnp.max(energy[:occupied])+jnp.min(energy[occupied:]))
    ef_a, ef_b = fermi(e_a, nocc_a), fermi(e_b, nocc_b)
    freqs, wts = scaled_legendre_grid(nw)

    def response_fn(omega):
        return rho_response_iw(omega, e_a, b_ov_a, spin_factor=2.0, conjugate=True) + rho_response_iw(
            omega, e_b, b_ov_b, spin_factor=2.0, conjugate=True
        )

    wmn_a = screened_w_imag_axis(b_a, response_fn, freqs, conjugate=True)
    wmn_b = screened_w_imag_axis(b_b, response_fn, freqs, conjugate=True)
    static_a = screened_w_imag_axis(b_a, response_fn, jnp.zeros(1), conjugate=True)[0]
    static_b = screened_w_imag_axis(b_b, response_fn, jnp.zeros(1), conjugate=True)[0]
    channels = ((e_a, b_ov_a, 1.0), (e_b, b_ov_b, 1.0))

    qp_a, sig_a, mask_a, conv_a, residual_a, weight_a, diagnostics_a = _qp_loop(
        mo_energy=e_a, b_mn=b_a, channels=channels, wmn=wmn_a, freqs=freqs, wts=wts,
        ef=ef_a, eta=float(eta), delta_v=delta_v_a, nocc=nocc_a, orbs=orbs,
        diff_mode=diff_mode, conjugate=True,
        wmn_static=static_a, return_diagnostics=True,
    )
    qp_b, sig_b, mask_b, conv_b, residual_b, weight_b, diagnostics_b = _qp_loop(
        mo_energy=e_b, b_mn=b_b, channels=channels, wmn=wmn_b, freqs=freqs, wts=wts,
        ef=ef_b, eta=float(eta), delta_v=delta_v_b, nocc=nocc_b, orbs=orbs,
        diff_mode=diff_mode, conjugate=True,
        wmn_static=static_b, return_diagnostics=True,
    )
    return GWResult(
        mo_energy=jnp.stack([qp_a, qp_b]),
        mo_coeff=jnp.stack([c_a, c_b]),
        converged=conv_a & conv_b,
        sigma_qp=jnp.stack([sig_a, sig_b]),
        converged_mask=jnp.stack([mask_a, mask_b]),
        nw=int(nw),
        qp_residual=jnp.stack([residual_a, residual_b]),
        qp_weight=jnp.stack((weight_a, weight_b)),
        qp_computed_mask=jnp.zeros((2, nmo), dtype=bool).at[:, jnp.asarray(orbs)].set(True),
        screening_energy=jnp.stack((e_a, e_b)),
        method='periodic_gamma_ugw_cd',
        **{name: jnp.stack((diagnostics_a[name], diagnostics_b[name])) for name in diagnostics_a},
    )


class KUGW:
    """Gamma-point periodic unrestricted GW facade (pbc UHF/UKS objects).

    The q -> 0 head/wing correction is not yet implemented for the
    unrestricted path; results therefore exclude it (``fc=False``
    semantics) and are labeled accordingly.
    """

    def __init__(self, mf, *, nw: int = 100, eta: float = 1e-3):
        if getattr(mf, "result", None) is None:
            raise RuntimeError("KUGW requires a converged periodic mean-field object; call mf.kernel() first.")
        kpts = jnp.asarray(mf.kpts)
        if kpts.shape != (1, 3) or bool(jnp.any(kpts != 0.0)):
            raise NotImplementedError(
                "KUGW currently supports the Gamma point only; k-point "
                "unrestricted GW is scheduled for a later Stage-2 slice."
            )
        self._scf = mf
        self.nw = int(nw)
        self.eta = float(eta)
        self.mo_energy = None
        self.mo_coeff = None
        self.converged = None
        self.result = None

    def kernel(self, orbs: Sequence[int] | None = None):
        mf = self._scf
        _validate_gw_reference(mf)
        result = mf.result
        mo_energy = jnp.asarray(result.mo_energy_spin[:, 0])
        mo_coeff = jnp.asarray(result.mo_coeff_spin[:, 0])
        occ = jnp.asarray(result.mo_occ_spin[:, 0])
        nocc_a = int(occ[0].sum())
        nocc_b = int(occ[1].sum())
        hfx_note = getattr(mf, "xc", "hf")
        res = g0w0_cd_gamma_unrestricted(
            inputs=mf.inputs,
            mo_energy=(mo_energy[0], mo_energy[1]),
            mo_coeff=(mo_coeff[0], mo_coeff[1]),
            nocc=(nocc_a, nocc_b),
            fock_matrix=None if hfx_note == "hf" else (result.fock_spin[0, 0], result.fock_spin[1, 0]),
            hcore_matrix=jnp.asarray(mf.inputs.hcore[0]),
            density_spin=jnp.asarray(result.density_spin[:, 0]),
            mesh=tuple(int(m) for m in mf.cell.mesh),
            nw=self.nw,
            eta=self.eta,
            orbs=orbs,
        )
        self.result = res
        self.converged = res.converged
        self.mo_energy = res.mo_energy
        self.mo_coeff = res.mo_coeff
        return self.mo_energy

    def run(self, orbs: Sequence[int] | None = None) -> "KUGW":
        self.kernel(orbs)
        return self


__all__ = ["g0w0_cd_gamma_unrestricted", "KUGW"]
