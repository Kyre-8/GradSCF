"""GW correlation self-energy on the real axis via contour deformation.

The self-energy matrix element for orbital ``p`` at real frequency ``omega``
splits into an imaginary-axis integral and a residue contribution from the
G0 poles enclosed by the deformed contour,

    Sigma_p(w) = Sigma_p^I(w) + Sigma_p^R(w)

    Sigma_p^I(w) = -(1/pi) sum_m ∫ dw' W_mp(iw')
                   (w - i*eta*sign(E_F - e_m) - e_m)
                   / ((w - i*eta*sign(E_F - e_m) - e_m)^2 + w'^2)

    Sigma_p^R(w) = ± sum_{m in poles} B[:,p,m] [(1-Pi(|e_m - w|))^{-1}-1] B[:,m,p]

where the sum runs over orbitals m whose energy lies between E_F and w
(+1 for w > E_F, -1 otherwise) and Pi is the retarded RPA response of
:mod:`gradscf.gw.polarizability`.

The pole-selection mask is genuinely discontinuous in ``omega`` (a pole
enters/leaves the contour); it is therefore wrapped in ``stop_gradient`` so
autodiff treats the active pole set as locally constant -- this is the
standard, physically correct treatment away from the discontinuities.

All functions are pure JAX and differentiable with respect to orbital
energies and low-rank factors.

References
----------
- R. W. Godby and R. J. Needs, "Metal-insulator transition in Kohn-Sham
  theory and quasiparticle theory", Phys. Rev. Lett. 62, 1169 (1989).
  DOI:10.1103/PhysRevLett.62.1169 (contour-deformation idea)
- T. Zhu and G. K.-L. Chan, J. Chem. Theory Comput. 14, 4856 (2018)
  (all-electron GW-CD formulation followed here, matching PySCF ``gw_cd``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.lax import Precision
from jaxtyping import Array

from .polarizability import rho_response_real
from .poles import screened_w_real_resolvent

# Memory threshold (bytes) above which batched evaluations fall back to a
# sequential lax.map: the residue path materializes O(naux^2) complex
# intermediates per batch element, which explodes for plane-wave product
# bases (naux ~ ngrid).
_BATCH_MEM_LIMIT = 1 << 31


def _response_solve_one(omega, channels, eta, naux, dtype, conjugate, response_scale=1.0):
    """RPA response at one frequency + (1-Pi)^{-1} - I (complex)."""
    pi = None
    for e_spin, b_ov_spin, spin_factor in channels:
        contrib = rho_response_real(
            omega, e_spin, b_ov_spin, eta=eta, spin_factor=spin_factor, conjugate=conjugate
        )
        pi = contrib if pi is None else pi + contrib
    pi = response_scale * pi
    eye = jnp.eye(naux, dtype=dtype)
    return jnp.linalg.solve(eye - pi, pi)



def _occupation_sign(mo_energy, ef, occupation_sign):
    sign = jnp.sign(mo_energy - ef) if occupation_sign is None else jnp.asarray(occupation_sign)
    return jax.lax.stop_gradient(sign)


def _static_residue_weights(omega, mo_energy, ef, occupation_sign):
    sign = _occupation_sign(mo_energy, ef, occupation_sign)
    return jax.lax.stop_gradient((jnp.sign(jnp.real(omega) - mo_energy) + sign) * .5)


def _imag_axis_weights(omega, mo_energy, ef, freqs, wts, eta):
    delta = omega - jnp.asarray(mo_energy) - 1j * eta * jnp.sign(ef - mo_energy)
    return jnp.asarray(wts)[None, :] * delta[:, None] / (delta[:, None] ** 2 + jnp.asarray(freqs)[None, :] ** 2)


def _imag_axis_integral(omega, wmn, mo_energy, ef, freqs, wts, eta,
                        wmn_static=None, occupation_sign=None):
    """Common scalar/matrix contraction; trailing W indices are spectators.

    Static subtraction integrates the own-G-pole limit analytically. The
    imaginary Green denominator uses eta=0; real-frequency W keeps its eta.
    """
    wmn = jnp.asarray(wmn)
    weights = _imag_axis_weights(omega, mo_energy, ef, freqs, wts,
                                 eta if wmn_static is None else 0.)
    if wmn_static is not None:
        wmn = wmn - jnp.asarray(wmn_static)[None, ...]
    value = -jnp.tensordot(weights.T, wmn, axes=((0, 1), (0, 1))) / jnp.pi
    if wmn_static is not None:
        sign = _occupation_sign(mo_energy, ef, occupation_sign)
        value += .5 * jnp.tensordot(sign, wmn_static, axes=1)
    return value


def sigma_imag_part(
    omega: float | Array,
    wmn_p: Array,
    mo_energy: Array,
    ef: float | Array,
    freqs: Array,
    wts: Array,
    eta: float,
    del_w: Array | None = None,
    p_index: int | Array | None = None,
    *,
    wmn_static: Array | None = None,
    occupation_sign: Array | None = None,
    del_w_static: Array | None = None,
) -> Array:
    """Imaginary-axis integral part of Sigma_p(omega).

    Parameters
    ----------
    omega:
        Real frequency at which Sigma is evaluated.
    wmn_p:
        Screened interaction column for orbital p, shape ``(nw, nmo)`` with
        entries ``W[w, m, p]``.
    mo_energy:
        Orbital energies of the Green's function channel, ``(nmo,)``.
    ef:
        Fermi-level estimate, conventionally ``(e_HOMO + e_LUMO) / 2``.
    freqs, wts:
        Imaginary-axis quadrature grid from
        :func:`gradscf.gw.freq.scaled_legendre_grid`.
    eta:
        Broadening (infinitesimal) of the non-interacting Green's function.

    Returns
    -------
    Complex scalar.
    """
    if wmn_static is not None and del_w is not None:
        if del_w_static is None or p_index is None:
            raise NotImplementedError("Static pole subtraction requires matching q0 static corrections")
        # The head/wing belongs only to the own-G-pole column. Treat body
        # and correction as one kernel, including the same static endpoint.
        wmn_p = jnp.asarray(wmn_p).at[:, p_index].add(del_w)
        wmn_static = jnp.asarray(wmn_static).at[p_index].add(del_w_static)
        del_w = None
    sigma = _imag_axis_integral(omega, wmn_p, mo_energy, ef, freqs, wts, eta,
                                wmn_static, occupation_sign)
    if del_w is not None:
        g0 = _imag_axis_weights(omega, mo_energy, ef, freqs, wts, eta)
        sigma = sigma - jnp.sum(del_w * g0[p_index]) / jnp.pi
    return sigma


def sigma_residue_part(
    omega: float | Array,
    mo_energy: Array,
    b_pm: Array,
    b_mp: Array,
    channels: tuple[tuple[Array, Array, float], ...],
    ef: float | Array,
    eta: float,
    conjugate: bool = False,
    q0: dict | None = None,
    response_scale: float = 1.0,
    wmn_static: Array | None = None,
    occupation_sign: Array | None = None,
    p_index: int | Array | None = None,
    resolvent_data: dict | None = None,
    g_indices: Array | None = None,
    del_w_static: Array | None = None,
) -> Array:
    """Residue part of Sigma_p(omega) from poles inside the contour.

    Parameters
    ----------
    omega:
        Real frequency at which Sigma is evaluated.
    mo_energy:
        Orbital energies of the Green's function channel, ``(nmo,)``.
    b_pm, b_mp:
        Factor slices ``B[:, p, m]`` and ``B[:, m, p]``, each of shape
        ``(naux, nmo)`` (m is the batched second axis).
    channels:
        Tuple of ``(mo_energy_spin, b_ov_spin, spin_factor)`` triples, one
        per spin channel.  Spin factors refer to the retarded response: a
        restricted calculation passes a single channel with factor 2.0
        (spin degeneracy), an unrestricted calculation passes the alpha and
        beta channels with factor 1.0 each.
    ef, eta:
        Fermi-level estimate and response broadening.
    response_scale:
        Weight of the channel sum in Pi (1/Nk for k-resolved screening).
        Applied to body, head and wings before dielectric inversion;
        independent of the outer q average of the self-energy.

    Returns
    -------
    Complex scalar.
    """
    mo_energy = jnp.asarray(mo_energy)
    b_pm = jnp.asarray(b_pm)
    b_mp = jnp.asarray(b_mp)
    eta = jnp.asarray(eta, dtype=jnp.float64)
    omega_r = jnp.asarray(omega, dtype=jnp.float64)

    above = omega_r > ef
    mask = jnp.where(
        above,
        (mo_energy < omega_r) & (mo_energy > ef),
        (mo_energy > omega_r) & (mo_energy < ef),
    )
    fm = jnp.where(above, 1.0, -1.0)
    # The active pole set is discontinuous in omega and in mo_energy; treat
    # it as locally constant for AD (see module docstring).
    mask = jax.lax.stop_gradient(mask)
    fm = jax.lax.stop_gradient(fm)

    if resolvent_data is not None:
        if q0 is not None or p_index is None:
            raise NotImplementedError("Pole residue expansion requires a molecular external orbital index")
        frequencies = jnp.abs(mo_energy - omega_r)
        indices = jnp.arange(mo_energy.size) if g_indices is None else g_indices
        from math import isqrt
        nmo = isqrt(resolvent_data["coupling"].shape[1])
        row = screened_w_real_resolvent(
            resolvent_data, frequencies, pairs=p_index*nmo+indices)
        weights = (_static_residue_weights(omega_r, mo_energy, ef, occupation_sign)
                   if wmn_static is not None else fm * mask.astype(row.dtype))
        if wmn_static is not None:
            row = row - jnp.asarray(wmn_static)
        return jnp.sum(weights * row)

    pole_freqs = jnp.abs(mo_energy - omega_r)  # (nmo,)
    weights = (_static_residue_weights(omega_r, mo_energy, ef, occupation_sign)
               if wmn_static is not None else mask.astype(jnp.complex128))
    active = weights != 0
    # An inactive G pole can coincide with a bare response pole. Inverting
    # its infinite Pi and subsequently multiplying by zero gives NaN. Keep
    # the batched solve, probing inactive lanes at the regular zero frequency.
    probe_freqs = jnp.where(active, pole_freqs, 0.)

    naux = b_pm.shape[0]
    dtype = jnp.complex128
    est_bytes = pole_freqs.shape[0] * naux * naux * 16 * 2
    solve_one = lambda w: _response_solve_one(
        w, channels, eta, naux, dtype, conjugate, response_scale
    )
    if est_bytes > _BATCH_MEM_LIMIT:
        screened_probe = jax.lax.map(solve_one, probe_freqs)
    else:
        screened_probe = jax.vmap(solve_one)(probe_freqs)  # (nmo, naux, naux)
    screened = jnp.where(active[:, None, None], screened_probe, 0.)
    first = b_pm.conj() if conjugate else b_pm
    contrib = jnp.einsum("Pm,mPQ,Qm->m", first, screened, b_mp, precision=Precision.HIGHEST)
    if wmn_static is not None:
        if q0 is not None and del_w_static is None:
            raise NotImplementedError("Static pole subtraction needs matching q0 corrections")
        # At eta=0 the subtracted residue vanishes at the contour boundary.
        # Signed weights include half the residue there and also handle an
        # empty/full spin channel without inventing a spin Fermi level.
        contrib = contrib - jnp.asarray(wmn_static)
    sigma_r = jnp.sum(weights * contrib)

    if q0 is not None:
        # q -> 0 head/wing residue correction for the m == p pole
        # (PRB 83, 245122 (2011); gradscf.gw.pbc.q0).  ``q0["qij"]` is
        # either a single array (one channel) or a tuple aligned with
        # ``channels`` (e.g. per-ki pair channels at k-point q = 0).
        qij = q0["qij"]
        q_abs = q0["q_abs"]
        volume = q0["volume"]
        p_index = q0["p_index"]
        nkpts = q0.get("nkpts", 1)
        qij_list = list(qij) if isinstance(qij, (tuple, list)) else [qij]
        if len(qij_list) != len(channels):
            raise NotImplementedError(
                "q0 residue correction requires qij entries aligned with the response channels."
            )
        eye = jnp.eye(naux, dtype=screened.dtype)
        # Reuse the finite probe inverse even when this correction is
        # inactive; replacing it by I could make an artificial zero Schur
        # complement under an outer vmap of the orbital callback.
        eps_body_inv = screened_probe[p_index] + eye
        q2 = jnp.sum(q_abs**2)
        q_norm = jnp.sqrt(q2)
        head_scale = (2.0 / jnp.pi) * (6.0 * jnp.pi**2 / volume / nkpts) ** (1.0 / 3.0)
        wings_const = jnp.sqrt(volume / 4.0 / jnp.pi**3) * (6.0 * jnp.pi**2 / volume / nkpts) ** (
            2.0 / 3.0
        )
        omega_c = jnp.asarray(probe_freqs[p_index], dtype=jnp.complex128)
        pi_00 = 0j
        pi_p0 = jnp.zeros(naux, dtype=jnp.complex128)
        for (e_spin, b_ov_spin, spin_factor), qij_c in zip(channels, qij_list):
            nocc_c = b_ov_spin.shape[1]
            if isinstance(e_spin, tuple):
                e_occ, e_virt = e_spin
            else:
                e_occ, e_virt = e_spin[:nocc_c], e_spin[nocc_c:]
            eia = e_occ[:, None] - e_virt[None, :]
            chi = 1.0 / (omega_c + eia + 2j * eta) + 1.0 / (-omega_c + eia)
            pi_00 = pi_00 + spin_factor * jnp.sum(qij_c.conj() * qij_c * chi)
            pi_p0 = pi_p0 + spin_factor * jnp.einsum(
                "Pia,ia->P", b_ov_spin, chi * qij_c.conj(), precision=Precision.HIGHEST
            )
        pi_00 = response_scale * pi_00
        pi_p0 = response_scale * pi_p0
        eps_00 = 1.0 - 4.0 * jnp.pi / q2 * pi_00
        eps_p0 = -jnp.sqrt(4.0 * jnp.pi) / q_norm * pi_p0
        schur = eps_00 - eps_p0.conj() @ eps_body_inv @ eps_p0
        eps_inv_00 = 1.0 / schur
        eps_inv_p0 = -eps_inv_00 * (eps_body_inv @ eps_p0)
        del00 = head_scale * (eps_inv_00 - 1.0)
        b_pp = b_pm[:, p_index]
        wing = wings_const * 2.0 * jnp.dot(b_pp.conj(), eps_inv_p0).real
        correction = del00 + wing
        correction = jnp.where(active[p_index], correction, 0.)
        if wmn_static is not None:
            sigma_r = sigma_r + weights[p_index]*(correction-del_w_static)
        else:
            sigma_r = sigma_r + mask.astype(contrib.dtype)[p_index]*correction

    return sigma_r if wmn_static is not None else fm * sigma_r


def sigma_cd(omega: float | Array, ctx: dict) -> Array:
    """Full contour-deformation self-energy Sigma_p(omega) from a context.

    ``ctx`` is a pytree dictionary with entries

    - ``mo_energy``: (nmo,) orbital energies of the GF channel
    - ``wmn_p``: (nw, nmo) imaginary-axis screened interaction column
    - optional ``wmn_static``: (nmo,) matching Wc(0) contractions; enables
      analytic static subtraction and the half-residue limit at a G pole
    - optional ``occupation_sign``: (nmo,), -1 occupied/+1 virtual; molecular
      drivers provide it explicitly, including empty/full spin channels
    - optional ``del_w_static``: scalar own-orbital q0 head/wing at zero
      frequency, matching ``del_w`` and ``q0`` when static subtraction is used
    - ``b_pm``, ``b_mp``: (naux, nmo) factor slices for orbital p
    - ``channels``: tuple of (mo_energy_spin, b_ov_spin, spin_factor)
      response channels (retarded-response spin factors: 2.0 restricted,
      1.0 per unrestricted spin)
    - ``ef``, ``eta``: Fermi estimate and broadening
    - ``freqs``, ``wts``: imaginary-axis quadrature

    With static subtraction, the imaginary Green denominator takes eta->0
    analytically. Finite eta is retained only in real-frequency W. This is
    an exact contour identity at eta=0; finite-eta boundary continuity and
    arbitrary higher derivatives are not implied. q0 head/wing contexts
    require the matching del_w_static correction in the same orbital frame.
    Contexts without wmn_static keep the legacy quadrature convention.

    Returns the complex scalar Sigma_p(omega).
    """
    static = ctx.get("wmn_static")
    corrected = ctx.get("q0") is not None or ctx.get("del_w") is not None
    if static is not None and corrected and ctx.get("del_w_static") is None:
        raise NotImplementedError("Static pole subtraction requires matching q0 static corrections")
    # Subtract Wc(0) before quadrature. Integrate its G-pole contribution
    # analytically together with the residue, including omega == epsilon_m.
    sigma_i = sigma_imag_part(
        omega,
        ctx["wmn_p"],
        ctx["mo_energy"],
        ctx["ef"],
        ctx["freqs"],
        ctx["wts"],
        ctx["eta"],
        del_w=ctx.get("del_w"),
        p_index=ctx.get("p_index"),
        wmn_static=static,
        occupation_sign=ctx.get("occupation_sign"),
        del_w_static=ctx.get("del_w_static"),
    )
    sigma_r = sigma_residue_part(
        omega,
        ctx["mo_energy"],
        ctx["b_pm"],
        ctx["b_mp"],
        ctx["channels"],
        ctx["ef"],
        ctx["eta"],
        conjugate=ctx.get("conjugate", False),
        q0=ctx.get("q0"),
        wmn_static=static,
        occupation_sign=ctx.get("occupation_sign"),
        p_index=ctx.get("p_index"),
        resolvent_data=ctx.get("resolvent_data"),
        g_indices=ctx.get("g_indices"),
        del_w_static=ctx.get("del_w_static"),
    )
    return sigma_i + sigma_r


__all__ = ["sigma_imag_part", "sigma_residue_part", "sigma_cd", "sigma_imag_matrix", "sigma_residue_matrix"]


def sigma_imag_matrix(
    omega: float | Array,
    wmn: Array,
    mo_energy: Array,
    ef: float | Array,
    freqs: Array,
    wts: Array,
    eta: float,
    *,
    wmn_static: Array | None = None,
    occupation_sign: Array | None = None,
) -> Array:
    """Off-diagonal imaginary-axis part of Sigma_mn(omega).

    Sigma_mn^I(w) = -(1/pi) sum_{q,w'} wts_w' W[q; m, n](iw')
                    emo_q / (emo_q^2 + w'^2),  emo_q = w - i*eta*sign(E_F - e_q) - e_q

    ``wmn`` is the full screened tensor ``W[q; m, n]`` of shape
    ``(nw, nq, nmo, nmo)`` from
    :func:`gradscf.gw.screened.screened_w_imag_axis_matrix`.
    """
    return _imag_axis_integral(omega, wmn, mo_energy, ef, freqs, wts, eta,
                               wmn_static, occupation_sign)


def sigma_residue_matrix(
    omega: float | Array,
    mo_energy: Array,
    b_mn: Array,
    channels: tuple[tuple[Array, Array, float], ...],
    ef: float | Array,
    eta: float,
    conjugate: bool = False,
    *,
    wmn_static: Array | None = None,
    occupation_sign: Array | None = None,
) -> Array:
    """Off-diagonal residue part of Sigma_mn(omega).

    Sigma_mn^R(w) = ± sum_{q in poles} conj(B_P[m,q]) [(1-Pi)^{-1}-1]_PQ B_Q[q,n]

    with Pi the retarded response at |e_q - w| (see sigma_residue_part).
    ``b_mn`` is the full factor tensor ``(naux, nmo, nmo)``.
    """
    mo_energy = jnp.asarray(mo_energy)
    b_mn = jnp.asarray(b_mn)
    eta = jnp.asarray(eta, dtype=jnp.float64)
    omega_r = jnp.asarray(omega, dtype=jnp.float64)

    above = omega_r > ef
    mask = jnp.where(
        above,
        (mo_energy < omega_r) & (mo_energy > ef),
        (mo_energy > omega_r) & (mo_energy < ef),
    )
    fm = jnp.where(above, 1.0, -1.0)
    mask = jax.lax.stop_gradient(mask)
    fm = jax.lax.stop_gradient(fm)

    pole_freqs = jnp.abs(mo_energy - omega_r)  # (nq,)
    weights = (fm*mask.astype(jnp.complex128) if wmn_static is None else
               _static_residue_weights(omega_r, mo_energy, ef, occupation_sign))
    active = weights != 0
    probe_freqs = jnp.where(active, pole_freqs, 0.)
    naux = b_mn.shape[0]
    est_bytes = pole_freqs.shape[0] * naux * naux * 16 * 2
    solve_one = lambda w: _response_solve_one(w, channels, eta, naux, jnp.complex128, conjugate)
    if est_bytes > _BATCH_MEM_LIMIT:
        screened = jax.lax.map(solve_one, probe_freqs)
    else:
        screened = jax.vmap(solve_one)(probe_freqs)  # (nq, naux, naux)

    first = b_mn.conj() if conjugate else b_mn
    # contrib[m,n] = sum_{q in poles} conj(B_P[m,q]) screened[q]_PQ B_Q[q,n]
    screened_masked = jnp.where(active[:, None, None], screened*weights[:, None, None], 0.)
    contrib = jnp.einsum("Pmq,qPQ,Qqn->mn", first, screened_masked, b_mn, precision=Precision.HIGHEST)
    if wmn_static is not None:
        contrib -= jnp.tensordot(weights, wmn_static, axes=1)
    return contrib
