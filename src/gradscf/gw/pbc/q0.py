"""q -> 0 head/wing corrections of the inverse dielectric matrix (Gamma).

The plane-wave product basis sets v(G=0) = 0, which removes the head of the
screened Coulomb interaction entirely.  For isolated systems and small
cells this shifts the occupied-state self-energy by ~1e-2 Ha.  Following
the finite-size correction of PySCF ``pbc.gw.krgw_cd`` (itself based on the
k.p formulation of the references below), the q -> 0 limit is restored
analytically:

- transition moments from k.p perturbation theory,

      qij_ia = (1/sqrt(Omega)) * (-i q.<psi_i|grad|psi_a>) / (e_a - e_i)

  evaluated at a small offset q = (1e-3, 0, 0);

- head (G=0, G'=0) and wing (G=P, G'=0) elements of the dielectric matrix,

      eps_00 = 1 - (4 pi/q^2) Pi_00,   eps_P0 = -sqrt(4 pi)/q * Pi_P0

- Schur-complement inverse given the body (G,G' != 0) inverse,

      eps_inv_00 = 1/(eps_00 - eps_P0^dagger eps_body_inv eps_P0)
      eps_inv_P0 = -eps_inv_00 eps_body_inv eps_P0

The real-frequency correction retains the PySCF finite-eta convention
for conjugating the opposite wing; it is not a general non-Hermitian
dielectric-block inversion.

- finite-size scaled correction terms added to the self-energy,

      Del_00(w) = (2/pi)(6 pi^2/Omega/Nk)^(1/3) (eps_inv_00 - 1)
      Del_P0[n](w) = sqrt(Omega/4 pi^3)(6 pi^2/Omega/Nk)^(2/3) * 2 Re sum_P B_P[n,n] eps_inv_P0[P]

References
----------
- J. Deslippe et al., Comput. Phys. Commun. 183, 1269 (2012).
  DOI:10.1016/j.cpc.2011.12.023
- q -> 0 head/wing formulation: Phys. Rev. B 83, 245122 (2011); see also
  T. Zhu and G. K.-L. Chan, arXiv:2007.03148 (2020) for the molecular
  analogue.
- M. S. Hybertsen and S. G. Louie, Phys. Rev. B 34, 5390 (1986).
  DOI:10.1103/PhysRevB.34.5390
"""

from __future__ import annotations

from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np
from jax.lax import Precision
from jaxtyping import Array

from ..polarizability import rho_response_iw

_Q_ABS = 1e-3


def gamma_cell_quadrature(lattice, kmesh, *, order=2):
    """Nonzero Gauss nodes in the reciprocal parallelepiped around Gamma.

    Cartesian points are in inverse Bohr. Weights integrate a normalized
    Brillouin-zone sum and add to 1/prod(kmesh). Only even orders are used,
    so the singular Coulomb origin is never sampled. This is a numerical
    cell integral, requiring order convergence; it is neither the exact-zero
    finite part of a slab/wire kernel nor the folded-OV k.p approximation.
    """
    if (isinstance(order, bool) or not isinstance(order, Integral)
            or order < 2 or order % 2):
        raise ValueError('Gamma-cell quadrature order must be a positive even integer >= 2')
    lattice, mesh = np.asarray(lattice), np.asarray(kmesh)
    if (lattice.shape != (3, 3) or np.iscomplexobj(lattice)
            or not np.isfinite(lattice).all() or np.linalg.det(lattice) <= 0
            or mesh.shape != (3,) or not np.issubdtype(mesh.dtype, np.integer)
            or np.any(mesh < 1)):
        raise ValueError('Use a finite positive-volume lattice and three positive k-mesh dimensions')
    nodes, weights = np.polynomial.legendre.leggauss(order)
    indices = np.stack(np.meshgrid(*(np.arange(order),)*3, indexing='ij'), axis=-1).reshape(-1, 3)
    fractions = nodes[indices]/(2*mesh[None, :])
    normalized_weights = np.prod(weights[indices]/(2*mesh[None, :]), axis=1)
    reciprocal = 2*np.pi*np.linalg.inv(lattice).T
    return fractions@reciprocal, normalized_weights


def transition_moments_kpoints(
    inputs,
    mo_energy_k: Array,
    mo_coeff_k: Array,
    nocc: int,
) -> tuple[Array, Array]:
    """k.p transition moments qij[ki, i, a] for every k point.

    Uses the per-k derivative channels of ``inputs.ao`` (which retain
    ``(grad + i k) u_k``), matching PySCF ``get_qij`` with
    ``eval_ao(kpt, deriv=1)``.  Returns ``(nk, nocc, nvir)`` and the
    offset vector q.
    """
    weights = jnp.asarray(inputs.weights)
    ngrid = weights.shape[0]
    volume = weights[0] * ngrid
    q_cart = jnp.asarray([_Q_ABS, 0.0, 0.0])
    qij_all = []
    for ki in range(len(mo_energy_k)):
        ao = jnp.asarray(inputs.ao[ki])  # (4, ngrid, nao)
        e = jnp.asarray(mo_energy_k[ki])
        c = jnp.asarray(mo_coeff_k[ki])
        nocc_i = int(nocc)
        ao_ao_grad = jnp.einsum(
            "g,gp,xgq->xpq", weights, ao[0].conj(), ao[1:4], precision=Precision.HIGHEST
        )
        q_ao_ao_grad = -1j * jnp.einsum("x,xpq->pq", q_cart, ao_ao_grad)
        q_mo = c[:, :nocc_i].T.conj() @ q_ao_ao_grad @ c[:, nocc_i:]
        enm = 1.0 / (e[nocc_i:, None] - e[None, :nocc_i])
        qij_all.append(enm.T * q_mo / jnp.sqrt(volume))
    return jnp.stack(qij_all), q_cart


def transition_moments_gamma(
    inputs,
    mo_energy: Array,
    mo_coeff: Array,
    nocc: int,
) -> tuple[Array, Array]:
    """k.p transition moments qij[i,a] at the Gamma point.

    Returns
    -------
    qij:
        Complex array ``(nocc, nvir)`` with
        ``qij_ia = (-i q.<psi_i|grad|psi_a>) / (e_a - e_i) / sqrt(Omega)``
        at ``q = (1e-3, 0, 0)``.
    q_abs:
        The offset vector ``q`` (Cartesian, Bohr^-1).
    """
    ao = jnp.asarray(inputs.ao[0])  # (4, ngrid, nao): value + 3 derivative channels
    weights = jnp.asarray(inputs.weights)
    mo_energy = jnp.asarray(mo_energy)
    mo_coeff = jnp.asarray(mo_coeff)
    nocc = int(nocc)
    nmo = mo_energy.shape[0]
    nvir = nmo - nocc
    ngrid = ao.shape[1]
    volume = weights[0] * ngrid

    # <ao_p | d/dx_alpha ao_q> on the uniform grid
    ao_ao_grad = jnp.einsum(
        "g,gp,xgq->xpq", weights, ao[0].conj(), ao[1:4], precision=Precision.HIGHEST
    )
    q_cart = jnp.asarray([_Q_ABS, 0.0, 0.0])
    q_ao_ao_grad = -1j * jnp.einsum("x,xpq->pq", q_cart, ao_ao_grad)
    q_mo_mo_grad = mo_coeff[:, :nocc].T.conj() @ q_ao_ao_grad @ mo_coeff[:, nocc:]
    enm = 1.0 / (mo_energy[nocc:, None] - mo_energy[None, :nocc])  # (nvir, nocc)
    qij = enm.T * q_mo_mo_grad / jnp.sqrt(volume)
    return qij.reshape(nocc, nvir), q_cart


def screened_w_imag_axis_head_wing(
    b: Array,
    b_ov: Array,
    mo_energy: Array,
    qij: Array,
    q_abs: Array,
    volume: float,
    freqs: Array,
    *,
    nkpts: int = 1,
) -> tuple[Array, Array, Array]:
    """Imaginary-axis screened W with head/wing corrections (Gamma).

    Returns
    -------
    wmn:
        Body screened interaction, shape ``(nw, nmo, nmo)`` (as
        :func:`gradscf.gw.screened.screened_w_imag_axis`).
    del00:
        Head correction term, shape ``(nw,)``.
    delP0:
        Wing correction per band, shape ``(nw, nmo)``.
    """
    b = jnp.asarray(b)
    b_ov = jnp.asarray(b_ov)
    mo_energy = jnp.asarray(mo_energy)
    qij = jnp.asarray(qij)
    freqs = jnp.asarray(freqs)
    naux = b.shape[0]
    q2 = jnp.sum(jnp.asarray(q_abs) ** 2)
    q_norm = jnp.sqrt(q2)
    head_scale = (2.0 / jnp.pi) * (6.0 * jnp.pi**2 / volume / nkpts) ** (1.0 / 3.0)
    wings_const = jnp.sqrt(volume / 4.0 / jnp.pi**3) * (6.0 * jnp.pi**2 / volume / nkpts) ** (
        2.0 / 3.0
    )

    def at_frequency(omega: Array):
        # body dielectric (same as the uncorrected path)
        pi = rho_response_iw(omega, mo_energy, b_ov, spin_factor=4.0, conjugate=True)
        eye = jnp.eye(naux, dtype=pi.dtype)
        eps_body_inv = jnp.linalg.solve(eye - pi, eye)
        screened = eps_body_inv - eye
        wmn = jnp.einsum("Pmn,PQ,Qmn->mn", b.conj(), screened, b, precision=Precision.HIGHEST)

        # head/wing dielectric elements
        nocc = b_ov.shape[1]
        eia = mo_energy[:nocc, None] - mo_energy[None, nocc:]
        chi = eia / (omega**2 + eia * eia)
        pi_00 = 4.0 * jnp.sum(qij.conj() * qij * chi) / nkpts
        eps_00 = 1.0 - 4.0 * jnp.pi / q2 * pi_00
        pi_p0 = (
            4.0
            * jnp.einsum(
                "Pia,ia->P", b_ov, chi * qij.conj(), precision=Precision.HIGHEST
            )
            / nkpts
        )
        eps_p0 = -jnp.sqrt(4.0 * jnp.pi) / q_norm * pi_p0

        # Schur-complement inverse
        schur = eps_00 - eps_p0.conj() @ eps_body_inv @ eps_p0
        eps_inv_00 = 1.0 / schur
        eps_inv_p0 = -eps_inv_00 * (eps_body_inv @ eps_p0)

        del00 = head_scale * (eps_inv_00 - 1.0)
        wn_p0 = jnp.einsum("Pnn,P->n", b.conj(), eps_inv_p0, precision=Precision.HIGHEST)
        delp0 = wings_const * 2.0 * wn_p0.real
        return wmn, del00, delp0

    est_bytes = freqs.shape[0] * naux * naux * 16 * 2
    if est_bytes > (1 << 31):
        wmn, del00, delp0 = jax.lax.map(at_frequency, freqs)
    else:
        wmn, del00, delp0 = jax.vmap(at_frequency)(freqs)
    return wmn, del00, delp0


def q0_residue_correction(
    omega_pole: Array,
    b_pp: Array,
    channels: tuple,
    qij: Array | tuple[Array, ...],
    q_abs: Array,
    volume: float,
    eta: float,
    *,
    nkpts: int = 1,
) -> Array:
    """Head/wing residue correction for the m == p pole (real frequency).

    ``omega_pole = |e_p - omega|``; ``channels`` are the retarded response
    channels of :func:`gradscf.gw.self_energy.sigma_residue_part`;
    ``b_pp`` is the factor column ``B[:, p, p]``.  Returns the complex
    scalar ``Del_00 + wings`` to be added (with the contour sign ``fm``).
    ``qij`` is a single array for one channel or a tuple aligned with
    ``channels``. All response channels are averaged by ``nkpts`` before
    dielectric inversion; finite-size prefactors are applied separately.
    """
    from ..polarizability import rho_response_real

    qij_list = list(qij) if isinstance(qij, (tuple, list)) else [qij]
    if len(qij_list) != len(channels):
        raise ValueError("qij must have one entry per response channel.")
    b_pp = jnp.asarray(b_pp)
    q2 = jnp.sum(jnp.asarray(q_abs) ** 2)
    q_norm = jnp.sqrt(q2)
    head_scale = (2.0 / jnp.pi) * (6.0 * jnp.pi**2 / volume / nkpts) ** (1.0 / 3.0)
    wings_const = jnp.sqrt(volume / 4.0 / jnp.pi**3) * (6.0 * jnp.pi**2 / volume / nkpts) ** (
        2.0 / 3.0
    )

    # body inverse dielectric at the pole frequency
    pi = None
    for e_spin, b_ov_spin, spin_factor in channels:
        contrib = rho_response_real(
            omega_pole, e_spin, b_ov_spin, eta=eta, spin_factor=spin_factor, conjugate=True
        )
        pi = contrib if pi is None else pi + contrib
    pi = pi / nkpts
    naux = pi.shape[0]
    eye = jnp.eye(naux, dtype=pi.dtype)
    eps_body_inv = jnp.linalg.solve(eye - pi, eye)

    # Match the same channel sum and normalization used for the body.
    omega_c = jnp.asarray(omega_pole, dtype=jnp.complex128)
    pi_00 = 0j
    pi_p0 = jnp.zeros(naux, dtype=jnp.complex128)
    for (energy, b_ov, spin_factor), qij_c in zip(channels, qij_list):
        nocc = b_ov.shape[1]
        e_occ, e_virt = energy if isinstance(energy, tuple) else (energy[:nocc], energy[nocc:])
        eia = e_occ[:, None] - e_virt[None, :]
        chi = 1.0 / (omega_c + eia + 2j * eta) + 1.0 / (-omega_c + eia)
        qij_c = jnp.asarray(qij_c)
        pi_00 = pi_00 + spin_factor * jnp.sum(qij_c.conj() * qij_c * chi)
        pi_p0 = pi_p0 + spin_factor * jnp.einsum(
            "Pia,ia->P", b_ov, chi * qij_c.conj(), precision=Precision.HIGHEST
        )
    pi_00 = pi_00 / nkpts
    pi_p0 = pi_p0 / nkpts
    eps_00 = 1.0 - 4.0 * jnp.pi / q2 * pi_00
    eps_p0 = -jnp.sqrt(4.0 * jnp.pi) / q_norm * pi_p0

    schur = eps_00 - eps_p0.conj() @ eps_body_inv @ eps_p0
    eps_inv_00 = 1.0 / schur
    eps_inv_p0 = -eps_inv_00 * (eps_body_inv @ eps_p0)

    del00 = head_scale * (eps_inv_00 - 1.0)
    wn_p0 = jnp.dot(b_pp.conj(), eps_inv_p0)
    wing = wings_const * 2.0 * wn_p0.real
    return del00 + wing


__all__ = [
    "gamma_cell_quadrature",
    "transition_moments_gamma",
    "transition_moments_kpoints",
    "screened_w_imag_axis_head_wing",
    "q0_residue_correction",
]
