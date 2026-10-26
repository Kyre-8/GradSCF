"""RPA density response (irreducible polarizability) in a low-rank basis.

Given low-rank ERI factors ``B_P[p,q]`` with
``(pq|rs) ~= sum_P B_P[p,q] B_P[r,s]`` (spectral factorization of the full
ERI, see :mod:`gradscf.integrals.molecular.factorization`), the independent-particle density response in
the auxiliary channel basis is

    Pi_PQ(iw) = spin_factor * sum_ia B_P[i,a] e_ia / (w^2 + e_ia^2) B_Q[i,a]

on the imaginary axis (Eq. for chi_0 at i*omega), and

    Pi_PQ(w)  = spin_factor * sum_ia B_P[i,a]
                [1/(w + e_ia + 2i*eta) + 1/(-w + e_ia)] B_Q[i,a]

at real frequencies (retarded form with broadening ``eta``), where
``e_ia = e_i - e_a < 0`` runs over occupied i and virtual a orbitals of one
spin channel.  ``spin_factor`` is 4 for a spin-restricted channel
(2 spins x 2 time orderings) and 2 for each unrestricted spin channel; the
total restricted-summed response is recovered by summing channels.

These are pure JAX functions and are differentiable with respect to
``mo_energy`` and ``b_ov`` under ``jax.grad``/``jax.vjp``.
For complex periodic factors (``conjugate=True``), the convention is
``Pi_PQ = sum_ia B_Pia chi_ia conj(B_Qia)`` on both frequency axes.

References
----------
- L. Hedin, Phys. Rev. 139, A796 (1965). DOI:10.1103/PhysRev.139.A796
- X. Ren et al., New J. Phys. 14, 053020 (2012).
  DOI:10.1088/1367-2630/14/5/053020 (RI form of the RPA polarizability)
- T. Zhu and G. K.-L. Chan, J. Chem. Theory Comput. 14, 4856 (2018)
  (retarded response used in contour-deformation GW; the factor
  2*eta broadening follows PySCF ``gw_cd.get_rho_response_R``).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.lax import Precision
from jaxtyping import Array


def rho_response_iw(
    omega: float | Array,
    mo_energy: Array,
    b_ov: Array,
    *,
    spin_factor: float = 4.0,
    conjugate: bool = False,
) -> Array:
    """Density response in auxiliary basis at imaginary frequency ``i*omega``.

    Parameters
    ----------
    omega:
        Real scalar; the response is evaluated at ``i * omega``.
    mo_energy:
        Mean-field orbital energies, shape ``(nmo,)``, occupied first.
        Optionally a tuple ``(e_occ, e_virt)`` supplying different energy
        sets for the occupied and virtual indices (k-point pairs).
    b_ov:
        Low-rank factors transformed to the occupied-virtual MO block,
        shape ``(naux, nocc, nvir)``.
    spin_factor:
        4.0 for a restricted channel, 2.0 for one unrestricted spin.

    Returns
    -------
    Real symmetric array of shape ``(naux, naux)``.
    """
    b_ov = jnp.asarray(b_ov)
    if isinstance(mo_energy, tuple):
        e_occ, e_virt = (jnp.asarray(x) for x in mo_energy)
    else:
        mo_energy = jnp.asarray(mo_energy)
        nocc = b_ov.shape[1]
        e_occ, e_virt = mo_energy[:nocc], mo_energy[nocc:]
    eia = e_occ[:, None] - e_virt[None, :]
    chi = eia / (omega**2 + eia * eia)
    weighted = b_ov * chi[None, :, :]
    second = b_ov.conj() if conjugate else b_ov
    return spin_factor * jnp.einsum("Pia,Qia->PQ", weighted, second, precision=Precision.HIGHEST)


def rho_response_real(
    omega: float | Array,
    mo_energy: Array,
    b_ov: Array,
    *,
    eta: float = 1e-3,
    spin_factor: float = 2.0,
    conjugate: bool = False,
) -> Array:
    """Retarded density response in auxiliary basis at real frequency ``omega``.

    Complex array of shape ``(naux, naux)``; see module docstring for the
    defining equation.  ``eta`` is the broadening entering as ``2*i*eta``
    (PySCF ``gw_cd`` convention).  ``mo_energy`` accepts the same tuple
    form as :func:`rho_response_iw`.
    """
    b_ov = jnp.asarray(b_ov)
    if isinstance(mo_energy, tuple):
        e_occ, e_virt = (jnp.asarray(x) for x in mo_energy)
    else:
        mo_energy = jnp.asarray(mo_energy)
        nocc = b_ov.shape[1]
        e_occ, e_virt = mo_energy[:nocc], mo_energy[nocc:]
    eia = e_occ[:, None] - e_virt[None, :]
    omega_c = jnp.asarray(omega, dtype=jnp.complex128)
    eta_c = jnp.asarray(eta, dtype=jnp.float64)
    chi = 1.0 / (omega_c + eia + 2j * eta_c) + 1.0 / (-omega_c + eia)
    weighted = b_ov * chi[None, :, :]
    # Pi = B chi B^dagger: conjugate the vertex, never the retarded weight.
    second = b_ov.conj() if conjugate else b_ov
    return spin_factor * jnp.einsum("Pia,Qia->PQ", weighted, second, precision=Precision.HIGHEST)


__all__ = ["rho_response_iw", "rho_response_real"]
