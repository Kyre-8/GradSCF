"""Plane-wave product basis for periodic GW (Gamma point first).

The periodic analogue of the molecular low-rank factors
``B_P[p,q]`` with ``(pq|rs) ~= sum_P B_P[p,q] B_P[r,s]`` is the plane-wave
product basis: orbital pair densities are Fourier transformed,

    pairs_G(m,n) = sum_r conj(psi_m(r)) psi_n(r) exp(-i G.r) (unnormalized DFT)

and the Coulomb interaction is absorbed into symmetric factors

    B_G(m,n) = sqrt(volume * v_G) / N_grid * pairs_G(m,n),   v_G = 4 pi /|G|^2

so that two-electron matrix elements recover the molecular form

    (mi|jn) = sum_G conj(B_G(i,m)) B_G(j,n)

(derivation: the FFT pair-potential contraction of
``gradscf.integrals.periodic.fft.get_jk`` expanded in G space; the
unnormalized fftn/ifftn pair contributes 1/N_grid and the grid weight
contributes volume/N_grid).  The G = 0 component has v_0 = 0 by
construction (see ``coulomb_kernel``); the q -> 0 head/wing corrections of
the inverse dielectric matrix are a separate Stage-2 slice
(``gradscf.gw.pbc.q0``).

Because the factors are complex, contractions that are conjugation-free in
the molecular code must conjugate the first factor here.

References
----------
- L. Hedin, Phys. Rev. 139, A796 (1965). DOI:10.1103/PhysRev.139.A796
- J. Deslippe et al., Comput. Phys. Commun. 183, 1269 (2012).
  DOI:10.1016/j.cpc.2011.12.023 (plane-wave product basis / BerkeleyGW)
- M. S. Hybertsen and S. G. Louie, Phys. Rev. B 34, 5390 (1986).
  DOI:10.1103/PhysRevB.34.5390
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from jax.lax import Precision
from jaxtyping import Array

from ...integrals.periodic.coulomb import coulomb_kernel


def gamma_product_factors(inputs, mo_coeff: Array, *, mesh: tuple[int, int, int]) -> Array:
    """Build plane-wave orbital-pair factors at the Gamma point.

    Parameters
    ----------
    inputs:
        :class:`gradscf.integrals.periodic.fft.PeriodicInputs` built for a
        single (Gamma) k point; uses ``inputs.ao[0, 0]`` (real-space orbital
        values), ``inputs.weights`` and ``inputs.gvectors``.
    mo_coeff:
        Gamma-point orbital coefficients, shape ``(nao, nmo)``.
    mesh:
        The FFT grid shape the inputs were built with (``cell.mesh``).

    Returns
    -------
    Complex array ``B_G(m, n)`` of shape ``(ngrid, nmo, nmo)`` such that
    ``(mi|jn) = sum_G conj(B_G(i,m)) B_G(j,n)`` (the pair indices
    coincide for real Gamma orbitals).
    """
    ao = jnp.asarray(inputs.ao[0, 0])  # (ngrid, nao), real at Gamma
    coeff = jnp.asarray(mo_coeff)
    gvectors = jnp.asarray(inputs.gvectors)
    ngrid = ao.shape[0]
    mesh = tuple(int(m) for m in mesh)
    if int(np.prod(mesh)) != ngrid:
        raise ValueError(f"mesh {mesh} does not match the AO grid size {ngrid}")
    # weights[g] = volume/ngrid uniformly; recover the cell volume:
    volume = float(jnp.asarray(inputs.weights[0])) * ngrid

    psi = ao @ coeff  # (ngrid, nmo)
    nmo = psi.shape[1]
    pairs = psi.conj()[:, :, None] * psi[:, None, :]  # (ngrid, nmo, nmo)
    pairs_g = jnp.fft.fftn(pairs.reshape(mesh + (nmo, nmo)), axes=(0, 1, 2)).reshape(
        ngrid, nmo, nmo
    )
    v_g = coulomb_kernel(gvectors)  # (ngrid,), v_0 = 0
    scale = jnp.sqrt(volume * v_g) / ngrid
    return (scale[:, None, None] * pairs_g).astype(jnp.complex128)


__all__ = ["gamma_product_factors", "kpoint_product_factors"]


def kpoint_product_factors(
    inputs,
    mo_coeff_k: Array,
    *,
    mesh: tuple[int, int, int],
    momentum_table: Array,
) -> list[Array]:
    """Plane-wave product factors for all momentum transfers q.

    For each momentum transfer index ``q`` and each ``ki`` (with partner
    ``ka = momentum_table[ki, q]``, i.e. ``ka = ki - q`` on the mesh),

        B_q[ki]_G(m, n) = sqrt(volume * v(G - q_cart)) / N_grid
                          * sum_r conj(u_ki,m(r)) u_ka,n(r)
                            exp(i(ka - ki + q_cart).r) exp(-i G.r)

    where ``u_k`` are the periodic parts stored in ``inputs.ao`` and
    ``q_cart = inputs.kpoints[q]``. The reciprocal wrap ``ka - ki + q``
    aligns every pair with the same auxiliary wavevectors G-q. No Bloch
    phase is added when that wrap vanishes: the stored u_k are periodic.

    Parameters
    ----------
    inputs:
        :class:`PeriodicInputs` built for the full k mesh.
    mo_coeff_k:
        Orbital coefficients per k point, shape ``(nk, nao, nmo)``.
    mesh:
        FFT grid shape (``cell.mesh``).
    momentum_table:
        From :func:`gradscf.pbc.momentum.momentum_transfer_table`.

    Returns
    -------
    List of length ``nk``; entry ``q`` has shape ``(nk, ngrid, nmo, nmo)``
    (index: ki, G, m, n), complex128.
    """
    ao = jnp.asarray(inputs.ao[:, 0])  # (nk, ngrid, nao), periodic parts u_k
    coeff_k = jnp.asarray(mo_coeff_k)
    gvectors = jnp.asarray(inputs.gvectors)
    kpoints = jnp.asarray(inputs.kpoints)
    nk, ngrid, nao = ao.shape
    mesh = tuple(int(m) for m in mesh)
    if int(np.prod(mesh)) != ngrid:
        raise ValueError(f"mesh {mesh} does not match the AO grid size {ngrid}")
    volume = float(jnp.asarray(inputs.weights[0])) * ngrid
    nmo = coeff_k.shape[-1]

    psi_k = jnp.einsum("kgp,kpm->kgm", ao, coeff_k, precision=Precision.HIGHEST)
    table = np.asarray(momentum_table)
    out = []
    for q in range(nk):
        ka = table[:, q]  # partner index per ki
        q_cart = kpoints[q]
        kernel = coulomb_kernel(gvectors - q_cart)  # canonical transfer -q
        scale = jnp.sqrt(volume * kernel) / ngrid
        # pairs[ki] = conj(u_ki,m) u_ka,n on the grid
        pairs = psi_k.conj()[:, :, :, None] * psi_k[jnp.asarray(ka)][:, :, None, :]
        wrap = kpoints[jnp.asarray(ka)] - kpoints + q_cart
        phase = jnp.exp(1j * (wrap @ jnp.asarray(inputs.coords).T))
        pairs = pairs * phase[:, :, None, None]
        pairs_g = jnp.fft.fftn(
            pairs.reshape(nk, *mesh, nmo, nmo), axes=(1, 2, 3)
        ).reshape(nk, ngrid, nmo, nmo)
        out.append((scale[None, :, None, None] * pairs_g).astype(jnp.complex128))
    return out
