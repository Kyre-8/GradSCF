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

from numbers import Integral
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
    if getattr(inputs, "boundary", None) is not None:
        raise NotImplementedError("Periodic GW product factors require three-dimensional inputs")
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


__all__ = ["gamma_product_factors", "kpoint_product_factors",
           "kpoint_product_factor_builder", "kpoint_transition_block"]


def kpoint_product_factor_builder(inputs, mo_coeff_k, *, mesh, momentum_table,
                                  max_block_bytes=None):
    """Cache periodic orbital values and build only requested (q,ki,m,n) pairs.

    The returned callable accepts one static q index, ``pairs`` of shape
    (npair,2), and optional ki indices. Its result has shape
    (nki,ngrid,npair). The same G-q kernel and reciprocal-wrap phase are used
    by the legacy full-factor wrapper. No q-dependent pair block is cached.
    max_block_bytes bounds the estimated pair/FFT/output/phase arrays before
    their allocation; it is an array bound, not a process-RSS guarantee.
    """
    if getattr(inputs, "boundary", None) is not None:
        raise NotImplementedError("Periodic GW product factors require three-dimensional inputs")
    ao, coeff = jnp.asarray(inputs.ao[:, 0]), jnp.asarray(mo_coeff_k)
    if ao.ndim != 3 or coeff.ndim != 3 or coeff.shape[:2] != (ao.shape[0], ao.shape[2]):
        raise ValueError("Use AO values (nk,ngrid,nao) and coefficients (nk,nao,nmo)")
    nk, ngrid, _ = ao.shape
    nmo = coeff.shape[-1]
    mesh = tuple(int(m) for m in mesh)
    if len(mesh) != 3 or min(mesh) < 1 or int(np.prod(mesh)) != ngrid:
        raise ValueError("The FFT mesh must match the AO grid")
    table = np.asarray(momentum_table)
    if (table.shape != (nk, nk) or not np.issubdtype(table.dtype, np.integer)
            or np.any(table < 0) or np.any(table >= nk)
            or not np.all(np.sort(table, axis=0) == np.arange(nk)[:, None])):
        raise ValueError("Each momentum-table column must be a permutation of all k points")
    if max_block_bytes is not None and (not isinstance(max_block_bytes, Integral) or max_block_bytes < 1):
        raise ValueError("max_block_bytes must be a positive integer or None")
    psi = jnp.einsum("kgp,kpm->kgm", ao, coeff, precision=Precision.HIGHEST)
    kpoints, gvectors, coords = map(jnp.asarray, (inputs.kpoints, inputs.gvectors, inputs.coords))
    volume = float(jnp.asarray(inputs.weights[0]))*ngrid

    def build(q, *, pairs, k_indices=None):
        if not isinstance(q, Integral) or not 0 <= q < nk:
            raise ValueError("q must be an in-range momentum-table index")
        pairs = np.asarray(pairs)
        indices = np.arange(nk) if k_indices is None else np.asarray(k_indices)
        if (pairs.ndim != 2 or pairs.shape[1] != 2 or not len(pairs)
                or not np.issubdtype(pairs.dtype, np.integer)
                or np.any(pairs < 0) or np.any(pairs >= nmo)):
            raise ValueError("pairs must be a nonempty integer (npair,2) orbital-index array")
        if (indices.ndim != 1 or not len(indices) or not np.issubdtype(indices.dtype, np.integer)
                or np.any(indices < 0) or np.any(indices >= nk)):
            raise ValueError("k_indices must be a nonempty in-range integer vector")
        estimate = 16*ngrid*(8*len(indices)*len(pairs)+len(indices)+1)
        if max_block_bytes is not None and estimate > max_block_bytes:
            raise ValueError(f"Requested product block exceeds max_block_bytes: {estimate} bytes")
        partner = table[indices, q]
        left = psi[jnp.asarray(indices)][:, :, jnp.asarray(pairs[:, 0])]
        right = psi[jnp.asarray(partner)][:, :, jnp.asarray(pairs[:, 1])]
        q_cart = kpoints[q]
        wrap = kpoints[jnp.asarray(partner)]-kpoints[jnp.asarray(indices)]+q_cart
        phase = jnp.exp(1j*(wrap@coords.T))
        values = left.conj()*right*phase[:, :, None]
        transformed = jnp.fft.fftn(values.reshape(len(indices), *mesh, len(pairs)),
                                   axes=(1, 2, 3)).reshape(len(indices), ngrid, len(pairs))
        scale = jnp.sqrt(volume*coulomb_kernel(gvectors-q_cart))/ngrid
        return (scale[None, :, None]*transformed).astype(jnp.complex128)
    return build


def kpoint_transition_block(build, mo_energy_k, nocc, momentum_table, q):
    """One complete restricted OV space for shared periodic GW/BSE screening.

    Returns factors B(ngrid,nk*nocc*nvir), gram=B^dagger B and unweighted
    e_occ(ki)-e_virt(ka), ordered (ki,i,a), ka=table[ki,q]. Frequency, spin
    and 1/Nk weights belong to the caller. Use B to project selected output
    pairs, then discard it before advancing q; this helper does not create
    an all-q cache or a QP/optical-window restriction. Callers must budget
    the complete OV factor/Gram storage, in addition to the builder's block
    array bound.
    """
    energy = jnp.asarray(mo_energy_k)
    if energy.ndim != 2 or not isinstance(nocc, Integral) or not 0 < nocc < energy.shape[1]:
        raise ValueError("Use a k-resolved spectrum with nonempty occupied and virtual spaces")
    nk, nmo = energy.shape
    table = np.asarray(momentum_table)
    if table.shape != (nk, nk) or not isinstance(q, Integral) or not 0 <= q < nk:
        raise ValueError("Transition momentum table/q does not match the spectrum")
    pairs = np.array([(i, a) for i in range(nocc) for a in range(nocc, nmo)], dtype=int)
    columns = [build(q, pairs=pairs, k_indices=[ki])[0] for ki in range(nk)]
    factors = jnp.concatenate(columns, axis=1)
    differences = jnp.concatenate([(energy[ki, :nocc, None]
        -energy[int(table[ki, q]), None, nocc:]).reshape(-1) for ki in range(nk)])
    return dict(factors=factors, gram=factors.conj().T@factors, differences=differences)


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
    if getattr(inputs, "boundary", None) is not None:
        raise NotImplementedError("Periodic GW product factors require three-dimensional inputs")
    nk, _, ngrid, _ = jnp.asarray(inputs.ao).shape
    coefficients = jnp.asarray(mo_coeff_k)
    nmo = coefficients.shape[-1]
    build = kpoint_product_factor_builder(inputs, coefficients, mesh=mesh,
                                          momentum_table=momentum_table)
    pairs = np.array([(m, n) for m in range(nmo) for n in range(nmo)], dtype=int)
    return [build(q, pairs=pairs).reshape(nk, ngrid, nmo, nmo) for q in range(nk)]
