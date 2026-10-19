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
import jax
import jax.numpy as jnp
from jax.lax import Precision
from jaxtyping import Array

from ...integrals.periodic.coulomb import coulomb_kernel
from .interaction import correlation_kernel, finite_head, _eager, _check_periodic_directions


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
           "kpoint_product_factor_builder", "kpoint_transition_block",
           "mixed_product_factors", "lehmann_weights"]


def _require_paired_lehmann(factors, differences, signs, *, label):
    """Certify the covariance pairing required by a folded even-W contour.

    Match the OV and VO spectral covariances at each opposite transition
    gap, allowing phases and rotations within equal-gap spaces. Thin QR
    avoids forming a G-space covariance and avoids Gram-norm cancellation.
    Validation is nondifferentiable; AD is valid only along parameter paths
    that preserve this pairing, not arbitrary complex orbital perturbations.
    """
    def validate(values, gaps, channels):
        values, gaps, channels = map(np.asarray, (values, gaps, channels))
        def unpaired():
            raise NotImplementedError(
                f'{label} requires paired OV/VO Lehmann covariances; '
                'use the full-response query path for its supported references.')
        if not all(np.isfinite(x).all() for x in (values, gaps, channels)):
            unpaired()
        used = np.zeros(len(gaps), dtype=bool)
        for index in np.flatnonzero(channels > 0):
            if used[index]:
                continue
            tolerance = 1e-10*(1+abs(gaps[index]))
            forward = np.flatnonzero((channels > 0) & (abs(gaps-gaps[index]) <= tolerance))
            reverse = np.flatnonzero((channels < 0) & (abs(gaps+gaps[index]) <= tolerance))
            if not len(reverse):
                unpaired()
            joined = values[:, np.r_[forward, reverse]]
            triangular = np.linalg.qr(joined, mode='r')
            left, right = triangular[:, :len(forward)], triangular[:, len(forward):]
            first, second = left@left.conj().T, right@right.conj().T
            scale = np.linalg.norm(first)+np.linalg.norm(second)
            if np.linalg.norm(first-second) > 1e-10*scale+1e-24:
                unpaired()
            used[forward], used[reverse] = True, True
        if not used.all():
            unpaired()

    values = jax.tree.map(jax.lax.stop_gradient, (factors, differences, signs))
    if any(isinstance(x, jax.core.Tracer) for x in jax.tree.leaves(values)):
        jax.debug.callback(validate, *values)
    else:
        validate(*values)


def _fourier_pair_factors(left, right, coords, gvectors, q, wrap, volume, mesh, *, kernel=None):
    """Shared batched periodic-u product, reciprocal wrap and Coulomb scale."""
    count, ngrid, npair = left.shape
    phase = jnp.exp(1j*(wrap@coords.T))
    values = left.conj()*right*phase[:, :, None]
    transformed = jnp.fft.fftn(values.reshape(count, *mesh, npair),
                               axes=(1, 2, 3)).reshape(count, ngrid, npair)
    interaction = coulomb_kernel(gvectors-q) if kernel is None else kernel
    scale = jnp.sqrt(volume*interaction)/ngrid
    return (scale[None, :, None]*transformed).astype(jnp.complex128)


def _check_lowdim_orbitals(inputs, values, coefficients=None):
    """Check raw zero-q heads in the actual FFT metric before deleting them.

    Full sampled Gram=I verifies both OV/VO zeros and VV/CC normalization.
    Two separately checked frames need not have an identity cross overlap.
    """
    if getattr(inputs, 'boundary', None) is None:
        return
    _eager(values, coefficients)
    finite_head(inputs)  # validates the common FFT volume and finite-part gauge
    if coefficients is not None:
        c = np.asarray(coefficients)
        metric = c.conj().swapaxes(-1, -2)@np.asarray(inputs.overlap)@c
        if not np.isfinite(metric).all() or not np.allclose(metric, np.eye(c.shape[-1]), atol=1e-9, rtol=0):
            raise ValueError('Low-dimensional orbitals must be orthonormal in the supplied AO metric.')
    psi = np.asarray(values)
    gram = np.einsum('g,...gm,...gn->...mn', np.asarray(inputs.weights), psi.conj(), psi, optimize=True)
    if not np.isfinite(gram).all() or not np.allclose(gram, np.eye(psi.shape[-1]), atol=1e-9, rtol=0):
        raise ValueError('Sampled FFT orbital metric must be I; raw transition heads cannot be discarded.')


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
    _check_periodic_directions(inputs, inputs.kpoints)
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
    if getattr(inputs, 'boundary', None) is not None:
        zero = np.flatnonzero(np.all(table == np.arange(nk)[:, None], axis=0))
        if len(zero) != 1 or np.any(np.asarray(inputs.kpoints)[zero[0]] != 0.):
            raise ValueError('The reduced-dimensional finite-head split requires an exact-zero source q.')
    if max_block_bytes is not None and (not isinstance(max_block_bytes, Integral) or max_block_bytes < 1):
        raise ValueError("max_block_bytes must be a positive integer or None")
    psi = jnp.einsum("kgp,kpm->kgm", ao, coeff, precision=Precision.HIGHEST)
    _check_lowdim_orbitals(inputs, psi, coeff)
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
        kernel = (None if getattr(inputs, 'boundary', None) is None else
                  correlation_kernel(inputs, gvectors-q_cart))
        return _fourier_pair_factors(left, right, coords, gvectors, q_cart, wrap, volume, mesh, kernel=kernel)
    return build


def mixed_product_factors(inputs, left_values, right_values, *, q_cart,
                          left_k, right_k, pairs, mesh, max_block_bytes=None):
    """Selected pairs between arbitrary allowed Bloch k points, shape (ngrid,npair).

    Values are periodic u_k arrays (ngrid,nmo), without an extra Bloch phase.
    All k/q vectors are Cartesian inverse Bohr. right_k-left_k+q_cart must
    be reciprocal; the shared FFT kernel applies that wrap and sqrt(V*v(G-q))/N.
    This helper does not construct orbitals or a screening/SCF k mesh.
    """
    left, right = jnp.asarray(left_values), jnp.asarray(right_values)
    ngrid = len(inputs.weights)
    if left.ndim != 2 or right.ndim != 2 or left.shape[0] != ngrid or right.shape[0] != ngrid:
        raise ValueError("Use periodic orbital values (ngrid,nmo) on the reference grid")
    mesh, pairs = tuple(int(m) for m in mesh), np.asarray(pairs)
    if len(mesh) != 3 or min(mesh) < 1 or int(np.prod(mesh)) != ngrid:
        raise ValueError("The FFT mesh must match the orbital grid")
    if (pairs.ndim != 2 or pairs.shape[1] != 2 or not len(pairs)
            or not np.issubdtype(pairs.dtype, np.integer) or np.any(pairs < 0)
            or np.any(pairs[:, 0] >= left.shape[1]) or np.any(pairs[:, 1] >= right.shape[1])):
        raise ValueError("pairs must contain in-range left/right orbital indices")
    q, kl, kr = (np.asarray(value) for value in (q_cart, left_k, right_k))
    if any(value.shape != (3,) or not np.isfinite(value).all() for value in (q, kl, kr)):
        raise ValueError("Use finite Cartesian k/q three-vectors")
    _check_periodic_directions(inputs, np.stack([q, kl, kr]))
    _check_lowdim_orbitals(inputs, left)
    _check_lowdim_orbitals(inputs, right)
    wrap = kr-kl+q
    fractional = wrap@np.asarray(inputs.lattice).T/(2*np.pi)
    if not np.allclose(fractional, np.rint(fractional), atol=1e-8, rtol=0):
        raise ValueError("The mixed pair must conserve momentum modulo a reciprocal vector")
    estimate = 16*ngrid*(8*len(pairs)+2)
    if max_block_bytes is not None and (not isinstance(max_block_bytes, Integral)
            or max_block_bytes < 1 or estimate > max_block_bytes):
        raise ValueError(f"Requested product block exceeds max_block_bytes: {estimate} bytes")
    kernel = (None if getattr(inputs, 'boundary', None) is None else
              correlation_kernel(inputs, jnp.asarray(inputs.gvectors)-jnp.asarray(q)))
    return _fourier_pair_factors(left[:, pairs[:, 0]][None], right[:, pairs[:, 1]][None],
        jnp.asarray(inputs.coords), jnp.asarray(inputs.gvectors), jnp.asarray(q),
        jnp.asarray(wrap)[None], float(inputs.weights[0])*ngrid, mesh, kernel=kernel)[0]


def kpoint_transition_block(build, mo_energy_k, nocc, momentum_table, q, *, full=False,
                            partner_energy=None):
    """One restricted transition space for shared periodic GW/BSE screening.

    The default returns the unchanged OV factors B(ngrid,nk*nocc*nvir),
    gram=B^dagger B and e_occ(ki)-e_virt(ka), ka=table[ki,q]. full=True adds
    VO columns after OV within each ki block, using the same (i,a) order
    with reversed pairs, and returns occupation-difference signs (+1,-1).
    Its rank and factor storage double; Gram storage quadruples.
    Frequency, spin and 1/Nk weights belong to the caller. Use B to project selected output
    pairs, then discard it before advancing q; this helper does not create
    an all-q cache or a QP/optical-window restriction. Callers must budget
    the complete OV factor/Gram storage, in addition to the builder's block
    array bound.

    partner_energy supplies the spectrum of a separately evaluated shifted
    frame with the same k-index labels. The builder must use that frame for
    its right orbital, and momentum_table identifies its corresponding row.
    Omitting it retains the mesh-closed source/source construction.
    """
    energy = jnp.asarray(mo_energy_k)
    if energy.ndim != 2 or not isinstance(nocc, Integral) or not 0 < nocc < energy.shape[1]:
        raise ValueError("Use a k-resolved spectrum with nonempty occupied and virtual spaces")
    nk, nmo = energy.shape
    partner = energy if partner_energy is None else jnp.asarray(partner_energy)
    if partner.shape != energy.shape:
        raise ValueError('Shifted partner energies must have the same (nk,nmo) shape as the source')
    table = np.asarray(momentum_table)
    if table.shape != (nk, nk) or not isinstance(q, Integral) or not 0 <= q < nk:
        raise ValueError("Transition momentum table/q does not match the spectrum")
    pairs = np.array([(i, a) for i in range(nocc) for a in range(nocc, nmo)], dtype=int)
    if full:
        pairs = np.concatenate((pairs, pairs[:, ::-1]))
    columns = [build(q, pairs=pairs, k_indices=[ki])[0] for ki in range(nk)]
    factors = jnp.concatenate(columns, axis=1)
    differences = jnp.concatenate([energy[ki, pairs[:, 0]]
        -partner[int(table[ki, q]), pairs[:, 1]] for ki in range(nk)])
    result = dict(factors=factors, gram=factors.conj().T@factors, differences=differences)
    if full:
        signs = jnp.ones(nocc*(nmo-nocc))
        result['signs'] = jnp.tile(jnp.concatenate((signs, -signs)), nk)
    return result


def lehmann_weights(frequency, differences, signs, nk, eta=None):
    """Full restricted Lehmann weights; frequency/differences/eta in Hartree.

    eta=None evaluates either imaginary branch: 2*s/[Nk*(i*frequency+delta)].
    At zero frequency this is the complete static BSE weight. The response
    at +/-i*frequency need not be even for general complex Bloch products.

    eta is only the G-pole residue convention, not an external real-frequency
    susceptibility: z=frequency+i*eta*s_m, with s_m=+1 at frequency>=0 and
    -1 otherwise. The time-ordered Lehmann denominator z+delta+i*eta*s is
    therefore frequency+delta+i*eta*(s_m+s). Positive-frequency OV/VO shifts
    are +2i*eta/0; negative-frequency shifts are 0/-2i*eta. When both channels
    have identical vertices and opposite differences, the positive branch
    recovers the legacy folded CD convention, including its finite eta.
    """
    frequency, differences, signs = map(jnp.asarray, (frequency, differences, signs))
    if (frequency.ndim or differences.ndim != 1 or signs.shape != differences.shape
            or any(jnp.iscomplexobj(value) for value in (frequency, differences, signs))):
        raise ValueError("Use a real scalar frequency and matching real difference/sign vectors")
    if isinstance(nk, bool) or not isinstance(nk, Integral) or nk < 1:
        raise ValueError("nk must be a positive integer")
    denominator = 1j*frequency+differences
    if eta is not None:
        eta = jnp.asarray(eta)
        if eta.ndim or jnp.iscomplexobj(eta):
            raise ValueError("eta must be a real scalar")
        pole_sign = jnp.where(frequency >= 0, 1., -1.)
        denominator = frequency+differences+1j*eta*(pole_sign+signs)
    return 2*signs/(nk*denominator)


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
