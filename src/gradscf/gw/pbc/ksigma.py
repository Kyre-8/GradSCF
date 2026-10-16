"""k-point resolved GW self-energy and G0W0 driver (Stage 2, k sampling).

Momentum-resolved G0W0 with contour deformation.  For orbital (kn, p),

    Sigma_p(kn, w) = -(1/pi)(1/nk) sum_q sum_m int dw' W_q[w', m, p]
                     g0[km, m](w, w')  +  (1/nk) sum_q residue(km poles)

with km = kn + q on the mesh (the inverse momentum_table), the pair factors
``B_q[ki]`` of :func:`gradscf.gw.pbc.product_basis.kpoint_product_factors`,
and the q = 0 head/wing correction (fc=True, PRB 83, 245122 (2011)).

The imaginary part is evaluated on the flattened (q, m) intermediate
space with :func:`gradscf.gw.self_energy.sigma_imag_part`; each q-block
of the residue reuses :func:`gradscf.gw.self_energy.sigma_residue_part`
with momentum-pair channels.

AD: eager path only; JAX transformations raise NotImplementedError explicitly.

References
----------
- L. Hedin, Phys. Rev. 139, A796 (1965). DOI:10.1103/PhysRev.139.A796
- J. Deslippe et al., Comput. Phys. Commun. 183, 1269 (2012).
  DOI:10.1016/j.cpc.2011.12.023
- q -> 0 head/wing: Phys. Rev. B 83, 245122 (2011).
"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np
from jax.lax import Precision
from jaxtyping import Array

from ...integrals.periodic.fft import get_kpoint_jk
from ..freq import scaled_legendre_grid
from ..polarizability import rho_response_iw
from ...solvers.nonlinear import ScalarRootConfig, solve_scalar_roots
from ...solvers import LinearSolverConfig
from ..self_energy import sigma_imag_part, sigma_residue_part
from ..screened import solve_transition_screening, transition_head_wing
from ..types import GWResult
from ...pbc.momentum import momentum_transfer_table
from .product_basis import (kpoint_product_factors, kpoint_product_factor_builder,
                            kpoint_transition_block)
from .q0 import transition_moments_kpoints

_MEM_LIMIT = 1 << 31


def _transition_weights(omega, differences, nk, eta=None):
    if eta is None:
        return (4.0/nk)*differences/(omega**2+differences**2)
    return (2.0/nk)*(1/(omega+differences+2j*eta)+1/(-omega+differences))


def _screening_memory_bytes(nk, ngrid, nmo, nocc, ntarget, norbs, nw, backend):
    """Additional primary-array estimate, excluding SCF inputs and runtime RSS.

    The transition estimate includes all retained reduced q blocks, orbital
    values, two OV factor copies during concatenation, a selected FFT block,
    and the shared solver's real representation. It is not a peak-RSS model.
    """
    if backend == "dense":
        frequency_batch = nw if nw*ngrid**2*32 <= _MEM_LIMIT else 1
        pole_batch = nmo if nmo*ngrid**2*32 <= _MEM_LIMIT else 1
        return 16*(nk**2*ngrid*nmo**2+nw*nk**2*nmo**2
                   +8*max(frequency_batch, pole_batch)*ngrid**2)
    nov, rank = nocc*(nmo-nocc), nk*nocc*(nmo-nocc)
    retained = nk*(16*(rank**2+rank*ntarget*nmo*norbs+nw*ntarget*nmo*norbs)+8*rank)
    orbitals = 16*nk*ngrid*nmo
    current_ov = 32*ngrid*rank
    fft_block = 16*ngrid*(8*max(nov, nmo*norbs)+2)
    real_solve = 48*rank**2
    return retained+orbitals+current_ov+fft_block+real_solve


def _screened_transition_kpoints(inputs, c_k, e_k, nocc, table, mesh, freqs,
                                 orbs, targets, config, max_memory_bytes, fc_q0):
    """Stream G-space factors by q/ki; retain only Gram/projection/frequency data."""
    nk, nmo = len(e_k), len(e_k[0])
    rank = nk*nocc*(nmo-nocc)
    output_pairs = np.array([(m, p) for m in range(nmo) for p in orbs], dtype=int)
    inverse = np.argsort(table, axis=0)
    zero_q = int(np.flatnonzero(np.all(table == np.arange(nk)[:, None], axis=0))[0])
    build = kpoint_product_factor_builder(inputs, c_k, mesh=mesh,
        momentum_table=table, max_block_bytes=max_memory_bytes)
    blocks, interactions = [], []
    head = wings = None
    for q in range(nk):
        transition = kpoint_transition_block(build, jnp.stack(e_k), nocc, table, q)
        factors, gram, differences = (transition[key] for key in ("factors", "gram", "differences"))
        projections = []
        for kn in targets:
            pair = build(q, pairs=output_pairs, k_indices=[int(inverse[kn, q])])[0]
            projections.append(factors.conj().T@pair)
        projected = jnp.stack(projections, axis=1).reshape(rank, len(targets), nmo, len(orbs))
        # No G-space arrays may remain alive through the QP root iteration.
        jax.block_until_ready((gram, projected))
        del factors, transition, pair, projections
        flat = projected.reshape(rank, -1)
        diagonal = jnp.stack([projected[:, kpos, p, ppos]
            for kpos in range(len(targets)) for ppos, p in enumerate(orbs)], axis=1)
        def at_frequency(omega):
            weights = _transition_weights(omega, differences, nk)
            solved = solve_transition_screening(gram, weights, flat, config=config)
            value = (jnp.sum(flat.conj()*solved.solution, axis=0)/nk).reshape(len(targets), nmo, len(orbs))
            if fc_q0 is None or q != zero_q:
                return value, None, None, solved.converged
            qij, q_abs, volume = fc_q0
            correction = transition_head_wing(gram, weights, jnp.asarray(qij).reshape(-1),
                diagonal, q_abs, volume, nkpts=nk, config=config)
            return value, correction.head, correction.wings.reshape(len(targets), len(orbs)), solved.converged & correction.converged
        values, d00, dp0, valid = jax.lax.map(at_frequency, freqs)
        if not bool(jnp.all(valid)):
            raise RuntimeError("A reduced imaginary-frequency screening solve failed its true residual check")
        interactions.append(values)
        blocks.append(dict(gram=gram, differences=differences, projected=projected))
        if fc_q0 is not None and q == zero_q:
            head, wings = d00, dp0
    return blocks, interactions, head, wings


def _sigma_residue_transition(omega, p, ppos, kn, kpos, e_k, blocks, table,
                              ef, eta, nk, config, q0_data=None):
    """Same contour pole selection and eta convention using reduced q blocks."""
    inverse = np.argsort(table, axis=0)
    above = omega > ef
    sign = jax.lax.stop_gradient(jnp.where(above, 1., -1.))
    total = jnp.zeros((), dtype=jnp.complex128)
    for q, block in enumerate(blocks):
        km = int(inverse[kn, q])
        energies = e_k[km]
        mask = jax.lax.stop_gradient(jnp.where(above,
            (energies < omega) & (energies > ef), (energies > omega) & (energies < ef)))
        frequencies = jnp.abs(energies-omega)
        pairs = block["projected"][:, kpos, :, ppos].T
        def at_pole(args):
            active, frequency, projected = args
            def evaluate(_):
                weights = _transition_weights(frequency, block["differences"], nk, eta)
                solved = solve_transition_screening(block["gram"], weights, projected, config=config)
                return jnp.vdot(projected, solved.solution)
            # The strict contour mask is unchanged. An excluded pole has an
            # exactly zero contribution and needs no dielectric solve.
            return jax.lax.cond(active, evaluate,
                lambda _: jnp.zeros((), dtype=jnp.complex128), operand=None)
        values = jax.lax.map(at_pole, (mask, frequencies, pairs))
        total += sign*jnp.sum(values)/nk
    if q0_data is not None:
        qij, q_abs, volume = q0_data
        zero_q = int(np.flatnonzero(np.all(table == np.arange(nk)[:, None], axis=0))[0])
        block = blocks[zero_q]
        km = int(inverse[kn, zero_q])
        energy = e_k[km][p]
        selected = jax.lax.stop_gradient(jnp.where(above,
            (energy < omega) & (energy > ef), (energy > omega) & (energy < ef)))
        def evaluate_head(_):
            weights = _transition_weights(jnp.abs(energy-omega), block["differences"], nk, eta)
            correction = transition_head_wing(block["gram"], weights, jnp.asarray(qij).reshape(-1),
                block["projected"][:, kpos, p, ppos], q_abs, volume, nkpts=nk, config=config)
            return correction.head+correction.wings
        total += sign*jax.lax.cond(selected, evaluate_head,
            lambda _: jnp.zeros((), dtype=jnp.complex128), operand=None)
    return total


def _response_iw_kpoint(omega, q, b_q, e_k, nocc, table, nk):
    """RPA response at momentum transfer q, imaginary axis (factor 4/pair)."""
    pi = None
    for ki in range(nk):
        ka = int(table[ki, q])
        b_ov = b_q[q][ki][:, :nocc, nocc:]
        contrib = rho_response_iw(
            omega, (e_k[ki][:nocc], e_k[ka][nocc:]), b_ov, spin_factor=4.0, conjugate=True
        )
        pi = contrib if pi is None else pi + contrib
    return pi / nk


def _screened_w_kpoint(b_q, e_k, nocc, table, freqs, nk, fc_q0=None):
    """Screened W per momentum transfer.

    Returns (w_q, del00, delP0): w_q[q] has shape (nw, nk, nmo, nmo) with
    W_q(kn)[w, m, p], including the 1/nk self-energy weight; del00 (nw,) /
    delP0 (nw, nk, nmo) are the separately scaled q=0 head/wing terms
    when fc_q0 is given, else None.
    """
    w_all = []
    del00 = delP0 = None
    inverse_table = np.argsort(table, axis=0)
    zero_q = int(np.flatnonzero(np.all(table == np.arange(nk)[:, None], axis=0))[0])
    for q in range(nk):
        def at_frequency(omega, q=q):
            pi = _response_iw_kpoint(omega, q, b_q, e_k, nocc, table, nk)
            naux = pi.shape[0]
            eye = jnp.eye(naux, dtype=pi.dtype)
            eps_inv = jnp.linalg.solve(eye - pi, eye)
            screened = eps_inv - eye
            w_kn = []
            for kn in range(nk):
                km = int(inverse_table[kn, q])
                b = b_q[q][km]
                w_kn.append(
                    jnp.einsum("Gmp,GH,Hmp->mp", b.conj(), screened, b, precision=Precision.HIGHEST) / nk
                )
            if fc_q0 is None or q != zero_q:
                return jnp.stack(w_kn), None, None
            # q -> 0 head/wing (PRB 83, 245122): k-summed responses
            qij, q_abs, volume = fc_q0
            q2 = jnp.sum(q_abs**2)
            q_norm = jnp.sqrt(q2)
            pi_00 = 0j
            pi_p0 = jnp.zeros(naux, dtype=jnp.complex128)
            for ki in range(nk):
                ka = int(table[ki, q])
                eia = e_k[ki][:nocc, None] - e_k[ka][None, nocc:]
                chi = eia / (omega**2 + eia * eia)
                pi_00 = pi_00 + 4.0 * jnp.sum(qij[ki].conj() * qij[ki] * chi)
                b_ov = b_q[q][ki][:, :nocc, nocc:]
                pi_p0 = pi_p0 + 4.0 * jnp.einsum(
                    "Pia,ia->P", b_ov, chi * qij[ki].conj(), precision=Precision.HIGHEST
                )
            pi_00 = pi_00 / nk
            pi_p0 = pi_p0 / nk
            eps_00 = 1.0 - 4.0 * jnp.pi / q2 * pi_00
            eps_p0 = -jnp.sqrt(4.0 * jnp.pi) / q_norm * pi_p0
            schur = eps_00 - eps_p0.conj() @ eps_inv @ eps_p0
            eps_inv_00 = 1.0 / schur
            eps_inv_p0 = -eps_inv_00 * (eps_inv @ eps_p0)
            head_scale = (2.0 / jnp.pi) * (6.0 * jnp.pi**2 / volume / nk) ** (1.0 / 3.0)
            wings_const = jnp.sqrt(volume / 4.0 / jnp.pi**3) * (6.0 * jnp.pi**2 / volume / nk) ** (
                2.0 / 3.0
            )
            d00 = head_scale * (eps_inv_00 - 1.0)
            dp0 = []
            for kn in range(nk):
                wn_p0 = jnp.einsum("Pnn,P->n", b_q[q][kn].conj(), eps_inv_p0, precision=Precision.HIGHEST)
                dp0.append(wings_const * 2.0 * wn_p0.real)
            return jnp.stack(w_kn), d00, jnp.stack(dp0)

        est = freqs.shape[0] * b_q[q].shape[1] ** 2 * 16 * 2
        if est > _MEM_LIMIT:
            w, d00, dp0 = jax.lax.map(at_frequency, freqs)
        else:
            w, d00, dp0 = jax.vmap(at_frequency)(freqs)
        w_all.append(w)
        if fc_q0 is not None and q == zero_q:
            del00, delP0 = d00, dp0
    return w_all, del00, delP0


def _sigma_residue_kpoint(
    omega, p, kn, e_k, b_q, channels_q, table, ef, eta, nk, q0_data=None
):
    """Residue part summed over momentum transfers (real frequency).

    The q = 0 head/wing correction (``q0_data``) is added undivided (its
    scaling constants already carry 1/nk), while each q block of the body
    carries the 1/nk momentum average.
    """
    total = jnp.zeros((), dtype=jnp.complex128)
    inverse_table = np.argsort(table, axis=0)
    for q in range(nk):
        km = int(inverse_table[kn, q])
        pair = b_q[q][km][:, :, p]
        total = total + sigma_residue_part(
            omega,
            e_k[km],
            pair,
            pair,
            channels_q[q],
            ef,
            eta,
            conjugate=True,
            response_scale=1.0 / nk,
        )
    total = total / nk
    if q0_data is not None:
        qij_t, q_abs, volume = q0_data
        zero_q = int(np.flatnonzero(np.all(table == np.arange(nk)[:, None], axis=0))[0])
        km0 = int(inverse_table[kn, zero_q])
        q0_dict = {
            "qij": qij_t,
            "q_abs": q_abs,
            "volume": volume,
            "p_index": p,
            "nkpts": nk,
        }
        with_q0 = sigma_residue_part(
            omega,
            e_k[km0],
            b_q[zero_q][km0][:, :, p],
            b_q[zero_q][km0][:, :, p],
            channels_q[zero_q],
            ef,
            eta,
            conjugate=True,
            q0=q0_dict,
            response_scale=1.0 / nk,
        )
        without_q0 = sigma_residue_part(
            omega,
            e_k[km0],
            b_q[zero_q][km0][:, :, p],
            b_q[zero_q][km0][:, :, p],
            channels_q[zero_q],
            ef,
            eta,
            conjugate=True,
            response_scale=1.0 / nk,
        )
        total = total + (with_q0 - without_q0)
    return total


def g0w0_cd_kpoints(
    *,
    inputs,
    kpts_frac: Array,
    mo_energy_k: Array,
    mo_coeff_k: Array,
    nocc: int,
    fock_k: Array,
    hcore_k: Array,
    density_spin: Array,
    mesh: tuple[int, int, int],
    nw: int = 100,
    eta: float = 1e-3,
    orbs: Sequence[int] | None = None,
    fc: bool = True,
    qp_kpoints: Sequence[int] | None = None,
    screening_backend: str = "transition",
    screening_config: LinearSolverConfig | None = None,
    max_memory_bytes: int = 1 << 31,
) -> GWResult:
    """k-point restricted G0W0 with contour deformation (eager path).

    Parameters
    ----------
    inputs:
        :class:`PeriodicInputs` for the full k mesh.
    kpts_frac:
        Fractional coordinates of the k mesh, shape ``(nk, 3)``.
    mo_energy_k, mo_coeff_k:
        Band energies ``(nk, nmo)`` and coefficients ``(nk, nao, nmo)``.
    nocc:
        Occupied bands per spin (restricted).
    fock_k, hcore_k:
        Converged Fock/core matrices per k point (AO), each ``(nk, nao, nao)``.
    density_spin:
        Converged spin densities ``(2, nk, nao, nao)``.
    mesh:
        FFT grid (``cell.mesh``).
    nw, eta, orbs:
        As in the Gamma driver.
    fc:
        Include the q = 0 head/wing correction (default True).
    qp_kpoints:
        Optional output k indices. Together with orbs this selects QP outputs
        only; all occupied/virtual screening transitions and internal m states
        are retained. Uncomputed levels retain their mean-field values and
        are explicitly excluded by qp_computed_mask.
    screening_backend:
        ``transition`` streams one q/ki factor block and uses the shared exact
        complex transition action. ``dense`` retains the bounded full-auxiliary
        reference. Both retain the existing finite-eta q0 convention.
    screening_config:
        Shared complex-solve configuration for the transition backend only.
    max_memory_bytes:
        Preflight bound for estimated additional primary tensors, including
        retained reduced q blocks. This is not a process-RSS guarantee.

    Returns
    -------
    :class:`GWResult` with ``mo_energy`` of shape ``(nk, nmo)``.
    """
    if getattr(inputs, "boundary", None) is not None:
        raise NotImplementedError("Periodic GW currently requires three-dimensional inputs")
    leaves = jax.tree_util.tree_leaves(
        (inputs, kpts_frac, mo_energy_k, mo_coeff_k, fock_k, hcore_k, density_spin)
    )
    if any(isinstance(leaf, jax.core.Tracer) for leaf in leaves):
        raise NotImplementedError("k-point GW currently supports eager evaluation only; AD/JIT is not implemented.")
    weights=getattr(inputs,'kweights',None)
    if weights is not None and not np.allclose(weights,1/len(weights),atol=1e-12,rtol=0):
        raise NotImplementedError('k-point GW currently requires equal k weights.')
    energy = jnp.asarray(mo_energy_k, dtype=jnp.float64)
    if energy.ndim != 2 or not energy.shape[0] or energy.shape[1] < 2:
        raise ValueError("Use a nonempty (nk,nmo) spectrum with occupied and virtual states")
    e_k = [e for e in energy]
    c_k = jnp.asarray(mo_coeff_k, dtype=jnp.complex128)
    nk = len(e_k)
    nmo = e_k[0].shape[0]
    if not isinstance(nocc, Integral) or not 0 < nocc < nmo:
        raise ValueError("nocc must leave nonempty occupied and virtual spaces")
    nocc = int(nocc)
    def selected(values, size, name):
        values = tuple(range(size)) if values is None else tuple(values)
        if (not values or any(not isinstance(value, Integral) or not 0 <= value < size for value in values)
                or len(set(values)) != len(values)):
            raise ValueError(f"{name} must contain distinct in-range integer indices")
        return tuple(map(int, values))
    orbs, targets = selected(orbs, nmo, "orbs"), selected(qp_kpoints, nk, "qp_kpoints")
    if screening_backend not in {"transition", "dense"}:
        raise ValueError("screening_backend must be 'transition' or 'dense'")
    if not isinstance(max_memory_bytes, Integral) or max_memory_bytes < 1:
        raise ValueError("max_memory_bytes must be a positive integer")
    if screening_backend == "dense" and screening_config is not None:
        raise ValueError("screening_config applies only to transition screening")
    config = screening_config or LinearSolverConfig(method="direct", rtol=1e-11, atol=1e-13)
    if not isinstance(config, LinearSolverConfig):
        raise TypeError("screening_config must be a LinearSolverConfig")
    ngrid = jnp.asarray(inputs.ao).shape[2]
    if len(mesh) != 3 or int(np.prod(mesh)) != ngrid:
        raise ValueError("The FFT mesh must match the AO grid")
    estimate = _screening_memory_bytes(nk, ngrid, nmo, nocc, len(targets), len(orbs), nw, screening_backend)
    if estimate > max_memory_bytes:
        raise ValueError(f"Periodic GW primary-array memory estimate {estimate} bytes exceeds max_memory_bytes")
    if screening_backend == "transition" and config.method == "direct" and 2*nk*nocc*(nmo-nocc) > config.max_dense:
        raise ValueError("The real representation of transition screening exceeds max_dense")

    table = momentum_transfer_table(np.asarray(kpts_frac))
    inverse_table = np.argsort(table, axis=0)
    b_q = (kpoint_product_factors(inputs, c_k, mesh=mesh, momentum_table=table)
           if screening_backend == "dense" else None)

    j_k, k_ewald = get_kpoint_jk(
        inputs, jnp.asarray(density_spin), mesh=mesh, exxdiv="ewald", with_k=True
    )
    k_ewald = k_ewald[0]  # restricted: K_alpha == K_beta
    delta_v_k = []
    for kn in range(nk):
        c = c_k[kn]
        vmf = c.conj().T @ (jnp.asarray(fock_k[kn]) - jnp.asarray(hcore_k[kn]) - j_k[kn]) @ c
        vk = c.conj().T @ k_ewald[kn] @ c
        delta_v_k.append(jnp.diag(-(vk + vmf)).real)

    ef = jnp.asarray(
        0.5 * (max(float(e[nocc - 1]) for e in e_k) + min(float(e[nocc]) for e in e_k))
    )
    freqs, wts = scaled_legendre_grid(nw)

    fc_q0 = None
    qij = q_abs = volume = None
    if fc:
        volume = float(jnp.asarray(inputs.weights[0])) * int(np.prod(mesh))
        qij, q_abs = transition_moments_kpoints(inputs, jnp.stack(e_k), c_k, nocc)
        fc_q0 = (qij, q_abs, volume)

    channels_q, blocks = [], None
    if screening_backend == "dense":
        w_q, del00, delP0 = _screened_w_kpoint(b_q, e_k, nocc, table, freqs, nk, fc_q0=fc_q0)
        for q in range(nk):
            chans = []
            for ki in range(nk):
                ka = int(table[ki, q])
                b_ov = b_q[q][ki][:, :nocc, nocc:]
                chans.append(((e_k[ki][:nocc], e_k[ka][nocc:]), b_ov, 2.0))
            channels_q.append(tuple(chans))
    else:
        blocks, w_q, del00, delP0 = _screened_transition_kpoints(inputs, c_k, e_k, nocc,
            table, mesh, freqs, orbs, targets, config, max_memory_bytes, fc_q0)

    def qp_residual(omega, kn, kpos, p, ppos, e_flat, wmn_p_flat):
        sigma_i = sigma_imag_part(omega, wmn_p_flat, e_flat, ef, freqs, wts, eta)
        q0_data = (tuple(qij[ki] for ki in range(nk)), q_abs, volume) if fc else None
        if screening_backend == "dense":
            sigma_r = _sigma_residue_kpoint(omega, p, kn, e_k, b_q, channels_q,
                table, ef, eta, nk, q0_data=q0_data)
        else:
            sigma_r = _sigma_residue_transition(omega, p, ppos, kn, kpos, e_k, blocks,
                table, ef, eta, nk, config, q0_data=q0_data)
        if fc and del00 is not None:
            # q=0 head/wing imag-axis correction (m == p channel); the
            # scaling constants already carry 1/nk.
            del_w = del00 + (delP0[:, kn, p] if screening_backend == "dense" else delP0[:, kpos, ppos])
            sign = jnp.sign(ef - e_k[kn][p])
            emo = omega - 1j * jnp.asarray(eta) * sign - e_k[kn][p]
            g0 = wts * emo / (emo**2 + freqs**2)
            sigma_i = sigma_i - jnp.sum(del_w * g0) / jnp.pi
        return omega - e_k[kn][p] - (jnp.real(sigma_i + sigma_r) + delta_v_k[kn][p])

    qp_energy = jnp.stack(e_k).copy()
    converged_mask = jnp.ones_like(qp_energy, dtype=bool)
    computed_mask = jnp.zeros_like(qp_energy, dtype=bool)
    residuals = jnp.zeros_like(qp_energy)
    converged = True
    for kpos, kn in enumerate(targets):
        e_flat = jnp.concatenate([e_k[int(inverse_table[kn, q])] for q in range(nk)])
        for ppos, p in enumerate(orbs):
            wmn_p_flat = jnp.concatenate([w_q[q][:, kn, :, p] if screening_backend == "dense"
                else w_q[q][:, kpos, :, ppos] for q in range(nk)], axis=1)
            f = lambda w: jnp.asarray([qp_residual(w[0], kn, kpos, p, ppos, e_flat, wmn_p_flat)])
            x0 = e_k[kn][p] + (-1e-2 if p < nocc else 1e-2)
            solved = solve_scalar_roots(f, x0[None], config=ScalarRootConfig())
            root, conv = solved.roots, solved.converged
            converged = converged and bool(conv[0])
            converged_mask = converged_mask.at[kn, p].set(bool(conv[0]))
            qp_energy = qp_energy.at[kn, p].set(root[0])
            computed_mask = computed_mask.at[kn, p].set(True)
            residuals = residuals.at[kn, p].set(solved.residual[0])

    return GWResult(
        mo_energy=qp_energy,
        mo_coeff=c_k,
        converged=converged,
        sigma_qp=None,
        converged_mask=converged_mask,
        nw=int(nw),
        qp_residual=residuals,
        qp_computed_mask=computed_mask,
        screening_energy=jnp.stack(e_k),
        g_orbitals=tuple(range(nmo)),
        screening_occupied=tuple(range(nocc)),
        screening_virtual=tuple(range(nocc, nmo)),
        method="periodic_cd_"+screening_backend,
    )


__all__ = ["g0w0_cd_kpoints"]
