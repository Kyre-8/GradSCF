"""Static singlet Delta+2J-W on a full, equally weighted periodic k mesh.

The bounded transition matrix is dense; auxiliary dielectric matrices are not.
Each q reuses the GW product builder and transition-space screened solve, then
releases its G-space factors. Reduced-D bare finite heads are retained;
q-cell/head-wing integration is absent in every dimension.
"""
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from ...gw.pbc.product_basis import (kpoint_product_factor_builder,
                                    kpoint_transition_block, lehmann_weights)
from ...gw.screened import solve_transition_screening
from ...gw.pbc.interaction import finite_head, _check_periodic_directions
from ...pbc.momentum import momentum_transfer_table
from ...pbc.kpoints import weights_for
from ...solvers import LinearSolverConfig


def _eager(*values):
    if any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(values)):
        raise NotImplementedError('Periodic BSE supports eager forward evaluation only; AD/JIT is not implemented.')


def _bands(selection, first, stop):
    values = np.arange(first, stop) if selection is None else np.asarray(selection)
    if (values.ndim != 1 or not len(values) or not np.issubdtype(values.dtype, np.integer)
            or len(np.unique(values)) != len(values) or np.any(values < first) or np.any(values >= stop)):
        raise ValueError('Band windows must contain unique in-range occupied or virtual indices.')
    return values


def build_tda_matrix(inputs, mo_coeff, screening_energies, qp_energies, nocc, *, mesh,
                     valence=None, conduction=None, screened=True, linear_config=None,
                     max_transitions=256, max_memory_bytes=512*1024**2):
    """Return the complex body-only singlet TDA matrix, ordered (k,v,c).

    Orbitals are cell-normalized and supplied from a converged canonical restricted
    reference. Screening uses every supplied occupied/virtual band; optical
    windows do not shrink it. QP energies are required for selected bands at
    every k; unselected QP entries may be NaN. No SCF or GW is performed here.

    For uniform w=1/Nk, Euclidean amplitudes x=sqrt(w)*A have x^dagger x=1.
    H=diag(Delta_QP)+(2J-W)/Nk. Screening's separate response 1/Nk is already
    included in D0=2*(f_m-f_n)/(Nk*(e_m-e_n)), summing both OV and VO
    transitions. No time-reversal reduction is assumed. All energies are Hartree.
    screened=False sets W=v for the HF-TDA algebraic limit. Comparing that
    limit requires the same orbital gaps and Coulomb/exxdiv convention.

    In 3D, both interactions use the G=0-excluded body. Slab/wire inputs
    use their matching finite-part Coulomb gauge and require exxdiv=None
    reference data; a removed negative bare head is restored once as
    -v0/(Nk*V)*I. No q-cell average or 3D finite-size correction is implied.
    The missing direct q->0 integral precludes production binding energies
    or material spectra. max_memory_bytes bounds estimated primary
    arrays, not process RSS; max_transitions bounds the dense exciton matrix.
    """
    _eager(inputs, mo_coeff, screening_energies, qp_energies)
    _check_periodic_directions(inputs, inputs.kpoints)
    c = jnp.asarray(mo_coeff, dtype=jnp.complex128)
    e, qp = jnp.asarray(screening_energies), jnp.asarray(qp_energies)
    if c.ndim != 3 or e.ndim != 2 or qp.shape != e.shape or (c.shape[0], c.shape[2]) != e.shape:
        raise ValueError('Use coefficients (nk,nao,nmo) and energy arrays (nk,nmo).')
    nk, nmo = e.shape
    if not nk or isinstance(nocc, bool) or not isinstance(nocc, Integral) or not 0 < nocc < nmo:
        raise ValueError('A nonempty insulating occupied/virtual space is required.')
    if jnp.iscomplexobj(e) or jnp.iscomplexobj(qp) or not np.isfinite(e).all():
        raise ValueError('Screening energies must be finite and real; QP energies must be real.')
    if float(jnp.min(e[:, nocc:])-jnp.max(e[:, :nocc])) <= 0.:
        raise ValueError('Static screening requires a positive insulating gap.')
    if not np.allclose(weights_for(inputs), 1/nk, atol=1e-12, rtol=0):
        raise NotImplementedError('Periodic BSE currently requires equal k weights.')
    metric = c.conj().swapaxes(-1, -2)@jnp.asarray(inputs.overlap)@c
    if not np.isfinite(c).all() or not np.allclose(metric, np.eye(nmo), atol=1e-7, rtol=0):
        raise ValueError('Reference orbitals must be finite and S-orthonormal.')
    occupied, virtual = _bands(valence, 0, nocc), _bands(conduction, nocc, nmo)
    nv, nc = len(occupied), len(virtual)
    dimension, rank = nk*nv*nc, 2*nk*nocc*(nmo-nocc)
    if not isinstance(max_transitions, Integral) or max_transitions < 1 or dimension > max_transitions:
        raise ValueError('The dense BSE matrix exceeds max_transitions.')
    gaps = qp[:, virtual][:, None, :]-qp[:, occupied][:, :, None]
    if not np.isfinite(gaps).all() or np.any(np.asarray(gaps) <= 0.):
        raise ValueError('All selected QP transitions need finite positive gaps.')
    ngrid = len(inputs.weights)
    screen_rank = rank if screened else 0
    estimate = (16*dimension**2 + 32*ngrid*(screen_rank+dimension+nk*nmo)
                + 128*screen_rank**2 + 32*screen_rank*nk*(nv**2+2*nc**2))
    if not isinstance(max_memory_bytes, Integral) or max_memory_bytes < 1 or estimate > max_memory_bytes:
        raise ValueError(f'Periodic BSE primary arrays exceed max_memory_bytes: {estimate} bytes.')
    fractional = np.asarray(inputs.kpoints)@np.asarray(inputs.lattice).T/(2*np.pi)
    table = momentum_transfer_table(fractional)
    zero = np.flatnonzero(np.all(table == np.arange(nk)[:, None], axis=0))
    if len(zero) != 1 or np.linalg.norm(np.asarray(inputs.kpoints)[zero[0]]) > 1e-10:
        raise NotImplementedError('Use a full Gamma-centered mesh with an explicit zero q point.')
    cfg = linear_config or LinearSolverConfig(method='direct', rtol=1e-11, atol=1e-13)
    build = kpoint_product_factor_builder(inputs, c, mesh=mesh, momentum_table=table,
                                         max_block_bytes=max_memory_bytes)
    ov = np.array([(v, a) for v in occupied for a in virtual], dtype=int)
    exchange = jnp.concatenate([build(int(zero[0]), pairs=ov, k_indices=[k])[0]
                                for k in range(nk)], axis=1)
    matrix = jnp.diag(gaps.reshape(-1)).astype(jnp.complex128)+2*(exchange.conj().T@exchange)/nk
    jax.block_until_ready(matrix)
    del exchange
    vv = np.array([(v, w) for v in occupied for w in occupied], dtype=int)
    cc = np.array([(a, b) for a in virtual for b in virtual], dtype=int)
    for q in range(nk):
        block = kpoint_transition_block(build, e, nocc, table, q, full=True) if screened else None
        bare, projected_left, projected_right = [], [], []
        for k in range(nk):
            left = build(q, pairs=vv, k_indices=[k])[0]
            right = build(q, pairs=cc, k_indices=[k])[0]
            bare.append(left.conj().T@right)
            if block is not None:
                projected_left.append(block['factors'].conj().T@left)
                projected_right.append(block['factors'].conj().T@right)
        if block is not None:
            weights = lehmann_weights(0., block['differences'], block['signs'], nk)
            solved = solve_transition_screening(block['gram'], weights,
                jnp.concatenate(projected_right, axis=1), config=cfg)
            if not bool(solved.converged):
                raise RuntimeError(f'Static BSE screening failed at q={q}: residual={float(solved.residual_norm)}')
        for k, partner in enumerate(table[:, q]):
            direct = bare[k]
            if block is not None:
                direct += projected_left[k].conj().T@solved.solution[:, k*nc**2:(k+1)*nc**2]
            direct = direct.reshape(nv, nv, nc, nc).transpose(0, 2, 1, 3).reshape(nv*nc, nv*nc)
            rows, cols = slice(k*nv*nc, (k+1)*nv*nc), slice(partner*nv*nc, (partner+1)*nv*nc)
            matrix = matrix.at[rows, cols].add(-direct/nk)
        jax.block_until_ready(matrix)
        del block, left, right, bare, projected_left, projected_right
    if getattr(inputs, 'boundary', None) is not None:
        head = float(finite_head(inputs))
        if head < 0.:
            # Positive heads already remain in the factors and are not added again.
            matrix = matrix-(head/nk)*jnp.eye(dimension, dtype=matrix.dtype)
    if not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.conj().T, atol=1e-9, rtol=1e-10):
        raise ValueError('The static periodic BSE body must be finite and Hermitian; no symmetrization is applied.')
    return matrix
