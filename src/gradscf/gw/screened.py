"""Screened Coulomb interaction W on the imaginary frequency axis.

Within the random phase approximation the screened Coulomb interaction in a
low-rank auxiliary representation is

    W_mn(iw) = sum_PQ B_P[m,n] [(1 - Pi(iw))^{-1} - 1]_PQ B_Q[m,n]

evaluated with the density response ``Pi`` from
:mod:`gradscf.gw.polarizability`.  The matrix inverse is written as the
linear solve ``solve(1 - Pi, Pi)`` (algebraically identical to
``inv(1 - Pi) - I``) which keeps the operation smooth for JAX autodiff.

References
----------
- L. Hedin, Phys. Rev. 139, A796 (1965). DOI:10.1103/PhysRev.139.A796
- X. Ren et al., New J. Phys. 14, 053020 (2012).
  DOI:10.1088/1367-2630/14/5/053020
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.lax import Precision
from jaxtyping import Array
from dataclasses import dataclass
from math import isfinite
from numbers import Integral
from typing import NamedTuple
from ..solvers import LinearOperator, LinearSolverConfig, solve_linear, solve_complex
from ..scf._pytree import pytree_dataclass
from ..solvers.diagnostics import require_converged_derivative


@pytree_dataclass(static_fields=("config",))
@dataclass(frozen=True)
class StaticScreening:
    """Full static W metric, stored densely or through transition factors."""
    dielectric: object
    valid: Array
    min_gap: Array
    transition_factors: object = None
    inverse_gaps: object = None
    config: object = None
    cholesky: object = None

    @property
    def naux(self):
        data = self.dielectric if self.dielectric is not None else self.transition_factors
        return data.shape[0]

    def operator(self):
        if self.dielectric is not None:
            from ..solvers.operators import as_operator
            return as_operator(self.dielectric)
        l, inverse_gaps = self.transition_factors, self.inverse_gaps

        def apply(v):
            return v + 4 * (l @ (inverse_gaps * (l.T @ v)))

        diagonal = 1 + 4 * jnp.sum(l**2 * inverse_gaps, axis=1)
        return LinearOperator((self.naux, self.naux), l.dtype, apply,
                              diagonal=diagonal, transpose_matvec=apply)


def build_static_screening(
    mo_energy, mo_factors, *, occupied, virtual, gap_tol=1e-10, max_aux=1024,
    config=None
):
    """Static RPA epsilon=I-Pi(0) for metric-whitened real factors.

    Restricted energies/factors have shape (nmo,)/(naux,nmo,nmo). Spin inputs
    have a leading axis of length two and paired occupied/virtual windows;
    both spin responses contribute to one charge dielectric.

    Screening indices are independent of the optical excitation window. Empty
    screening transitions give epsilon=I (bare interaction). Real symmetric
    MO factors and positive screening gaps are required. Invalid numerical
    inputs retain diagnostics with valid=False; there is no gap clipping model.
    A GMRES LinearSolverConfig stores only transition factors and gaps, never
    the dense dielectric. Direct screening is the default bounded reference.
    """
    e, l = jnp.asarray(mo_energy), jnp.asarray(mo_factors)
    unrestricted = e.ndim == 2
    if unrestricted:
        if e.shape[0] != 2 or l.ndim != 4 or l.shape[0] != 2:
            raise ValueError("Spin screening requires two energy/factor channels")
        if len(occupied) != 2 or len(virtual) != 2:
            raise ValueError("Spin screening requires alpha/beta windows")
        energies, factors = e, l
        occs, virs = occupied, virtual
    else:
        energies, factors = (e,), (l,)
        occs, virs = (occupied,), (virtual,)
    if jnp.iscomplexobj(e) or jnp.iscomplexobj(l):
        raise NotImplementedError("Static molecular screening requires real inputs")
    if (not isfinite(gap_tol) or gap_tol <= 0
            or not isinstance(max_aux, Integral) or max_aux < 1):
        raise ValueError("Invalid static screening tolerance or auxiliary limit")
    if config is not None and not isinstance(config, LinearSolverConfig):
        raise TypeError("screening config must be a LinearSolverConfig")
    cfg = config or LinearSolverConfig(method="direct", rtol=1e-11, atol=1e-13,
                                       max_dense=max_aux)
    dtype = jnp.result_type(e, l, 1.0)
    valid, minimum = jnp.asarray(True), jnp.asarray(jnp.inf, dtype)
    vertices, weights = [], []
    for energy, factor, occupied, virtual in zip(energies, factors, occs, virs):
        if energy.ndim != 1 or factor.ndim != 3 or factor.shape[1:] != (energy.size,) * 2:
            raise ValueError("Screening requires energy (nmo,) and factors (naux,nmo,nmo)")
        naux = factor.shape[0]
        if naux > max_aux:
            raise ValueError("Static screening exceeds max_aux")
        if cfg.method == "direct" and naux > cfg.max_dense:
            raise ValueError("Static screening direct solve exceeds max_dense")
        occ, vir = tuple(occupied), tuple(virtual)
        if any(not isinstance(p, Integral) or not 0 <= p < energy.size for p in occ + vir
               ) or len(set(occ + vir)) != len(occ) + len(vir):
            raise ValueError("Screening indices must be distinct, disjoint and in range")
        energy, factor = energy.astype(dtype), factor.astype(dtype)
        oi, va = jnp.asarray(occ, dtype=jnp.int32), jnp.asarray(vir, dtype=jnp.int32)
        gaps = energy[va][None, :] - energy[oi][:, None]
        minimum = jnp.minimum(minimum, jnp.min(gaps, initial=jnp.inf))
        scale = jnp.maximum(1.0, jnp.max(jnp.abs(factor), initial=0.0))
        valid &= (jnp.all(jnp.isfinite(factor)) & jnp.all(jnp.isfinite(gaps))
                  & jnp.all(gaps > gap_tol)
                  & (jnp.max(jnp.abs(factor - factor.swapaxes(1, 2)), initial=0.)
                     <= 64 * jnp.finfo(dtype).eps * scale))
        lov = factor[:, oi[:, None], va[None, :]].reshape(naux, len(occ) * len(vir))
        vertices.append(lov)
        # StaticScreening uses coefficient 4: each unrestricted spin contributes 2.
        weights.append(((0.5 if unrestricted else 1.) /
                        jnp.where(gaps > gap_tol, gaps, 1.)).reshape(-1))
    lov = jnp.concatenate(vertices, axis=1)
    inverse_gaps = jnp.concatenate(weights)
    if cfg.method == "gmres":
        return StaticScreening(None, valid, minimum, lov, inverse_gaps, cfg)
    dielectric = jnp.eye(lov.shape[0], dtype=dtype) + 4 * (lov * inverse_gaps) @ lov.T
    return StaticScreening(dielectric, valid, minimum, config=cfg,
                           cholesky=jnp.linalg.cholesky(dielectric))


def solve_static_screening(state, values):
    """Return the checked full-W solve and residual diagnostics.

    Direct mode reuses the state's dense Cholesky factor; GMRES processes
    columns sequentially with the shared implicit primal/transpose rules.
    The solution alone remains a linear action in RHS for kernel transposes.
    Input validity and true solve residuals are included in converged/status.
    """
    values = jnp.asarray(values)
    naux = state.naux
    if values.ndim < 1 or values.shape[0] != naux:
        raise ValueError("Screening RHS must have leading dimension naux")
    columns = 1
    for n in values.shape[1:]:
        columns *= n
    rhs = values.reshape(naux, columns)
    cfg = state.config or LinearSolverConfig(
        method="direct", rtol=1e-11, atol=1e-13, max_dense=max(1, naux)
    )
    op = state.operator()
    # Use the physical residual metric. Left diagonal preconditioning can
    # satisfy JAX's stopping test before the unpreconditioned residual passes.
    out = solve_linear(op, rhs, config=cfg, cholesky=state.cholesky)
    valid = state.valid & out.converged
    solution = require_converged_derivative(out.solution, state.valid)
    solution = jnp.where(state.valid, solution, jnp.nan).reshape(values.shape)
    return out._replace(solution=solution, converged=valid, status=jnp.where(valid, 0, 1))


def apply_static_screening(state, values):
    """Apply epsilon^-1 with physical input and primal/AD validity guards."""
    out = solve_static_screening(state, values)
    valid = state.valid & out.converged
    result = require_converged_derivative(out.solution, valid)
    return jnp.where(valid, result, jnp.nan)


def screened_w_imag_axis(
    b_mn: Array,
    response_fn: Callable[[Array], Array],
    freqs: Array,
    *,
    conjugate: bool = False,
) -> Array:
    """Compute ``W_mn(iw)`` for every imaginary grid frequency.

    Parameters
    ----------
    b_mn:
        Low-rank factors in the full MO basis, shape ``(naux, nmo, nmo)``.
    response_fn:
        Callable mapping a real scalar frequency to the density response
        matrix ``Pi`` of shape ``(naux, naux)`` (real or complex dtype; the
        dtype propagates to the output).
    freqs:
        1D array of imaginary-axis quadrature nodes.

    Returns
    -------
    Array of shape ``(nw, nmo, nmo)`` with ``W_mn`` at each frequency.
    """
    b_mn = jnp.asarray(b_mn)
    freqs = jnp.asarray(freqs)
    naux = b_mn.shape[0]

    def at_frequency(omega: Array) -> Array:
        pi = response_fn(omega)
        eye = jnp.eye(naux, dtype=pi.dtype)
        # (1 - Pi)^{-1} - I == solve(1 - Pi, Pi); solve has a well-defined
        # adjoint and avoids explicitly forming the inverse.
        screened = jnp.linalg.solve(eye - pi, pi)
        first = b_mn.conj() if conjugate else b_mn
        return jnp.einsum("Pmn,PQ,Qmn->mn", first, screened, b_mn, precision=Precision.HIGHEST)

    # Batch over frequencies sequentially when the batched (naux, naux)
    # intermediates would exceed the memory budget (plane-wave bases).
    est_bytes = freqs.shape[0] * naux * naux * 16 * 2
    if est_bytes > (1 << 31):
        return jax.lax.map(at_frequency, freqs)
    return jax.vmap(at_frequency)(freqs)


__all__ = ["screened_w_imag_axis", "screened_w_imag_axis_matrix",
           "StaticScreening", "build_static_screening", "apply_static_screening",
           "solve_static_screening", "solve_transition_screening",
           "TransitionHeadWingResult", "transition_head_wing"]


def screened_w_imag_axis_matrix(
    b_mn: Array,
    response_fn: Callable[[Array], Array],
    freqs: Array,
    *,
    conjugate: bool = False,
) -> Array:
    """Full screened tensor ``W[q; m, n](iw)`` for off-diagonal self-energies.

    W[q; m, n] = sum_PQ conj(B_P[m,q]) [(1-Pi)^{-1} - 1]_PQ B_Q[q,n]

    Returns shape ``(nw, nq, nmo, nmo)`` (q = intermediate state index).
    """
    b_mn = jnp.asarray(b_mn)
    freqs = jnp.asarray(freqs)
    naux = b_mn.shape[0]

    def at_frequency(omega: Array) -> Array:
        pi = response_fn(omega)
        eye = jnp.eye(naux, dtype=pi.dtype)
        screened = jnp.linalg.solve(eye - pi, pi)
        first = b_mn.conj() if conjugate else b_mn
        return jnp.einsum("Pmq,PQ,Qqn->qmn", first, screened, b_mn, precision=Precision.HIGHEST)

    est_bytes = freqs.shape[0] * naux * naux * 16 * 2
    if est_bytes > (1 << 31):
        return jax.lax.map(at_frequency, freqs)
    return jax.vmap(at_frequency)(freqs)


def solve_transition_screening(gram, weights, projected_rhs, *, config=None):
    """Solve the exact transition-space screened action with complex weights.

    For Pi=B diag(d) B^dagger, M=B^dagger B and C=B^dagger X, return the
    checked shared LinearResult for Z=(I-diag(d) M)^-1 diag(d) C. Then
    (epsilon^-1-I) X=B Z and paired matrix elements are sum(conj(C)*Z,0).
    Only M, d and C are needed; no auxiliary-space square matrix is formed.

    Weights include the chosen frequency, spin and response k average.
    Complex retarded weights are not conjugated or made Hermitian. No inverse
    weight, inverse Gram, rank truncation or new linear solver is used, so
    zero weights and dependent transition columns remain valid. The reported
    residual is that of the reduced equation, not an auxiliary-space norm.
    Direct solves have the shared solve_complex bound on 2*ntransition.
    This primitive alone does not enable periodic-GW streaming or its AD.
    """
    gram, weights, rhs = map(jnp.asarray, (gram, weights, projected_rhs))
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1] or not gram.shape[0]:
        raise ValueError("Transition Gram matrix must be nonempty and square")
    size = gram.shape[0]
    if weights.shape != (size,) or rhs.ndim not in (1, 2) or rhs.shape[0] != size:
        raise ValueError("Use weights (ntransition,) and projected RHS (ntransition[,nrhs])")
    cfg = config or LinearSolverConfig(method="direct", rtol=1e-11, atol=1e-13)
    if not isinstance(cfg, LinearSolverConfig):
        raise TypeError("config must be a LinearSolverConfig")
    dtype = jnp.result_type(gram, weights, rhs, 1j)
    gram, weights, rhs = (value.astype(dtype) for value in (gram, weights, rhs))
    def multiply_weights(value):
        return weights*value if value.ndim == 1 else weights[:, None]*value
    def apply(value):
        return value-multiply_weights(gram@value)
    operator = LinearOperator((size, size), dtype, apply, matmat=apply)
    solved = solve_complex(operator, multiply_weights(rhs), config=cfg)
    tolerance = 64*jnp.finfo(gram.real.dtype).eps*jnp.maximum(1., jnp.max(jnp.abs(gram)))
    valid = (solved.converged & jnp.all(jnp.isfinite(gram))
             & jnp.all(jnp.isfinite(weights)) & jnp.all(jnp.isfinite(rhs))
             & (jnp.max(jnp.abs(gram-gram.conj().T)) <= tolerance))
    solution = require_converged_derivative(jnp.where(valid, solved.solution, jnp.nan), valid)
    return solved._replace(solution=solution, converged=valid, status=jnp.where(valid, 0, 1))


class TransitionHeadWingResult(NamedTuple):
    head: Array
    wings: Array
    schur: Array
    residual_norm: Array
    converged: Array
    status: Array


def transition_head_wing(gram, weights, qij, projected_diagonal, q_abs, volume,
                         *, nkpts=1, config=None):
    """Compressed q=0 head/wing contraction preserving the existing convention.

    M=B^dagger B, d already includes the response 1/Nk, qij is ordered with
    those same transition columns, and projected_diagonal=B^dagger b_pp for
    each selected external orbital. nkpts enters only the finite-size head/
    wing prefactors here; do not divide these outputs by an outer q average.

    This exactly retains q0_residue_correction's finite-eta convention:
    schur=eps00-conj(epsP0)@eps_body^-1@epsP0 and twice the real wing.
    For complex retarded d this is not a general non-Hermitian dielectric
    block inverse. Agreement with dense code validates the compression, not
    the physical correctness of that opposite-wing convention. No change to
    the existing q0 prescription or residue pole selection is made here.
    """
    gram, weights, qij, diagonal, q_abs, volume = map(jnp.asarray,
        (gram, weights, qij, projected_diagonal, q_abs, volume))
    if gram.ndim != 2 or qij.shape != (gram.shape[0],):
        raise ValueError("qij must match the transition Gram dimension")
    if diagonal.ndim not in (1, 2) or diagonal.shape[0] != gram.shape[0]:
        raise ValueError("Projected diagonal pairs must have a transition leading dimension")
    if q_abs.shape != (3,) or volume.ndim or jnp.iscomplexobj(q_abs) or jnp.iscomplexobj(volume):
        raise ValueError("Use real q_abs (3,) and scalar volume")
    if not isinstance(nkpts, Integral) or nkpts < 1:
        raise ValueError("nkpts must be a positive integer")
    solved = solve_transition_screening(gram, weights, qij.conj(), config=config)
    q2 = jnp.sum(q_abs**2)
    parameters_valid = (jnp.isfinite(q2) & (q2 > 0) & jnp.isfinite(volume)
                        & (volume > 0) & jnp.all(jnp.isfinite(qij))
                        & jnp.all(jnp.isfinite(diagonal)))
    safe_q2 = jnp.where(parameters_valid, q2, 1.)
    safe_volume = jnp.where(parameters_valid, volume, 1.)
    factor = jnp.sqrt(4*jnp.pi/safe_q2)
    # epsP0=B h and eps_body^-1 epsP0=B u; retain all transition head terms
    # even if some body factors are zero or linearly dependent.
    h = -factor*weights*qij.conj()
    u = -factor*solved.solution
    epsilon00 = 1-factor**2*jnp.sum(weights*qij.conj()*qij)
    schur = epsilon00-jnp.vdot(h, gram@u)
    schur_valid = jnp.isfinite(schur) & (jnp.abs(schur) > 0)
    inverse00 = 1/jnp.where(schur_valid, schur, 1.)
    inverse_wing_coefficients = -inverse00*u
    head_scale = (2/jnp.pi)*(6*jnp.pi**2/safe_volume/nkpts)**(1/3)
    wing_scale = jnp.sqrt(safe_volume/(4*jnp.pi**3))*(6*jnp.pi**2/safe_volume/nkpts)**(2/3)
    head = head_scale*(inverse00-1)
    wings = 2*wing_scale*jnp.real(inverse_wing_coefficients@diagonal.conj())
    valid = (solved.converged & parameters_valid & schur_valid
             & jnp.isfinite(head) & jnp.all(jnp.isfinite(wings)))
    head = require_converged_derivative(jnp.where(valid, head, jnp.nan), valid)
    wings = require_converged_derivative(jnp.where(valid, wings, jnp.nan), valid)
    return TransitionHeadWingResult(head, wings, schur, solved.residual_norm,
                                    valid, jnp.where(valid, 0, 1))
