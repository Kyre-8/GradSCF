"""Structured real RPA Davidson and metric eigenvalue response."""
from __future__ import annotations
from collections.abc import Callable
from numbers import Integral
import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
from jaxtyping import Array
from ..diagnostics import require_converged_derivative
from .davidson import (
    _symmetrize,
    _MATMUL_PRECISION, _DEFAULT_MATMUL_PRECISION, _solver_dtype,
    _davidson_search_nroots, _davidson_max_subspace,
    _residual_tol_with_dtype_slack, _all_roots_converged, _matmul,
    _safe_preconditioner_denominator, _resolve_symmetric_linear_operator,
    _davidson_lowest_symmetric,
)

DEFAULT_RPA_TOL = 1e-5
DEFAULT_RPA_MAXITER = 512
__all__ = ["implicit_differential_davidson_lowest_tdhf"]


def _davidson_lowest_tdhf(
    vind: Callable[[Array], Array],
    *,
    nroots: int,
    size: int,
    diag: Array,
    tol: float = DEFAULT_RPA_TOL,
    max_iter: int = DEFAULT_RPA_MAXITER,
    max_subspace: int | None = None,
    matrix_eps: float = 1e-10,
    preconditioner_floor: float = 1e-8,
    preconditioner_level_shift: float = 0.0,
    orth_eps: float = 1e-10,
    initial_vectors: Array | None = None,
    initial_guess_count: int | None = None,
) -> tuple[Array, Array, Array, Array]:
    """Real structured Davidson for stable TDHF/TDDFT problems in JAX.

    ``vind`` follows PySCF ``gen_tdhf_operation``: input rows are ``[X, Y]`` and
    output rows are ``[AX + BY, -(BX + AY)]``.
    """

    if initial_guess_count is not None and (
        isinstance(initial_guess_count, bool)
        or not isinstance(initial_guess_count, Integral)
        or initial_guess_count < 1
    ):
        raise ValueError("initial_guess_count must be a positive integer")

    diag = jnp.asarray(diag)
    dim = int(size)
    dtype = _solver_dtype(diag.dtype)
    diag = diag.astype(dtype).reshape(dim)
    index_dtype = jnp.asarray(0).dtype

    if dim == 0:
        empty = jnp.zeros((0, 0), dtype=dtype)
        return jnp.zeros((0,), dtype=dtype), empty, empty, jnp.asarray(True)

    nroots = max(1, min(int(nroots), dim))
    if max_subspace is None:
        max_subspace = min(dim, max(8 * nroots, 128))
    else:
        max_subspace = _davidson_max_subspace(nroots, dim, max_subspace)

    def apply_pair(v_cols: Array, w_cols: Array) -> tuple[Array, Array]:
        rows = jnp.concatenate([v_cols.T, w_cols.T], axis=-1)
        with jax.default_matmul_precision(_DEFAULT_MATMUL_PRECISION):
            applied = jnp.asarray(vind(rows), dtype=dtype).reshape(-1, 2 * dim)
        return applied[:, :dim].T, -applied[:, dim:].T

    if initial_vectors is None:
        guess_dim = nroots if initial_guess_count is None else max(nroots, int(initial_guess_count))
        guess_dim = min(dim, max_subspace, guess_dim)
        guess_idx = jnp.argsort(diag)[:guess_dim]
        guess_v = jax.nn.one_hot(guess_idx, dim, dtype=dtype).T
    else:
        guess_v = jax.lax.stop_gradient(jnp.asarray(initial_vectors, dtype=dtype))
        if (guess_v.ndim != 2 or guess_v.shape[0] != dim
                or not nroots <= guess_v.shape[1] <= max_subspace):
            raise ValueError("initial_vectors must fit the requested RPA subspace")
        guess_dim = guess_v.shape[1]
        guess_v, _ = jnp.linalg.qr(guess_v, mode="reduced")
    guess_w = jnp.zeros_like(guess_v)
    guess_u1, guess_u2 = apply_pair(guess_v, guess_w)

    v_basis = jnp.zeros((dim, max_subspace), dtype=dtype)
    w_basis = jnp.zeros((dim, max_subspace), dtype=dtype)
    u1_basis = jnp.zeros((dim, max_subspace), dtype=dtype)
    u2_basis = jnp.zeros((dim, max_subspace), dtype=dtype)
    v_basis = jax.lax.dynamic_update_slice(v_basis, guess_v, (0, 0))
    w_basis = jax.lax.dynamic_update_slice(w_basis, guess_w, (0, 0))
    u1_basis = jax.lax.dynamic_update_slice(u1_basis, guess_u1, (0, 0))
    u2_basis = jax.lax.dynamic_update_slice(u2_basis, guess_u2, (0, 0))
    active_mask = jnp.zeros((max_subspace,), dtype=bool)
    active_mask = jax.lax.dynamic_update_slice(
        active_mask,
        jnp.ones((guess_dim,), dtype=bool),
        (0,),
    )

    inactive_shift = jnp.asarray(
        (jnp.maximum(jnp.max(jnp.abs(diag)), 1.0) + 1.0) * 1.0e6,
        dtype=dtype,
    )
    tol_arr = _residual_tol_with_dtype_slack(tol, dtype)
    eps_arr = jnp.asarray(matrix_eps, dtype=dtype)
    orth_eps_arr = jnp.asarray(orth_eps, dtype=dtype)
    full_diag = jnp.concatenate([diag, -diag])

    basis_dim0 = jnp.asarray(guess_dim, dtype=index_dtype)
    best_w0 = jnp.zeros((nroots,), dtype=dtype)
    best_x0 = jnp.zeros((dim, nroots), dtype=dtype)
    best_y0 = jnp.zeros((dim, nroots), dtype=dtype)
    best_residual0 = jnp.asarray(jnp.inf, dtype=dtype)
    converged0 = jnp.asarray(False)
    done0 = jnp.asarray(False)

    def _solve_subspace(
        v_in: Array,
        w_in: Array,
        u1_in: Array,
        u2_in: Array,
        active_in: Array,
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        primary_basis = jnp.concatenate([v_in.T.conj(), w_in.T.conj()], axis=1)
        dual_basis = jnp.concatenate([w_in.T.conj(), v_in.T.conj()], axis=1)
        primary_action = jnp.concatenate([u1_in.T, -u2_in.T], axis=1)
        dual_action = jnp.concatenate([u2_in.T, -u1_in.T], axis=1)
        projected_basis = jnp.concatenate([primary_basis, dual_basis], axis=0)
        projected_action = jnp.concatenate([primary_action, dual_action], axis=0)
        projected_active = jnp.concatenate([active_in, active_in])
        active_outer = projected_active[:, None] & projected_active[None, :]
        # Preserve the RPA metric under projection. Q.T R Q is not a
        # structure-preserving reduction when range(Q) is not J-invariant.
        # Instead solve (Q.T H Q)c = omega (Q.T J Q)c. H is positive definite
        # for the stable problem; Cholesky turns the inverse-frequency pencil
        # into a small real symmetric eigenproblem, without forming physical H.
        physical_metric = jnp.concatenate(
            [jnp.ones(dim, dtype=dtype), -jnp.ones(dim, dtype=dtype)]
        )
        projected_h = _symmetrize(
            _matmul(projected_basis, (projected_action * physical_metric).T)
        )
        projected_j = _symmetrize(
            _matmul(projected_basis * physical_metric, projected_basis.T)
        )
        inactive_sign = jnp.concatenate(
            [
                jnp.ones((max_subspace,), dtype=dtype),
                -jnp.ones((max_subspace,), dtype=dtype),
            ]
        )
        projected_h = jnp.where(active_outer, projected_h, 0.0)
        projected_j = jnp.where(active_outer, projected_j, 0.0)
        projected_h += jnp.diag((~projected_active).astype(dtype) * inactive_shift)
        projected_j += jnp.diag((~projected_active).astype(dtype) * inactive_sign)
        chol = jnp.linalg.cholesky(jax.lax.stop_gradient(projected_h))
        reduced = solve_triangular(chol, jax.lax.stop_gradient(projected_j), lower=True)
        reduced = solve_triangular(chol, reduced.T, lower=True).T
        inverse_values, vectors = jnp.linalg.eigh(_symmetrize(reduced))
        root_valid = jnp.isfinite(inverse_values) & (inverse_values > 2.0 / inactive_shift)
        frequencies = 1.0 / jnp.where(root_valid, inverse_values, 1.0)
        order = jnp.argsort(jnp.where(root_valid, frequencies, jnp.inf))
        selected = order[:nroots]
        selected_valid = root_valid[selected]
        omega = jnp.where(selected_valid, frequencies[selected], eps_arr)
        coefficients = solve_triangular(chol.T, vectors[:, selected], lower=False)
        coefficients = jnp.where(
            selected_valid[None, :], coefficients * jnp.sqrt(omega)[None, :], 0.0
        )
        primary_coeff = coefficients[:max_subspace]
        dual_coeff = coefficients[max_subspace:]
        x_full = _matmul(v_in, primary_coeff) + _matmul(w_in, dual_coeff)
        y_full = _matmul(w_in, primary_coeff) + _matmul(v_in, dual_coeff)
        applied_x, applied_y = apply_pair(x_full, y_full)
        r_x = applied_x - x_full * omega[None, :]
        r_y = applied_y + y_full * omega[None, :]
        residual_norms = jnp.sqrt(
            jnp.sum(jnp.abs(r_x) ** 2, axis=0) + jnp.sum(jnp.abs(r_y) ** 2, axis=0)
        )
        residual_norms = jnp.where(selected_valid, residual_norms, jnp.inf)
        return omega, x_full, y_full, r_x, r_y, residual_norms

    def _append_new_pairs(
        v_in: Array,
        w_in: Array,
        u1_in: Array,
        u2_in: Array,
        active_in: Array,
        basis_dim_in: Array,
        new_x: Array,
        new_y: Array,
        new_mask: Array,
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        def orthogonalize_pair(
            x: Array,
            y: Array,
            v_cols: Array,
            w_cols: Array,
        ) -> tuple[Array, Array]:
            direct = _matmul(v_cols.T.conj(), x) + _matmul(w_cols.T.conj(), y)
            dual = _matmul(w_cols.T.conj(), x) + _matmul(v_cols.T.conj(), y)
            return (
                x - _matmul(v_cols, direct) - _matmul(w_cols, dual),
                y - _matmul(w_cols, direct) - _matmul(v_cols, dual),
            )

        def body_fun(col_idx: int, carry):
            x_cols, y_cols, mask_cols, count = carry
            accept_seed = jax.lax.dynamic_index_in_dim(
                new_mask,
                col_idx,
                axis=0,
                keepdims=False,
            )
            x = jax.lax.dynamic_slice(new_x, (0, col_idx), (dim, 1)).reshape(dim)
            y = jax.lax.dynamic_slice(new_y, (0, col_idx), (dim, 1)).reshape(dim)
            # Reorthogonalize small residual corrections before normalization;
            # one pass can amplify roundoff into false independent directions.
            for _ in range(2):
                x, y = orthogonalize_pair(x, y, v_in, w_in)
                x, y = orthogonalize_pair(x, y, x_cols, y_cols)
            plus_norm2 = jnp.sum(jnp.abs(x + y) ** 2)
            minus_norm2 = jnp.sum(jnp.abs(x - y) ** 2)
            independent = jnp.minimum(plus_norm2, minus_norm2) > orth_eps_arr**2
            accept = accept_seed & independent
            plus_scale = jax.lax.rsqrt(jnp.where(independent, plus_norm2, 1.0))
            minus_scale = jax.lax.rsqrt(jnp.where(independent, minus_norm2, 1.0))
            direct_scale = 0.5 * (plus_scale + minus_scale)
            dual_scale = 0.5 * (plus_scale - minus_scale)
            x, y = (
                direct_scale * x + dual_scale * y,
                direct_scale * y + dual_scale * x,
            )

            def do_update(update_carry):
                x_upd, y_upd, mask_upd, count_upd = update_carry
                x_upd = jax.lax.dynamic_update_slice(x_upd, x[:, None], (0, count_upd))
                y_upd = jax.lax.dynamic_update_slice(y_upd, y[:, None], (0, count_upd))
                mask_upd = jax.lax.dynamic_update_slice(
                    mask_upd,
                    jnp.asarray([True]),
                    (count_upd,),
                )
                return x_upd, y_upd, mask_upd, count_upd + jnp.asarray(1, dtype=index_dtype)

            return jax.lax.cond(accept, do_update, lambda z: z, carry)

        init_x = jnp.zeros_like(new_x)
        init_y = jnp.zeros_like(new_y)
        init_mask = jnp.zeros((new_x.shape[1],), dtype=bool)
        x_pairs, y_pairs, pair_mask, pair_count = jax.lax.fori_loop(
            0,
            new_x.shape[1],
            body_fun,
            (init_x, init_y, init_mask, jnp.asarray(0, dtype=index_dtype)),
        )
        pair_u1, pair_u2 = apply_pair(x_pairs, y_pairs)

        def scatter_body(col_idx: int, carry):
            v_cur, w_cur, u1_cur, u2_cur, active_cur, offset = carry
            accept = jax.lax.dynamic_index_in_dim(
                pair_mask,
                col_idx,
                axis=0,
                keepdims=False,
            )
            target = basis_dim_in + offset
            accept = accept & (target < max_subspace)
            x = jax.lax.dynamic_slice(x_pairs, (0, col_idx), (dim, 1))
            y = jax.lax.dynamic_slice(y_pairs, (0, col_idx), (dim, 1))
            u1 = jax.lax.dynamic_slice(pair_u1, (0, col_idx), (dim, 1))
            u2 = jax.lax.dynamic_slice(pair_u2, (0, col_idx), (dim, 1))

            def do_update(update_carry):
                v_upd, w_upd, u1_upd, u2_upd, active_upd, offset_upd = update_carry
                v_upd = jax.lax.dynamic_update_slice(v_upd, x, (0, target))
                w_upd = jax.lax.dynamic_update_slice(w_upd, y, (0, target))
                u1_upd = jax.lax.dynamic_update_slice(u1_upd, u1, (0, target))
                u2_upd = jax.lax.dynamic_update_slice(u2_upd, u2, (0, target))
                active_upd = jax.lax.dynamic_update_slice(
                    active_upd,
                    jnp.asarray([True]),
                    (target,),
                )
                return (
                    v_upd,
                    w_upd,
                    u1_upd,
                    u2_upd,
                    active_upd,
                    offset_upd + jnp.asarray(1, dtype=index_dtype),
                )

            return jax.lax.cond(accept, do_update, lambda z: z, carry)

        v_out, w_out, u1_out, u2_out, active_out, accepted_count = jax.lax.fori_loop(
            0,
            x_pairs.shape[1],
            scatter_body,
            (
                v_in,
                w_in,
                u1_in,
                u2_in,
                active_in,
                jnp.asarray(0, dtype=index_dtype),
            ),
        )
        return v_out, w_out, u1_out, u2_out, active_out, basis_dim_in + accepted_count

    def _restart_from_roots(x_full: Array, y_full: Array):
        return _append_new_pairs(
            jnp.zeros((dim, max_subspace), dtype=dtype),
            jnp.zeros((dim, max_subspace), dtype=dtype),
            jnp.zeros((dim, max_subspace), dtype=dtype),
            jnp.zeros((dim, max_subspace), dtype=dtype),
            jnp.zeros((max_subspace,), dtype=bool),
            jnp.asarray(0, dtype=index_dtype),
            x_full[:, :nroots],
            y_full[:, :nroots],
            jnp.ones((nroots,), dtype=bool),
        )

    def _step(
        _iter: int,
        state: tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array, Array, Array],
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array, Array, Array]:
        (
            v_cur,
            w_cur,
            u1_cur,
            u2_cur,
            active_cur,
            basis_dim_cur,
            best_w_cur,
            best_x_cur,
            best_y_cur,
            best_residual_cur,
            converged_cur,
            done_cur,
        ) = state

        def do_iteration(iter_state):
            (
                v_it,
                w_it,
                u1_it,
                u2_it,
                active_it,
                basis_dim_it,
                best_w_it,
                best_x_it,
                best_y_it,
                best_residual_it,
                converged_it,
                _done_it,
            ) = iter_state
            omega, x_full, y_full, r_x, r_y, residual_norms = _solve_subspace(
                v_it,
                w_it,
                u1_it,
                u2_it,
                active_it,
            )
            max_residual = jnp.max(residual_norms)
            converged_now = _all_roots_converged(residual_norms, tol_arr)
            improve_best = (max_residual < best_residual_it) | converged_now
            best_w_next = jnp.where(improve_best, omega, best_w_it)
            best_x_next = jnp.where(improve_best, x_full, best_x_it)
            best_y_next = jnp.where(improve_best, y_full, best_y_it)
            best_residual_next = jnp.where(improve_best, max_residual, best_residual_it)

            denom_base = full_diag[:, None] - (
                omega[None, :] - jnp.asarray(preconditioner_level_shift, dtype=dtype)
            )
            denom_sign = jnp.where(denom_base < 0.0, -1.0, 1.0)
            denom = jnp.where(
                jnp.abs(denom_base) < preconditioner_floor,
                denom_sign * preconditioner_floor,
                denom_base,
            )
            residual_stacked = jnp.concatenate([r_x, -r_y], axis=0)
            correction = residual_stacked / denom
            new_x = correction[:dim, :]
            new_y = correction[dim:, :]
            new_mask = ~jnp.isfinite(residual_norms) | (residual_norms > tol_arr)
            no_new = ~jnp.any(new_mask)
            # Fill the remaining slots before restarting. Restarting whenever
            # a whole correction block does not fit wastes the tail of the
            # allocated subspace (e.g. repeatedly using only 6 of 8 slots).
            overflow = basis_dim_it >= jnp.asarray(max_subspace, dtype=index_dtype)

            def keep_current(_):
                return v_it, w_it, u1_it, u2_it, active_it, basis_dim_it

            def grow_or_restart(_):
                return jax.lax.cond(
                    overflow,
                    lambda __: _append_new_pairs(
                        *_restart_from_roots(x_full, y_full),
                        new_x,
                        new_y,
                        new_mask,
                    ),
                    lambda __: _append_new_pairs(
                        v_it,
                        w_it,
                        u1_it,
                        u2_it,
                        active_it,
                        basis_dim_it,
                        new_x,
                        new_y,
                        new_mask,
                    ),
                    operand=None,
                )

            v_next, w_next, u1_next, u2_next, active_next, basis_dim_next = jax.lax.cond(
                converged_now | no_new,
                keep_current,
                grow_or_restart,
                operand=None,
            )
            done_next = converged_now | no_new
            converged_flag_next = converged_it | converged_now
            return (
                v_next,
                w_next,
                u1_next,
                u2_next,
                active_next,
                basis_dim_next,
                best_w_next,
                best_x_next,
                best_y_next,
                best_residual_next,
                converged_flag_next,
                done_next,
            )

        return jax.lax.cond(done_cur, lambda s: s, do_iteration, state)

    final_state = jax.lax.fori_loop(
        0,
        int(max_iter),
        _step,
        (
            v_basis,
            w_basis,
            u1_basis,
            u2_basis,
            active_mask,
            basis_dim0,
            best_w0,
            best_x0,
            best_y0,
            best_residual0,
            converged0,
            done0,
        ),
    )
    (
        _v_basis,
        _w_basis,
        _u1_basis,
        _u2_basis,
        _active_mask,
        _basis_dim,
        best_w,
        best_x,
        best_y,
        _best_residual,
        converged,
        _done,
    ) = final_state
    return best_w, best_x, best_y, converged


def implicit_differential_davidson_lowest_tdhf(
    vind: Callable[[Array], Array],
    *,
    nroots: int,
    size: int,
    diag: Array,
    tol: float = DEFAULT_RPA_TOL,
    max_iter: int = DEFAULT_RPA_MAXITER,
    max_subspace: int | None = None,
    matrix_eps: float = 1e-10,
    preconditioner_floor: float = 1e-8,
    preconditioner_level_shift: float = 0.0,
    orth_eps: float = 1e-10,
    initial_guess_count: int | None = None,
) -> tuple[Array, Array, Array, Array]:
    """Return TDHF Davidson roots with implicit eigenvalue differentiation.

    An explicit initial count broadens the diagonal seed independently of the
    requested root count. Residual convergence alone does not certify that the
    initial space reaches every lower-energy symmetry sector.
    """

    dim = int(size)

    def solver_vind(values: Array) -> Array:
        return jax.lax.stop_gradient(vind(values))

    omega, x_vecs, y_vecs, converged = _davidson_lowest_tdhf(
        solver_vind,
        nroots=nroots,
        size=dim,
        diag=diag,
        tol=tol,
        max_iter=max_iter,
        max_subspace=max_subspace,
        matrix_eps=matrix_eps,
        preconditioner_floor=preconditioner_floor,
        preconditioner_level_shift=preconditioner_level_shift,
        orth_eps=orth_eps,
        initial_guess_count=initial_guess_count,
    )
    x_vecs = jax.lax.stop_gradient(x_vecs)
    y_vecs = jax.lax.stop_gradient(y_vecs)
    applied = vind(jnp.concatenate([x_vecs.T, y_vecs.T], axis=-1))
    top = applied[:, :dim].T
    bottom = -applied[:, dim:].T
    numerator = jnp.sum(x_vecs * top, axis=0) + jnp.sum(y_vecs * bottom, axis=0)
    denominator = jnp.sum(x_vecs * x_vecs, axis=0) - jnp.sum(y_vecs * y_vecs, axis=0)
    denominator = jnp.where(
        jnp.abs(denominator) > jnp.asarray(1e-30, dtype=x_vecs.dtype),
        denominator,
        jnp.asarray(1e-30, dtype=x_vecs.dtype),
    )
    values = require_converged_derivative(numerator / denominator, converged)
    return values, x_vecs, y_vecs, converged


def solve_dense_rpa(a, b, *, nroots, tol=1e-7, min_frequency=1e-8, max_dense=256):
    """Bounded complex RPA solve with metric-normalized eigenvalue response.

    Requires A=A^H and B=B^T. Positive, real-frequency, positive-metric roots
    are selected. Only isolated-root energy derivatives are supplied; X/Y are
    stopped, as for the real RPA Davidson energy-only interface.
    """
    a, b = jnp.asarray(a), jnp.asarray(b)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or b.shape != a.shape:
        raise ValueError("RPA requires square A and B blocks of the same shape")
    n = a.shape[0]
    if n > max_dense:
        raise ValueError("Dense RPA exceeds max_dense")
    if not 1 <= nroots <= n:
        raise ValueError("Invalid number of RPA roots")
    matrix = jnp.block([[a, b], [-b.conj(), -a.conj()]])
    values, vectors = jnp.linalg.eig(jax.lax.stop_gradient(matrix))
    eligible = (values.real > min_frequency) & (jnp.abs(values.imag) < tol)
    order = jnp.argsort(jnp.where(eligible, values.real, jnp.inf))[:nroots]
    selected, vec = values[order], vectors[:, order]
    applied = matrix @ vec
    residual = jnp.linalg.norm(applied - vec * selected, axis=0)
    norm = jnp.sum(jnp.abs(vec[:n])**2 - jnp.abs(vec[n:])**2, axis=0)
    valid = eligible[order] & (residual < tol) & (norm > 0)
    symmetry_tol = 32*jnp.finfo(a.real.dtype).eps*jnp.maximum(1.,jnp.linalg.norm(matrix))
    valid = valid & (jnp.linalg.norm(a-a.conj().T) <= symmetry_tol) & (jnp.linalg.norm(b-b.T) <= symmetry_tol)
    metric_applied = jnp.concatenate([applied[:n], -applied[n:]])
    quotient = jnp.sum(vec.conj()*metric_applied, axis=0).real / jnp.where(norm > 0, norm, 1.)
    energies = selected.real + quotient - jax.lax.stop_gradient(quotient)
    energies = require_converged_derivative(energies, valid)
    vec = vec / jnp.sqrt(jnp.where(norm > 0, norm, 1.))[None, :]
    return energies, vec[:n], vec[n:], valid
