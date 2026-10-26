"""Restarted two-sided residual expansion in a real orthonormal search space.

Storage is O(n*m + m*m), m <= max_space. No physical dense matrix is assembled.
The projected matrix is never symmetrized. Spectral ordering/isolation outside
this finite search space is not certified by a small Ritz residual.
"""

from typing import NamedTuple
from jaxtyping import Array
import numpy as np
import jax
import jax.numpy as jnp
from ._nonhermitian_ritz import paired_eigenvectors
from ..memory import apply_columns


class IterativeRitz(NamedTuple):
    spectrum: Array
    right: Array
    left: Array
    iterations: Array
    dimension: Array
    guard_residuals: Array
    guard_valid: Array
    restarts: Array


def _append(basis, count, candidates):
    width = basis.shape[1]
    eps = 64 * jnp.finfo(basis.dtype).eps

    def add(i, state):
        v, used = state
        x = candidates[:, i]
        length = jnp.linalg.norm(x)
        x = x / jnp.maximum(length, jnp.finfo(x.dtype).tiny)
        for _ in range(2):
            x = x - v @ (v.T @ x)
        norm = jnp.linalg.norm(x)
        accept = (used < width) & (length > eps) & (norm > eps) & jnp.isfinite(norm)
        column = x / jnp.maximum(norm, jnp.finfo(x.dtype).tiny)
        v = jax.lax.cond(accept, lambda v: v.at[:, used].set(column), lambda v: v, v)
        return v, used + accept.astype(used.dtype)

    return jax.lax.fori_loop(0, candidates.shape[1], add, (basis, count))


def davidson_ritz(op, cfg, initial_vectors=None, *, batch_size=None):
    n = op.shape[0]
    width = min(n, cfg.max_space)
    count = min(n, cfg.nroots + cfg.guard_roots)
    if width < n and width < 4 * count + 2:
        raise ValueError("max_space must be at least 4*(nroots+guard_roots)+2")
    if cfg.nroots > width:
        raise ValueError("nroots exceeds max_space")
    transpose = op.T
    apply = lambda x: jax.lax.stop_gradient(apply_columns(op, x, batch_size))
    apply_transpose = lambda x: jax.lax.stop_gradient(apply_columns(transpose, x, batch_size))
    rng = np.random.default_rng(cfg.seed)
    random = jnp.asarray(
        rng.normal(size=(n, min(width, count))),
        dtype=op.dtype,
    )
    if initial_vectors is None:
        if op.diagonal is None:
            seeds = random
        else:
            canonical = jax.nn.one_hot(
                jnp.argsort(op.diagonal)[:count], n, dtype=op.dtype
            ).T
            # Separate random columns alone still leave exact canonical Ritz
            # roots in the span, allowing premature convergence in a wrong sector.
            noise = jnp.asarray(rng.normal(size=canonical.shape), dtype=op.dtype)
            seeds = jnp.concatenate(
                [canonical + 1e-3 * noise / np.sqrt(n), random], axis=1
            )
    else:
        if jnp.iscomplexobj(initial_vectors):
            raise NotImplementedError("Initial vectors must be real")
        seeds = jnp.asarray(initial_vectors, dtype=op.dtype)
        if seeds.ndim != 2 or seeds.shape[0] != n or not 1 <= seeds.shape[1] <= width:
            raise ValueError("initial_vectors must have shape (n, 1..max_space)")
        # Preserve explicitly supplied guesses, while supplementing them with
        # independent full-support exploration directions.
        seeds = jnp.concatenate([seeds, random], axis=1)
    seeds = jax.lax.stop_gradient(seeds)
    basis, used = _append(
        jnp.zeros((n, width), op.dtype), jnp.asarray(0, jnp.int32), seeds
    )

    def project(v, used):
        av, atv = apply(v), apply_transpose(v)
        h = v.T @ av
        # Gershgorin bound places inactive padded roots above every active root.
        shift = 1 + 2 * jnp.max(jnp.sum(jnp.abs(h), axis=1))
        h += jnp.diag(jnp.where(jnp.arange(width) < used, 0.0, shift))
        values, cr, cl = paired_eigenvectors(h, count)
        r, l = v @ cr, v @ cl
        rn, ln = jnp.linalg.norm(r, axis=0), jnp.linalg.norm(l, axis=0)
        rden, lden = jnp.maximum(rn, 1e-30), jnp.maximum(ln, 1e-30)
        r, l = r / rden, l / lden
        rr = (av @ cr) / rden - r * values[:count]
        lr = (atv @ cl) / lden - l * values[:count]
        norms = jnp.maximum(jnp.linalg.norm(rr, axis=0), jnp.linalg.norm(lr, axis=0))
        present = (rn > 0.5) & (ln > 0.5) & (jnp.arange(count) < used)
        norms = jnp.where(present, norms, jnp.inf)
        return values, r, l, rr, lr, norms

    def body(state):
        iteration, v, used, _, restarts = state
        values, r, l, rr, lr, norms = project(v, used)
        done = jnp.all(norms <= cfg.atol) & jnp.all(jnp.isfinite(values))

        def expand(args):
            v, used = args
            # Preserve both Ritz spaces, including complex conjugate directions.
            retained = jnp.concatenate([r.real, r.imag, l.real, l.imag], axis=1)
            v, used = jax.lax.cond(
                used == width,
                lambda _: _append(
                    jnp.zeros_like(v), jnp.asarray(0, jnp.int32), retained
                ),
                lambda _: (v, used),
                None,
            )
            if op.diagonal is not None:
                denom = values[None, :count] - jax.lax.stop_gradient(
                    op.diagonal[:, None]
                )
                denom = jnp.where(
                    jnp.abs(denom) > cfg.preconditioner_floor,
                    denom,
                    cfg.preconditioner_floor,
                )
                rr0, lr0 = rr / denom, lr / denom
            else:
                rr0, lr0 = rr, lr
            active = norms > cfg.atol * 0.1
            corrections = jnp.concatenate(
                [
                    rr0.real * active,
                    rr0.imag * active,
                    lr0.real * active,
                    lr0.imag * active,
                ],
                axis=1,
            )
            old_used = used
            v, used = _append(v, used, corrections)
            # Exact diagonal preconditioning can give a vector parallel to the
            # Ritz vector. Expand with the unpreconditioned residual if stalled.
            raw = jnp.concatenate(
                [
                    rr.real * active,
                    rr.imag * active,
                    lr.real * active,
                    lr.imag * active,
                ],
                axis=1,
            )
            return jax.lax.cond(
                used == old_used,
                lambda args: _append(*args, raw),
                lambda args: args,
                (v, used),
            )

        restarted = (used == width) & ~done & (iteration + 1 < cfg.maxiter)
        v, used = jax.lax.cond(
            done | (iteration + 1 >= cfg.maxiter), lambda args: args, expand, (v, used)
        )
        return iteration + 1, v, used, done, restarts + restarted.astype(restarts.dtype)

    iterations, basis, used, _, restarts = jax.lax.while_loop(
        lambda state: (state[0] < cfg.maxiter) & ~state[3],
        body,
        (
            jnp.asarray(0, jnp.int32),
            basis,
            used,
            jnp.asarray(False),
            jnp.asarray(0, jnp.int32),
        ),
    )
    values, right, left, _, _, norms = project(basis, used)
    spectrum = jnp.where(jnp.arange(width) < used, values, jnp.nan + 0j)
    guards = norms[cfg.nroots :]
    return IterativeRitz(
        spectrum,
        right[:, : cfg.nroots],
        left[:, : cfg.nroots],
        iterations,
        used,
        guards,
        jnp.all(guards <= cfg.atol),
        restarts,
    )
