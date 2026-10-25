"""Public JAX GMRES with scaled right-hand sides and numerical solve margin."""
import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import gmres


def gmres_solve(matvec, rhs, *, rtol, atol, maxiter, restart, preconditioner=None):
    # Keep scaling in the opaque numerical solve, not the differentiated map.
    scale = jnp.max(jnp.abs(rhs))
    def nonzero(_):
        scaled_rhs = rhs / scale
        # Leave room for rounding in response operators and scaling recovery.
        # The caller's true-residual acceptance tolerance remains unchanged.
        solve_rtol, solve_atol = .1 * rtol, .1 * atol / scale
        target = atol/scale + rtol*jnp.linalg.norm(scaled_rhs)

        def attempt(best, method, iterations):
            previous, previous_norm = best
            initial = jnp.where(jnp.all(jnp.isfinite(previous)), previous,
                                jnp.zeros_like(previous))
            candidate, _ = gmres(matvec, scaled_rhs, x0=initial, tol=solve_rtol,
                atol=solve_atol, restart=restart, maxiter=iterations,
                M=preconditioner, solve_method=method)
            norm = jnp.linalg.norm(matvec(candidate)-scaled_rhs)
            use = jnp.isfinite(norm) & (norm < previous_norm)
            return jnp.where(use, candidate, previous), jnp.where(use, norm, previous_norm)

        def accepted(best):
            return jnp.isfinite(best[1]) & (best[1] <= target)

        def retry(best):
            best = attempt(best, "batched", maxiter)
            # Retain the full incremental recovery budget if batched fails.
            return jax.lax.cond(accepted(best), lambda value: value,
                lambda value: attempt(value, "incremental", maxiter), best)

        # This predicate stays inside checked_linear_solve's opaque numerical
        # solve. Its differentiated operator remains linear, including VJPs.
        invalid = (jnp.full_like(rhs, jnp.nan), jnp.asarray(jnp.inf, rhs.real.dtype))
        # A short probe avoids spending every restart on a stalled QR path.
        best = attempt(invalid, "incremental", min(maxiter, 2))
        unit, _ = jax.lax.cond(accepted(best), lambda value: value, retry, best)
        return unit * scale
    return jax.lax.cond(scale > 0, nonzero, lambda _: jnp.zeros_like(rhs), None)
