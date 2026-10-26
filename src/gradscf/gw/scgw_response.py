"""Implicit response of the coupled real restricted Matsubara scGW state.

The state contains symmetric F, complex symmetric Sigma at positive
frequencies, the tail moment, and mu. Charge is a separate residual block;
mu is never frozen. Forward iteration is stopped before attaching the root
rule. Tangent/adjoint systems use the shared checked GMRES solver and
a scalar charge Schur complement. Invalid or unresolved response solves
return NaNs, following the SCF differentiation policy.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from gradscf.integrals.molecular.jk import build_jk_from_df
from ..solvers.linear import solve_scalar_border
from ..solvers import LinearSolverConfig
from ..solvers.nonlinear import attach_root
from .matsubara import dyson_green_and_density
from .scgw import _scgw_step


def _symmetric(values, nmo):
    rows, cols = jnp.triu_indices(nmo)
    matrix = jnp.zeros(values.shape[:-1] + (nmo, nmo), dtype=values.dtype)
    matrix = matrix.at[..., rows, cols].set(values)
    return matrix.at[..., cols, rows].set(values)


def _pack_state(fock, sigma, moment, mu, grid):
    rows, cols = jnp.triu_indices(fock.shape[0])
    positive = sigma[grid.nw:, rows, cols]
    return jnp.concatenate((fock[rows, cols], positive.real.ravel(), positive.imag.ravel(),
                            moment[rows, cols] / (jnp.pi / grid.beta), jnp.atleast_1d(mu)))


def _unpack_state(state, nmo, grid):
    size = nmo * (nmo + 1) // 2
    frequency_size = grid.nw * size
    fock = _symmetric(state[:size], nmo)
    real = state[size:size + frequency_size].reshape(grid.nw, size)
    imag = state[size + frequency_size:size + 2 * frequency_size].reshape(grid.nw, size)
    positive = _symmetric(real + 1j * imag, nmo)
    sigma = jnp.concatenate((positive[::-1].conj(), positive), axis=0)
    moment = _symmetric(state[size + 2 * frequency_size:-1], nmo) * (jnp.pi / grid.beta)
    return fock, sigma, moment, state[-1]


def _residual(state, params, grid, nocc):
    fock, sigma, moment, mu = _unpack_state(state, params["hcore"].shape[0], grid)
    green, per_spin = dyson_green_and_density(fock, sigma, mu, grid)
    mapped = _scgw_step(green, fock, mu, params["b"], params["hcore"],
                        2 * per_spin, grid, moment, params.get('phonons'))
    fock_new = mapped['fock']
    result = state - _pack_state(fock_new, mapped["sigma_iw"], mapped["sigma_moment"], mu, grid)
    return result.at[-1].set((2 * jnp.trace(per_spin) - 2 * nocc) / grid.beta)


def _chemical_potential(fock, sigma, guess, grid, target, particle_tol):
    def number(mu):
        _, density = dyson_green_and_density(fock, sigma, mu, grid)
        return 2 * jnp.trace(density)

    initial_number = number(guess)
    initial_ok = jnp.isfinite(initial_number) & (jnp.abs(initial_number - target) < particle_tol)

    def search(_):
        levels = jnp.linalg.eigvalsh(fock)
        span = jnp.maximum(1.0, jnp.maximum(levels[-1] - levels[0], jnp.max(jnp.abs(sigma))))
        def bracket_cond(s):
            width, low, high, iteration = s
            return (iteration < 20) & jnp.isfinite(low + high) & ~((low <= target) & (target <= high))
        def bracket_body(s):
            width, _, _, iteration = s
            width = 2 * width
            return width, number(guess - width), number(guess + width), iteration + 1
        span, nlo, nhi, _ = jax.lax.while_loop(
            bracket_cond, bracket_body, (span, number(guess - span), number(guess + span), jnp.array(0))
        )
        valid = jnp.isfinite(nlo + nhi) & (nlo <= target) & (target <= nhi)
        def bisect_cond(s):
            lo, hi, nl, nh, mu, done, ok, iteration = s
            return (iteration < 80) & ~done & ok
        def bisect_body(s):
            lo, hi, nl, nh, _, _, ok, iteration = s
            mu = 0.5 * (lo + hi)
            count = number(mu)
            ok = ok & jnp.isfinite(count) & (count >= nl - particle_tol) & (count <= nh + particle_tol)
            done = jnp.abs(count - target) < particle_tol
            below = count < target
            return (jnp.where(below, mu, lo), jnp.where(below, hi, mu),
                    jnp.where(below, count, nl), jnp.where(below, nh, count), mu, done, ok, iteration + 1)
        out = jax.lax.while_loop(
            bisect_cond, bisect_body,
            (guess - span, guess + span, nlo, nhi, guess, jnp.array(False), valid, jnp.array(0)),
        )
        return out[4], out[5] & out[6]

    return jax.lax.cond(initial_ok, lambda _: (guess, jnp.array(True)), search, operand=None)


@partial(jax.jit, static_argnames=("nocc", "max_iter", "tol", "mixing", "particle_tol"))
def _solve_state(hcore, b, energy_guess, grid, *, nocc, max_iter, tol, mixing, particle_tol,
                 phonons=None):
    nmo = hcore.shape[0]
    density = jnp.diag(jnp.where(jnp.arange(nmo) < nocc, 2.0, 0.0))
    jmat, kmat = build_jk_from_df(b, density)
    fock = hcore + jmat - 0.5 * kmat
    sigma = jnp.zeros((2 * grid.nw, nmo, nmo), dtype=jnp.complex128)
    moment = jnp.zeros_like(fock)
    mu = 0.5 * (energy_guess[nocc - 1] + energy_guess[nocc])

    def cond(s):
        fock, sigma, moment, mu, iteration, residual, done, valid = s
        return (iteration < max_iter) & ~done & valid
    def body(s):
        fock, sigma, moment, mu, iteration, _, _, _ = s
        mu, number_ok = _chemical_potential(fock, sigma, mu, grid, 2 * nocc, particle_tol)
        green, density = dyson_green_and_density(fock, sigma, mu, grid)
        mapped = _scgw_step(green, fock, mu, b, hcore, 2 * density, grid, moment, phonons)
        new_fock = mapped['fock']
        new_sigma, new_moment = mapped["sigma_iw"], mapped["sigma_moment"]
        residual = jnp.maximum(jnp.max(jnp.abs(new_fock - fock)), jnp.max(jnp.abs(new_sigma - sigma)))
        residual = jnp.maximum(residual, jnp.max(jnp.abs(new_moment - moment)) / (jnp.pi / grid.beta))
        valid = number_ok & jnp.isfinite(residual)
        done = valid & (residual < tol)
        # Preserve the evaluated state when it has passed all forward gates.
        weight = jnp.where(done, 0.0, mixing)
        return (fock + weight * (new_fock - fock), sigma + weight * (new_sigma - sigma),
                moment + weight * (new_moment - moment), mu, iteration + 1, residual, done, valid)

    out = jax.lax.while_loop(cond, body, (fock, sigma, moment, mu, jnp.array(0),
                                       jnp.array(jnp.inf), jnp.array(False), jnp.array(True)))
    return _pack_state(*out[:4], grid), out[4], out[5], out[6] & out[7]


def _charge_linear_solve(matvec, rhs, config, beta, charge_response_tol, converged):
    """Require a resolved positive dN/dmu for a stable finite-T branch."""
    linear_config = LinearSolverConfig(rtol=config.tolerance, maxiter=config.max_iter,
                                        restart=20 if config.restart is None else config.restart)
    return solve_scalar_border(matvec, rhs, config=linear_config, converged=converged,
        schur_valid=lambda schur: beta * schur > charge_response_tol)


def _raise_nonconverged(residual):
    raise ArithmeticError(f"scGW did not converge in the compiled forward solve (residual {float(residual):.3e} Ha).")


def implicit_scgw(*, coeff, energy_guess, hcore, b, nocc, nuclear_repulsion, grid,
                  max_iter, tol, mixing, particle_tol, config, charge_response_tol,
                  phonons=None):
    from .scgw import _assemble_scgw_result
    if config.mode != "implicit":
        raise ValueError("scGW differentiation currently supports mode='implicit' only.")
    if not config.require_converged or config.regularization != 0:
        raise ValueError("scGW implicit response requires converged states and zero regularization.")
    if not np.isfinite(charge_response_tol) or charge_response_tol < 0:
        raise ValueError("charge_response_tol must be finite and nonnegative (electrons/Ha).")
    stopped = jax.tree_util.tree_map(jax.lax.stop_gradient, (hcore, b, energy_guess, phonons))
    seed, iteration, error, converged = _solve_state(
        *stopped[:3], grid, nocc=nocc, max_iter=max_iter, tol=tol, mixing=mixing,
        particle_tol=particle_tol, phonons=stopped[3],
    )
    jax.lax.cond(converged, lambda: None, lambda: jax.debug.callback(_raise_nonconverged, error))
    seed = jax.lax.stop_gradient(seed)
    params = {"hcore": hcore, "b": b, "phonons": phonons}
    residual = lambda state: _residual(state, params, grid, nocc)
    tangent_solve = lambda matvec, rhs: _charge_linear_solve(
        matvec, rhs, config, grid.beta, charge_response_tol, converged
    )
    state = attach_root(residual, seed, tangent_solve=tangent_solve)
    fock, sigma, moment, mu = _unpack_state(state, hcore.shape[0], grid)
    return _assemble_scgw_result(
        coeff=coeff, hcore=hcore, b=b, nuclear_repulsion=nuclear_repulsion,
        fock=fock, sigma=sigma, moment=moment, mu=mu, grid=grid,
        nocc=nocc, n_iter=iteration, converged=converged,
        phonons=phonons,
    )
