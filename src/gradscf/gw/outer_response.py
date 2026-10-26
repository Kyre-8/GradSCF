"""Shared attachment policy for molecular GW self-consistency response."""
import jax
import jax.numpy as jnp

from ..solvers import LinearSolverConfig


def require_real(*inputs):
    if any(jnp.iscomplexobj(x) for x in jax.tree.leaves(inputs)):
        raise NotImplementedError('Molecular GW outer response requires real inputs.')


def linear_config(config):
    if config.mode != 'implicit':
        raise ValueError("GW outer differentiation requires mode='implicit'.")
    if not config.require_converged or config.regularization != 0:
        raise ValueError('GW outer response requires converged states and zero regularization.')
    return LinearSolverConfig(rtol=config.tolerance, maxiter=config.max_iter,
                              restart=20 if config.restart is None else config.restart)


def require_valid(valid, residual, method):
    # These are failure certificates, not physical observables. A live
    # callback payload can retain the eager-seed loop in reverse-mode AD
    # despite attaching the final root through its implicit rule.
    valid = jax.lax.stop_gradient(jnp.asarray(valid))
    residual = jax.lax.stop_gradient(jnp.asarray(residual))
    def fail(value):
        raise ArithmeticError(f'{method} outer solve failed convergence/response validity (residual {float(value):.3e}).')
    jax.lax.cond(valid, lambda: None, lambda: jax.debug.callback(fail, residual))


def require_occupation_gap(energy, nocc, method):
    """Validate fixed occupied/virtual labels without ordering each sector.

    A zero-temperature molecular G/W spectrum needs a positive gap between
    every occupied and virtual level. Degeneracy within either sector is
    permitted. Empty/full individual spin sectors have no gap condition.
    """
    energy = jnp.asarray(energy)
    if isinstance(nocc, tuple):
        if energy.ndim != 2 or len(nocc) != energy.shape[0]:
            raise ValueError('Spin occupations must match the stacked spectra.')
        for spin, occupied in enumerate(nocc):
            require_occupation_gap(energy[spin], occupied, method)
        return
    if not 0 <= nocc <= energy.shape[-1]:
        raise ValueError('Occupation count must lie between zero and nmo.')
    gap = (jnp.min(energy[..., nocc:], axis=-1)
           - jnp.max(energy[..., :nocc], axis=-1)) if 0 < nocc < energy.shape[-1] else jnp.inf
    valid = jnp.all(jnp.isfinite(energy)) & jnp.all(gap > 0)
    require_valid(valid, jnp.min(gap), f'{method} occupation gap')
