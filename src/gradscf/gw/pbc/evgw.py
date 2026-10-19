"""Full-response source-k G0W0 and eigenvalue-only self-consistent GW.

evGW retains the reference orbitals and density, updates every band in both
G and W, and solves the diagonal on-shell fixed point. Complete reference
KS-degenerate blocks use a trace average. G0W0 also offers the full matrix
Dyson window. Neither path implies periodic outer AD or a q-cell correction.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from ..evgw import _converged_result
from ..qp import solve_quasiparticle_roots
from ..types import GWResult
from ...solvers.nonlinear import ScalarRootConfig, NonlinearConfig, iterate_eager
from .query import _query_subspaces, _solve_subspace_qp
from .source import (build_correlation, correlation_at, correlation_diagonal, hartree_exchange,
                     require_eager, validate_model)


def _blocks(model, tolerance):
    nmo = model.energy.shape[-1]
    return _query_subspaces(model.energy, np.arange(nmo), tolerance)


def _diagonal_functions(model, cache, k, groups, exchange):
    """One trace-averaged diagonal equation per complete reference cluster."""
    for group in groups:
        bands = jnp.asarray(group)
        base = jnp.mean(model.energy[k, bands])
        shift = jnp.mean(jnp.diag(exchange[k]-model.reference_potential[k])[bands]).real
        def correlation(omega, bands=bands):
            return correlation_diagonal(cache, k, omega, bands)
        def equation(omega, base=base, shift=shift, correlation=correlation):
            return omega-base-shift-correlation(omega[0]).real
        yield bands, correlation, equation


@partial(jax.jit, static_argnames=('nk', 'linear_config', 'root_config'))
def _diagonal_roots(numerical_cache, reference, shift, bands, k, initial_guess,
                    *, nk, linear_config, root_config):
    """Reuse the complete shared selector across k/band values of one size.

    Numerical cache leaves, source k and band indices are dynamic operands.
    The same six seeds, slopes and branch certificate remain in the shared
    QP selector; only its enclosing executable is reused.
    """
    cache = dict(numerical_cache, nk=nk, linear_config=linear_config)
    correlation = lambda omega: correlation_diagonal(cache, k, omega, bands)
    equation = lambda omega: omega-reference-shift-correlation(omega[0]).real
    solved = solve_quasiparticle_roots(equation, reference,
        initial_guess=initial_guess, target=initial_guess, config=root_config)
    return solved, correlation(solved.roots[0])


def _evaluate_diagonal(model, energy, *, nw, eta, linear_config=None, gap_tol=1e-7):
    """Evaluate a single coupled evGW state, without an inner Dyson solve."""
    cache = build_correlation(model, energy, nw=nw, eta=eta, linear_config=linear_config)
    _, exchange = hartree_exchange(model, model.reference_density)
    e = jnp.asarray(energy)
    residual = jnp.zeros_like(e)
    sigma = jnp.zeros(e.shape, dtype=jnp.complex128)
    weight = jnp.full_like(e, jnp.nan)
    on_shell = jnp.full_like(e, jnp.nan)
    groups = _blocks(model, gap_tol)
    for k in range(len(e)):
        for bands, correlation, _ in _diagonal_functions(model, cache, k, groups[k], exchange):
            omega = jnp.mean(e[k, bands])
            value, derivative = jax.jvp(correlation, (omega,), (jnp.ones_like(omega),))
            base = jnp.mean(model.energy[k, bands])
            shift = jnp.mean(jnp.diag(exchange[k]-model.reference_potential[k])[bands]).real
            error = omega-base-shift-value.real
            slope = 1-derivative.real
            # A split numerical guess does not redefine a reference degenerate
            # space. The shared trace equation drives its members together.
            residual = residual.at[k, bands].set(e[k, bands]-omega+error)
            sigma = sigma.at[k, bands].set(value)
            z = jnp.where(jnp.isfinite(slope) & (slope > 1e-10), 1/slope, jnp.nan)
            weight = weight.at[k, bands].set(z)
            on_shell = on_shell.at[k, bands].set(omega-error)
    branch = jnp.isfinite(weight) & (weight > 0)
    return GWResult(mo_energy=e, mo_coeff=model.coeff, converged=False,
        sigma_qp=sigma, converged_mask=jnp.zeros_like(e, dtype=bool), nw=int(nw),
        qp_residual=residual, qp_computed_mask=jnp.zeros_like(e, dtype=bool),
        qp_weight=weight, qp_on_shell=on_shell, qp_branch_valid=branch,
        qp_candidate_count=jnp.zeros_like(e, dtype=jnp.int32), screening_energy=e,
        g_orbitals=tuple(range(e.shape[-1])), screening_occupied=tuple(range(model.nocc)),
        screening_virtual=tuple(range(model.nocc, e.shape[-1])),
        method='periodic_evgw_full_cd_diagonal')


def g0w0_model(model, *, nw=32, eta=1e-3, self_energy='matrix',
                root_config=None, linear_config=None, initial_guess=None, gap_tol=1e-7):
    """All-band source-k G0W0 for the common Hamiltonian.

    Matrix self-energy uses the entire orbital space at every source k.
    The reference potential is the actual native F-h-J; bare exchange uses
    the declared finite-q product model. Uncertified roots retain masks.
    ``initial_guess`` is both a numerical seed and a continuation target.
    """
    validate_model(model)
    require_eager(initial_guess)
    if self_energy not in ('matrix', 'diagonal'):
        raise ValueError("self_energy must be 'matrix' or 'diagonal'")
    if not np.isfinite(gap_tol) or gap_tol <= 0:
        raise ValueError('gap_tol must be finite and positive')
    config = root_config or ScalarRootConfig(method='hybrid', ftol=1e-9, xtol=1e-9, maxiter=120)
    energy = jnp.asarray(model.energy)
    if initial_guess is not None:
        initial_guess = jnp.asarray(initial_guess)
        if initial_guess.shape != energy.shape or not np.isfinite(initial_guess).all():
            raise ValueError('initial_guess must match the complete source-k spectrum')
    cache = build_correlation(model, energy, nw=nw, eta=eta, linear_config=linear_config)
    numerical_cache = {name: value for name, value in cache.items()
                       if name not in ('nk', 'linear_config')}
    _, exchange = hartree_exchange(model, model.reference_density)
    qp, residual = energy, jnp.zeros_like(energy)
    sigma = jnp.zeros(energy.shape, dtype=jnp.complex128)
    valid = jnp.zeros(energy.shape, dtype=bool)
    weight, on_shell, linearized = (jnp.full_like(energy, jnp.nan) for _ in range(3))
    count = jnp.zeros_like(energy, dtype=jnp.int32)
    branch = jnp.zeros_like(valid)
    vectors = jnp.zeros(energy.shape+(energy.shape[-1],), dtype=jnp.complex128)
    groups = _blocks(model, gap_tol)
    for k in range(len(energy)):
        if self_energy == 'matrix':
            out = _solve_subspace_qp(energy[k], exchange[k], model.reference_potential[k],
                lambda omega: correlation_at(cache, k, omega), config=config,
                occupied=jnp.arange(energy.shape[-1]) < model.nocc,
                initial_guess=None if initial_guess is None else initial_guess[k])
            qp = qp.at[k].set(out['roots'])
            residual = residual.at[k].set(out['residual'])
            sigma = sigma.at[k].set(out['sigma'])
            valid = valid.at[k].set(out['converged'])
            weight = weight.at[k].set(out['weight'])
            on_shell = on_shell.at[k].set(out['on_shell'])
            linearized = linearized.at[k].set(out['linearized'])
            count = count.at[k].set(out['candidate_count'])
            branch = branch.at[k].set(out['branch_valid'])
            vectors = vectors.at[k].set(out['vectors'])
            continue
        for group in groups[k]:
            bands = jnp.asarray(group)
            reference = jnp.mean(energy[k, bands])[None]
            shift = jnp.mean(jnp.diag(exchange[k]-model.reference_potential[k])[bands]).real
            guess = None if initial_guess is None else jnp.mean(initial_guess[k, bands])[None]
            solved, value = _diagonal_roots(numerical_cache, reference, shift, bands, jnp.asarray(k), guess,
                nk=cache['nk'], linear_config=cache['linear_config'], root_config=config)
            qp = qp.at[k, bands].set(solved.roots[0])
            residual = residual.at[k, bands].set(solved.residual[0])
            sigma = sigma.at[k, bands].set(value)
            valid = valid.at[k, bands].set(solved.converged[0])
            weight = weight.at[k, bands].set(solved.weight[0])
            on_shell = on_shell.at[k, bands].set(solved.on_shell[0])
            linearized = linearized.at[k, bands].set(solved.linearized[0])
            count = count.at[k, bands].set(solved.candidate_count[0])
            branch = branch.at[k, bands].set(solved.branch_valid[0])
            vectors = vectors.at[k, bands, bands].set(1.)
    return GWResult(mo_energy=qp, mo_coeff=model.coeff, converged=jnp.all(valid),
        sigma_qp=sigma, converged_mask=valid, qp_computed_mask=jnp.ones_like(valid),
        nw=int(nw), qp_residual=residual, qp_weight=weight, qp_on_shell=on_shell,
        qp_linearized=linearized, qp_candidate_count=count, qp_branch_valid=branch,
        qp_vectors=vectors, screening_energy=energy,
        g_orbitals=tuple(range(energy.shape[-1])), screening_occupied=tuple(range(model.nocc)),
        screening_virtual=tuple(range(model.nocc, energy.shape[-1])),
        method=f'periodic_g0w0_full_cd_{self_energy}')


def evgw_cd_kpoints(model, *, nw=32, eta=1e-3, max_iter=30, tol=1e-6,
                     damping=.2, initial_guess=None, linear_config=None, gap_tol=1e-7,
                     diis_space=6, diis_start=2, callback=None, differentiation=None):
    """Converge all source-k diagonal QP energies with fixed orbitals/density.

    G and full OV+VO W both use each current spectrum. Convergence is the
    undamped coupled on-shell Dyson residual, checked at the returned state.
    Shared eager DIIS owns the numerical trajectory; diis_space=0 selects a
    plain damped map. Invalid finite-gap proposals backtrack before building
    W. callback(iteration,energy,residual_norm) observes the actual states.
    No subset update, frozen W, Gamma-cell integration or outer AD is implied.
    """
    validate_model(model)
    require_eager(initial_guess)
    if differentiation is not None:
        raise NotImplementedError('Periodic evGW is eager forward only; outer AD is unsupported')
    if not isinstance(diis_space, int) or isinstance(diis_space, bool) or diis_space < 0:
        raise ValueError('diis_space must be a nonnegative integer')
    energy = jnp.asarray(model.energy if initial_guess is None else initial_guess)
    if energy.shape != model.energy.shape or jnp.iscomplexobj(energy):
        raise ValueError('initial_guess must be a real complete source-k spectrum')
    shape = energy.shape
    last = {}
    def valid(value):
        spectrum = np.asarray(value).reshape(shape)
        return np.min(spectrum[:, model.nocc:])-np.max(spectrum[:, :model.nocc]) > 0
    def physical(value):
        spectrum = value.reshape(shape)
        result = _evaluate_diagonal(model, spectrum, nw=nw, eta=eta,
                                    linear_config=linear_config, gap_tol=gap_tol)
        last['result'] = result
        return (spectrum-result.qp_residual).reshape(-1)
    observer = (None if callback is None else
                lambda iteration, state, norm: callback(iteration, state.reshape(shape), norm))
    solved = iterate_eager(physical, energy.reshape(-1), valid_state=valid, callback=observer,
        config=NonlinearConfig(maxiter=max_iter, residual_tol=tol, energy_tol=tol,
            damping=damping, diis_space=max(1, diis_space), diis_start=diis_start))
    if not solved.converged:
        raise ArithmeticError(f'Periodic evGW did not converge in {solved.iterations} iterations '
            f'(unmixed Dyson residual {solved.residual_norm:.3e} Ha > {tol}); no fallback result is returned')
    result = last['result']
    return _converged_result(result, jnp.ones_like(result.mo_energy, dtype=bool), tol, True)


__all__ = ['g0w0_model', 'evgw_cd_kpoints']
