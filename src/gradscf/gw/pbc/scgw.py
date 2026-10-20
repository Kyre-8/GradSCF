"""Eager full complex finite-temperature matrix scGW on a closed k mesh.

The dynamical state is F[k], Sigma_c[k,iw] and its 1/iw moment in one fixed
orthonormal source-MO frame. Density from the interacting Dyson G sets both
Hartree/exchange and the common chemical potential. A source-mesh q sum is
used; off-mesh query and q->0 head/wing completion are not supplied here.
No quasiparticle poles are inferred from the static Fock spectrum.
"""
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from ...solvers.nonlinear.eager import iterate_eager
from ...solvers.nonlinear.iterate import NonlinearConfig
from ...solvers.nonlinear.scalar import bisect
from .matsubara import (MatsubaraGrid, matsubara_grid, dyson_green_and_density,
                       gw_matsubara_step, _moment_is_psd)
from .source import hartree_exchange, validate_model


@dataclass(frozen=True)
class PeriodicSCGWResult:
    """One returned Dyson state and its unmixed physical map.

    ``density_mo`` and ``density_matrix`` are SPIN-SUMMED; the kernel and
    source Hartree/exchange helpers instead accept single-spin density.
    Arrays carrying k have the source mesh ordering. Frequencies include
    both signs in grid.fermion. Energies and static_fock eigenvalues are in
    Ha, beta in Ha^-1, charge_response in electrons/Ha.

    ``mo_energy`` is None until a separate causal spectral calculation has
    established poles. ``static_mo_energy`` only diagonalizes static F.
    No periodic total/internal energy functional is claimed in this slice.
    """
    mo_energy: None
    mo_coeff: object
    chemical_potential: object
    density_mo: object
    density_matrix: object
    green_iw: object
    self_energy_iw: object
    sigma_moment: object
    fock_mo: object
    mapped_fock_mo: object
    mapped_self_energy_iw: object
    mapped_sigma_moment: object
    static_mo_energy: object
    static_mo_coeff: object
    fixed_point_residual: object
    particle_number_error: object
    charge_response: object
    grid: MatsubaraGrid
    converged: bool
    n_iter: int
    nw: int
    beta: float
    polarizability_inu: object
    screened_inu: object
    correlation_energy: None = None
    electronic_energy: None = None
    total_energy: None = None
    method: str = 'periodic_scgw_matsubara'
    time_oversampling: int = 2
    time_quadrature: str = 'uniform'

    @property
    def sigma_c_iw(self):
        return self.self_energy_iw

    @property
    def full_fock(self):
        return self.fock_mo

    @property
    def per_spin_density_mo(self):
        return .5*self.density_mo


def _count(density):
    return 2*jnp.mean(jnp.trace(density, axis1=-2, axis2=-1).real)


@partial(jax.jit, static_argnames=('particle_tol',))
def _bracketed_number_root(fock, sigma, lower, upper, target, grid, sigma_moment=None, *, particle_tol):
    def residual(mu):
        return _count(dyson_green_and_density(fock, sigma, mu, grid, sigma_moment)[1])-target
    return bisect(residual, lower, upper, tolerance=particle_tol, maxiter=64)


@jax.jit
def _number_charge(fock, sigma, mu, grid, sigma_moment=None):
    def count(value):
        return _count(dyson_green_and_density(fock, sigma, value, grid, sigma_moment)[1])
    return jax.jvp(count, (jnp.asarray(mu),), (jnp.ones_like(jnp.asarray(mu)),))


def _solve_number(fock, sigma, guess, grid, target, particle_tol, sigma_moment=None):
    """Physics-specific bracketing around the shared scalar root solver."""
    def evaluate(mu):
        green, density = dyson_green_and_density(fock, sigma, mu, grid, sigma_moment)
        count = float(_count(density))
        if not np.isfinite(count):
            raise ArithmeticError('Nonfinite periodic scGW electron number.')
        return count, green, density

    count, green, density = evaluate(guess)
    corrected = abs(count-target) > particle_tol
    if corrected:
        spectrum = np.asarray(jnp.linalg.eigvalsh(fock))
        span = max(1., float(np.ptp(spectrum)), float(jnp.max(jnp.abs(sigma))))
        for _ in range(20):
            lower, upper = guess-span, guess+span
            nlower, _, _ = evaluate(lower)
            nupper, _, _ = evaluate(upper)
            if nlower <= target <= nupper:
                break
            span *= 2
        else:
            raise ArithmeticError('Could not bracket periodic scGW electron number; increase nw.')
        mu, converged = _bracketed_number_root(fock, sigma, lower, upper,
            target, grid, sigma_moment, particle_tol=particle_tol)
        if not bool(converged):
            raise ArithmeticError('Periodic scGW number root did not meet particle_tol.')
        guess = float(mu)
        count, green, density = evaluate(guess)
    _, charge = _number_charge(fock, sigma, guess, grid, sigma_moment)
    charge = float(charge)
    # A thermally unresolved insulating plateau is accepted only when the
    # initial mu already has the required number. It carries no mu response.
    if not np.isfinite(charge) or charge < -1e-10 or (corrected and charge <= 0):
        raise ArithmeticError('Nonpositive/nonfinite periodic scGW charge response; increase nw.')
    if abs(count-target) > particle_tol:
        raise ArithmeticError('Periodic scGW returned density violates particle_tol.')
    return guess, green, density, charge


def _physical_map(model, green, fock, mu, density, moment, grid):
    step = gw_matsubara_step(green, fock, mu, model, grid,
                            sigma_moment=moment, density=density,
                            time_diagnostics=False, time_batch_size=256)
    hartree, exchange = hartree_exchange(model, density)
    step['fock'] = model.hcore+hartree+exchange
    return step


def _pack(fock, sigma, moment, frequency):
    value = jnp.concatenate((fock.reshape(-1), sigma.reshape(-1), (moment/frequency).reshape(-1)))
    return jnp.concatenate((value.real, value.imag))


def _unpack(value, fshape, sshape, frequency):
    half = value.size//2
    value = value[:half]+1j*value[half:]
    nf, ns = int(np.prod(fshape)), int(np.prod(sshape))
    fock = value[:nf].reshape(fshape)
    sigma = value[nf:nf+ns].reshape(sshape)
    moment = frequency*value[nf+ns:].reshape(fshape)
    return .5*(fock+fock.conj().swapaxes(-1, -2)), sigma, .5*(moment+moment.conj().swapaxes(-1, -2))


def scgw_matsubara_kpoints(model, *, nw=64, beta=100., time_oversampling=2,
                           time_quadrature='uniform', max_iter=60,
                           tol=1e-6, mixing=.5, particle_tol=1e-9,
                           diis_space=6, diis_start=3, differentiation=None,
                           callback=None):
    """Solve full complex restricted matrix GW, returning imaginary-axis data.

    ``model`` is a PeriodicGWModel or a converged restricted periodic mean
    field accepted by build_periodic_model. Source factors contain no 1/Nk.
    ``mixing`` is the accepted fraction of the unmixed physical update.
    Shared nonlinear DIIS acts on real/imaginary packed F, Sigma and its
    tail moment; convergence always checks the unmixed map in Ha, including
    moment/(pi/beta). Chemical potential is solved from that same G with a
    shared scalar bisection and a checked forward charge susceptibility.
    max_iter bounds accepted updates; up to max_iter+1 physical maps are
    evaluated, including the initial state. Returned n_iter counts these
    map evaluations rather than updates.

    Only eager source-mesh evaluation with fc=False is currently supported.
    Increase nw at fixed beta before taking a low-temperature limit.
    time_oversampling pads Nt=2*nw*time_oversampling without increasing the
    frequency window; 2 resolves retained quadratic modes, 1 is the baseline.
    time_quadrature='gauss' selects positive symmetric Gauss-Legendre time
    integration; 'uniform' retains the midpoint rule. Both require convergence.
    Optional callback(iteration, residual, mu, particle_error) receives host
    scalars after each evaluated physical map, including the final iterate.
    Residual/mu use Ha and particle_error uses electrons per cell.
    """
    from .source import PeriodicGWModel, build_periodic_model
    if (differentiation is not None or any(isinstance(value, jax.core.Tracer)
            for value in (beta, nw, time_oversampling))):
        raise NotImplementedError('Periodic scGW differentiation/AD is not implemented; use eager forward evaluation.')
    if not isinstance(model, PeriodicGWModel):
        model = build_periodic_model(model, fc=False)
    physical = (model.energy, model.coeff, model.hcore, model.factors)
    if any(isinstance(x, jax.core.Tracer) for x in jax.tree_util.tree_leaves(physical)):
        raise NotImplementedError('Periodic scGW is eager only; differentiation/AD is not implemented.')
    validate_model(model)
    if model.metadata and model.metadata.get('fc', False):
        raise NotImplementedError('Periodic scGW q0 head/wing completion requires fc=False in this implementation.')
    if (max_iter < 1 or diis_space < 1 or diis_start < 1
            or not np.isfinite(tol) or tol <= 0
            or not np.isfinite(particle_tol) or particle_tol <= 0
            or not np.isfinite(mixing) or not 0 < mixing <= 1):
        raise ValueError('Invalid periodic scGW iteration, tolerance or mixing controls.')
    grid = matsubara_grid(nw=nw, beta=beta, time_oversampling=time_oversampling,
                          time_quadrature=time_quadrature)
    fock = jnp.asarray(model.reference_fock, dtype=jnp.complex128)
    sigma = jnp.zeros((model.energy.shape[0], 2*grid.nw)+fock.shape[1:], dtype=jnp.complex128)
    moment = jnp.zeros_like(fock)
    mu = float(model.chemical_potential)
    frequency = jnp.pi/grid.beta
    packed = _pack(fock, sigma, moment, frequency)
    target = 2*model.nocc
    fshape, sshape = fock.shape, sigma.shape
    last = dict(mu=mu)

    def physical_map(value):
        fock, sigma, moment = _unpack(value, fshape, sshape, frequency)
        # The previous map's Pi/W buffers are not inputs to the next map.
        # Drop them before building new frequency/time diagrams.
        last.pop('step', None)
        mu, green, density, charge = _solve_number(
            fock, sigma, last['mu'], grid, target, particle_tol, moment)
        step = _physical_map(model, green, fock, mu, density, moment, grid)
        # Complex absolute errors (not separate real/imaginary bounds).
        residual = max(float(jnp.max(jnp.abs(step['fock']-fock))),
            float(jnp.max(jnp.abs(step['sigma_iw']-sigma))),
            float(jnp.max(jnp.abs(step['sigma_moment']-moment)))/float(frequency))
        if not np.isfinite(residual):
            raise ArithmeticError('Nonfinite periodic scGW physical map.')
        last.update(mu=mu, fock=fock, sigma=sigma, moment=moment, green=green,
            density=density, charge=charge, step=step, residual=residual,
            particle_error=float(_count(density)-target))
        return _pack(step['fock'], step['sigma_iw'], step['sigma_moment'], frequency)

    def check(_state, _mapped, _norm):
        return last['residual'] < tol and abs(last['particle_error']) <= particle_tol

    def observe(iteration, _state, _norm):
        callback(iteration+1, last['residual'], last['mu'], last['particle_error'])

    def valid_state(value):
        return bool(_moment_is_psd(_unpack(value, fshape, sshape, frequency)[2]))

    iteration = iterate_eager(physical_map, packed,
        config=NonlinearConfig(maxiter=int(max_iter), residual_tol=float(tol),
            damping=1-float(mixing), diis_space=int(diis_space), diis_start=int(diis_start)),
        valid_state=valid_state, check=check, callback=observe if callback is not None else None)
    if not iteration.converged:
        raise ArithmeticError(f'Periodic scGW did not converge in {max_iter} updates '
            f"(unmixed matrix/tail residual {last['residual']:.3e} Ha > {tol}).")
    # The shared driver returns its last evaluated state; retain the exact
    # Dyson G, density and map from that evaluation without another solve.
    fock, sigma, moment = last['fock'], last['sigma'], last['moment']
    mu, green, density, charge = last['mu'], last['green'], last['density'], last['charge']
    step = last['step']
    static_energy, rotation = jnp.linalg.eigh(fock)
    spin_density = 2*density
    ao_density = jnp.einsum('kam,kmn,kbn->kab', model.coeff, spin_density, model.coeff.conj())
    return PeriodicSCGWResult(mo_energy=None, mo_coeff=model.coeff,
        chemical_potential=jnp.asarray(mu), density_mo=spin_density,
        density_matrix=ao_density, green_iw=green, self_energy_iw=sigma,
        sigma_moment=moment, fock_mo=fock, mapped_fock_mo=step['fock'],
        mapped_self_energy_iw=step['sigma_iw'], mapped_sigma_moment=step['sigma_moment'],
        static_mo_energy=static_energy, static_mo_coeff=model.coeff@rotation,
        fixed_point_residual=jnp.asarray(last['residual']), particle_number_error=_count(density)-target,
        charge_response=jnp.asarray(charge), grid=grid, converged=True,
        n_iter=iteration.iterations+1, nw=grid.nw, beta=grid.beta,
        time_oversampling=int(time_oversampling),
        time_quadrature=time_quadrature,
        polarizability_inu=step['polarizability_inu'], screened_inu=step['screened_inu'])


__all__ = ['PeriodicSCGWResult', 'scgw_matsubara_kpoints']
