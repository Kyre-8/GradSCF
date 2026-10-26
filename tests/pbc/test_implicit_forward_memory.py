"""Implicit SCF must not stage a reverse tape for its numerical trajectory.

This checks the AD dataflow, not a platform-dependent RSS threshold. Implicit
mode must not add trajectory arrays beyond the primal solver's diagnostics;
explicit mode deliberately retains the differentiable iteration trajectory.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.integrals.periodic.fft import build_inputs
from gradscf.pbc import gto, scf
from gradscf.scf import SCFDifferentiationConfig


def _scans(jaxpr):
    if hasattr(jaxpr, 'jaxpr'):
        jaxpr = jaxpr.jaxpr
    if not hasattr(jaxpr, 'eqns'):
        return
    for equation in jaxpr.eqns:
        if equation.primitive.name == 'scan':
            yield equation
        for value in equation.params.values():
            for child in value if isinstance(value, (tuple, list)) else (value,):
                yield from _scans(child)


@pytest.mark.parametrize('nk', [1, 2])
@pytest.mark.parametrize('mode', ['implicit', 'explicit'])
def test_scf_trajectory_tape_is_absent_only_for_implicit_mode(nk, mode):
    cell = gto.M(atom='H .4 .5 .6; H 1.8 .8 .9', a=np.eye(3)*6.,
                 unit='Bohr', mesh=(7,)*3)
    inputs = build_inputs(cell, kpts=cell.make_kpts((nk, 1, 1)))
    direction = jnp.diag(jnp.array([.13, -.07]))
    seed = jnp.broadcast_to(jnp.eye(2)*.5, (2, nk, 2, 2))
    config = SCFDifferentiationConfig(mode=mode)
    cycles = 17  # Distinct from response-solver iteration lengths.

    def energy(value):
        live = inputs._replace(hcore=inputs.hcore+value*direction,
                               overlap=inputs.overlap+.02*value*jnp.eye(2))
        initial = seed+.01*value*direction
        solver = scf.run_gamma_scf if nk == 1 else scf.run_kpoint_scf
        result = solver(live, mesh=cell.mesh, nelec=cell.nelec, xc='pbe',
                        max_cycle=cycles, init_density=initial[:, 0] if nk == 1 else initial,
                        differentiation=config)
        return result.total_energy

    # Abstract tracing only: no claim of convergence from this short trajectory.
    primal = [eq for eq in _scans(jax.make_jaxpr(energy)(.03))
              if eq.params['length'] == cycles]
    derivative = [eq for eq in _scans(jax.make_jaxpr(jax.value_and_grad(energy))(.03))
                  if eq.params['length'] == cycles and not eq.params['reverse']]
    assert len(primal) == len(derivative) == 1
    primal_stacked = primal[0].outvars[primal[0].params['num_carry']:]
    stacked = derivative[0].outvars[derivative[0].params['num_carry']:]
    if mode == 'implicit':
        assert len(stacked) == len(primal_stacked), (
            f'Implicit numerical SCF stages {len(stacked)-len(primal_stacked)} extra trajectory arrays.')
    else:
        assert len(stacked) > len(primal_stacked), 'Explicit SCF must preserve its iteration derivatives.'


@pytest.mark.parametrize('nk', [1, 2])
def test_implicit_final_orbitals_and_density_keep_live_metric_response(nk):
    cell = gto.M(atom='H .4 .5 .6; H 1.8 .8 .9', a=np.eye(3)*6.,
                 unit='Bohr', mesh=(7,)*3)
    inputs = build_inputs(cell, kpts=cell.make_kpts((nk, 1, 1)))
    direction = jnp.array([[.12, .03], [.03, -.08]])
    config = SCFDifferentiationConfig(mode='implicit', tolerance=1e-11, max_iter=80)

    def solve(value):
        live = inputs._replace(hcore=inputs.hcore+value*direction,
                               overlap=inputs.overlap+.02*value*jnp.eye(2))
        solver = scf.run_gamma_scf if nk == 1 else scf.run_kpoint_scf
        result = solver(live, mesh=cell.mesh, nelec=cell.nelec,
                        max_cycle=100, damping=.1, conv_tol=1e-12,
                        conv_tol_density=1e-11, conv_tol_grad=1e-10,
                        differentiation=config)
        return result, live.overlap

    result, overlap = solve(.03)
    assert result.converged

    def metric(value):
        state, overlap = solve(value)
        coefficients = state.mo_coeff_spin
        return coefficients.conj().swapaxes(-1, -2)@overlap@coefficients

    identity, tangent = jax.jit(lambda value: jax.jvp(metric, (value,), (jnp.ones_like(value),)))(.03)
    np.testing.assert_allclose(identity, jnp.broadcast_to(jnp.eye(2), identity.shape), atol=2e-12, rtol=0)
    np.testing.assert_allclose(tangent, 0., atol=2e-10, rtol=0)

    def observable(value):
        state, _ = solve(value)
        # The density term requires the implicit state response at both orders.
        return state.total_energy+.02*jnp.sum(jnp.abs(state.density_spin)**2)

    gradient = jax.jit(jax.grad(observable))
    curvature = jax.jit(jax.grad(jax.grad(observable)))(.03)
    step = 1e-4
    finite = (gradient(.03+step)-gradient(.03-step))/(2*step)
    assert jnp.isfinite(curvature)
    np.testing.assert_allclose(curvature, finite, atol=3e-6, rtol=2e-5)
