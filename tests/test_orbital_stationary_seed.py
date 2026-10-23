"""Stationary seeds skip L-BFGS but retain finite-iterate Newton response."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.solvers.nonlinear.minimize import solve_orbitals


CURVATURE = 1e-6
TOLERANCE = 1e-10
SEED = 1e-4


def _energy(angles, base, args):
    parameter = args['parameter']
    target = parameter + parameter**2
    return .5*args['curvature']*jnp.sum((base+angles-target)**2)


def _rotate(angles, base):
    return base+angles


def _solve(parameter, *, curvature=CURVATURE, seed=SEED, max_iterations=4):
    return solve_orbitals(dict(parameter=parameter, curvature=curvature),
        jnp.array([seed]), energy=_energy,
        rotate=_rotate, dimension=1, max_iterations=max_iterations,
        gradient_tolerance=TOLERANCE)


@pytest.mark.parametrize('curvature,seed', [(CURVATURE, SEED), (1., 1e-10)],
                         ids=['shift_limited', 'well_conditioned'])
def test_stationary_seed_skips_lbfgs_and_keeps_three_newton_refinements(curvature, seed):
    initial_norm = .5*curvature*seed
    assert .1*TOLERANCE < initial_norm < TOLERANCE
    result = _solve(0., curvature=curvature, seed=seed)
    assert result.iterations == 0
    assert all(stage['lbfgs_iterations'] == 0 for stage in result.stages)
    assert result.stages[0]['polishing_steps'] == 3
    assert result.stages[1]['polishing_steps'] == 0
    assert all(step['accepted'] and step['refinement'] for step in result.polishing[:3])
    assert all(not step['accepted'] for step in result.polishing[3:])
    contraction = (1e-6/(curvature+1e-6))**3
    np.testing.assert_allclose(result.coeff, [seed*contraction], atol=seed*1e-16, rtol=1e-10)
    np.testing.assert_allclose(result.gradient_norm, initial_norm*contraction,
                               atol=initial_norm*1e-16, rtol=1e-10)


@pytest.mark.parametrize('curvature,seed', [(CURVATURE, SEED), (1., 1e-10)],
                         ids=['shift_limited', 'well_conditioned'])
def test_stationary_seed_explicit_jvp_vjp_and_directional_hvp_follow_newton_iterates(curvature, seed):
    value = lambda parameter: _solve(parameter, curvature=curvature, seed=seed).coeff[0]
    parameter, direction, cotangent = 0., .3, 1.2
    # This is the finite explicit trajectory, including the fixed Newton shift.
    contraction = (1e-6/(curvature+1e-6))**3
    response = 1-contraction
    (actual, tangent) = jax.jvp(value, (parameter,), (direction,))
    _, pullback = jax.vjp(value, parameter)
    _, hessian_vector = jax.jvp(jax.grad(value), (parameter,), (direction,))
    np.testing.assert_allclose(actual, seed*contraction, atol=seed*1e-16, rtol=1e-10)
    np.testing.assert_allclose(tangent, response*direction, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(pullback(cotangent)[0], response*cotangent, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(hessian_vector, 2*response*direction, atol=1e-12, rtol=1e-12)
    # Keep both finite-difference points on the stationary-seed branch.
    step = .1*seed
    finite = (value(parameter+step)-value(parameter-step))/(2*step)
    np.testing.assert_allclose(tangent, finite*direction, atol=1e-12, rtol=1e-12)


def test_above_tolerance_seed_still_runs_bounded_lbfgs():
    result = _solve(0., seed=4*SEED, max_iterations=1)
    assert result.iterations == 1
    assert result.stages[0]['lbfgs_iterations'] == 1
    assert result.gradient_norm > TOLERANCE
