"""Private primal scaling preserves physical Newton equations and finite AD."""
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.solvers.nonlinear import minimize
from gradscf._memory import jaxpr_temporary_bytes


DIMENSION = 48
TOLERANCE = 1e-10
SHIFT = 1e-6


def _require_callback():
    assert 'hessian_diagonal' in inspect.signature(minimize.solve_orbitals).parameters, (
        'The private primal Hessian-diagonal callback is missing')


def _energy(angles, base, args):
    target = args['parameter'] + args['parameter']**2
    return .5*jnp.sum(args['diagonal']*(base+angles-target)**2)


def _rotate(angles, base):
    return base+angles


def _diagonal(base, args):
    return args['diagonal']


def _inputs(scale):
    diagonal = scale*jnp.geomspace(1e-6, 1e-3, DIMENSION)
    # Half the physical gradient norm is 6e-11 Eh: stationary, but all
    # three shifted refinements retain a RHS well above float64 epsilon.
    seed = jnp.full(DIMENSION, 1.2e-10/jnp.sqrt(DIMENSION))/diagonal
    return diagonal, seed


def _solve(parameter, diagonal, seed, callback):
    return minimize.solve_orbitals(
        dict(parameter=parameter, diagonal=diagonal*(1+.2*parameter)), seed,
        energy=_energy, rotate=_rotate, dimension=DIMENSION, max_iterations=1,
        gradient_tolerance=TOLERANCE, hessian_diagonal=callback)


def _expected(parameter, diagonal, seed):
    target = parameter+parameter**2
    contraction = (SHIFT/(diagonal*(1+.2*parameter)+SHIFT))**3
    return target+(seed-target)*contraction


def _assert_refinements(result):
    assert result.iterations == 0
    assert result.stages[0]['polishing_steps'] == 3
    assert result.stages[1]['polishing_steps'] == 0
    assert all(row['accepted'] and row['refinement'] for row in result.polishing[:3])
    assert all(not row['accepted'] for row in result.polishing[3:])
    np.testing.assert_allclose([row['hessian_shift'] for row in result.polishing[:3]],
                               SHIFT, atol=0., rtol=0.)
    assert all(row['linear_relative_residual'] <= .1 for row in result.polishing[:3])


@pytest.mark.parametrize('scale', [.5, 2.])
def test_right_diagonal_reduces_actual_gmres_actions_and_keeps_physical_refinements(scale, monkeypatch):
    _require_callback()
    diagonal, seed = _inputs(scale)
    original = minimize.gmres
    events = []

    def record(kind, value):
        events.append((kind, float(value)))

    def counted(operator, rhs, **kwargs):
        assert kwargs['tol'] == 1e-11 and kwargs['atol'] == 0.
        assert kwargs['restart'] == 40 and kwargs['maxiter'] == 10
        assert kwargs['solve_method'] == 'incremental' and kwargs.get('M') is None
        jax.debug.callback(lambda value: record('rhs', value), jnp.linalg.norm(rhs), ordered=True)

        def action(vector):
            # Observe only calls made by the real GMRES, with no HVP probes.
            jax.debug.callback(lambda: record('action', 0.), ordered=True)
            return operator(vector)
        return original(action, rhs, **kwargs)

    def counted_diagonal(base, args):
        jax.debug.callback(lambda: record('diagonal', 0.), ordered=True)
        return _diagonal(base, args)

    monkeypatch.setattr(minimize, 'gmres', counted)
    # The instrumented global is captured at trace time; isolate each scale.
    minimize.solve_orbitals.clear_cache()
    baseline = _solve(0., diagonal, seed, None)
    baseline.coeff.block_until_ready()
    jax.effects_barrier()
    baseline_events = list(events)
    events.clear()
    scaled = _solve(0., diagonal, seed, counted_diagonal)
    scaled.coeff.block_until_ready()
    jax.effects_barrier()
    minimize.solve_orbitals.clear_cache()

    for result in (baseline, scaled):
        _assert_refinements(result)
        assert jnp.all(jnp.isfinite(result.coeff))
    # JAX's absolute epsilon cutoff can stop the unscaled tiny-RHS solve
    # before an exact Newton step. Use the analytic trajectory as the oracle.
    expected = _expected(0., diagonal, seed)
    np.testing.assert_allclose(scaled.coeff, expected, atol=2e-17, rtol=2e-8)
    assert max(row['linear_relative_residual'] for row in scaled.polishing[:3]) <= 1e-10
    for observed in (baseline_events, events):
        rhs = [value for kind, value in observed if kind == 'rhs']
        assert len(rhs) == 3
        assert all(value > 100*jnp.finfo(jnp.float64).eps for value in rhs)
        assert rhs[0] == pytest.approx(1.2e-10, rel=1e-12)
    assert sum(kind == 'diagonal' for kind, _ in events) == 3
    baseline_actions = sum(kind == 'action' for kind, _ in baseline_events)
    scaled_actions = sum(kind == 'action' for kind, _ in events)
    assert 0 < scaled_actions < .5*baseline_actions, (baseline_actions, scaled_actions)
    print(dict(scale=scale, baseline_actions=baseline_actions, right_actions=scaled_actions,
        baseline_max_error=float(jnp.max(jnp.abs(baseline.coeff-expected))),
        right_max_error=float(jnp.max(jnp.abs(scaled.coeff-expected))),
        baseline_residuals=[float(row['linear_relative_residual']) for row in baseline.polishing[:3]],
        right_residuals=[float(row['linear_relative_residual']) for row in scaled.polishing[:3]]))


@pytest.mark.parametrize('scale', [.5, 2.])
def test_right_diagonal_finite_newton_value_jvp_vjp_and_hvp_match_analytic_trajectory(scale):
    _require_callback()
    diagonal, seed = _inputs(scale)
    weights = jnp.linspace(.5, 1.5, DIMENSION)/DIMENSION
    actual = lambda parameter: jnp.vdot(weights, _solve(parameter, diagonal, seed, _diagonal).coeff)
    expected = lambda parameter: jnp.vdot(weights, _expected(parameter, diagonal, seed))
    parameter, direction, cotangent = 0., .3, 1.2
    value, tangent = jax.jvp(actual, (parameter,), (direction,))
    reference, reference_tangent = jax.jvp(expected, (parameter,), (direction,))
    _, pullback = jax.vjp(actual, parameter)
    _, reference_pullback = jax.vjp(expected, parameter)
    _, hvp = jax.jvp(jax.grad(actual), (parameter,), (direction,))
    _, reference_hvp = jax.jvp(jax.grad(expected), (parameter,), (direction,))
    np.testing.assert_allclose(value, reference, atol=2e-17, rtol=2e-8)
    np.testing.assert_allclose(tangent, reference_tangent, atol=2e-10, rtol=2e-10)
    np.testing.assert_allclose(pullback(cotangent)[0], reference_pullback(cotangent)[0],
                               atol=2e-10, rtol=2e-10)
    np.testing.assert_allclose(hvp, reference_hvp, atol=2e-9, rtol=2e-9)


def test_exhausted_forward_budget_precedes_diagonal_or_objective():
    _require_callback()

    def forbidden(*args):
        pytest.fail('A known exhausted budget reached the primal callback/objective')
    with pytest.raises(ValueError, match='memory|budget|resident'):
        minimize.solve_orbitals({}, jnp.zeros(DIMENSION), energy=forbidden,
            rotate=_rotate, hessian_diagonal=forbidden, dimension=DIMENSION,
            max_iterations=1, gradient_tolerance=TOLERANCE,
            max_memory=.001, reserved_bytes=4096)


def test_forward_preflight_charges_diagonal_graph_and_four_live_vectors(monkeypatch):
    _require_callback()
    diagonal, seed = _inputs(1.)
    args = dict(parameter=0., diagonal=diagonal, projection=jnp.eye(32))
    charges = []
    original = minimize.plan_linear

    def record(size, dtype, config, maximum, **kwargs):
        charges.append(kwargs['resident_bytes']+kwargs['operator_min_bytes'])
        assert kwargs['operator_min_bytes'] >= 4096
        return original(size, dtype, config, maximum, **kwargs)

    def projected_diagonal(base, args):
        # Model live Fock-projection temporaries without executing a solve.
        projection = args['projection'] @ args['projection']
        return args['diagonal']+1e-12*jnp.sum(projection)

    monkeypatch.setattr(minimize, 'plan_linear', record)
    controls = dict(energy=_energy, rotate=_rotate, dimension=DIMENSION,
        max_iterations=1, gradient_tolerance=TOLERANCE, max_memory=64.,
        reserved_bytes=2048, operator_min_bytes=4096)
    minimize.solve_orbitals.lower(args, seed, hessian_diagonal=None, **controls)
    minimize.solve_orbitals.lower(args, seed, hessian_diagonal=projected_diagonal, **controls)
    assert len(charges) == 2
    scratch = jaxpr_temporary_bytes(jax.make_jaxpr(projected_diagonal)(seed, args))
    assert scratch > 32*32*8
    # d, shifted/scaled D, right unknown y and recovered physical step coexist.
    assert charges[1] >= charges[0]+scratch+4*DIMENSION*8
