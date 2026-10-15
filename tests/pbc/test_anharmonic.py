"""Projected energy diagnostics against analytic potentials (CPU float64)."""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize('order', range(2, 7))
def test_polynomial_coefficients_and_derivative_factorials(order):
    from gradscf.pbc.anharmonic import fit_frozen_mode
    s = np.linspace(-.3, .4, 19)
    origin = .07
    expected = np.array([-3., .2, .8, -.3, .6, -.1, .2])[:order+1]
    energy = np.polynomial.polynomial.polyval(s-origin, expected)
    result = fit_frozen_mode(s, energy, order=order, origin_bohr=origin)
    np.testing.assert_allclose(result.coefficients, expected, rtol=2e-9, atol=1e-10)
    np.testing.assert_allclose(result.derivatives, expected*np.array([math.factorial(n) for n in range(order+1)]), rtol=2e-9, atol=1e-9)
    assert result.rank == order+1
    assert result.rms_residual < 1e-13
    assert result.scale_bohr == pytest.approx(.37)


def test_scaled_coordinates_small_displacements_and_even_opt_in():
    from gradscf.pbc.anharmonic import fit_frozen_mode
    s = np.linspace(-1e-4, 1e-4, 13)
    result = fit_frozen_mode(s, 2*s**2 + 3e4*s**3 + 5e7*s**4)
    np.testing.assert_allclose(result.coefficients[2:], [2., 3e4, 5e7], rtol=1e-10)
    assert result.condition_number < 30
    even = fit_frozen_mode(s, 2*s**2 + 5e7*s**4, even=True)
    np.testing.assert_array_equal(even.coefficients[jnp.array([1, 3])], [0., 0.])
    assert even.rank == 3


@pytest.mark.parametrize('s,e,kwargs,match', [
    ([0., .1], [0., .1], {}, 'samples'),
    ([0.]*6, [0.]*6, {}, 'span|rank'),
    ([-1., -1., 0., 0., 1., 1.], [0.]*6, {}, 'rank'),
    (np.linspace(-1., 1., 9), [0.]*9, {'max_condition': 2.}, 'condition'),
    (np.linspace(-1., 1., 9), [0., 0., 0., 0., 1., 0., 0., 0., 0.], {'max_residual_hartree': 1e-4}, 'residual'),
    ([0., .1, .2, .3, np.nan], [0.]*5, {}, 'finite'),
    ([0., .1, .2, .3, .4], [0., 0., np.inf, 0., 0.], {}, 'finite'),
    ([0., .1, .2, .3, .4], [0.]*5, {'order': True}, 'order'),
    ([0., .1, .2, .3, .4], [0.]*5, {'order': 7}, 'order'),
    ([0., .1, .2, .3, .4], [0.]*5, {'origin_bohr': np.nan}, 'finite'),
])
def test_invalid_fit_inputs(s, e, kwargs, match):
    from gradscf.pbc.anharmonic import fit_frozen_mode
    with pytest.raises(ValueError, match=match):
        fit_frozen_mode(s, e, **kwargs)


def test_fit_jit_parameter_gradient():
    from gradscf.pbc.anharmonic import fit_frozen_mode
    s = jnp.linspace(-.2, .2, 13)
    def loss(p):
        return fit_frozen_mode(s, -2.+p*s**2 + p**2*s**4).coefficients[4]
    np.testing.assert_allclose(jax.jit(jax.value_and_grad(loss))(1.3), [1.69, 2.6], rtol=1e-9)


@pytest.mark.parametrize('case', ['energy_offset', 'physical_coefficients', 'derivatives', 'residual_rms'])
def test_fit_rejects_nonfinite_outputs_from_finite_inputs(case):
    from gradscf.pbc.anharmonic import fit_frozen_mode
    s = np.linspace(-1., 1., 9)
    if case == 'energy_offset':
        energy = np.array([-1e308, 1e308, 0., 0., 0., 0., 0., 0., 0.])
    elif case == 'physical_coefficients':
        s = s*1e-100
        energy = np.linspace(-1., 1., 9)**4
    elif case == 'derivatives':
        s = s*1e-77
        energy = 1e307*s**4
    else:
        energy = np.array([0., 0., 0., 0., 1e200, 0., 0., 0., 0.])
    assert np.isfinite(s).all() and np.isfinite(energy).all()
    with pytest.raises(ValueError, match='finite'):
        fit_frozen_mode(s, energy)


def test_directional_jet_nonzero_origin_parameters_and_finite_difference():
    from gradscf.pbc.anharmonic import directional_energy_taylor
    x = jnp.array([.2, -.3])
    direction = jnp.array([.4, .7])
    def energy(y, p):
        return p*y[0]**4 + y[0]*y[1]**2
    p = 1.7
    result = directional_energy_taylor(energy, x, direction, order=4, args=(p,))
    a = np.polynomial.Polynomial([.2, .4])
    b = np.polynomial.Polynomial([-.3, .7])
    expected = (p*a**4+a*b**2).coef
    np.testing.assert_allclose(result.coefficients, expected, atol=1e-13)
    np.testing.assert_allclose(result.derivatives, expected*np.array([1, 1, 2, 6, 24]), atol=1e-12)
    h = 1e-4
    fd2 = (energy(x+h*direction,p)-2*energy(x,p)+energy(x-h*direction,p))/h**2
    np.testing.assert_allclose(result.derivatives[2], fd2, atol=1e-8)
    derivative = jax.jit(jax.grad(lambda parameter: directional_energy_taylor(energy, x, direction, order=4, args=(parameter,)).coefficients[4]))(p)
    np.testing.assert_allclose(derivative, .4**4, atol=1e-13)


def test_jet_shape_validation_and_explicit_unsupported_primitive():
    from gradscf.pbc.anharmonic import directional_energy_taylor
    with pytest.raises(ValueError, match='shape'):
        directional_energy_taylor(lambda x: jnp.sum(x), jnp.ones(2), jnp.ones(3))
    with pytest.raises(ValueError, match='scalar'):
        directional_energy_taylor(lambda x: x*x, jnp.ones(2), jnp.ones(2))
    def callback(x):
        return jax.pure_callback(lambda v: np.asarray(v*v), jax.ShapeDtypeStruct((), x.dtype), x)
    with pytest.raises((KeyError, NotImplementedError), match='pure_callback|Taylor|jet'):
        directional_energy_taylor(callback, jnp.array(.2), jnp.array(1.))


def test_directional_jet_rejects_factorial_derivative_overflow():
    from gradscf.pbc.anharmonic import directional_energy_taylor
    with pytest.raises(ValueError, match='finite'):
        directional_energy_taylor(lambda x: 1e307*x**4, jnp.array(0.), jnp.array(1.))
