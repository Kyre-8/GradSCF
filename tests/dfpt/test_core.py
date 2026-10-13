import importlib.util
import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_response_and_normalized_taylor_coefficients():
    assert importlib.util.find_spec('gradscf.dfpt') is not None, 'DFPT module is missing'
    from gradscf.dfpt import Problem
    p = Problem(lambda x, a: x*x-a, jnp.array([1.]), jnp.array([1.]),
                lambda x, a: {'density': x, 'energy': jnp.sum(x**3)})
    result = jax.jit(lambda: p.response(jnp.ones(1)))()
    assert result.converged
    np.testing.assert_allclose(result.density, [.5], atol=1e-12)
    np.testing.assert_allclose(result.energy, 1.5, atol=1e-12)
    series = p.series(lambda t: p.inputs+t, order=3)
    assert series.converged
    np.testing.assert_allclose(series.state[:, 0], [.5, -.125, .0625], atol=1e-11)
    np.testing.assert_allclose(series.energy, [1.5, .375, -.0625], atol=1e-11)


def test_ks_screening_and_occupied_degeneracy():
    from gradscf.dfpt import ks_problem
    # Two degenerate occupied orbitals; response must not divide by their gap.
    h = jnp.diag(jnp.array([-1., -1., 1.]))
    b = jnp.array([[0., 0., 1.], [0., 0., 0.], [1., 0., 0.]])
    def run(u):
        args = {'h': h[None], 's': jnp.eye(3)[None], 'u': u}
        def fock(d, a):
            return a['h'] + a['u'] * jnp.sum(b*d[0]) * b[None]
        def energy(d, a):
            return jnp.sum(a['h']*d) + .5*a['u']*jnp.sum(b*d[0])**2
        p = ks_problem(jnp.eye(3)[None], (2,), args, fock, energy,
                       overlap=lambda a: a['s'], occupation=2.)
        direction = {'h': b[None], 's': jnp.zeros((1, 3, 3)), 'u': 0.}
        result = p.response(direction)
        return result, p, direction
    result, p, direction = run(.3)
    assert result.converged
    np.testing.assert_allclose(result.density[0, 0, 2], -2/(2+4*.3), atol=1e-11)
    np.testing.assert_allclose(jnp.trace(result.density[0]), 0., atol=1e-12)
    np.testing.assert_allclose(jax.grad(lambda u: run(u)[0].density[0, 0, 2])(.3),
                                8/(2+4*.3)**2, atol=1e-9)
    series = p.series(lambda t: jax.tree.map(lambda a, v: a+t*v, p.inputs, direction), order=2)
    assert series.converged
    np.testing.assert_allclose(2*series.energy[1], -4/(2+4*.3), atol=1e-10)


def test_invalid_base_is_not_a_response():
    from gradscf.dfpt import Problem
    p = Problem(lambda x, a: x*x-a, jnp.array([2.]), jnp.array([1.]), lambda x,a: {'density':x})
    out = p.response(jnp.ones(1))
    assert not out.converged
    assert np.isnan(out.density).all()


def test_nested_observables_nonlinear_path_and_invalid_path():
    from gradscf.dfpt import Problem
    p = Problem(lambda x, a: x*x-a, jnp.ones(1), jnp.ones(1),
                lambda x, a: {'nested': (x, {'explicit': a})})
    result = jax.jit(lambda: p.series(lambda t: p.inputs+2*t+t*t, order=3))()
    assert result.converged
    np.testing.assert_allclose(result.state[:, 0], [1., 0., 0.], atol=1e-11)
    np.testing.assert_allclose(result.values['nested'][1]['explicit'][:, 0], [2., 1., 0.], atol=1e-11)
    invalid = p.series(lambda t: p.inputs+1+t, order=2)
    assert not invalid.converged
    assert np.isnan(invalid.values['nested'][1]['explicit']).all()
    with pytest.raises(ValueError, match='positive integer'):
        p.series(lambda t: p.inputs+t, order=0)


@pytest.mark.parametrize('complex_coefficients', [False, True])
def test_ks_metric_response_and_complex_perturbation(complex_coefficients):
    from gradscf.dfpt import ks_problem
    h = jnp.diag(jnp.array([-1., 1.])).astype(complex)[None]
    args = {'h': h, 's': jnp.eye(2, dtype=complex)[None]}
    p = ks_problem(jnp.eye(2, dtype=complex if complex_coefficients else float)[None], (1,), args,
                   lambda d,a:a['h'], lambda d,a:jnp.einsum('bij,bji->', d, a['h']).real,
                   overlap=lambda a:a['s'], occupation=2.)
    b = jnp.array([[0., 1j], [-1j, 0.]])[None]
    response = p.response({'h': b, 's': jnp.zeros_like(h)})
    assert response.converged
    np.testing.assert_allclose(response.density, -b, atol=1e-12)
    s1 = jnp.array([[.2, .1], [.1, -.1]], dtype=complex)[None]
    metric = p.response({'h': jnp.zeros_like(h), 's': s1})
    assert metric.converged
    np.testing.assert_allclose(metric.values['electron_number'], 0., atol=1e-12)
    np.testing.assert_allclose(metric.energy, .4, atol=1e-12)


def test_ks_rejects_occupied_virtual_degeneracy():
    from gradscf.dfpt import ks_problem
    p = ks_problem(jnp.eye(2)[None], (1,), jnp.zeros((1,2,2)),
                   lambda d,a:a, lambda d,a:jnp.sum(d*a), overlap=lambda a:jnp.eye(2)[None])
    response = p.response(jnp.zeros((1,2,2)))
    assert not response.converged
    assert np.isnan(response.density).all()
