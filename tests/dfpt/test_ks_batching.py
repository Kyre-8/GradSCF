"""Block coordinates retain their order without unrolling every spin/k block."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.dfpt import ks_problem
from gradscf.dfpt.ks import _Hermitian
from gradscf.solvers import LinearSolverConfig


def problem(nocc, complex_values=False, occupation=1.):
    dtype = complex if complex_values else float
    eye = jnp.broadcast_to(jnp.eye(2, dtype=dtype), (len(nocc), 2, 2))
    h = jnp.broadcast_to(jnp.diag(jnp.array([-1., 1.], dtype=dtype)), eye.shape)
    return ks_problem(eye, nocc, {'h': h, 's': eye}, lambda d, a: a['h'],
        lambda d, a: jnp.einsum('bij,bji->', d, a['h']).real,
        overlap=lambda a: a['s'], occupation=occupation)


def test_ks_rejects_empty_blocks():
    with pytest.raises(ValueError, match='complete square orbital blocks'):
        problem(())


@pytest.mark.parametrize('complex_values', [False, True])
def test_hermitian_coordinate_order(complex_values):
    matrix = np.array([[1., 2., 3.], [2., 4., 7.], [3., 7., 9.]])
    expected = [1., 4., 9., 2., 3., 7.]
    if complex_values:
        matrix = matrix + 1j*np.array([[0., 5., 6.], [-5., 0., 8.], [-6., -8., 0.]])
        expected += [5., 6., 8.]
    chart = _Hermitian(3, complex_values)
    matrices = jnp.asarray(np.stack([matrix, 2*matrix]))
    packed = jax.jit(jax.vmap(chart.pack))(matrices)
    np.testing.assert_array_equal(packed, np.array([expected, 2*np.array(expected)]))
    np.testing.assert_array_equal(jax.jit(jax.vmap(chart.unpack))(packed), matrices)


@pytest.mark.parametrize('complex_values', [False, True])
def test_residual_graph_does_not_unroll_spin_k_blocks(complex_values):
    counts = []
    for blocks in (2, 32):
        p = problem((1,)*blocks, complex_values)
        residual = jax.make_jaxpr(p.residual)(p.solution, p.inputs)
        def derivative(x, args, v):
            return jax.jvp(lambda y: p.residual(y, args), (x,), (v,))[1]
        response = jax.make_jaxpr(derivative)(p.solution, p.inputs, jnp.ones_like(p.solution))
        counts.append((len(residual.jaxpr.eqns), len(response.jaxpr.eqns)))
    print('KS block graph sizes (residual, JVP):', counts)
    # A block axis changes array shapes, not the number of coordinate operations.
    assert all(large <= small+30 for small, large in zip(*counts)), counts


@pytest.mark.parametrize('complex_values', [False, True])
def test_mixed_occupation_residual_and_ad_match_matrix_equations(complex_values):
    p = problem((0, 1, 2), complex_values)
    size = 4 if complex_values else 3
    expected_initial = np.zeros((3, size))
    expected_initial[1, 0] = 1.
    expected_initial[2, :2] = 1.
    np.testing.assert_array_equal(p.solution, expected_initial.ravel())
    rng = np.random.default_rng(61)
    x = np.asarray(p.solution)+.15*rng.normal(size=p.solution.shape)
    v = rng.normal(size=x.shape)
    h = np.asarray(p.inputs['h']) + .1*np.ones((3, 2, 2))
    s = np.asarray(p.inputs['s']) + .05*np.ones((3, 2, 2))
    args = {'h': jnp.asarray(h), 's': jnp.asarray(s)}

    def reference(vector):
        result = []
        for block, (values, no) in enumerate(zip(vector.reshape(3, size), (0, 1, 2))):
            upper = values[2]+(1j*values[3] if complex_values else 0.)
            q = np.array([[values[0], upper], [np.conj(upper), values[1]]])
            idem = q@s[block]@q-q
            comm = h[block]@q@s[block]-s[block]@q@h[block]
            upper_equation = np.conj(comm[1, 0]) if no == 1 else idem[0, 1]
            packed = [idem[0, 0].real, idem[1, 1].real, upper_equation.real]
            if complex_values:
                packed.append(upper_equation.imag)
            result.extend(packed)
        return np.array(result)

    evaluate = jax.jit(lambda y: p.residual(y, args))
    np.testing.assert_allclose(evaluate(x), reference(x), atol=2e-14)
    tangent = jax.jvp(evaluate, (jnp.asarray(x),), (jnp.asarray(v),))[1]
    finite = (reference(x+1e-4*v)-reference(x-1e-4*v))/(2e-4)
    np.testing.assert_allclose(tangent, finite, atol=2e-11)
    weights = jnp.arange(x.size, dtype=float)
    gradient = jax.grad(lambda y: jnp.vdot(weights, evaluate(y)))(jnp.asarray(x))
    np.testing.assert_allclose(jnp.vdot(gradient, v), jnp.vdot(weights, tangent), atol=2e-13)


@pytest.mark.parametrize('complex_values', [False, True])
@pytest.mark.parametrize('occupation', [1., 2.])
def test_mixed_occupation_response_and_taylor(complex_values, occupation):
    p = problem((0, 1, 2), complex_values, occupation)
    z = .3+.4j if complex_values else .5
    b = jnp.asarray([[0., z], [np.conj(z), 0.]])
    perturbation = jnp.broadcast_to(b, p.inputs['h'].shape)
    direction = {'h': perturbation, 's': jnp.zeros_like(p.inputs['s'])}
    config = LinearSolverConfig(method='direct', rtol=1e-11, atol=1e-12, max_dense=64)
    response = jax.jit(lambda: p.response(direction, config=config))()
    series = jax.jit(lambda: p.series(
        lambda t: {'h': p.inputs['h']+t*perturbation, 's': p.inputs['s']},
        order=3, config=config))()
    assert response.converged & series.converged
    # Exact lower projector of [[-1,t*z],[t*z*,1]] is (I-H/sqrt(1+|z*t|²))/2.
    expected = np.zeros((3, 3, 2, 2), dtype=complex if complex_values else float)
    expected[0, 1] = -.5*occupation*np.asarray(b)
    expected[1, 1] = .25*occupation*abs(z)**2*np.diag([-1., 1.])
    expected[2, 1] = .25*occupation*abs(z)**2*np.asarray(b)
    np.testing.assert_allclose(response.density, expected[0], atol=2e-12)
    np.testing.assert_allclose(series.density, expected, atol=2e-12)
    np.testing.assert_allclose(series.energy, [0., -.5*occupation*abs(z)**2, 0.], atol=2e-12)
