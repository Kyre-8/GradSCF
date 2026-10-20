"""Reuse a real symmetric factorization without eigenvector derivatives."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gradscf.solvers.linear.shifted import factor_shifted, solve_shifted


@pytest.mark.parametrize('shift', [.2, .2+.1j])
def test_degenerate_shifted_solve_response(shift):
    matrix = jnp.eye(3) * 1.7
    direction = jnp.array([[.2, .1, -.3], [.1, -.4, .2], [-.3, .2, .1]])
    rhs = jnp.arange(6., dtype=float).reshape(3, 2) / 5
    def solve(t, reference=False):
        a = matrix + t * direction
        b = rhs * (1+.1*t)
        z = shift + .03*t
        return (jnp.linalg.solve(a-z*jnp.eye(3), b) if reference else
                solve_shifted(factor_shifted(a), b, z))
    actual = jax.jit(solve)(0.)
    np.testing.assert_allclose(actual, solve(0., True), atol=2e-13)
    tangent = jax.jit(lambda t: jax.jvp(solve, (t,), (1.,))[1])(0.)
    expected = jax.jvp(lambda t: solve(t, True), (0.,), (1.,))[1]
    np.testing.assert_allclose(tangent, expected, atol=2e-12)
    loss = lambda t: jnp.real(jnp.vdot(solve(t), solve(t)))
    expected_loss = lambda t: jnp.real(jnp.vdot(solve(t, True), solve(t, True)))
    np.testing.assert_allclose(jax.jit(jax.grad(loss))(0.), jax.grad(expected_loss)(0.), atol=2e-12)
    np.testing.assert_allclose(jax.grad(loss)(0.), (loss(1e-5)-loss(-1e-5))/2e-5, atol=2e-9)


def test_degenerate_basis_rotation_and_multiple_shifts():
    matrix = jnp.eye(3)*2
    state = factor_shifted(matrix)
    q, _ = jnp.linalg.qr(jnp.array([[.1, .3, .7], [.8, .4, .2], [.6, .9, .5]]))
    rotated = state._replace(vectors=state.vectors @ q)
    rhs = jnp.array([1., -.2, .5])
    values = jax.jit(jax.vmap(lambda z: solve_shifted(rotated, rhs, z)))(jnp.array([.1, .5, 1.]))
    np.testing.assert_allclose(values, rhs[None, :]/(2-jnp.array([.1, .5, 1.]))[:,None], atol=2e-13)


@pytest.mark.parametrize('multiple_rhs', [False, True])
def test_cached_refinement_preserves_original_equation_and_response(multiple_rhs):
    """A slightly inaccurate cache must not invalidate a solvable response RHS."""
    matrix = jnp.array([[1.7+.2j, .13-.02j], [-.04+.08j, 2.1-.1j]])
    direction = jnp.array([[.02+.01j, -.03j], [.04-.02j, -.04+.02j]])
    rhs = jnp.array([[.2+.1j, -.1+.3j], [.7-.3j, .5+.2j]])
    if not multiple_rhs:
        rhs = rhs[:, 0]
    def solve(t, exact=False):
        a, b, z = matrix+t*direction, rhs*(1+.03*t), .2+.1j+.02*t
        if exact:
            return jnp.linalg.solve(a-z*jnp.eye(2), b)
        state = factor_shifted(a, symmetric=False)
        # Controlled eigenvalue error makes the raw cache fail the original
        # residual tolerance. A residual correction still solves the ORIGINAL
        # matrix; the cache is never promoted to a changed physical operator.
        state = state._replace(values=state.values+1e-8)
        return solve_shifted(state, b, z)
    actual = jax.jit(solve)(0.)
    np.testing.assert_allclose(actual, solve(0., True), atol=2e-13)
    np.testing.assert_allclose(jax.jvp(solve, (0.,), (1.,))[1],
        jax.jvp(lambda t: solve(t, True), (0.,), (1.,))[1], atol=2e-12)
    loss = lambda t: jnp.real(jnp.vdot(solve(t), solve(t)))
    exact_loss = lambda t: jnp.real(jnp.vdot(solve(t, True), solve(t, True)))
    np.testing.assert_allclose(jax.jit(jax.grad(loss))(0.),
                               jax.grad(exact_loss)(0.), atol=2e-12)


@pytest.mark.parametrize('defective', [False, True])
def test_complex_symmetric_shifted_response_and_defective_fallback(defective):
    # The second matrix has a nonzero nilpotent block and no complete eigenbasis.
    matrix = (jnp.array([[1j, 1.], [1., -1j]])+2*jnp.eye(2) if defective else
              jnp.array([[1.7+.2j, .13-.02j], [.13-.02j, 2.1-.1j]]))
    direction = jnp.array([[.02+.01j, -.03j], [-.03j, -.04+.02j]])
    rhs = jnp.array([.2+.1j, .7-.3j])
    if defective:
        assert not bool(factor_shifted(matrix).spectral_valid)
    def solve(t, reference=False):
        a, b, z = matrix+t*direction, rhs*(1+.03*t), .2+.1j+.02*t
        return (jnp.linalg.solve(a-z*jnp.eye(2), b) if reference else
                solve_shifted(factor_shifted(a), b, z))
    np.testing.assert_allclose(jax.jit(solve)(0.), solve(0., True), atol=2e-12)
    np.testing.assert_allclose(jax.jvp(solve, (0.,), (1.,))[1],
        jax.jvp(lambda t: solve(t, True), (0.,), (1.,))[1], atol=2e-12)
    loss = lambda t: jnp.real(jnp.vdot(solve(t), solve(t)))
    exact = lambda t: jnp.real(jnp.vdot(solve(t, True), solve(t, True)))
    np.testing.assert_allclose(jax.jit(jax.grad(loss))(0.), jax.grad(exact)(0.), atol=2e-12)


@pytest.mark.parametrize('case', ['real_general', 'complex_general', 'defective', 'degenerate'])
def test_general_shifted_equation_forward_jvp_and_nonconjugating_vjp(case):
    real = case == 'real_general'
    matrix = (jnp.array([[1.6, .7], [-.12, 2.1]]) if real else
              jnp.array([[1.6+.2j, .7-.1j], [-.12+.31j, 2.1-.3j]]))
    if case == 'defective':
        matrix = jnp.array([[1.7+.1j, .8+.2j], [0., 1.7+.1j]])
    elif case == 'degenerate':
        matrix = jnp.eye(2, dtype=jnp.complex128)*1.7
    direction = (jnp.array([[.1, .3], [-.2, -.04]]) if real else
                 jnp.array([[.02+.03j, .05-.07j], [-.04+.06j, -.03-.02j]]))
    rhs = jnp.array([.2, .7]) if real else jnp.array([.2+.1j, .7-.3j])
    shift = .2 if real else .2+.11j
    state = factor_shifted(matrix, symmetric=False)
    if case == 'defective':
        assert not bool(state.spectral_valid)
    def solve(t, exact=False):
        a, b, z = matrix+t*direction, rhs*(1+.03*t), shift+.02*t
        return (jnp.linalg.solve(a-z*jnp.eye(2), b) if exact else
                solve_shifted(factor_shifted(a, symmetric=False), b, z))
    np.testing.assert_allclose(jax.jit(solve)(0.), solve(0., True), atol=2e-12, rtol=0)
    np.testing.assert_allclose(jax.jvp(solve, (0.,), (1.,))[1],
        jax.jvp(lambda t: solve(t, True), (0.,), (1.,))[1], atol=3e-12, rtol=0)
    loss = lambda t: jnp.real(jnp.vdot(solve(t), solve(t)))
    expected = lambda t: jnp.real(jnp.vdot(solve(t, True), solve(t, True)))
    np.testing.assert_allclose(jax.jit(jax.grad(loss))(0.), jax.grad(expected)(0.), atol=3e-12, rtol=0)
    cotangent = jnp.array([.13, -.21]) if real else jnp.array([.13+.17j, -.21+.08j])
    value, actual_pullback = jax.vjp(lambda b: solve_shifted(state, b, shift), rhs)
    # General real eigencaches promote the operator solve to complex, even
    # when its unique solution is real. Compare VJPs with matching codomains.
    _, exact_pullback = jax.vjp(lambda b: jnp.linalg.solve(
        matrix.astype(value.dtype)-shift*jnp.eye(2), b), rhs)
    cotangent = cotangent.astype(value.dtype)
    np.testing.assert_allclose(actual_pullback(cotangent)[0], exact_pullback(cotangent)[0], atol=3e-12, rtol=0)


def test_shifted_solver_checks_actual_equation_not_only_cached_basis_flag():
    matrix = jnp.diag(jnp.array([1.3, 2.1]))
    state = factor_shifted(matrix)
    # A stale spectral cache cannot silently solve a different physical matrix.
    stale = state._replace(values=state.values+.15)
    answer = solve_shifted(stale, jnp.array([.2, .7]), .3)
    assert np.isnan(np.asarray(answer)).all()
