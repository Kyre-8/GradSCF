"""Higher spectral response: closed-form isolated roots and degenerate clusters."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.solvers import EigenSolverConfig, EigenResponseConfig, LinearSolverConfig, solve_hermitian


@pytest.mark.parametrize("order", [2, 3, 4])
def test_isolated_energy_higher_derivatives_match_two_by_two_formula(order):
    cfg = EigenSolverConfig(method="dense", nroots=1, atol=1e-12)
    policy = EigenResponseConfig(linear_config=LinearSolverConfig(method="direct", rtol=1e-12, atol=1e-14))

    def energy(t):
        a = jnp.array([[-1.+.2*t, .6*t], [.6*t, 2.-.1*t]])
        return solve_hermitian(a, config=cfg, response=policy).values[0]

    def oracle(t):
        return .5*(1.+.1*t-jnp.sqrt((-3.+.3*t)**2+4*(.6*t)**2))

    actual, expected = energy, oracle
    for _ in range(order):
        actual, expected = jax.grad(actual), jax.grad(expected)
    np.testing.assert_allclose(jax.jit(actual)(.13), expected(.13), atol=3e-10, rtol=3e-9)


@pytest.mark.parametrize("method", ["dense", "davidson"])
def test_degenerate_cluster_projector_and_energy_sum_hessian(method):
    base = jnp.diag(jnp.array([1., 1., 3., 5.]))
    perturbation = jnp.array([[.2, .6, .4, .1], [.6, -.3, .2, .3],
                              [.4, .2, .1, -.2], [.1, .3, -.2, .4]])
    probe = jnp.array([.3, .4, .5, .2])
    policy = EigenResponseConfig(target="subspace",
        linear_config=LinearSolverConfig(method="direct", rtol=1e-12, atol=1e-14))
    cfg = EigenSolverConfig(method=method, nroots=2, atol=1e-11, max_subspace=4)

    def observable(t):
        result = solve_hermitian(base+t*perturbation, config=cfg, response=policy, probes=probe)
        return probe@result.projection + .3*result.eigenvalue_sum

    def dense_value(t):
        values, vectors = np.linalg.eigh(np.asarray(base+t*perturbation))
        return np.sum((vectors[:, :2].T @ np.asarray(probe))**2) + .3*values[:2].sum()

    step = 3e-4
    expected = (dense_value(step)-2*dense_value(0.)+dense_value(-step))/step**2
    actual = jax.jit(jax.grad(jax.grad(observable)))(0.)
    np.testing.assert_allclose(actual, expected, atol=8e-8, rtol=1e-5)


def test_rank_one_eigenvector_normalization_through_second_order():
    policy = EigenResponseConfig(target="eigenpairs",
        linear_config=LinearSolverConfig(method="direct", rtol=1e-12, atol=1e-14))
    cfg = EigenSolverConfig(method="dense", nroots=1, atol=1e-12)

    def vector(t):
        return solve_hermitian(jnp.array([[1., t], [t, 3.]]), config=cfg, response=policy).vectors[:, 0]

    norm = lambda t: jnp.sum(vector(t)**2)
    np.testing.assert_allclose(jax.jit(jax.grad(jax.grad(norm)))(0.), 0., atol=1e-11)


@pytest.mark.parametrize("order", [2, 3, 4])
def test_exactly_degenerate_cluster_higher_derivatives(order):
    cfg = EigenSolverConfig(method="dense", nroots=2, atol=1e-12)
    policy = EigenResponseConfig(target="subspace",
        linear_config=LinearSolverConfig(method="direct", rtol=1e-12, atol=1e-14))
    probe = jnp.array([.3, .4, .5])

    def observable(t):
        a = jnp.array([[1., 0., t], [0., 1., 0.], [t, 0., 3.]])
        result = solve_hermitian(a, config=cfg, response=policy, probes=probe)
        return probe@result.projection + .3*result.eigenvalue_sum

    def oracle(t):
        radius = jnp.sqrt(1+t*t)
        return (.5*(probe[0]**2+probe[2]**2)
                + .5*(probe[0]**2-probe[2]**2)/radius
                - probe[0]*probe[2]*t/radius + probe[1]**2 + .3*(3-radius))

    actual, expected = observable, oracle
    for _ in range(order):
        actual, expected = jax.grad(actual), jax.grad(expected)
    np.testing.assert_allclose(jax.jit(actual)(0.), expected(0.), atol=5e-10, rtol=1e-8)


def test_multiple_isolated_roots_retain_third_response_when_batched():
    cfg = EigenSolverConfig(method="dense", nroots=2, atol=1e-12)
    policy = EigenResponseConfig(linear_config=LinearSolverConfig(method="direct", rtol=1e-12, atol=1e-14))
    base = jnp.diag(jnp.array([-2., -.5, 1., 3.]))
    perturbation = jnp.array([[.2, .3, .1, -.1], [.3, -.1, .2, .4],
                              [.1, .2, .1, .1], [-.1, .4, .1, -.2]])
    weights = jnp.array([.7, -.3])
    actual = lambda t: weights@solve_hermitian(base+t*perturbation, config=cfg, response=policy).values
    expected = lambda t: weights@jnp.linalg.eigvalsh(base+t*perturbation)[:2]
    for _ in range(3):
        actual, expected = jax.grad(actual), jax.grad(expected)
    np.testing.assert_allclose(jax.jit(actual)(.17), expected(.17), atol=5e-10, rtol=1e-8)


@pytest.mark.parametrize("nroots", [1, 4])
def test_energy_forward_and_first_derivative_do_not_require_response_capacity(nroots):
    """Only the second energy derivative requires the four-dimensional solve."""
    cfg = EigenSolverConfig(method="dense", nroots=nroots, atol=1e-12)
    policy = EigenResponseConfig(linear_config=LinearSolverConfig(
        method="direct", max_dense=2, rtol=1e-12, atol=1e-14))

    def energy(t):
        a = jnp.diag(jnp.array([1., 2., 3., 4.]))
        a = a.at[0, 1].set(t).at[1, 0].set(t)
        return solve_hermitian(a, config=cfg, response=policy).values[0]

    t = .1
    expected = 1.5-jnp.sqrt(.25+t*t)
    expected_first = -t/jnp.sqrt(.25+t*t)
    np.testing.assert_allclose(energy(t), expected, atol=1e-12)
    np.testing.assert_allclose(jax.jit(energy)(t), expected, atol=1e-12)
    np.testing.assert_allclose(jax.jit(jax.grad(energy))(t), expected_first, atol=1e-12)
    with pytest.raises(ValueError, match="max_dense"):
        jax.jit(jax.grad(jax.grad(energy)))(t)


def test_mixed_parameter_gmres_energy_hessian_matches_dense_oracle():
    cfg = EigenSolverConfig(method="dense", nroots=2, atol=1e-12)
    policy = EigenResponseConfig(linear_config=LinearSolverConfig(
        method="gmres", rtol=1e-11, atol=1e-13, restart=4, maxiter=10))
    base = jnp.diag(jnp.array([-2., -.5, 1., 3.]))
    b = jnp.array([[.2, .3, .1, -.1], [.3, -.1, .2, .4],
                   [.1, .2, .1, .1], [-.1, .4, .1, -.2]])
    c = jnp.array([[.1, -.2, .2, .05], [-.2, .3, .1, .1],
                   [.2, .1, -.1, .3], [.05, .1, .3, .1]])
    weights = jnp.array([.7, -.3])

    def matrix(x):
        return base+x[0]*b+x[1]*c+x[0]*x[1]*jnp.diag(jnp.array([.1, .2, -.1, .3]))

    actual = lambda x: weights@solve_hermitian(matrix(x), config=cfg, response=policy).values
    expected = lambda x: weights@jnp.linalg.eigvalsh(matrix(x))[:2]
    point = jnp.array([.17, -.11])
    hessian = jax.jit(jax.hessian(actual))(point)
    np.testing.assert_allclose(hessian, jax.hessian(expected)(point), atol=3e-10, rtol=3e-9)
    np.testing.assert_allclose(hessian, hessian.T, atol=1e-11)


def test_invalid_response_flag_remains_authoritative_for_structural_higher_zeros():
    """A zero inactive higher tangent cannot certify an invalid response."""
    cfg = EigenSolverConfig(method="dense", nroots=2, atol=1e-12)
    policy = EigenResponseConfig(target="subspace",
        linear_config=LinearSolverConfig(method="direct", rtol=1e-12, atol=1e-14))

    def result(t):
        a = jnp.array([[1.+t*t, .3], [0., 2.]])  # Deliberately non-Hermitian.
        return solve_hermitian(a, config=cfg, response=policy)

    assert not bool(result(.1).response_valid)
    first = jax.jit(jax.grad(lambda t: result(t).eigenvalue_sum))(.1)
    assert bool(jnp.isnan(first))
    # Do not require every higher derivative to be NaN: AD may eliminate a
    # structurally inactive path. response_valid is required at every order.


@pytest.mark.parametrize("inner_jit", [False, True])
@pytest.mark.parametrize("nroots", [1, 4])
def test_mixed_third_derivative_of_batched_eigenvectors(inner_jit, nroots):
    """A canonical-orbital observable crosses an inner polarizability JIT."""
    cfg = EigenSolverConfig(method="dense", nroots=nroots, atol=1e-12)
    policy = EigenResponseConfig(target="eigenpairs",
        # Capacity is four per root, even for the four-root (4,4) chart.
        linear_config=LinearSolverConfig(method="direct", max_dense=4, rtol=1e-12, atol=1e-14))
    base = jnp.diag(jnp.array([-2., -.5, 1., 3.]))
    b = jnp.array([[.2, .3, .1, -.1], [.3, -.1, .2, .4],
                   [.1, .2, .1, .1], [-.1, .4, .1, -.2]])
    c = jnp.array([[.1, -.2, .2, .05], [-.2, .3, .1, .1],
                   [.2, .1, -.1, .3], [.05, .1, .3, .1]])
    probe, weights = jnp.array([.3, .4, .5, .2]), jnp.array([.7, -.3, .2, .5])

    def observable(r, field, native):
        matrix = base+r*b+field*c+.2*r*field*b
        vectors = (solve_hermitian(matrix, config=cfg, response=policy).vectors
                   if native else jnp.linalg.eigh(matrix)[1])
        return weights[:nroots] @ (probe @ vectors[:, :nroots])**2

    def derivative(native):
        energy = lambda r, f: observable(r, f, native)
        response = lambda r: jax.grad(jax.grad(energy, 1), 1)(r, -.11)
        return jax.jit(jax.grad(jax.jit(response) if inner_jit else response))(.17)

    np.testing.assert_allclose(derivative(True), derivative(False), atol=5e-10, rtol=1e-8)
