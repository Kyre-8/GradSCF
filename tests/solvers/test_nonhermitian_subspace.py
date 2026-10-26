"""Invariant real non-Hermitian cluster response, including second order."""
from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.solvers import (EigenResponseConfig, LinearOperator, LinearSolverConfig,
                            NonHermitianSolverConfig, solve_nonhermitian)

LINEAR = LinearSolverConfig(method='direct', rtol=1e-11, atol=1e-12, max_dense=32)
RESPONSE = EigenResponseConfig(target='subspace', linear_config=LINEAR)


def model(t):
    base = jnp.array([[1., .2, .5, -.1], [0., 1., -.3, .2],
                      [.1, 0., 1., .2], [0., .1, 0., 1.]])
    k = jnp.array([[.1, .3, .5, -.2], [-.2, .1, -.3, .4],
                   [.2, -.4, -.1, .2], [.3, .1, -.2, .1]])
    s = base @ (jnp.eye(4)+t*k+.2*t*t*(k@k))
    values = jnp.array([-1.-.3*t+.1*t*t, -1.+.4*t+.2*t*t, 1.+.2*t, 2.])
    inverse = jnp.linalg.inv(s)
    return s @ jnp.diag(values) @ inverse, s[:, :2] @ inverse[:2], values[:2].sum()


def result(t, method='dense', **kwargs):
    a, _, _ = model(t)
    config = NonHermitianSolverConfig(nroots=2, method=method, max_space=4,
                                       atol=1e-10, max_dense=8)
    op = LinearOperator(a.shape, a.dtype, lambda x: a@x,
                        transpose_matvec=lambda x: a.T@x, diagonal=jnp.diag(a))
    return solve_nonhermitian(op, config=config, response=RESPONSE,
                              probes=jnp.eye(4), **kwargs)


@pytest.mark.parametrize('method', ['dense', 'davidson'])
def test_cluster_invariants_jit_jvp_vjp_and_hvp(method):
    def observables(t):
        solved = result(t, method)
        return jnp.concatenate((solved.projection.reshape(-1), solved.eigenvalue_sum[None]))
    def exact(t):
        _, projection, trace = model(t)
        return jnp.concatenate((projection.reshape(-1), trace[None]))
    value = jax.jit(lambda t: result(t, method))(0.)
    assert bool(value.response_valid)
    assert value.values is None and value.right_vectors is None and value.left_vectors is None
    np.testing.assert_allclose(observables(0.), exact(0.), atol=3e-10)
    actual = jax.jit(lambda t: jax.jvp(observables, (t,), (jnp.array(1.),))[1])(0.)
    expected = jax.jvp(exact, (0.,), (1.,))[1]
    np.testing.assert_allclose(actual, expected, atol=4e-9)
    weights = jnp.arange(17, dtype=jnp.float64)/17.
    loss = lambda t: jnp.dot(weights, observables(t))
    exact_loss = lambda t: jnp.dot(weights, exact(t))
    np.testing.assert_allclose(jax.jit(jax.grad(loss))(0.), jax.grad(exact_loss)(0.), atol=4e-9)
    curvature = jax.jit(jax.grad(jax.grad(loss)))(0.)
    np.testing.assert_allclose(curvature, jax.grad(jax.grad(exact_loss))(0.), atol=2e-8)
    for step in (2e-3, 1e-3):
        finite = (loss(step)-2*loss(0.)+loss(-step))/(step*step)
        np.testing.assert_allclose(curvature, finite, atol=2e-6)


def test_cluster_gmres_is_one_coupled_system():
    response = replace(RESPONSE, linear_config=LinearSolverConfig(
        rtol=1e-11, atol=1e-12, restart=12, maxiter=4))
    a, projection, _ = model(0.)
    solved = solve_nonhermitian(a, config=NonHermitianSolverConfig(nroots=2),
                                response=response, probes=jnp.eye(4))
    np.testing.assert_allclose(solved.projection, projection, atol=2e-10)
    gradient = jax.grad(lambda t: solve_nonhermitian(model(t)[0],
        config=NonHermitianSolverConfig(nroots=2), response=response,
        probes=jnp.eye(4)).projection[2, 0])(0.)
    np.testing.assert_allclose(gradient, jax.grad(lambda t: model(t)[1][2, 0])(0.), atol=2e-8)


@pytest.mark.parametrize('matrix, roots', [
    (jnp.diag(jnp.array([-1., -1., 2.])), 1),
    (jnp.array([[-1., 1., 0.], [0., -1., 0.], [0., 0., 2.]]), 2),
    (jnp.array([[-1., -1., 0.], [1., -1., 0.], [0., 0., 2.]]), 2),
])
def test_invalid_cluster_guard(matrix, roots):
    def objective(t):
        value = solve_nonhermitian(matrix+t*jnp.eye(3),
            config=NonHermitianSolverConfig(nroots=roots), response=RESPONSE)
        return value.eigenvalue_sum
    solved = solve_nonhermitian(matrix, config=NonHermitianSolverConfig(nroots=roots),
                                response=RESPONSE)
    assert not bool(solved.response_valid)
    assert not np.isfinite(jax.jit(jax.grad(objective))(0.))


def test_fullspan_jordan_trace_and_projection_bypass_root(monkeypatch):
    import gradscf.solvers.eigen._nonhermitian_subspace as module
    def forbidden(*args, **kwargs):
        raise AssertionError('Fullspan must not attach an eigenvector root')
    monkeypatch.setattr(module, 'attach_root', forbidden)
    probe = jnp.arange(6., dtype=jnp.float64).reshape(3, 2)
    def solve(t):
        a = jnp.array([[t*t, 1., 0.], [0., t*t, 1.], [0., 0., t*t]])
        op = LinearOperator((3, 3), a.dtype, lambda x: a@x,
                            transpose_matvec=lambda x: a.T@x, diagonal=jnp.zeros(3))
        return solve_nonhermitian(op, config=NonHermitianSolverConfig(
            nroots=3, method='davidson', max_space=3, max_dense=1),
            response=RESPONSE, probes=probe, max_memory=1.)
    solved = jax.jit(solve)(.2)
    assert bool(solved.response_valid)
    np.testing.assert_allclose(solved.projection, probe, atol=0.)
    np.testing.assert_allclose(solved.eigenvalue_sum, .12, atol=1e-12)
    np.testing.assert_allclose(jax.jit(jax.grad(jax.grad(lambda t: solve(t).eigenvalue_sum)))(0.), 6., atol=1e-12)


def test_subspace_explicit_api_conflicts_and_coupled_capacity():
    a = model(0.)[0]
    cfg = NonHermitianSolverConfig(nroots=2)
    with pytest.raises(ValueError, match='legacy|controls'):
        solve_nonhermitian(a, config=cfg, response=RESPONSE, vector_response=True)
    with pytest.raises(ValueError, match='subspace'):
        solve_nonhermitian(a, config=cfg, probes=jnp.ones(4))
    with pytest.raises(ValueError, match='max_dense|coupled'):
        solve_nonhermitian(a, config=cfg,
            response=replace(RESPONSE, linear_config=replace(LINEAR, max_dense=11)))
    with pytest.raises(ValueError, match='memory|budget'):
        solve_nonhermitian(a, config=cfg, response=RESPONSE, max_memory=.0001)


def test_cluster_general_basis_change_and_probe_response(monkeypatch):
    import gradscf.solvers.eigen.nonhermitian as module
    original = module.paired_eigenvectors
    change = jnp.array([[1.4, .5], [-.2, .7]])
    def rotated(matrix, count):
        values, right, left = original(matrix, count)
        return values, right@change, left@jnp.linalg.inv(change).T
    before = jax.jit(jax.grad(jax.grad(lambda t: result(t).projection[2, 0])))(0.)
    monkeypatch.setattr(module, 'paired_eigenvectors', rotated)
    after = jax.jit(jax.grad(jax.grad(lambda t: result(t).projection[2, 0])))(0.)
    np.testing.assert_allclose(after, before, atol=2e-8)
    a, exact, _ = model(0.)
    cfg = NonHermitianSolverConfig(nroots=2)
    apply = lambda p: solve_nonhermitian(a, config=cfg, response=RESPONSE, probes=p).projection
    probe = jnp.arange(4., dtype=jnp.float64)
    np.testing.assert_allclose(jax.jit(jax.jacfwd(apply))(probe), exact, atol=3e-10)
    np.testing.assert_allclose(jax.jit(jax.jacrev(apply))(probe), exact, atol=3e-10)


def test_tangent_failure_and_sum_only_right_root(monkeypatch):
    import gradscf.solvers.eigen._nonhermitian_subspace as module
    response = replace(RESPONSE, linear_config=LinearSolverConfig(
        rtol=1e-14, restart=1, maxiter=1))
    value = jax.jit(jax.grad(lambda t: solve_nonhermitian(model(t)[0],
        config=NonHermitianSolverConfig(nroots=2), response=response,
        probes=jnp.eye(4)).projection[2, 0]))(0.)
    assert not np.isfinite(value)
    calls = []
    original = module.attach_root
    def counted(*args, **kwargs):
        calls.append(args[1].shape)
        return original(*args, **kwargs)
    monkeypatch.setattr(module, 'attach_root', counted)
    solved = solve_nonhermitian(model(0.)[0], config=NonHermitianSolverConfig(nroots=2),
                                response=RESPONSE, max_memory=2.)
    assert solved.projection is None and bool(solved.response_valid)
    assert calls == [(12,)]


def test_response_budget_retains_opaque_owners_before_attach(monkeypatch):
    import gradscf.solvers.eigen._nonhermitian_subspace as module
    def forbidden(*args, **kwargs):
        raise AssertionError('Joint memory rejection must precede root attachment')
    monkeypatch.setattr(module, 'attach_root', forbidden)
    with pytest.raises(ValueError, match='memory|budget'):
        solve_nonhermitian(model(0.)[0], config=NonHermitianSolverConfig(nroots=2),
            response=RESPONSE, probes=jnp.eye(4), _response_budget=dict(
                max_memory=.1, reserved_bytes=40_000, operator_min_bytes=40_000,
                operator_bytes_per_column=0, action_columns=2))


def test_float32_trace_preserves_dtype():
    a = jnp.diag(jnp.array([-1., -1., 2.], dtype=jnp.float32))
    solved = solve_nonhermitian(a, config=NonHermitianSolverConfig(nroots=2, atol=1e-5),
                                response=RESPONSE)
    assert solved.eigenvalue_sum.dtype == jnp.float32


def test_original_probe_owner_counted_before_promotion(monkeypatch):
    import gradscf.solvers.eigen.nonhermitian as module
    seen = []
    def stop(*args, **kwargs):
        seen.append(kwargs['resident_bytes'])
        raise RuntimeError('captured primal budget')
    monkeypatch.setattr(module, 'plan_nonhermitian', stop)
    probe = np.ones((4, 2), dtype=np.float32)
    with pytest.raises(RuntimeError, match='captured primal budget'):
        solve_nonhermitian(model(0.)[0], config=NonHermitianSolverConfig(nroots=2),
            response=RESPONSE, probes=probe, max_memory=2., reserved_bytes=100)
    # Matrix + diagonal + original float32 probe + promoted float64 probe.
    assert seen == [100+(16+4+8)*8+8*4]


def test_ill_conditioned_selected_projector_rejects_response():
    a = jnp.array([[-1., 0., 1e6], [0., -1., 0.], [0., 0., 2.]])
    cfg = NonHermitianSolverConfig(nroots=2, max_condition=1000., atol=1e-7)
    solved = solve_nonhermitian(a, config=cfg, response=RESPONSE)
    assert float(solved.boundary_gap) > 2.9
    assert not bool(solved.response_valid)
    assert not np.isfinite(jax.grad(lambda t: solve_nonhermitian(
        a+t*jnp.eye(3), config=cfg, response=RESPONSE).eigenvalue_sum)(0.))


def test_explicit_response_gap_controls_apply_to_legacy_state_targets():
    a = jnp.diag(jnp.array([0., .01, 2.]))
    cfg = NonHermitianSolverConfig(nroots=1)
    for target in ('eigenvalues', 'eigenpairs'):
        response = EigenResponseConfig(target=target, gap_atol=.02, linear_config=LINEAR)
        solved = solve_nonhermitian(a, config=cfg, response=response)
        assert not bool(jnp.all(solved.response_valid))


def test_joint_response_budget_rejects_before_seed_action(monkeypatch):
    import gradscf.solvers.eigen._nonhermitian_subspace as module
    def forbidden(*args, **kwargs):
        raise AssertionError('Response seed action ran before joint budget rejection')
    monkeypatch.setattr(module, 'apply_columns', forbidden)
    with pytest.raises(ValueError, match='memory|budget'):
        solve_nonhermitian(model(0.)[0], config=NonHermitianSolverConfig(nroots=2),
            response=RESPONSE, probes=jnp.ones((4, 2)), _response_budget=dict(
                max_memory=.1, reserved_bytes=40_000, operator_min_bytes=40_000,
                operator_bytes_per_column=0, action_columns=2))


def test_proper_cluster_bounded_davidson_operator():
    n = 30
    probe = jnp.arange(2*n, dtype=jnp.float64).reshape(n, 2)/n
    cfg = NonHermitianSolverConfig(nroots=2, method='davidson', max_space=14,
                                   block_size=3, atol=1e-9, maxiter=80)
    response = replace(RESPONSE, linear_config=LinearSolverConfig(
        rtol=1e-10, atol=1e-12, restart=20, maxiter=15))
    def solve(t):
        diagonal = jnp.concatenate((jnp.array([-1.-.2*t+t*t, -1.+.2*t+t*t]),
                                    jnp.arange(1., n-1.)))
        coupling = (.3+t)*(diagonal[:2]-diagonal[2:4])
        def action(v):
            return (diagonal*v).at[2:4].add(coupling*v[:2])
        def transpose(v):
            return (diagonal*v).at[:2].add(coupling*v[2:4])
        def block(v):
            assert v.shape[1] <= 14, 'A full physical matrix was requested'
            return jax.vmap(action, in_axes=1, out_axes=1)(v)
        op = LinearOperator((n, n), diagonal.dtype, action, diagonal=diagonal,
                            matmat=block, transpose_matvec=transpose)
        return solve_nonhermitian(op, config=cfg, response=response, probes=probe,
                                  max_memory=8., _batch_size=3)
    solved = jax.jit(solve)(0.)
    assert bool(solved.response_valid) and not bool(solved.raw_result.spectrum_complete)
    assert int(solved.raw_result.subspace_dimension) <= 14
    expected = jnp.zeros_like(probe).at[:2].set(probe[:2]).at[2:4].set(.3*probe[:2])
    np.testing.assert_allclose(solved.projection, expected, atol=3e-8)
    tangent = jax.jit(lambda t: jax.jvp(lambda x: solve(x).projection, (t,), (1.,))[1])(0.)
    np.testing.assert_allclose(tangent, jnp.zeros_like(probe).at[2:4].set(probe[:2]), atol=3e-8)


@pytest.mark.parametrize('side', ['right', 'left'])
def test_qr_chart_rechecks_actual_block_residual(side, monkeypatch):
    import gradscf.solvers.eigen.nonhermitian as module
    # Accurate individual approximate pairs can span an inaccurate QR chart.
    # The second column carries only 3e-11 eigen-residual but normalization of
    # its independent component amplifies that to 3e-6 in the selected span.
    r = jnp.array([[1., 1.], [0., 1e-5], [0., 1e-11]])
    l = jnp.array([[1., 0.], [-1e5, 1e5], [0., 0.]])
    if side == 'left':
        r, l = l, r
    def approximate(matrix, count):
        return jnp.array([-1., -1., 2.], dtype=jnp.complex128), r.astype(jnp.complex128), l.astype(jnp.complex128)
    monkeypatch.setattr(module, 'paired_eigenvectors', approximate)
    solved = solve_nonhermitian(jnp.diag(jnp.array([-1., -1., 2.])),
        config=NonHermitianSolverConfig(nroots=2, atol=1e-9), response=RESPONSE,
        probes=jnp.ones((3, 1)))
    assert bool(jnp.all(solved.raw_result.converged)), 'The old per-root guard must pass this counterexample'
    assert not bool(solved.response_valid), 'A non-root QR chart must not receive implicit AD'
    value = jax.grad(lambda t: solve_nonhermitian(
        jnp.diag(jnp.array([-1., -1., 2.]))+t*jnp.eye(3),
        config=NonHermitianSolverConfig(nroots=2, atol=1e-9), response=RESPONSE,
        probes=jnp.ones((3, 1))).eigenvalue_sum)(0.)
    assert not np.isfinite(value)
