"""Existing EE cluster forward invariants; no non-Hermitian cluster AD claim."""
import jax
import jax.numpy as jnp
import numpy as np

from gradscf import cc
from gradscf.cc.integrals import prepare_integrals


def test_ee_complete_cluster_forward_invariant_and_response_guard():
    config = cc.CCConfig(conv_tol=1e-12, residual_tol=1e-11)
    interaction = jnp.zeros((3,)*4)
    change = jnp.array([[1.4, .3], [-.2, .8]])
    expected = jnp.diag(jnp.array([1., 1., 0., 0., 0.]))
    projectors = []

    for method in ('dense', 'davidson'):
        eom = cc.EOMConfig(sector='ee', nroots=2, solver=method,
                           max_space=12, conv_tol=1e-11)

        def states(displacement):
            h = jnp.diag(jnp.array([-1., .4+displacement, .4-displacement]))
            data = prepare_integrals(h, interaction, nocc=1)
            ground = cc.run_cc_from_integrals(data, config=config)
            return cc.run_eom_from_integrals(data, ground, config=eom,
                                             cc_config=config, max_memory=4.)

        result = jax.jit(states)(0.)
        assert np.all(result.converged) and not np.any(result.response_valid)
        np.testing.assert_allclose(result.energies, [1.4, 1.4], atol=2e-12, rtol=0.)
        assert float(jnp.min(result.raw_eigenvalues[2:].real)) > 2.7
        right, left = result.right_vectors, result.left_vectors
        assert np.isfinite(right).all() and np.isfinite(left).all()
        np.testing.assert_allclose(left.T @ right, jnp.eye(2), atol=2e-11, rtol=0.)
        projector = right @ left.T
        transformed = (right @ change) @ (left @ jnp.linalg.inv(change).T).T
        np.testing.assert_allclose(projector, expected, atol=2e-11, rtol=0.)
        np.testing.assert_allclose(transformed, projector, atol=2e-11, rtol=0.)
        np.testing.assert_allclose(projector @ projector, projector, atol=2e-11, rtol=0.)
        np.testing.assert_allclose(jnp.trace(projector), 2., atol=2e-11, rtol=0.)
        np.testing.assert_allclose(jnp.sum(result.energies), 2.8, atol=2e-12, rtol=0.)
        # Individual sorted roots split as 1.4 +/- t. The current solver
        # supports only isolated-pair AD and rejects the whole selected cluster,
        # even for its mathematically smooth trace; this is an explicit limit.
        assert not np.isfinite(jax.jit(jax.grad(lambda t: states(t).energies[0]))(0.))
        assert not np.isfinite(jax.jit(jax.grad(lambda t: states(t).energies.sum()))(0.))
        projectors.append(projector)
    np.testing.assert_allclose(projectors[0], projectors[1], atol=2e-11, rtol=0.)


def test_ee_complete_cluster_explicit_invariant_response():
    from gradscf.solvers import EigenResponseConfig, LinearSolverConfig
    response = EigenResponseConfig(target='subspace', linear_config=LinearSolverConfig(
        method='direct', max_dense=32, rtol=1e-11, atol=1e-12))
    config = cc.CCConfig(conv_tol=1e-12, residual_tol=1e-11)
    for method in ('dense', 'davidson'):
        eom = cc.EOMConfig(sector='ee', nroots=2, solver=method, max_space=12,
                           conv_tol=1e-11, response=response)
        def states(t):
            h = jnp.diag(jnp.array([-1., .4+t+t*t, .4-t+t*t]))
            data = prepare_integrals(h, jnp.zeros((3,)*4), nocc=1)
            ground = cc.run_cc_from_integrals(data, config=config)
            return cc.run_eom_from_integrals(data, ground, config=eom, cc_config=config,
                                             probes=jnp.eye(5), max_memory=4.)
        value = jax.jit(states)(0.)
        assert bool(value.response_valid) and bool(value.ground_valid)
        assert value.energies is None and value.right_vectors is None
        np.testing.assert_allclose(value.eigenvalue_sum, 2.8, atol=2e-11)
        np.testing.assert_allclose(value.projection, jnp.diag(jnp.array([1.,1.,0.,0.,0.])), atol=2e-11)
        trace = lambda t: states(t).eigenvalue_sum
        np.testing.assert_allclose(jax.jit(jax.grad(trace))(0.), 0., atol=2e-10)
        np.testing.assert_allclose(jax.jit(jax.grad(jax.grad(trace)))(0.), 4., atol=2e-9)
        np.testing.assert_allclose(jax.jit(jax.jacfwd(lambda t: states(t).projection))(0.), 0., atol=2e-9)


def test_charged_complete_cluster_explicit_response():
    from gradscf.solvers import EigenResponseConfig, LinearSolverConfig
    response = EigenResponseConfig(target='subspace', linear_config=LinearSolverConfig(
        method='direct', max_dense=32, rtol=1e-11, atol=1e-12))
    config = cc.CCConfig(conv_tol=1e-12, residual_tol=1e-11)
    for sector, nocc in (('ip', 2), ('ea', 1)):
        eom = cc.EOMConfig(sector=sector, nroots=2, response=response, conv_tol=1e-11)
        def solve(t):
            diagonal = (jnp.array([-.5+t-.1*t, -.5-t-.1*t, .8]) if sector=='ip'
                        else jnp.array([-1., .4+t+.1*t, .4-t+.1*t]))
            data = prepare_integrals(jnp.diag(diagonal), jnp.zeros((3,)*4), nocc=nocc)
            ground = cc.run_cc_from_integrals(data, config=config)
            return cc.run_eom_from_integrals(data, ground, config=eom, cc_config=config,
                                             max_memory=4.)
        solved = jax.jit(solve)(0.)
        assert bool(solved.response_valid) and solved.projection is None
        np.testing.assert_allclose(solved.eigenvalue_sum, 1. if sector=='ip' else .8, atol=2e-10)
        np.testing.assert_allclose(jax.jit(jax.grad(lambda t: solve(t).eigenvalue_sum))(0.), .2, atol=2e-9)


def test_eom_eager_subspace_returns_invariants():
    import pytest
    from gradscf.solvers import EigenResponseConfig
    h = jnp.diag(jnp.array([-1., .4, .4]))
    ground = cc.CCSD(cc.CCReference(h, jnp.zeros((3,)*4), 1),
                    conv_tol=1e-12, residual_tol=1e-11, max_memory=4.).run()
    obj = ground.EOMEE(nroots=2, response=EigenResponseConfig(target='subspace'))
    value = obj.kernel(probes=jnp.eye(5))
    assert value is obj.result and bool(value.response_valid)
    assert obj.e is obj.right_vectors is obj.left_vectors is None
    np.testing.assert_allclose(obj.eigenvalue_sum, 2.8, atol=2e-10)
    assert obj.diagnostics().eom is value.raw_result
    with pytest.raises(ValueError, match='invariants'):
        obj.amplitudes()


def test_free_dyson_rejects_subspace_before_state_access():
    import pytest
    result = cc.EOMSubspaceResult(None, jnp.asarray(0.), jnp.asarray(True),
                                 jnp.asarray(1.), jnp.asarray(True), None)
    with pytest.raises(ValueError, match='[Ss]ubspace|invariant'):
        cc.dyson_orbitals(None, None, result, sector='ip')
