"""EOM's existing primal plan is relayed to pair response without replanning."""
import importlib
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import cc
from gradscf.solvers import LinearOperator, NonHermitianSolverConfig
from test_eom_dyson import h2


def test_eom_relays_response_budget_without_replanning_its_primal(monkeypatch):
    module = importlib.import_module('gradscf.cc.eom.operators')
    planned = NonHermitianSolverConfig(nroots=2, block_size=2)
    memory = SimpleNamespace(max_memory=3.25, resident_bytes=737,
        operator_min_bytes=128, operator_bytes_per_column=256,
        config=planned, solver_reserved_bytes=99999, action_columns=2)
    op = LinearOperator((3, 3), jnp.float64, lambda vector: vector)
    class Captured(Exception):
        pass
    def solve(*args, **kwargs):
        assert kwargs.get('max_memory') is None, 'EOM replans its existing primal'
        assert kwargs['config'] is planned and kwargs['_batch_size'] == 2
        assert kwargs.get('_response_budget') == dict(max_memory=3.25,
            reserved_bytes=737, operator_min_bytes=128,
            operator_bytes_per_column=256, action_columns=2)
        raise Captured
    monkeypatch.setattr(module, 'solve_nonhermitian', solve)
    with pytest.raises(Captured):
        module._run_operator(op, jnp.asarray(True),
            cc.EOMConfig(nroots=2, vector_response=True), memory)


@pytest.mark.parametrize('sector', ['ip', 'ea'])
def test_prepared_pair_budget_keeps_dyson_response_and_finite_difference(h2, sector, monkeypatch):
    from gradscf.cc.integrals import prepare_integrals
    from gradscf.solvers import LinearSolverConfig
    h, g = h2
    direction = jnp.array([[.3, .2], [.2, -.1]])
    cfg = cc.CCConfig(conv_tol=1e-12, residual_tol=1e-11)
    eom_cfg = cc.EOMConfig(sector=sector, nroots=2, conv_tol=1e-11,
        vector_response=True, response_config=LinearSolverConfig(
            method='direct', max_dense=3, rtol=1e-11))
    owner = importlib.import_module('gradscf.cc.eom.operators')
    generic = importlib.import_module('gradscf.solvers.eigen.nonhermitian')
    def forbidden(*args, **kwargs):
        pytest.fail('Prepared response re-prepared integrals or replanned the EOM primal')
    monkeypatch.setattr(owner, 'prepare_integrals', forbidden)
    monkeypatch.setattr(generic, 'plan_nonhermitian', forbidden)

    def states(t, budget):
        data = prepare_integrals(h+t*direction, g, nocc=1)
        ground = cc.run_cc_from_integrals(data, config=cfg)
        left = cc.solve_lambda_from_integrals(data, ground, config=cfg)
        eom = cc.run_eom_from_integrals(data, ground, config=eom_cfg,
                                      cc_config=cfg, max_memory=budget)
        return ground, left, eom

    def observable(states):
        ground, left, eom = states
        dyson = cc.dyson_orbitals(ground, left, eom, sector=sector)
        residue = dyson.right[:, None, :]*dyson.left[None, :, :]
        return jnp.concatenate((eom.energies, dyson.weights, residue.ravel()))

    def response(budget):
        values, tangents = jax.jit(lambda t: jax.jvp(lambda x: states(x, budget),
            (t,), (jnp.ones_like(t),)))(jnp.asarray(0.))
        return values, tangents, jax.jvp(observable, (values,), (tangents,))

    values, tangents, (value, derivative) = response(4.)
    _, _, expected = response(None)
    for a, e in zip((value, derivative), expected):
        np.testing.assert_allclose(a, e, atol=2e-9, rtol=2e-8)
    assert np.all(values[2].response_valid) and values[0].converged and values[1].converged
    function = jax.jit(lambda t: observable(states(t, 4.)))
    for step in (3e-4, 1e-4):
        finite_difference = (function(step)-function(-step))/(2*step)
        np.testing.assert_allclose(derivative, finite_difference, atol=2e-7, rtol=2e-6)
    weights = jnp.array([.3, -.2, .7, -.4, 0., 0., .23, 0., 0., 0., 0., 0.])
    reverse = jax.jit(jax.grad(lambda t: weights@observable(states(t, 4.))))(0.)
    np.testing.assert_allclose(reverse, weights@derivative, atol=2e-9)
    frozen = tangents[2]._replace(right_vectors=jnp.zeros_like(values[2].right_vectors),
                                 left_vectors=jnp.zeros_like(values[2].left_vectors))
    frozen_derivative = jax.jvp(observable, (values,), ((tangents[0], tangents[1], frozen),))[1]
    # At this parity-symmetric point vector mixing is off-diagonal, so the
    # trace weights alone cannot detect frozen vectors to first order.
    assert np.linalg.norm(derivative[4:]-frozen_derivative[4:]) > 1e-5
