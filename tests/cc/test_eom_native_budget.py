"""Native EOM construction/actions share source owners and opaque scratch."""
from dataclasses import replace
import importlib
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import cc
from gradscf._memory import array_storage_bytes
from gradscf.ao2mo import VirtualERI
from gradscf.ao2mo.actions import _native_action_workspace
from gradscf.solvers import LinearSolverConfig
from gradscf.solvers.memory import plan_nonhermitian
from test_native_eom_action import _prepared
from test_native_virtual_action import _problem, _assert_trees


def test_shared_counter_retains_nested_array_bytes_and_adds_declared_opaque_scratch():
    from gradscf._memory import jaxpr_temporary_bytes
    assert 'primitive_bytes' in inspect.signature(jaxpr_temporary_bytes).parameters
    inner = jax.jit(lambda x: x+x)
    graph = jax.make_jaxpr(lambda x: inner(x)*x)(jax.ShapeDtypeStruct((3, 2), jnp.float64))
    assert jaxpr_temporary_bytes(graph) == 144
    assert jaxpr_temporary_bytes(graph, primitive_bytes={'add': 37, 'mul': 61}) == 242


@pytest.mark.parametrize('charges,error', [({'add': -1}, ValueError),
    ({'add': True}, ValueError), ({1: 3}, TypeError)])
def test_shared_counter_rejects_invalid_opaque_byte_declarations(charges, error):
    from gradscf._memory import jaxpr_temporary_bytes
    assert 'primitive_bytes' in inspect.signature(jaxpr_temporary_bytes).parameters
    graph = jax.make_jaxpr(lambda x: x+x)(jax.ShapeDtypeStruct((2,), jnp.float64))
    with pytest.raises(error, match='primitive_bytes'):
        jaxpr_temporary_bytes(graph, primitive_bytes=charges)


@pytest.fixture(scope='module')
def native_state():
    _, basis, coefficients, _ = _problem()
    coefficients = .7*coefficients
    config = cc.CCConfig(conv_tol=1e-12, residual_tol=1e-11, adjoint_tol=1e-11)
    data = _prepared(coefficients, basis, 'direct')
    ground = cc.run_cc_from_integrals(data, config=config)
    assert ground.converged
    return data, ground, config


@pytest.mark.parametrize('sector', ['ee', 'ip', 'ea'])
def test_native_source_reservation_rejects_before_build(native_state, sector, monkeypatch):
    module = importlib.import_module('gradscf.cc.eom.operators')
    data, ground, config = native_state
    data = data._replace(vvvv=replace(data.vvvv, reserved_bytes=2_000_000))
    # The provider fits its original 8 MB budget; its opaque upstream owners
    # exceed the new 2 MB EOM working budget before any operator is constructed.
    assert np.isfinite(data.vvvv.contract(jnp.ones((1, 2, 2)))).all()
    monkeypatch.setattr(module, '_build_operator', lambda *a, **k: pytest.fail(
        'Uncharged native source owners reached EOM construction'))
    with pytest.raises(ValueError, match='memory|budget|resident'):
        cc.run_eom_from_integrals(data, ground,
            config=cc.EOMConfig(sector=sector, solver='dense'), cc_config=config,
            max_memory=2.)


@pytest.mark.parametrize('sector', ['ee', 'ip', 'ea'])
def test_native_optimizer_is_charged_before_construction(native_state, sector, monkeypatch):
    module = importlib.import_module('gradscf.cc.eom.operators')
    data, ground, config = native_state
    data = data._replace(vvvv=replace(data.vvvv, reserved_bytes=0))
    eom = cc.EOMConfig(sector=sector, solver='dense')
    space = module._checked_space(ground, eom, config)
    native, _ = _native_action_workspace(data.vvvv.basis, data.nvir)
    base = array_storage_bytes(*jax.tree.leaves((data, ground)))
    _, solver, _ = plan_nonhermitian(space.size, ground.t1.dtype,
                                     module._solver_config(eom), 8.)
    minimum = solver+2*space.size*ground.t1.dtype.itemsize*eom.nroots
    assert minimum < native//2
    budget = (base+native//2)*1.25/1e6
    monkeypatch.setattr(module, '_build_operator', lambda *a, **k: pytest.fail(
        'Uncharged native optimizer reached physical EOM construction'))
    with pytest.raises(ValueError, match='memory|budget|resident'):
        cc.run_eom_from_integrals(data, ground, config=eom, cc_config=config,
                                  max_memory=budget)


@pytest.mark.parametrize('sector', ['ee', 'ip', 'ea'])
def test_native_provider_uses_eom_budget_and_preserves_source_owners(native_state, sector, monkeypatch):
    module = importlib.import_module('gradscf.cc.eom.operators')
    data, ground, config = native_state
    old = data.vvvv
    seen = []
    original = VirtualERI.contract

    def observe(self, rhs):
        seen.append((self.max_memory, self.reserved_bytes))
        return original(self, rhs)

    monkeypatch.setattr(VirtualERI, 'contract', observe)
    context = module._operator_context_from_integrals(data, ground,
        config=cc.EOMConfig(sector=sector, solver='dense'), cc_config=config,
        max_memory=4., reserved_bytes=1234)
    assert seen and all(memory == 4. and reservation >= old.reserved_bytes
                        for memory, reservation in seen)
    assert data.vvvv is old and old.max_memory == 8.
    base = array_storage_bytes(*jax.tree.leaves((data, ground)))
    assert context.memory.resident_bytes >= 1234+old.reserved_bytes+base
    # IP's native residual validation belongs to construction only. EE and
    # EA keep native actions in the root/response operator.
    if sector == 'ip':
        assert context.memory.operator_min_bytes == 0
    else:
        native, _ = _native_action_workspace(old.basis, data.nvir)
        assert context.memory.operator_min_bytes >= native
    memory, n = context.memory, context.space.size
    total = (memory.resident_bytes+memory.solver_reserved_bytes+memory.operator_min_bytes
             +memory.action_columns*(2*n*context.operator.dtype.itemsize
                                     +memory.operator_bytes_per_column))
    assert total <= 4_000_000*.8


def test_native_eom_uses_one_capacity_snapshot_for_forecast_and_action(native_state, monkeypatch):
    module = importlib.import_module('gradscf.cc.eom.operators')
    actions = importlib.import_module('gradscf.ao2mo.actions')
    data, ground, config = native_state

    def changed_capacity(*args, **kwargs):
        pytest.fail('Native EOM re-queried capacity after the shared phase snapshot')

    monkeypatch.setattr(actions, 'workspace_bytes', changed_capacity)
    context = module._operator_context_from_integrals(data, ground,
        config=cc.EOMConfig(sector='ee', solver='dense'), cc_config=config,
        max_memory=4.)
    result = context.operator.apply(jnp.ones(context.space.size))
    assert context.valid and np.isfinite(result).all()


def test_native_snapshot_is_static_and_cannot_exceed_a_smaller_override(native_state, monkeypatch):
    data, _, _ = native_state
    assert '_working_bytes' in inspect.signature(VirtualERI).parameters
    provider = replace(data.vvvv, _working_bytes=3_200_000)
    assert len(jax.tree.leaves(provider)) == len(jax.tree.leaves(data.vvvv))
    monkeypatch.setattr(type(provider.basis), 'get_jk', lambda *a, **k: pytest.fail(
        'A stale snapshot bypassed the smaller max_memory override'))
    with pytest.raises(ValueError, match='memory|budget|resident'):
        replace(provider, max_memory=.0001).contract(jnp.ones((1, 2, 2)))


@pytest.mark.parametrize('sector', ['ee', 'ip', 'ea'])
def test_native_davidson_uses_the_same_budgeted_physical_operator(native_state, sector):
    data, ground, config = native_state
    results = [cc.run_eom_from_integrals(data, ground,
        config=cc.EOMConfig(sector=sector, solver=method, max_space=12,
                            conv_tol=1e-10), cc_config=config, max_memory=4.)
        for method in ('dense', 'davidson')]
    for result in results:
        assert np.all(result.converged) and np.all(result.response_valid)
        assert all(np.isfinite(value).all() for value in (result.energies,
            result.right_vectors, result.left_vectors, result.residual_norms,
            result.left_residual_norms, result.condition_numbers))
    np.testing.assert_allclose(results[1].energies, results[0].energies,
                               atol=3e-10, rtol=3e-9)
    np.testing.assert_allclose(results[1].right_vectors @ results[1].left_vectors.T,
                               results[0].right_vectors @ results[0].left_vectors.T,
                               atol=3e-9, rtol=3e-8)


def test_native_isolated_pair_budget_keeps_vector_response():
    _, basis, coefficients, _ = _problem()
    coefficients = .7*coefficients
    config = cc.CCConfig(conv_tol=1e-12, residual_tol=1e-11, adjoint_tol=1e-11)
    eom = cc.EOMConfig(sector='ea', solver='dense', conv_tol=1e-10,
        vector_response=True, response_config=LinearSolverConfig(
            method='direct', max_dense=7, rtol=1e-11))

    def quantity(scale, mode):
        data = _prepared(coefficients*(1+.03*scale), basis, mode)
        ground = cc.run_cc_from_integrals(data, config=config)
        result = cc.run_eom_from_integrals(data, ground, config=eom,
                                         cc_config=config, max_memory=8.)
        projector = result.right_vectors[:, :1] @ result.left_vectors[:, :1].T
        weight = jnp.linspace(-.07, .05, projector.size).reshape(projector.shape)
        return result.energies[0]+jnp.sum(weight*projector)

    direct = lambda x: quantity(x, 'direct')
    dense = lambda x: quantity(x, 'dense')
    actual = jax.jit(jax.value_and_grad(direct))(0.)
    expected = jax.jit(jax.value_and_grad(dense))(0.)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves((actual, expected)))
    _assert_trees(actual, expected, atol=3e-9, rtol=3e-8)
    for step in (3e-4, 1e-4):
        finite = (direct(step)-direct(-step))/(2*step)
        assert np.isfinite(finite)
        np.testing.assert_allclose(actual[1], finite, atol=3e-8, rtol=3e-6)
