"""EOM known-operator storage shares the original primal MB budget."""
import importlib
import inspect
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import cc
from gradscf._memory import array_storage_bytes
from gradscf.cc.integrals import prepare_integrals
from gradscf.solvers.memory import plan_nonhermitian
from test_prepared_properties import _problem, _config


def _run(ints, ground, **kwargs):
    from gradscf.cc.eom import run_eom_from_integrals
    assert 'max_memory' in inspect.signature(run_eom_from_integrals).parameters, (
        'Missing EOM primal memory-budget entry')
    return run_eom_from_integrals(ints, ground, **kwargs)


def _context(ints, ground, **kwargs):
    module = importlib.import_module('gradscf.cc.eom.operators')
    function = getattr(module, '_operator_context_from_integrals', None)
    assert callable(function), 'Missing private EOM operator memory context'
    return function(ints, ground, **kwargs)


@pytest.mark.parametrize('sector', ['ee', 'ip', 'ea'])
def test_eom_rejects_impossible_budget_before_operator_intermediates(sector, monkeypatch):
    module = importlib.import_module('gradscf.cc.eom.operators')
    h, g = _problem()
    cfg = _config()
    ints = prepare_integrals(h, g, nocc=2)
    ground = cc.run_cc_from_integrals(ints, config=cfg)

    def forbidden(*args, **kwargs):
        pytest.fail('EOM allocated operator intermediates before its minimum budget check')

    monkeypatch.setattr(module, 'intermediates', forbidden)
    monkeypatch.setattr(jax, 'linearize', forbidden)
    with pytest.raises(ValueError, match='memory|budget|resident|minimum'):
        _run(ints, ground, config=cc.EOMConfig(sector=sector), cc_config=cfg, max_memory=.0001)


@pytest.mark.parametrize('sector', ['ee', 'ip', 'ea'])
def test_eom_context_charges_inputs_captured_arrays_and_same_original_budget(sector):
    module = importlib.import_module('gradscf.cc.eom.operators')
    h, g = _problem()
    cfg = _config()
    ints = prepare_integrals(h, g, nocc=2)
    ground = cc.run_cc_from_integrals(ints, config=cfg)
    eom_cfg = cc.EOMConfig(sector=sector)
    context = _context(ints, ground, config=eom_cfg, cc_config=cfg,
                       max_memory=4., reserved_bytes=4096)
    memory = context.memory
    base = array_storage_bytes(*jax.tree.leaves((ints, ground)))
    assert memory.resident_bytes >= base+4096
    assert memory.closure_bytes > 0 and memory.solver_reserved_bytes > 0
    n = context.operator.shape[0]
    total = (memory.resident_bytes+memory.solver_reserved_bytes+memory.operator_min_bytes
             +(2*n*context.operator.dtype.itemsize+memory.operator_bytes_per_column)
             *memory.action_columns)
    assert total <= 4_000_000*.8
    assert memory.config.nroots == eom_cfg.nroots
    assert memory.config.guard_roots == eom_cfg.guard_roots
    assert memory.config.maxiter == eom_cfg.max_cycle
    public = module.build_eom_operator_from_integrals(ints, ground, config=eom_cfg,
                                                     cc_config=cfg, max_memory=4.)
    assert len(public) == 3


@pytest.mark.parametrize('frozen', [None, [0, 3]])
def test_raw_eom_checks_selected_payload_before_preparing(frozen, monkeypatch):
    module = importlib.import_module('gradscf.cc.eom.operators')
    h, g = _problem()
    cfg = _config()
    ints = prepare_integrals(h, g, nocc=2, frozen=frozen)
    ground = cc.run_cc_from_integrals(ints, config=cfg)
    eom_cfg = cc.EOMConfig()
    space = module._checked_space(ground, eom_cfg, cfg)
    _, solver, _ = plan_nonhermitian(space.size, h.dtype,
                                     module._solver_config(eom_cfg), 4.)
    raw = array_storage_bytes(h, g, *jax.tree.leaves(ground))
    selected = array_storage_bytes(*jax.tree.leaves(ints))
    # Raw inputs and the mandatory solver fit; selected storage does not.
    minimum = solver+2*space.size*h.dtype.itemsize*eom_cfg.nroots
    budget = (raw+minimum+selected//2)*1.25/1e6
    original = module.prepare_integrals

    def forbidden(*args, **kwargs):
        if not isinstance(args[0], jax.core.Tracer):
            pytest.fail('Raw EOM allocated selected integral blocks before budget rejection')
        return original(*args, **kwargs)

    monkeypatch.setattr(module, 'prepare_integrals', forbidden)
    with pytest.raises(ValueError, match='memory|budget|resident|minimum'):
        cc.run_eom(h, g, ground, nocc=2, frozen=frozen, config=eom_cfg,
                   cc_config=cfg, max_memory=budget)


@pytest.mark.parametrize('sector', ['ee', 'ip', 'ea'])
@pytest.mark.parametrize('method', ['dense', 'davidson'])
def test_eom_budgeted_roots_jvp_vjp_match_unbudgeted(sector, method):
    h, g = _problem()
    indices = jnp.array([1, 2])
    h, g = h[jnp.ix_(indices, indices)], g[jnp.ix_(indices, indices, indices, indices)]
    cfg = _config()
    eom_cfg = cc.EOMConfig(sector=sector, solver=method, max_space=12)
    direction = jnp.array([[.02, .01], [.01, -.03]])

    def value(x, budgeted):
        ints = prepare_integrals(h+x*direction, g*(1+.13*x), nocc=1)
        state = cc.run_cc_from_integrals(ints, config=cfg)
        options = dict(config=eom_cfg, cc_config=cfg)
        out = (_run(ints, state, max_memory=2., **options) if budgeted else
               cc.run_eom_from_integrals(ints, state, **options))
        return out.energies[0]

    actual, expected = lambda x: value(x, True), lambda x: value(x, False)
    for operation in (lambda f: jax.jit(lambda x: jax.jvp(f, (x,), (1.,)))(0.),
                      lambda f: jax.jit(jax.value_and_grad(f))(0.)):
        for a, e in zip(operation(actual), operation(expected)):
            np.testing.assert_allclose(a, e, atol=3e-10, rtol=3e-9)
    finite = (actual(1e-4)-actual(-1e-4))/2e-4
    np.testing.assert_allclose(jax.grad(actual)(0.), finite, atol=3e-8, rtol=3e-6)


def test_eom_facade_inherits_cc_budget_and_explicit_override_can_reset():
    ground = SimpleNamespace(max_memory=8.)
    obj = cc.EOMEE(ground)
    assert getattr(obj, 'max_memory', None) == 8., 'EOM must inherit the CC memory budget'
    ground.max_memory = 12.
    assert obj.max_memory == 12.
    obj.max_memory = 3.
    ground.max_memory = 16.
    assert obj.max_memory == 3.
    obj.max_memory = None
    assert obj.max_memory == 16.
    assert cc.EOMIP(ground, max_memory=4.).max_memory == 4.
    for invalid in (0., -1., True, float('nan')):
        with pytest.raises(ValueError):
            cc.EOMEA(ground, max_memory=invalid)


def test_eom_budget_change_invalidates_only_eom_cache_and_failure_clears_result():
    h, g = _problem()
    indices = jnp.array([1, 2])
    h, g = h[jnp.ix_(indices, indices)], g[jnp.ix_(indices, indices, indices, indices)]
    ground = cc.CCSD(cc.CCReference(h, g, 1), conv_tol=1e-12,
                    residual_tol=1e-11, max_memory=4.).run()
    obj = ground.EOMEE().run()
    state = ground.result
    ground.max_memory = 3.
    with pytest.raises(RuntimeError, match='changed'):
        obj.check_source()
    obj.run()
    assert ground.result is state
    obj.max_memory = 3.
    ground.max_memory = 2.
    obj.check_source()
    obj.max_memory = .0001
    with pytest.raises(ValueError, match='memory|budget|resident|minimum'):
        obj.kernel()
    assert obj.result is obj.e is obj.right_vectors is obj.left_vectors is None
    with pytest.raises(RuntimeError, match='kernel'):
        obj.check_source()
