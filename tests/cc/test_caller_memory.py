"""Original shared-solver budgets include the prepared CC caller's owners."""
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import cc
from gradscf._memory import tree_storage_bytes
from gradscf.cc.integrals import prepare_integrals
from gradscf.cc.uccsd import prepare_ucc_integrals
from test_prepared_properties import _problem, _config
from test_spin_virtual_consumers import _compact_raw
from test_scf_prepared import source as rhf_source
from test_scf_unrestricted_prepared import source as uhf_source


def _frame(unrestricted):
    h, g = _problem()
    if unrestricted:
        ints, no, nv = prepare_ucc_integrals((h, h), (g, g, g), nocc=(2, 1))
        ints = _compact_raw(ints, no, nv, (g, g, g))
        return ints, lambda **kw: cc.run_ucc_from_integrals(ints, no, nv, config=_config(), **kw)
    ints = prepare_integrals(h, g, nocc=2)
    return ints, lambda **kw: cc.run_cc_from_integrals(ints, config=_config(), **kw)


@pytest.mark.parametrize('unrestricted', [False, True])
def test_prepared_ground_rejects_live_payload_before_numerical_solver(unrestricted, monkeypatch):
    ints, solve = _frame(unrestricted)
    owner = importlib.import_module('gradscf.cc.uccsd' if unrestricted else 'gradscf.cc.ground')
    def forbidden(*args, **kwargs):
        pytest.fail('Insufficient CC budget reached the numerical solver')
    monkeypatch.setattr(owner, 'solve_nonlinear', forbidden)
    with pytest.raises(ValueError, match='memory|budget|resident'):
        solve(max_memory=.000001)


@pytest.mark.parametrize('unrestricted', [False, True])
def test_prepared_ground_forwards_original_budget_and_actual_integral_leaves(unrestricted, monkeypatch):
    ints, solve = _frame(unrestricted)
    owner = importlib.import_module('gradscf.cc.uccsd' if unrestricted else 'gradscf.cc.ground')
    class Captured(Exception):
        pass
    def checked(*args, **kwargs):
        assert kwargs['max_memory'] == 32.
        assert kwargs['reserved_bytes'] >= 123+tree_storage_bytes(ints)
        assert kwargs['operator_min_bytes'] > 0
        raise Captured
    monkeypatch.setattr(owner, 'solve_nonlinear', checked)
    with pytest.raises(Captured):
        solve(max_memory=32., reserved_bytes=123)


@pytest.mark.parametrize('unrestricted', [False, True])
def test_prepared_lambda_budget_includes_saved_vjp_owners(unrestricted, monkeypatch):
    ints, solve = _frame(unrestricted)
    state = solve()
    owner = importlib.import_module('gradscf.cc.lambda_equations')
    original = owner.jax.vjp
    saved = []
    def capture(*args, **kwargs):
        values, pullback = original(*args, **kwargs)
        # Metadata traces precede the actual VJP; only its last pullback is
        # retained by the numerical solve, rather than every tracing object.
        saved[:] = jax.tree.leaves(pullback)
        return values, pullback
    class Captured(Exception):
        pass
    def checked(operator, rhs, **kwargs):
        assert saved
        assert kwargs['max_memory'] == 32.
        assert kwargs['reserved_bytes'] >= 123+tree_storage_bytes(ints, state, saved, exclude=(rhs,))
        assert kwargs['operator_min_bytes'] > 0
        raise Captured
    monkeypatch.setattr(owner.jax, 'vjp', capture)
    monkeypatch.setattr(owner, 'solve_linear', checked)
    with pytest.raises(Captured):
        cc.solve_lambda_from_integrals(ints, state, config=_config(), max_memory=32., reserved_bytes=123)


@pytest.mark.parametrize('unrestricted', [False, True])
def test_prepared_triples_budget_rejects_before_residual_or_amplitude_expansion(unrestricted, monkeypatch):
    ints, solve = _frame(unrestricted)
    state = solve()
    owner = importlib.import_module('gradscf.cc._spin_triples' if unrestricted else 'gradscf.cc.triples')
    def forbidden(*args, **kwargs):
        pytest.fail('Insufficient triples budget reached physical tensor construction')
    monkeypatch.setattr(owner, 'residual', forbidden)
    if unrestricted:
        monkeypatch.setattr(owner.SpinAmplitudeSpace, 'from_blocks', forbidden)
    with pytest.raises(ValueError, match='memory|budget|resident'):
        cc.evaluate_triples_from_integrals(ints, state, max_memory=.000001)


@pytest.mark.parametrize('unrestricted', [False, True])
def test_roomy_ground_and_lambda_budget_preserves_joint_response(unrestricted):
    h, g = _problem()
    direction = jnp.asarray(.01)
    cfg = _config()
    def value(x, budget):
        if unrestricted:
            data, no, nv = prepare_ucc_integrals((h+x*jnp.eye(4), h),
                (g, g*(1+.02*x), g), nocc=(2, 1))
            data = _compact_raw(data, no, nv, (g, g*(1+.02*x), g))
            state = cc.run_ucc_from_integrals(data, no, nv, config=cfg, max_memory=budget)
        else:
            data = prepare_integrals(h+x*jnp.eye(4), g*(1+.02*x), nocc=2)
            state = cc.run_cc_from_integrals(data, config=cfg, max_memory=budget)
        left = cc.solve_lambda_from_integrals(data, state, config=cfg, max_memory=budget)
        return state.total_energy+.01*jnp.sum(left.adjoint**2)
    def bundle(budget):
        return jax.jit(lambda x: jax.jvp(jax.value_and_grad(lambda y: value(y, budget)),
                                        (x,), (direction,)))(jnp.asarray(.02))
    for actual, expected in zip(jax.tree.leaves(bundle(32.)), jax.tree.leaves(bundle(None))):
        assert np.isfinite(np.asarray(actual)).all()
        assert np.isfinite(np.asarray(expected)).all()
        if jax.default_backend() == 'gpu':
            assert actual.devices() == {jax.devices('gpu')[0]}
        np.testing.assert_allclose(actual, expected, atol=3e-10, rtol=3e-8, equal_nan=False)


def _native_frame():
    from test_native_virtual_action import _problem as native_problem
    from gradscf.cc.integrals import from_orbitals
    _, basis, _, _ = native_problem()
    c = jnp.asarray(np.random.default_rng(6561).normal(scale=.08, size=(basis.nao, 5)))
    fock = jnp.diag(jnp.asarray([-1.3, -.8, .3, .6, .9]))
    return tuple(from_orbitals(c, fock, 2, jnp.asarray(-2.), direct_basis=basis,
                               max_memory=32., vvvv=mode) for mode in ('dense', 'direct'))


def test_native_ground_uses_actual_reduced_phase_capacities(monkeypatch):
    from dataclasses import replace
    from gradscf.cc.ground import linear_config
    from gradscf.solvers.memory import plan_linear
    _, data = _native_frame()
    cfg = replace(_config(), diis_space=12, adjoint_restart=100)
    owner = importlib.import_module('gradscf.cc.ground')
    records = []
    class Captured(Exception):
        pass
    def capture(residual, initial, **kwargs):
        records.append((initial, kwargs))
        # Trace the same physical callback, so a native minimum/tile failure
        # cannot hide behind a successful outer solver preflight.
        jax.make_jaxpr(residual)(initial)
        raise Captured
    monkeypatch.setattr(owner, 'solve_nonlinear', capture)
    with pytest.raises(Captured):
        cc.run_cc_from_integrals(data, config=cfg, max_memory=32.)
    initial, wide = records[-1]
    _, working = plan_linear(initial.size, initial.dtype,
        replace(linear_config(cfg), restart=8), 32.,
        operator_min_bytes=wide['operator_min_bytes'])
    required = wide['reserved_bytes']+3*initial.size*initial.dtype.itemsize+working
    memory = (int(np.ceil(required*1.25))+16)/1_000_000
    with pytest.raises(Captured):
        cc.run_cc_from_integrals(data, config=cfg, max_memory=memory)
    _, narrow = records[-1]
    assert narrow['max_memory'] == memory
    assert narrow['linear_config'].restart < min(initial.size, cfg.adjoint_restart)
    assert narrow['config'].diis_space <= cfg.diis_space
    assert (cfg.diis_space, cfg.adjoint_restart) == (12, 100)


def test_native_budgeted_ground_and_left_keep_joint_response():
    dense, native = _native_frame()
    probe = jnp.arange(25., dtype=jnp.float64).reshape(5, 5)/250
    def value(x, data):
        fock = data.fock+x*probe
        moving = data._replace(fock=fock, mo_energy=jnp.diag(fock))
        state = cc.run_cc_from_integrals(moving, config=_config(), max_memory=32.)
        left = cc.solve_lambda_from_integrals(moving, state, config=_config(), max_memory=32.)
        return state.total_energy+.01*jnp.sum(left.adjoint**2)
    def bundle(data):
        return jax.jit(lambda x: jax.jvp(jax.value_and_grad(lambda y: value(y, data)),
            (x,), (jnp.asarray(.01),)))(jnp.asarray(.02))
    for actual, expected in zip(jax.tree.leaves(bundle(native)), jax.tree.leaves(bundle(dense))):
        np.testing.assert_allclose(actual, expected, atol=3e-10, rtol=3e-8)


@pytest.mark.parametrize('unrestricted', [False, True])
def test_public_cc_consumers_inherit_one_budget_and_reserve_visible_source(
        unrestricted, request, monkeypatch):
    from gradscf.scf.reference import reference_from_source, unrestricted_reference_from_source
    source = request.getfixturevalue('uhf_source' if unrestricted else 'rhf_source')
    raw = (unrestricted_reference_from_source if unrestricted else reference_from_source)(source)
    options = dict(conv_tol=1e-12, residual_tol=1e-10, max_memory=source.max_memory)
    expected = cc.CCSD(raw, **options).run()
    density, triples = expected.make_rdm1(), expected.ccsd_t()
    api = importlib.import_module('gradscf.cc.api')
    seen = set()
    names = ('run_ucc_from_integrals' if unrestricted else 'run_cc_from_integrals',
             'solve_lambda_from_integrals', 'evaluate_triples_from_integrals',
             '_density_ucc_from_integrals' if unrestricted else '_density_from_integrals')
    for name in names:
        original = getattr(api, name)
        def checked(*args, _name=name, _original=original, **kwargs):
            assert kwargs.get('max_memory') == source.max_memory, 'CC consumer lost the original MB budget'
            assert kwargs.get('reserved_bytes', 0) > 0, 'CC consumer lost visible SCF/source owners'
            seen.add(_name)
            return _original(*args, **kwargs)
        monkeypatch.setattr(api, name, checked)
    actual = cc.CCSD(source, **options).run()
    actual_density = actual.make_rdm1()
    np.testing.assert_allclose(actual.e_tot, expected.e_tot, atol=3e-10)
    for a, b in zip(jax.tree.leaves(actual_density), jax.tree.leaves(density)):
        np.testing.assert_allclose(a, b, atol=2e-8)
    np.testing.assert_allclose(actual.ccsd_t(), triples, atol=3e-10)
    assert seen == set(names)


def test_raw_ucc_preflight_pays_full_mo_dtype_promotions(monkeypatch):
    module = importlib.import_module('gradscf.cc._memory')
    h, g = _problem()
    reservations = []
    class Captured(Exception):
        pass
    def record(memory, *, resident_bytes):
        reservations.append(resident_bytes)
        raise Captured
    monkeypatch.setattr(module, 'workspace_bytes', record)
    for integrals in (g, g.astype(jnp.float32)):
        with pytest.raises(Captured):
            module.preparation_reservation(lambda a, b: prepare_ucc_integrals(a, b, nocc=(2, 1)),
                ((h, h), (integrals, integrals, integrals)), 32., 0)
    # Raw UCC explicitly promotes each of its three supplied G arguments.
    # Its one aliased resident G becomes smaller while those three f64 copies appear.
    assert reservations[1]-reservations[0] == 3*g.size*8-g.size*4


def test_dense_virtual_scratch_counts_promoted_operand_dtype():
    from gradscf.cc._memory import residual_workspace
    h, g = _problem()
    data = prepare_integrals(h, g, nocc=2)
    low = data._replace(vvvv=data.vvvv.astype(jnp.float32))
    assert residual_workspace(low, 2, 2, 'ccsd', dtype=jnp.float64) == residual_workspace(
        data, 2, 2, 'ccsd', dtype=jnp.float64)
    # Fock/initial precision does not lower contractions with f64 ERI blocks.
    low = data._replace(fock=data.fock.astype(jnp.float32),
                        mo_energy=data.mo_energy.astype(jnp.float32))
    assert residual_workspace(low, 2, 2, 'ccsd', dtype=jnp.float32) == residual_workspace(
        data, 2, 2, 'ccsd', dtype=jnp.float64)


def test_lambda_rejects_the_measured_transpose_workspace_minimum(monkeypatch):
    from gradscf.cc.integrals import CCIntegrals
    shape = lambda *s: jax.ShapeDtypeStruct(s, jnp.float64)
    # The fixed 44 MO/8 occupied DF-RHF frame's shapes; metadata alone must
    # reject its measured 123.52 MB counterexample before numerical GMRES.
    data = CCIntegrals(shape(44, 44), shape(44), shape(8, 8, 8, 8),
        shape(8, 36, 8, 8), shape(8, 36, 8, 36), shape(8, 8, 36, 36),
        shape(8, 36, 36, 8), shape(8, 36, 36, 36), shape(36, 36, 36, 36), shape())
    state = cc.CCResult(shape(), shape(), shape(), shape(8, 36), shape(8, 8, 36, 36),
        shape(), shape(), jax.ShapeDtypeStruct((), jnp.int64),
        jax.ShapeDtypeStruct((), jnp.bool_), shape(), jax.ShapeDtypeStruct((), jnp.int32))
    owner = importlib.import_module('gradscf.cc.lambda_equations')
    def forbidden(*args, **kwargs):
        pytest.fail('Underestimated transpose workspace reached numerical GMRES')
    monkeypatch.setattr(owner, 'solve_linear', forbidden)
    with pytest.raises(ValueError, match='memory|budget|working'):
        jax.make_jaxpr(lambda i, s: cc.solve_lambda_from_integrals(i, s,
            config=cc.CCConfig(conv_tol=1e-12, residual_tol=1e-10, adjoint_tol=1e-10),
            max_memory=123.521238, reserved_bytes=123904))(data, state)


def test_raw_semicanonical_budget_precedes_same_spin_t2_bridge():
    shape = lambda *s: jax.ShapeDtypeStruct(s, jnp.float64)
    h, g = shape(4, 4), shape(4, 4, 4, 4)
    state = cc.CCResult(shape(), shape(), shape(), shape(2, 2), shape(2, 2, 2, 2),
        shape(), shape(), jax.ShapeDtypeStruct((), jnp.int64),
        jax.ShapeDtypeStruct((), jnp.bool_), shape(), jax.ShapeDtypeStruct((), jnp.int32))
    reserved = 37
    # Supplied inputs fit, but the one complete AA/BB bridge copy does not.
    # ShapeDtypeStruct has no swapaxes: reaching it exposes a late preflight.
    required = reserved+tree_storage_bytes(h, g, state)+state.t2.size*8//2
    memory = required*1.25/1_000_000
    with pytest.raises(ValueError, match='memory|budget|resident'):
        cc.evaluate_triples(h, g, state, nocc=2, orbital_basis='semicanonical',
            algorithm='auto', max_memory=memory, reserved_bytes=reserved)
