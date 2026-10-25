"""Restricted selected AO blocks and the shared prepared CC solver entry."""
from dataclasses import replace
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import cc, integrals
from gradscf.cc.integrals import CCIntegrals, prepare_integrals
from gradscf.cc.rccsd import residual, correlation_energy
from gradscf.integrals.backends.native.jk import NativeDirectBasis


REQUESTS = ((0, 0, 0, 0), (0, 1, 0, 0), (0, 1, 0, 1),
            (0, 0, 1, 1), (0, 1, 1, 0), (0, 1, 1, 1), (1, 1, 1, 1))
MODELS = ('ccsd', 'ccd', 'cc2', 'qcisd')


def _prepare(c, f, nocc, energy, **kwargs):
    module = importlib.import_module('gradscf.cc.integrals')
    function = getattr(module, 'from_orbitals', None)
    assert callable(function), 'Missing selected restricted CC from_orbitals entry'
    return function(c, f, nocc, energy, **kwargs)


def _solve(ints, **kwargs):
    function = getattr(cc, 'run_cc_from_integrals', None)
    assert callable(function), 'Missing shared prepared restricted CC solver entry'
    return function(ints, **kwargs)


def _assert_trees(actual, expected, *, atol=2e-12, rtol=2e-11):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for a, e in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        assert a.shape == e.shape
        np.testing.assert_allclose(a, e, atol=atol, rtol=rtol)


def _oracle(c, f, nocc, energy, ao_eri, frozen=None):
    # Full tiny MO tensor is an independent test oracle, never production prep.
    g = jnp.einsum('pqrs,pi,qj,rk,sl->ijkl', ao_eri, c, c, c, c,
                   precision='highest', optimize='optimal')
    removed = set(range(frozen)) if isinstance(frozen, int) else set(frozen or ())
    o = jnp.asarray([i for i in range(nocc) if i not in removed], dtype=jnp.int32)
    v = jnp.asarray([i for i in range(nocc, c.shape[1]) if i not in removed], dtype=jnp.int32)
    active = jnp.concatenate((o, v))
    f = f[jnp.ix_(active, active)]
    spaces = (o, v)
    blocks = tuple(g[jnp.ix_(*(spaces[i] for i in request))] for request in REQUESTS)
    return CCIntegrals(f, jnp.diag(f), *blocks, energy)


def _array_problem(representation, dtype):
    rng = np.random.default_rng(6301)
    c = jnp.asarray(rng.normal(scale=.2, size=(5, 7)).astype(dtype))
    f = jnp.asarray(rng.normal(scale=.1, size=(7, 7)).astype(dtype))
    energy = jnp.asarray(-3.4, dtype=dtype)
    shape = {'dense': (5,)*4, 'packed': (15, 15), 'df': (4, 5, 5)}[representation]
    source = jnp.asarray(rng.normal(scale=.2, size=shape).astype(dtype))
    keyword = {'dense': 'eri', 'packed': 'eri_pair_matrix', 'df': 'df_factors'}[representation]

    def expand(value):
        if representation == 'dense':
            return value
        if representation == 'df':
            return jnp.einsum('Lpq,Lrs->pqrs', value, value, precision='highest')
        rows, cols = np.tril_indices(5)
        indices = np.empty((5, 5), dtype=np.int32)
        indices[rows, cols] = indices[cols, rows] = np.arange(15)
        return value[indices[:, :, None, None], indices[None, None, :, :]]

    return c, f, energy, source, keyword, expand


@pytest.mark.parametrize('representation', ['dense', 'packed', 'df'])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('frozen', [None, 1, [0, 5], [0, 1, 2], [3, 4, 5, 6]])
def test_selected_array_blocks_preserve_frozen_frame_and_dtype(representation, dtype, frozen):
    c, f, energy, source, keyword, expand = _array_problem(representation, dtype)
    actual = _prepare(c, f, 3, energy, frozen=frozen, **{keyword: source}, max_memory=2.)
    expected = _oracle(c, f, 3, energy, expand(source), frozen)
    assert all(x.dtype == dtype for x in jax.tree.leaves(actual))
    tolerance = 3e-6 if dtype == np.float32 else 2e-12
    _assert_trees(actual, expected, atol=tolerance, rtol=tolerance)
    if frozen is None:
        assert np.linalg.norm(np.asarray(actual.fock-actual.fock.T)) > .1


@pytest.mark.parametrize('representation', ['dense', 'packed', 'df'])
def test_selected_array_raw_chart_joint_jvp_vjp_hvp(representation):
    c, f, energy, source, keyword, expand = _array_problem(representation, np.float64)
    rng = np.random.default_rng(6302)
    point = (c, f, energy, source)
    direction = tuple(jnp.asarray(rng.normal(scale=.03, size=x.shape)) for x in point)

    def loss(cs, fm, e, data, selected):
        ints = (_prepare(cs, fm, 3, e, frozen=[0, 5], **{keyword: data}, max_memory=2.)
                if selected else _oracle(cs, fm, 3, e, expand(data), [0, 5]))
        t1 = jnp.full((ints.nocc, ints.nvir), .02)
        t2 = jnp.full((ints.nocc, ints.nocc, ints.nvir, ints.nvir), .01)
        r1, r2 = residual(t1, t2, ints)
        return ints.reference_energy+correlation_energy(t1, t2, ints)+.1*jnp.sum(r1*r1)+.2*jnp.sum(r2*r2)

    actual = lambda *x: loss(*x, True)
    expected = lambda *x: loss(*x, False)
    for operation in (lambda fn: jax.jvp(fn, point, direction),
                      lambda fn: jax.value_and_grad(fn, argnums=(0, 1, 2, 3))(*point),
                      lambda fn: jax.jvp(jax.grad(fn, argnums=(0, 1, 2, 3)), point, direction)[1]):
        _assert_trees(operation(actual), operation(expected), atol=3e-10, rtol=3e-9)


def _native_problem(dtype=jnp.float64):
    topology, parameters = integrals.prepare_basis('H 0 0 0; H .2 -.1 .8', basis='3-21g')
    basis = NativeDirectBasis(integrals.make_plan(topology, backend='native'), parameters)
    rng = np.random.default_rng(6303)
    c = jnp.asarray(rng.normal(scale=.15, size=(topology.nao, 4)), dtype=dtype)
    f = jnp.diag(jnp.asarray([-1.2, .3, .5, .8], dtype=dtype))
    return basis, c, f, jnp.asarray(-1.4, dtype=dtype)


def test_native_preparation_is_one_shared_batch_with_external_residency(monkeypatch):
    module = importlib.import_module('gradscf.cc.integrals')
    function = getattr(module, 'transform_eri_blocks', None)
    assert callable(function), 'CC preparation must use the shared AO2MO batch entry'
    basis, c, f, energy = _native_problem(jnp.float32)
    seen = []

    def capture(coefficients, requests, **kwargs):
        seen.append((coefficients, requests, kwargs))
        return function(coefficients, requests, **kwargs)

    monkeypatch.setattr(module, 'transform_eri_blocks', capture)
    actual = _prepare(c, f, 1, energy, frozen=[3], direct_basis=basis,
                      max_memory=.25, reserved_bytes=4096)
    assert len(seen) == 1 and tuple(seen[0][1]) == REQUESTS
    assert tuple(x.shape for x in seen[0][0]) == ((4, 1), (4, 2))
    assert seen[0][2]['max_memory'] == .25
    assert seen[0][2]['reserved_bytes'] >= 4096+c.nbytes+f.nbytes+energy.nbytes
    assert all(x.dtype == jnp.float64 for x in jax.tree.leaves(actual))
    expected = _oracle(c.astype(jnp.float64), f.astype(jnp.float64), 1,
        energy.astype(jnp.float64), basis.plan.evaluate('eri', basis.parameters), [3])
    _assert_trees(actual, expected)


def _equations(value):
    value = getattr(value, 'jaxpr', value)
    if hasattr(value, 'eqns'):
        for equation in value.eqns:
            yield equation
            yield from _equations(equation.params)
    elif isinstance(value, dict):
        for child in value.values():
            yield from _equations(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _equations(child)


def test_native_preparation_has_one_slab_site_and_no_full_ao_or_mo_tensor():
    basis, c, f, energy = _native_problem()
    function = lambda cs: _prepare(cs, f, 1, energy, direct_basis=basis, max_memory=.25)
    graphs = (jax.make_jaxpr(function)(c), jax.make_jaxpr(jax.grad(
        lambda cs: sum(jnp.sum(x*x) for x in function(cs)[2:9])))(c))
    assert sum(e.primitive.name == 'gradscf_eri_slab_value' for e in _equations(graphs[0])) == 1
    for graph in graphs:
        for equation in _equations(graph):
            for variable in equation.outvars:
                assert getattr(getattr(variable, 'aval', None), 'shape', None) != (4,)*4


def test_native_preparation_geometry_coefficient_response_and_prepared_solve():
    basis, c, f, energy = _native_problem()
    parameters = basis.parameters
    coords = jnp.concatenate((parameters.nuclear_coords, parameters.centers))
    rng = np.random.default_rng(6304)
    point = (coords, c)
    direction = (jnp.asarray(rng.normal(scale=.03, size=coords.shape)),
                 jnp.asarray(rng.normal(scale=.03, size=c.shape)))
    config = cc.CCConfig(conv_tol=1e-12, residual_tol=1e-11, adjoint_tol=1e-11)
    initial = _prepare(c, f, 1, energy, direct_basis=basis, max_memory=.25)
    assert _solve(initial, config=config).converged

    def evaluate(r, cs, direct):
        moving = NativeDirectBasis(basis.plan, replace(parameters,
            nuclear_coords=r[:2], centers=r[2:]))
        ints = (_prepare(cs, f, 1, energy, direct_basis=moving, max_memory=.25) if direct
                else _oracle(cs, f, 1, energy, moving.plan.evaluate('eri', moving.parameters)))
        # Fock and HF energy are independent supplied inputs in this pure chart.
        return _solve(ints, config=config).total_energy

    actual, expected = (lambda r, cs: evaluate(r, cs, True)), (lambda r, cs: evaluate(r, cs, False))
    for operation in (lambda fn, x: jax.value_and_grad(fn, argnums=(0, 1))(*x),
                      lambda fn, x: jax.jvp(jax.grad(fn, argnums=(0, 1)), x, direction)[1]):
        _assert_trees(jax.jit(lambda r, cs: operation(actual, (r, cs)))(*point),
                      operation(expected, point),
                      atol=3e-9, rtol=3e-8)
    slope = sum(jnp.vdot(g, d) for g, d in zip(jax.tree.leaves(jax.grad(
        actual, argnums=(0, 1))(*point)), jax.tree.leaves(direction)))
    for step in (3e-4, 1e-4):
        plus = jax.tree.map(lambda x, d: x+step*d, point, direction)
        minus = jax.tree.map(lambda x, d: x-step*d, point, direction)
        np.testing.assert_allclose(slope, (actual(*plus)-actual(*minus))/(2*step),
                                   atol=3e-8, rtol=3e-7)


@pytest.mark.parametrize('model', MODELS)
def test_prepared_solver_and_original_api_share_all_model_behavior(model):
    rng = np.random.default_rng(6305)
    n, nocc = 5, 2
    factors = rng.normal(scale=.025, size=(3, n, n))
    factors = factors+factors.swapaxes(1, 2)
    g = jnp.asarray(np.einsum('Lpq,Lrs->pqrs', factors, factors))
    target_fock = jnp.diag(jnp.array([-1.3, -.8, .3, .5, .9]))
    h = target_fock-2*jnp.einsum('pqii->pq', g[:, :, :nocc, :nocc])+jnp.einsum('piiq->pq', g[:, :nocc, :nocc, :])
    config = cc.CCConfig(method=model, conv_tol=1e-12, residual_tol=1e-11)
    ints = prepare_integrals(h, g, nocc=nocc, nuclear_repulsion=.7, frozen=[4])
    selected = _prepare(jnp.eye(n), target_fock, nocc, ints.reference_energy,
                         eri=g, frozen=[4], max_memory=1.)
    actual = jax.jit(lambda: _solve(selected, config=config))()
    expected = cc.run_cc(h, g, nocc=nocc, nuclear_repulsion=.7, frozen=[4], config=config)
    assert actual.converged and expected.converged
    _assert_trees(actual, expected, atol=2e-11, rtol=2e-10)
    r1, r2 = residual(actual.t1, actual.t2, selected, model=model)
    assert jnp.max(jnp.abs(r1), initial=0.) < config.residual_tol
    assert jnp.max(jnp.abs(r2), initial=0.) < config.residual_tol


def test_budget_reservation_rejects_before_native_integrals(monkeypatch):
    from gradscf.ao2mo import native
    basis, c, f, energy = _native_problem()

    def forbidden(*args, **kwargs):
        pytest.fail('CC preparation budget failure reached native integrals')

    monkeypatch.setattr(native, 'evaluate_eri_slab', forbidden)
    with pytest.raises(ValueError, match='(?i)(memory|budget|resident)'):
        _prepare(c, f, 1, energy, direct_basis=basis, max_memory=.25, reserved_bytes=200_001)


@pytest.mark.parametrize('nocc', [-1, 5, True, 1.5])
def test_invalid_occupation_is_rejected(nocc):
    basis, c, f, energy = _native_problem()
    with pytest.raises(ValueError, match='nocc'):
        _prepare(c, f, nocc, energy, direct_basis=basis)


@pytest.mark.parametrize('keyword,message', [
    ('eri', 'AO eri must have shape'),
    ('eri_pair_matrix', 'square s4 AO pair matrix'),
    ('df_factors', 'df_factors must have shape'),
])
def test_array_source_shape_precedes_local_budget(keyword, message):
    with pytest.raises(ValueError, match=message):
        _prepare(jnp.eye(3), jnp.eye(3), 1, jnp.asarray(-1.),
                 **{keyword: jnp.asarray(1.)}, max_memory=.000001)


def test_prepared_solver_never_prepares_integrals_again(monkeypatch):
    module = importlib.import_module('gradscf.cc.ground')
    ints = prepare_integrals(jnp.diag(jnp.array([-1., .5])), jnp.zeros((2,)*4), nocc=1)

    def forbidden(*args, **kwargs):
        pytest.fail('Prepared solver repeated integral preparation')

    monkeypatch.setattr(module, 'prepare_integrals', forbidden)
    result = _solve(ints)
    assert result.converged
    np.testing.assert_array_equal(result.total_energy, ints.reference_energy)
