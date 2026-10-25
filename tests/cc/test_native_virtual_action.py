"""Exact native bare-vvvv action and restricted prepared ground integration."""
from dataclasses import replace
import importlib
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import cc, integrals
from gradscf.cc import _intermediates as imd, rccsd
from gradscf.cc.integrals import from_orbitals
from gradscf.integrals.backends.native.jk import (
    NativeDirectBasis, ProjectedNativeDirectBasis,
)


def _action(c, basis, **kwargs):
    module = importlib.import_module('gradscf.ao2mo')
    cls = getattr(module, 'VirtualERI', None)
    assert cls is not None, 'Missing bounded native VirtualERI action'
    return cls(c, basis, **kwargs)


def _problem(projected=False):
    topology, parameters = integrals.prepare_basis(
        'H 0 0 0; H .2 -.1 .8', basis='3-21g')
    primitive = NativeDirectBasis(integrals.make_plan(topology, backend='native'), parameters)
    rng = np.random.default_rng(6410)
    transform = jnp.asarray(np.eye(topology.nao)+rng.normal(scale=.04, size=(topology.nao,)*2))
    basis = ProjectedNativeDirectBasis(primitive, transform) if projected else primitive
    c = jnp.asarray(rng.normal(scale=.2, size=(basis.nao, 3)))
    return primitive, basis, c, transform


def _dense(primitive, c, transform=None):
    if transform is not None:
        c = transform@c
    g = primitive.plan.evaluate('eri', primitive.parameters)
    return jnp.einsum('pqrs,pa,qc,rb,sd->acbd', g, c, c, c, c, precision='highest')


def _assert_trees(actual, expected, atol=3e-10, rtol=3e-9):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for a, e in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(a, e, atol=atol, rtol=rtol)


def test_ao2mo_provider_imports_before_scf_and_cc():
    result = subprocess.run([sys.executable, '-c',
        'import sys; import gradscf.ao2mo; assert "gradscf.scf" not in sys.modules; import gradscf.cc'],
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('projected', [False, True])
@pytest.mark.parametrize('leading', [(), (5,), (2, 3)])
def test_bare_native_action_matches_exchange_for_general_rhs(projected, leading):
    primitive, basis, c, transform = _problem(projected)
    rng = np.random.default_rng(6411)
    x = jnp.asarray(rng.normal(size=leading+(3, 3)))
    g = _dense(primitive, c, transform if projected else None)
    action = _action(c, basis, max_memory=.5)
    actual = jax.jit(lambda operator, rhs: operator.contract(rhs))(action, x)
    expected = jnp.einsum('acbd,...cd->...ab', g, x)
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-11)
    anti = x-jnp.swapaxes(x, -1, -2)
    result = action.contract(anti)
    np.testing.assert_allclose(result, jnp.einsum('acbd,...cd->...ab', g, anti),
                               atol=2e-12, rtol=2e-11)
    assert np.max(np.abs(result)) > 1e-5, 'Antisymmetric RHS was lost as in a Coulomb action'


@pytest.mark.parametrize('projected', [False, True])
def test_native_action_joint_geometry_coefficient_rhs_projection_ad(projected, monkeypatch):
    primitive, basis, c, transform = _problem(projected)
    parameters = primitive.parameters
    nnuc = len(primitive.plan.topology.nuclear_charges)
    coords = jnp.concatenate((parameters.nuclear_coords, parameters.centers))
    rng = np.random.default_rng(6412)
    rhs = jnp.asarray(rng.normal(size=(2, 3, 3)))
    point = (coords, c, rhs, transform)
    tangent = tuple(jnp.asarray(rng.normal(scale=.03, size=p.shape)) for p in point)
    probe = jnp.asarray(rng.normal(size=rhs.shape))

    def evaluate(values, direct):
        r, coeff, x, projection = values
        moving = NativeDirectBasis(primitive.plan, replace(parameters,
            nuclear_coords=r[:nnuc], centers=r[nnuc:]))
        if direct:
            provider = ProjectedNativeDirectBasis(moving, projection) if projected else moving
            return _action(coeff, provider, max_memory=.5).contract(x)
        g = _dense(moving, coeff, projection if projected else None)
        return jnp.einsum('acbd,...cd->...ab', g, x)

    def bundle(values, direct):
        function = lambda v: evaluate(v, direct)
        value, jvp = jax.jvp(function, (values,), (tangent,))
        _, pullback = jax.vjp(function, values)
        vjp, = pullback(probe)
        loss = lambda v: jnp.sum(function(v)**2)
        hvp = jax.jvp(jax.grad(loss), (values,), (tangent,))[1]
        return value, jvp, vjp, hvp

    expected = bundle(point, False)
    original = type(primitive.plan).evaluate

    def forbid_full(self, operator, *args, **kwargs):
        if operator == 'eri':
            raise AssertionError('Native virtual action materialized AO ERIs')
        return original(self, operator, *args, **kwargs)

    monkeypatch.setattr(type(primitive.plan), 'evaluate', forbid_full)
    actual = jax.jit(lambda values: bundle(values, True))(point)
    _assert_trees(actual, expected, atol=3e-9, rtol=3e-8)
    gradient = jax.jit(jax.grad(lambda v: jnp.sum(evaluate(v, True)**2)))
    step = 1e-4
    plus = jax.tree.map(lambda v, d: v+step*d, point, tangent)
    minus = jax.tree.map(lambda v, d: v-step*d, point, tangent)
    fd = jax.tree.map(lambda a, b: (a-b)/(2*step), gradient(plus), gradient(minus))
    _assert_trees(actual[-1], fd, atol=3e-7, rtol=3e-6)


@pytest.mark.parametrize('shape', [(0, 3, 3), (2, 0, 0)])
def test_native_action_empty_output_and_early_budget_validation(shape, monkeypatch):
    _, basis, c, _ = _problem()
    if shape[-1] == 0:
        c = c[:, :0]
    x = jnp.zeros(shape)
    action = _action(c, basis, max_memory=.5)
    monkeypatch.setattr(NativeDirectBasis, 'get_jk', lambda *a, **kw: pytest.fail('Empty output called J/K'))
    result = action.contract(x)
    assert result.shape == shape and result.dtype == jnp.float64
    np.testing.assert_array_equal(result, 0.)
    with pytest.raises(ValueError, match='budget|resident|memory'):
        _action(c, basis, max_memory=.000001).contract(x)


def test_native_action_rejects_nested_projection_and_bad_budget_before_density(monkeypatch):
    _, basis, c, transform = _problem(True)
    monkeypatch.setattr(NativeDirectBasis, 'get_jk', lambda *a, **kw: pytest.fail('Invalid action called J/K'))
    nested = ProjectedNativeDirectBasis(basis, transform)
    with pytest.raises((TypeError, NotImplementedError), match='nested|single|NativeDirectBasis'):
        _action(c, nested, max_memory=.5).contract(jnp.ones((3, 3)))
    for kwargs in ({'max_memory': 0.}, {'max_memory': .000001}, {'reserved_bytes': -1}):
        with pytest.raises(ValueError):
            _action(c, basis, **kwargs).contract(jnp.ones((3, 3)))


@pytest.mark.parametrize('projected', [False, True])
def test_native_action_budget_splits_rhs_and_retains_tail(projected, monkeypatch):
    primitive, basis, c, transform = _problem(projected)
    rhs = jnp.asarray(np.random.default_rng(6415).normal(size=(5, 3, 3)))
    expected = jnp.einsum('acbd,kcd->kab',
        _dense(primitive, c, transform if projected else None), rhs)
    original = type(primitive.plan).get_jk
    seen = []

    def record(self, parameters, density, **kwargs):
        seen.append(density.shape[0])
        return original(self, parameters, density, **kwargs)

    monkeypatch.setattr(type(primitive.plan), 'get_jk', record)
    # Find a budget with a genuine nondividing batch size; this checks planner
    # behavior without reproducing its internal scratch formula in the test.
    for memory in np.linspace(.005, .1, 96):
        action = _action(c, basis, max_memory=float(memory))
        seen.clear()
        try:
            jax.make_jaxpr(lambda x: action.contract(x))(rhs)
        except ValueError as error:
            assert 'budget' in str(error) or 'resident' in str(error)
            continue
        if seen and 1 < seen[-1] < len(rhs) and len(rhs) % seen[-1]:
            actual = jax.jit(lambda x: action.contract(x))(rhs)
            np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-11)
            dense = _dense(primitive, c, transform if projected else None)
            actual_gradient = jax.jit(jax.grad(lambda x: jnp.sum(action.contract(x)**2)))(rhs)
            expected_gradient = jax.grad(lambda x: jnp.sum(jnp.einsum('acbd,kcd->kab', dense, x)**2))(rhs)
            np.testing.assert_allclose(actual_gradient, expected_gradient, atol=2e-11, rtol=2e-10)
            reverse = jax.make_jaxpr(jax.grad(lambda cs: jnp.sum(
                _action(cs, basis, max_memory=float(memory)).contract(rhs)**2)))(c)

            def scans(value):
                value = getattr(value, 'jaxpr', value)
                if hasattr(value, 'eqns'):
                    for equation in value.eqns:
                        if equation.primitive.name == 'scan':
                            yield equation
                        yield from scans(equation.params)
                elif isinstance(value, dict):
                    for child in value.values():
                        yield from scans(child)
                elif isinstance(value, (tuple, list)):
                    for child in value:
                        yield from scans(child)

            scan_equations = list(scans(reverse))
            assert scan_equations
            ao_shapes = {(basis.nao,)*2, (primitive.nao,)*2}
            for equation in scan_equations:
                for variable in equation.outvars:
                    shape = getattr(getattr(variable, 'aval', None), 'shape', ())
                    assert not (len(shape) >= 4 and shape[-2:] in ao_shapes), 'Reverse scan retains per-block AO density/K'
            break
    else:
        pytest.fail('No bounded RHS tail was exercised')


@pytest.mark.parametrize('model', ['ccd', 'ccsd', 'cc2', 'qcisd'])
def test_prepared_direct_action_matches_dense_residual_and_converged_ground(model, monkeypatch):
    _, basis, c, _ = _problem()
    c = jnp.concatenate((c[:, :1], .6*c[:, 1:]), axis=1)
    fock = jnp.diag(jnp.asarray([-1.2, .4, .7]))
    energy = jnp.asarray(-1.4)
    dense = from_orbitals(c, fock, 1, energy, direct_basis=basis, max_memory=.5)
    direct = from_orbitals(c, fock, 1, energy, direct_basis=basis, max_memory=.5, vvvv='direct')
    assert len(direct) == 10
    assert type(direct.vvvv).__name__ == 'VirtualERI'
    rng = np.random.default_rng(6413)
    t1 = jnp.asarray(rng.normal(scale=.02, size=(1, 2)))
    t2 = jnp.asarray(rng.normal(scale=.02, size=(1, 1, 2, 2)))
    _assert_trees(rccsd.residual(t1, t2, direct, model=model),
                  rccsd.residual(t1, t2, dense, model=model), atol=2e-12, rtol=2e-11)
    pair = (t1, .7*t1)
    x = jnp.einsum('ic,jd->ijcd', *pair)
    np.testing.assert_allclose(imd.contract_Wvvvv(t1, direct, pair),
        imd.contract_Wvvvv(t1, dense, x), atol=2e-12, rtol=2e-11)
    config = cc.CCConfig(method=model, conv_tol=1e-12, residual_tol=1e-11)
    actual = cc.run_cc_from_integrals(direct, config=config)
    expected = cc.run_cc_from_integrals(dense, config=config)
    assert actual.converged and expected.converged
    _assert_trees(actual, expected, atol=2e-11, rtol=2e-10)


@pytest.mark.parametrize('model', ['ccd', 'ccsd'])
def test_prepared_direct_action_implicit_ground_coefficient_response(model):
    _, basis, c, _ = _problem()
    c = .7*c
    fock, energy = jnp.diag(jnp.asarray([-1.2, .4, .7])), jnp.asarray(-1.4)
    config = cc.CCConfig(method=model, conv_tol=1e-12, residual_tol=1e-11, adjoint_tol=1e-11)
    direction = .03*jnp.sin(c)

    def evaluate(coeff, mode):
        prepared = from_orbitals(coeff, fock, 1, energy, direct_basis=basis,
                                 max_memory=.5, vvvv=mode)
        return cc.run_cc_from_integrals(prepared, config=config).total_energy

    def bundle(coeff, mode):
        function = lambda value: evaluate(value, mode)
        return jax.value_and_grad(function)(coeff), jax.jvp(jax.grad(function),
            (coeff,), (direction,))[1]

    _assert_trees(jax.jit(lambda value: bundle(value, 'direct'))(c), bundle(c, 'dense'),
                  atol=3e-9, rtol=3e-8)


def test_direct_preparation_never_requests_vvvv_or_full_integrals(monkeypatch):
    _, basis, c, _ = _problem()
    module = importlib.import_module('gradscf.cc.integrals')
    original = module.transform_eri_blocks
    requests = []

    def checked(coefficients, blocks, **kwargs):
        requests.extend(blocks)
        assert (1, 1, 1, 1) not in blocks
        return original(coefficients, blocks, **kwargs)

    monkeypatch.setattr(module, 'transform_eri_blocks', checked)
    prepared = from_orbitals(c, jnp.diag(jnp.asarray([-1., .4, .7])), 1, jnp.asarray(-1.),
                             direct_basis=basis, max_memory=.5, vvvv='direct')
    assert len(requests) == 6
    graph = jax.make_jaxpr(lambda p, x: p.contract(x))(prepared.vvvv, jnp.ones((2, 2)))

    def equations(value):
        value = getattr(value, 'jaxpr', value)
        if hasattr(value, 'eqns'):
            for equation in value.eqns:
                yield equation
                yield from equations(equation.params)
        elif isinstance(value, dict):
            for child in value.values():
                yield from equations(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                yield from equations(child)

    for equation in equations(graph):
        for variable in equation.outvars:
            shape = getattr(getattr(variable, 'aval', None), 'shape', None)
            assert shape not in ((basis.nao,)*4, (c.shape[1],)*4, (2,)*4)


def test_direct_mode_requires_native_and_valid_mode():
    c, f, g = jnp.eye(3), jnp.eye(3), jnp.zeros((3,)*4)
    with pytest.raises((TypeError, ValueError), match='native|direct_basis'):
        from_orbitals(c, f, 1, jnp.asarray(-1.), eri=g, vvvv='direct')
    with pytest.raises(ValueError, match='vvvv'):
        from_orbitals(c, f, 1, jnp.asarray(-1.), eri=g, vvvv='typo')
