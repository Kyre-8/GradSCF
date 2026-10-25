"""Multi-request AO->MO values, shared work, memory and off-manifold AD."""
from dataclasses import replace
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import ao2mo, integrals
from gradscf.integrals.backends.native.jk import NativeDirectBasis, ProjectedNativeDirectBasis


REQUESTS = ((0, 1, 2, 3), (0, 1, 4, 3), (0, 4, 2, 3),
            (4, 1, 2, 3), (0, 1, 2, 3), (1, 0, 3, 2), (0, 1, 2, 5))


def _batch(coefficients, requests, **kwargs):
    function = getattr(ao2mo, 'transform_eri_blocks', None)
    assert callable(function), 'Missing multi-request AO->MO transform_eri_blocks API'
    return function(coefficients, requests, **kwargs)


def _problem(cart=True, basis_spec='3-21g'):
    topology, parameters = integrals.prepare_basis(
        'H 0 0 0; H .2 -.1 .8', basis=basis_spec, cart=cart)
    basis = NativeDirectBasis(integrals.make_plan(topology, backend='native'), parameters)
    rng = np.random.default_rng(6201)
    # Spaces 0 and 4 have the same shape and different values. IDs, not
    # shape/equality guesses, determine sharing and coefficient derivatives.
    coefficients = tuple(jnp.asarray(rng.normal(scale=.25, size=(topology.nao, n)))
                         for n in (2, 3, 1, 4, 2, 5))
    return basis, coefficients


def _dense(eri, coefficients, requests):
    return tuple(jnp.einsum('pqrs,pP,qQ,rR,sS->PQRS', eri,
        *(coefficients[i] for i in request), precision='highest', optimize='optimal')
        for request in requests)


def _assert_trees(actual, expected, *, atol=2e-12, rtol=2e-11):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for a, e in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        assert a.shape == e.shape
        if os.environ.get('GRADSCF_TEST_GPU') == '1':
            assert a.devices() == {jax.devices('gpu')[0]}
        np.testing.assert_allclose(a, e, atol=atol, rtol=rtol)


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


@pytest.mark.parametrize('cart,basis_spec', [
    (True, '3-21g'),
    (True, {'H': [[0, [1.1, 1.]], [2, [.7, 1.]]]}),
    (False, {'H': [[0, [1.1, 1.]], [2, [.7, 1.]]]}),
], ids=['s', 'cartesian-d', 'spherical-d'])
def test_native_batch_asymmetric_spaces_match_dense(cart, basis_spec):
    basis, coefficients = _problem(cart, basis_spec)
    expected = _dense(basis.plan.evaluate('eri', basis.parameters), coefficients, REQUESTS)
    actual = jax.jit(lambda b, cs: _batch(cs, REQUESTS, direct_basis=b,
        max_memory=4.))(basis, coefficients)
    _assert_trees(actual, expected)


def test_native_batch_frozen_unequal_spin_spaces():
    basis, _ = _problem()
    rng = np.random.default_rng(6202)
    ca, cb = (jnp.asarray(rng.normal(scale=.25, size=(basis.nao, 6))) for _ in range(2))
    # nocc=(3,2), frozen alpha=(0,5), beta=(1,4): independent occupied
    # and virtual removals give unequal active alpha/beta dimensions.
    coefficients = (ca[:, (1, 2)], ca[:, (3, 4)], cb[:, (0,)], cb[:, (2, 3, 5)])
    requests = ((0, 1, 0, 1), (0, 1, 2, 3), (2, 3, 2, 3))
    actual = jax.jit(lambda cs: _batch(cs, requests, direct_basis=basis,
        max_memory=.25))(coefficients)
    eri = basis.plan.evaluate('eri', basis.parameters)
    full = _dense(eri, (ca, cb), ((0, 0, 0, 0), (0, 0, 1, 1), (1, 1, 1, 1)))
    expected = (full[0][jnp.ix_(jnp.array([1, 2]), jnp.array([3, 4]),
                               jnp.array([1, 2]), jnp.array([3, 4]))],
                full[1][jnp.ix_(jnp.array([1, 2]), jnp.array([3, 4]),
                               jnp.array([0]), jnp.array([2, 3, 5]))],
                full[2][jnp.ix_(jnp.array([0]), jnp.array([2, 3, 5]),
                               jnp.array([0]), jnp.array([2, 3, 5]))])
    _assert_trees(actual, expected)


def test_native_batch_eight_physical_eri_permutations():
    basis, coefficients = _problem()
    permutations = ((0, 1, 2, 3), (1, 0, 2, 3), (0, 1, 3, 2), (1, 0, 3, 2),
                    (2, 3, 0, 1), (3, 2, 0, 1), (2, 3, 1, 0), (3, 2, 1, 0))
    actual = jax.jit(lambda cs: _batch(cs, permutations, direct_basis=basis,
        max_memory=.25))(coefficients)
    eri = basis.plan.evaluate('eri', basis.parameters)
    _assert_trees(actual, _dense(eri, coefficients, permutations))


def test_native_batch_shares_slab_and_half_without_full_ao_tensor():
    basis, coefficients = _problem()
    function = lambda cs: _batch(cs, REQUESTS, direct_basis=basis, max_memory=.25)
    graph = jax.make_jaxpr(function)(coefficients)
    equations = list(_equations(graph))
    # One static slab site inside compact q/r loops, independent of request
    # count. An implementation making independent singular calls has seven.
    assert sum(e.primitive.name == 'gradscf_eri_slab_value' for e in equations) == 1
    # Five distinct canonical outputs require 2 first, 3 second, 4 half and
    # 5 final contractions. In particular the final request changes only the
    # last-contracted r space of the first request, sharing a genuine half.
    # Native ENV packing also normalizes contractions with rank<=3 dots.
    # Count the rank-four AO->MO contractions separately from that work.
    assert sum(e.primitive.name == 'dot_general' and
               len(e.outvars[0].aval.shape) == 4 for e in equations) == 14
    for graph in (graph, jax.make_jaxpr(jax.grad(
            lambda cs: sum(jnp.sum(x*x) for x in function(cs))))(coefficients)):
        for equation in _equations(graph):
            for variable in equation.outvars:
                assert getattr(getattr(variable, 'aval', None), 'shape', None) != (basis.nao,)*4


def test_native_batch_joint_geometry_coefficient_response():
    basis, coefficients = _problem()
    parameters = basis.parameters
    coords = jnp.concatenate((parameters.nuclear_coords, parameters.centers))
    rng = np.random.default_rng(6203)
    direction = (jnp.asarray(rng.normal(scale=.05, size=coords.shape)),
                 tuple(jnp.asarray(rng.normal(scale=.05, size=c.shape)) for c in coefficients))
    requests = (REQUESTS[0], REQUESTS[1], REQUESTS[0])

    def loss(r, cs, native):
        moving = NativeDirectBasis(basis.plan, replace(parameters,
            nuclear_coords=r[:2], centers=r[2:]))
        outputs = (_batch(cs, requests, direct_basis=moving, max_memory=.25) if native
                   else _dense(moving.plan.evaluate('eri', moving.parameters), cs, requests))
        # Duplicate requests receive different cotangents; the cross-output
        # term tests that sharing does not sever either response path.
        sums = tuple(jnp.sum(jnp.sin(g) + .2*g*g) for g in outputs)
        return sums[0] + .7*sums[1] - .3*sums[2] + .1*sums[0]*sums[1]

    actual, expected = (lambda r, cs: loss(r, cs, True)), (lambda r, cs: loss(r, cs, False))
    inputs = (coords, coefficients)
    for operation in (lambda f, x: jax.jvp(f, x, direction),
                      lambda f, x: jax.grad(f, argnums=(0, 1))(*x),
                      lambda f, x: jax.jvp(jax.grad(f, argnums=(0, 1)), x, direction)[1]):
        _assert_trees(jax.jit(lambda r, cs: operation(actual, (r, cs)))(*inputs),
                      operation(expected, inputs),
                      atol=3e-10, rtol=3e-9)
    gradient = jax.grad(actual, argnums=(0, 1))(*inputs)
    slope = sum(jnp.vdot(g, d) for g, d in zip(jax.tree.leaves(gradient), jax.tree.leaves(direction)))
    for step in (3e-4, 1e-4):
        plus = jax.tree.map(lambda x, d: x+step*d, inputs, direction)
        minus = jax.tree.map(lambda x, d: x-step*d, inputs, direction)
        finite_difference = (actual(*plus)-actual(*minus))/(2*step)
        np.testing.assert_allclose(slope, finite_difference, atol=3e-8, rtol=3e-7)


@pytest.mark.parametrize('representation', ['dense', 'packed', 'df'])
def test_array_batch_preserves_raw_input_chart(representation):
    rng = np.random.default_rng(6204)
    coefficients = tuple(jnp.asarray(rng.normal(size=(3, n))) for n in (2, 2, 1))
    # Every input is deliberately off the physical symmetry manifold.
    shape = {'dense': (3,)*4, 'packed': (6, 6), 'df': (4, 3, 3)}[representation]
    source = jnp.asarray(rng.normal(size=shape))
    direction = jnp.asarray(rng.normal(size=shape))
    keyword = {'dense': 'eri', 'packed': 'eri_pair_matrix', 'df': 'df_factors'}[representation]
    requests = ((0, 1, 2, 1), (1, 0, 2, 1), (2, 1, 0, 1), (0, 1, 2, 1))

    def objective(data, batch):
        if batch:
            values = _batch(coefficients, requests, **{keyword: data}, max_memory=.5)
        else:
            values = tuple(ao2mo.transform_eri_block(tuple(coefficients[i] for i in request),
                **{keyword: data}, max_memory=.5) for request in requests)
        return sum((i+.5)*jnp.sum(jnp.sin(value)+value**2) for i, value in enumerate(values))

    actual, expected = (lambda x: objective(x, True)), (lambda x: objective(x, False))
    _assert_trees(jax.value_and_grad(actual)(source), jax.value_and_grad(expected)(source),
                  atol=2e-10, rtol=2e-10)
    _assert_trees(jax.jvp(jax.grad(actual), (source,), (direction,))[1],
                  jax.jvp(jax.grad(expected), (source,), (direction,))[1],
                  atol=2e-9, rtol=2e-10)


@pytest.mark.parametrize('representation', ['dense', 'packed', 'df'])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_array_batch_reuses_normalized_source_for_every_request(
        monkeypatch, representation, dtype):
    from gradscf.ao2mo import blocks

    rng = np.random.default_rng(6207)
    coefficients = tuple(jnp.asarray(rng.normal(size=(3, n)).astype(dtype)) for n in (2, 1))
    shape = {'dense': (3,)*4, 'packed': (6, 6), 'df': (4, 3, 3)}[representation]
    source = rng.normal(size=shape).astype(dtype)
    keyword = {'dense': 'eri', 'packed': 'eri_pair_matrix', 'df': 'df_factors'}[representation]
    requests = ((0, 1, 0, 1), (1, 0, 0, 1), (0, 1, 0, 1))
    original_source, original_single = blocks._array_source, blocks.transform_eri_block
    normalized, forwarded = [], []

    def normalize(*args):
        result = original_source(*args)
        normalized.append(result[1])
        return result

    def single(cs, **kwargs):
        forwarded.append(kwargs[keyword])
        return original_single(cs, **kwargs)

    monkeypatch.setattr(blocks, '_array_source', normalize)
    monkeypatch.setattr(blocks, 'transform_eri_block', single)
    actual = _batch(coefficients, requests, **{keyword: source}, max_memory=.5)
    assert len(forwarded) == 2
    assert all(value is normalized[0] for value in forwarded)
    expected = tuple(original_single(tuple(coefficients[i] for i in request),
        **{keyword: normalized[0]}, max_memory=.5) for request in requests)
    assert all(value.dtype == dtype for value in actual)
    tolerance = 3e-6 if dtype == np.float32 else 2e-12
    _assert_trees(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize('max_memory', [60000., .000001], ids=['ample-budget', 'tiny-budget'])
@pytest.mark.parametrize('keyword,shape,message', [
    ('eri', (), 'AO eri must have shape'),
    ('eri', (3, 3), 'AO eri must have shape'),
    ('eri', (3, 3, 3, 2), 'AO eri must have shape'),
    ('eri_pair_matrix', (), 'square s4 AO pair matrix'),
    ('eri_pair_matrix', (6,), 'square s4 AO pair matrix'),
    ('eri_pair_matrix', (6, 5), 'square s4 AO pair matrix'),
    ('df_factors', (), 'df_factors must have shape'),
    ('df_factors', (3, 3), 'df_factors must have shape'),
    ('df_factors', (4, 3, 2), 'df_factors must have shape'),
])
def test_array_batch_shape_validation_precedes_budget_and_contraction(
        monkeypatch, keyword, shape, message, max_memory):
    coefficients = (jnp.ones((3, 2)), jnp.full((3, 1), .7))

    def forbidden(*args, **kwargs):
        pytest.fail('Invalid array source reached its first contraction')

    monkeypatch.setattr(jnp, 'einsum', forbidden)
    with pytest.raises(ValueError, match=message):
        _batch(coefficients, ((0, 1, 0, 1),), **{keyword: jnp.zeros(shape)},
               max_memory=max_memory)


@pytest.mark.parametrize('reserved', [-1, True, 1.5])
def test_batch_rejects_invalid_reserved_bytes(reserved):
    basis, coefficients = _problem()
    with pytest.raises(ValueError, match='reserved_bytes'):
        _batch(coefficients, REQUESTS, direct_basis=basis, max_memory=.25,
               reserved_bytes=reserved)


@pytest.mark.parametrize('requests', [((0, 1, 2),), ((0, 1, 2, 6),),
                                      ((0, 1, 2, -1),), ((0, True, 2, 3),)])
def test_batch_rejects_invalid_static_labels(requests):
    basis, coefficients = _problem()
    with pytest.raises(ValueError, match='(?i)(request|index|label|integer)'):
        _batch(coefficients, requests, direct_basis=basis, max_memory=.25)


def test_batch_reservation_precedes_projection_allocation():
    basis, coefficients = _problem()

    class UnallocatedProjection:
        shape = (basis.nao, basis.nao)
        dtype = jnp.dtype(jnp.float64)
        ndim = 2
        nbytes = basis.nao**2*8

        def __jax_array__(self):
            pytest.fail('External reservation was not charged before projection allocation')

    with pytest.raises(ValueError, match='(?i)(memory|budget|resident|minimum)'):
        _batch(coefficients, REQUESTS,
            direct_basis=ProjectedNativeDirectBasis(basis, UnallocatedProjection()),
            max_memory=1., reserved_bytes=800_001)


def test_batch_budget_charges_simultaneously_retained_outputs(monkeypatch):
    # A single float64 12^4 output and its dense intermediates fit 0.6 MB;
    # two independent outputs and their existing two-buffer contract do not.
    coefficients = (jnp.ones((3, 12)), jnp.full((3, 12), .7))
    source = jnp.ones((3,)*4)
    request = (0, 0, 0, 0)
    single = ao2mo.transform_eri_block(tuple(coefficients[i] for i in request),
        eri=source, max_memory=.6)
    assert single.shape == (12,)*4

    def forbidden(*args, **kwargs):
        pytest.fail('Batch output budget was checked after a contraction allocated')

    monkeypatch.setattr(jnp, 'einsum', forbidden)
    with pytest.raises(ValueError, match='(?i)(memory|budget|output)'):
        _batch(coefficients, (request, (0, 0, 0, 1)), eri=source, max_memory=.6)


@pytest.mark.parametrize('dtype', [jnp.float32, jnp.float64])
def test_batch_projected_transform_keeps_shared_response(dtype):
    basis, coefficients = _problem()
    transform = (jnp.eye(basis.nao)[:, :3]+.03).astype(dtype)
    coefficients = tuple(c[:3] for c in coefficients)
    requests = (REQUESTS[0], REQUESTS[1], REQUESTS[0])
    eri = basis.plan.evaluate('eri', basis.parameters)

    def objective(t, native):
        values = (_batch(coefficients, requests,
            direct_basis=ProjectedNativeDirectBasis(basis, t), max_memory=.25) if native
            else _dense(eri, tuple(t@c for c in coefficients), requests))
        return sum((i+1)*jnp.sum(value**2) for i, value in enumerate(values))

    tolerance = 3e-6 if dtype == jnp.float32 else 2e-10
    _assert_trees(jax.value_and_grad(lambda t: objective(t, True))(transform),
                  jax.value_and_grad(lambda t: objective(t, False))(transform),
                  atol=tolerance/10, rtol=tolerance)


def test_empty_batch_and_empty_spaces_do_not_evaluate_integrals(monkeypatch):
    basis, coefficients = _problem()
    from gradscf.ao2mo import native

    def forbidden(*args, **kwargs):
        pytest.fail('Empty AO->MO request evaluated native integrals')

    monkeypatch.setattr(native, 'evaluate_eri_slab', forbidden)
    assert _batch(coefficients, (), direct_basis=basis, max_memory=.25) == ()
    coefficients = (*coefficients, coefficients[0][:, :0])
    actual = _batch(coefficients, ((6, 1, 2, 3), (0, 1, 6, 3)),
                    direct_basis=basis, max_memory=.25)
    assert tuple(x.shape for x in actual) == ((0, 3, 1, 4), (2, 3, 0, 4))
    assert all(x.dtype == jnp.float64 for x in actual)


def _portable_native_response():
    """Small CPU/CUDA parity payload; physical/AD oracles are tested above."""
    basis, coefficients = _problem()
    parameters = basis.parameters
    coords = jnp.concatenate((parameters.nuclear_coords, parameters.centers))
    requests = (REQUESTS[0], REQUESTS[-1], REQUESTS[0])
    rng = np.random.default_rng(6206)
    direction = (jnp.asarray(rng.normal(scale=.05, size=coords.shape)),
                 tuple(jnp.asarray(rng.normal(scale=.03, size=c.shape)) for c in coefficients))

    def native_values(r, cs):
        moving = NativeDirectBasis(basis.plan, replace(parameters,
            nuclear_coords=r[:2], centers=r[2:]))
        return _batch(cs, requests, direct_basis=moving, max_memory=.25)

    def loss(r, cs):
        values = native_values(r, cs)
        sums = tuple(jnp.sum(jnp.sin(g)+.2*g*g) for g in values)
        return sums[0]+.7*sums[1]-.3*sums[2]+.1*sums[0]*sums[1]

    inputs = (coords, coefficients)
    values = jax.jit(native_values)(*inputs)
    response = (jax.jit(lambda r, cs: jax.jvp(loss, (r, cs), direction))(*inputs),
                jax.jit(jax.grad(loss, argnums=(0, 1)))(*inputs),
                jax.jit(lambda r, cs: jax.jvp(jax.grad(loss, argnums=(0, 1)),
                    (r, cs), direction)[1])(*inputs))
    single = jax.jit(lambda cs: ao2mo.transform_eri_block(
        tuple(cs[i] for i in REQUESTS[0]), direct_basis=basis, max_memory=.25))(coefficients)
    return values, response, single


@pytest.mark.skipif(os.environ.get('GRADSCF_TEST_GPU') != '1',
                    reason='Explicitly selected CUDA GPU required')
def test_native_batch_gpu_device_parity_and_pyscf_values():
    """GPU values/JVP/VJP/HVP match a separately executed CPU payload.

    CPU native AD already has exact dense and finite-difference gates above.
    Here geometry AD is device parity, while PySCF supplies the value oracle.
    No full native ERI operation is evaluated in the CUDA process.
    """
    pyscf = pytest.importorskip('pyscf')
    actual = _portable_native_response()
    assert all(x.devices() == {jax.devices('gpu')[0]} for x in jax.tree.leaves(actual))
    with np.load(os.environ['GRADSCF_AO2MO_CPU_ORACLE']) as reference:
        expected = jax.tree.unflatten(jax.tree.structure(actual),
            [jnp.asarray(reference[f'arr_{i}']) for i in range(len(reference.files))])
    _assert_trees(actual, expected, atol=3e-10, rtol=3e-9)
    basis, coefficients = _problem()
    mol = pyscf.gto.M(atom=list(zip(('H', 'H'), np.asarray(basis.parameters.nuclear_coords))),
                      basis='3-21g', unit='Bohr', cart=True, verbose=0)
    requests = (REQUESTS[0], REQUESTS[-1], REQUESTS[0])
    eri = jnp.asarray(mol.intor('int2e', aosym='s1'))
    expected_values = _dense(eri, coefficients, requests)
    _assert_trees(actual[0], expected_values)
    _assert_trees((actual[2],), (actual[0][0],))

    def coefficient_loss(cs, native):
        values = (_batch(cs, requests, direct_basis=basis, max_memory=.25) if native
                  else _dense(eri, cs, requests))
        return sum((i+.5)*jnp.sum(jnp.sin(g)+.2*g*g) for i, g in enumerate(values))

    native_loss = lambda cs: coefficient_loss(cs, True)
    reference_loss = lambda cs: coefficient_loss(cs, False)
    direction = tuple(.03*c for c in coefficients)
    _assert_trees(jax.jit(jax.value_and_grad(native_loss))(coefficients),
                  jax.value_and_grad(reference_loss)(coefficients), atol=3e-10, rtol=3e-9)
    _assert_trees(jax.jit(lambda cs: jax.jvp(jax.grad(native_loss), (cs,), (direction,))[1])(
                      coefficients),
                  jax.jvp(jax.grad(reference_loss), (coefficients,), (direction,))[1],
                  atol=3e-10, rtol=3e-9)
