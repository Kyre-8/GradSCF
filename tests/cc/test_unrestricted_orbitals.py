"""Selected UHF AO preparation, including the unsymmetrized array AD chart."""
from dataclasses import replace
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.cc.uccsd import prepare_ucc_integrals, run_ucc


def _prepare(*args, **kwargs):
    module = importlib.import_module('gradscf.cc.unrestricted_integrals')
    return module.from_orbitals(*args, **kwargs)


def _problem(representation='dense', dtype=np.float64):
    rng = np.random.default_rng(6510)
    cs = tuple(jnp.asarray(rng.normal(scale=.2, size=(4, 4)), dtype) for _ in range(2))
    fs = tuple(jnp.asarray(rng.normal(scale=.02, size=(4, 4)), dtype)
               + jnp.diag(jnp.asarray(values, dtype)) for values in
               ((-1.3, -.8, .3, .7), (-1.2, .2, .5, .9)))
    shape = {'dense': (4,)*4, 'packed': (10, 10), 'df': (3, 4, 4)}[representation]
    source = jnp.asarray(rng.normal(scale=.04, size=shape), dtype)
    key = {'dense': 'eri', 'packed': 'eri_pair_matrix', 'df': 'df_factors'}[representation]

    def expand(value):
        if representation == 'dense':
            return value
        if representation == 'df':
            return jnp.einsum('Lpq,Lrs->pqrs', value, value, precision='highest')
        rows, cols = np.tril_indices(4)
        indices = np.empty((4, 4), dtype=np.int32)
        indices[rows, cols] = indices[cols, rows] = np.arange(10)
        return value[indices[:, :, None, None], indices[None, None, :, :]]

    return cs, fs, jnp.asarray(-3.4, dtype), source, key, expand


def _raw_frame(cs, fs, energy, ao_eri, nocc):
    # Tiny full MO tensors are independent oracles only. The beta-alpha chart
    # is exactly the existing pair transpose of gab, even for raw AO arrays.
    gs = tuple(jnp.einsum('pqrs,pi,qj,rk,sl->ijkl', ao_eri,
        cs[a], cs[a], cs[b], cs[b], precision='highest', optimize='optimal')
        for a, b in ((0, 0), (0, 1), (1, 1)))
    aa, ab, bb = gs
    na, nb = nocc
    va = (jnp.einsum('pqii->pq', aa[:, :, :na, :na])
          + jnp.einsum('pqii->pq', ab[:, :, :nb, :nb])
          - jnp.einsum('piiq->pq', aa[:, :na, :na, :]))
    vb = (jnp.einsum('pqii->pq', bb[:, :, :nb, :nb])
          + jnp.einsum('iipq->pq', ab[:na, :na])
          - jnp.einsum('piiq->pq', bb[:, :nb, :nb, :]))
    hs = fs[0]-va, fs[1]-vb
    electronic = .5*(jnp.trace((hs[0]+fs[0])[:na, :na])
                       + jnp.trace((hs[1]+fs[1])[:nb, :nb]))
    return hs, gs, energy-electronic


def _oracle(cs, fs, nocc, energy, ao_eri, frozen=None):
    hs, gs, enuc = _raw_frame(cs, fs, energy, ao_eri, nocc)
    return prepare_ucc_integrals(hs, gs, nocc=nocc, frozen=frozen, nuclear_repulsion=enuc)


def _assert_trees(actual, expected, *, atol=2e-12, rtol=2e-11):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for a, e in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        assert np.shape(a) == np.shape(e)
        np.testing.assert_allclose(a, e, atol=atol, rtol=rtol)


@pytest.mark.parametrize('representation', ['dense', 'packed', 'df'])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('nocc,frozen', [((2, 1), None), ((2, 2), 1),
                                        ((2, 1), ([0, 3], [2]))])
def test_selected_spin_blocks_match_raw_oracle(representation, dtype, nocc, frozen):
    cs, fs, energy, source, key, expand = _problem(representation, dtype)
    actual = _prepare(cs, fs, nocc, energy, frozen=frozen, max_memory=2., **{key: source})
    expected = _oracle(cs, fs, nocc, energy, expand(source), frozen)
    tolerance = 3e-6 if dtype == np.float32 else 2e-12
    _assert_trees(actual, expected, atol=tolerance, rtol=tolerance)
    assert all(x.dtype == dtype for x in actual[0])
    if frozen is None:
        assert np.linalg.norm(np.asarray(actual[0].fock-actual[0].fock.T)) > .01


@pytest.mark.parametrize('representation', ['dense', 'packed', 'df'])
def test_joint_coefficient_fock_energy_source_jvp_vjp_hvp(representation):
    cs, fs, energy, source, key, expand = _problem(representation)
    point = (cs, fs, energy, source)
    rng = np.random.default_rng(6511)
    direction = jax.tree.map(lambda x: jnp.asarray(rng.normal(scale=.03, size=x.shape)), point)

    def loss(values, direct):
        c, f, e, g = values
        ints = (_prepare(c, f, (2, 1), e, frozen=([0], [3]), max_memory=2., **{key: g})
                if direct else _oracle(c, f, (2, 1), e, expand(g), ([0], [3])))[0]
        return sum((i+1)*jnp.sum(block**2) for i, block in enumerate(ints))

    def bundle(values, direct):
        function = lambda x: loss(x, direct)
        value, tangent = jax.jvp(function, (values,), (direction,))
        _, pullback = jax.vjp(function, values)
        gradient, = pullback(jnp.asarray(1.))
        hvp = jax.jvp(jax.grad(function), (values,), (direction,))[1]
        return value, tangent, gradient, hvp

    _assert_trees(jax.jit(lambda x: bundle(x, True))(point),
                  jax.jit(lambda x: bundle(x, False))(point), atol=3e-10, rtol=3e-9)


def _native_problem(projected=False):
    from gradscf import integrals
    from gradscf.integrals.backends.native.jk import NativeDirectBasis, ProjectedNativeDirectBasis

    topology, parameters = integrals.prepare_basis('H 0 0 0; H .2 -.1 .8', basis='3-21g')
    primitive = NativeDirectBasis(integrals.make_plan(topology, backend='native'), parameters)
    transform = jnp.eye(4)+jnp.asarray(np.random.default_rng(6512).normal(scale=.03, size=(4, 4)))
    basis = ProjectedNativeDirectBasis(primitive, transform) if projected else primitive
    cs, fs, energy, _, _, _ = _problem()
    return primitive, basis, cs, fs, energy, transform


@pytest.mark.parametrize('projected', [False, True])
def test_native_is_one_batch_without_full_mo_or_spin_materialization(projected, monkeypatch):
    module = importlib.import_module('gradscf.cc.unrestricted_integrals')
    primitive, basis, cs, fs, energy, transform = _native_problem(projected)
    effective = tuple(transform@c for c in cs) if projected else cs
    expected = _oracle(effective, fs, (2, 1), energy,
        primitive.plan.evaluate('eri', primitive.parameters), ([0], [3]))
    original, seen = module.transform_eri_blocks, []

    def batch(coefficients, requests, **kwargs):
        seen.append((tuple(c.shape for c in coefficients), tuple(requests), kwargs))
        return original(coefficients, requests, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail('Selected UHF preparation requested complete AO/MO/spin integrals')

    monkeypatch.setattr(module, 'transform_eri_blocks', batch)
    monkeypatch.setattr(type(primitive.plan), 'evaluate', forbidden)
    from gradscf.ao2mo import reference, spin
    monkeypatch.setattr(reference, 'transform_unrestricted_integrals', forbidden)
    monkeypatch.setattr(spin, 'spin_orbital_integrals', forbidden)
    actual = _prepare(cs, fs, (2, 1), energy, frozen=([0], [3]), direct_basis=basis,
                      max_memory=2., reserved_bytes=4096)
    _assert_trees(actual, expected)
    assert len(seen) == 1 and len(seen[0][0]) == 4
    assert all(shape[1] < 4 for shape in seen[0][0])
    assert all(len(request) == 4 for request in seen[0][1])
    assert seen[0][2]['max_memory'] == 2.
    assert seen[0][2]['reserved_bytes'] > 4096+sum(x.nbytes for x in actual[0])


def test_native_joint_geometry_coefficient_fock_energy_response():
    from gradscf.integrals.backends.native.jk import NativeDirectBasis
    primitive, _, cs, fs, energy, _ = _native_problem()
    parameters = primitive.parameters
    coords = jnp.concatenate((parameters.nuclear_coords, parameters.centers))
    point = (coords, cs, fs, energy)
    rng = np.random.default_rng(6513)
    direction = jax.tree.map(lambda x: jnp.asarray(rng.normal(scale=.02, size=x.shape)), point)

    def loss(values, selected):
        r, c, f, e = values
        moving = NativeDirectBasis(primitive.plan, replace(parameters,
            nuclear_coords=r[:2], centers=r[2:]))
        ints = (_prepare(c, f, (2, 1), e, direct_basis=moving, max_memory=2.)
                if selected else _oracle(c, f, (2, 1), e,
                    moving.plan.evaluate('eri', moving.parameters)))[0]
        return sum((i+1)*jnp.sum(x*x) for i, x in enumerate(ints))

    def bundle(values, selected):
        function = lambda x: loss(x, selected)
        value, tangent = jax.jvp(function, (values,), (direction,))
        value2, gradient = jax.value_and_grad(function)(values)
        hvp = jax.jvp(jax.grad(function), (values,), (direction,))[1]
        return value, value2, tangent, gradient, hvp

    _assert_trees(jax.jit(lambda x: bundle(x, True))(point),
                  jax.jit(lambda x: bundle(x, False))(point), atol=3e-9, rtol=3e-8)


def test_final_spin_storage_budget_is_checked_before_transform_or_assembly(monkeypatch):
    module = importlib.import_module('gradscf.cc.unrestricted_integrals')
    cs = (jnp.ones((2, 8)),)*2
    fs, energy, source = (jnp.eye(8),)*2, jnp.asarray(-3.), jnp.ones((2,)*4)

    def forbidden(*args, **kwargs):
        pytest.fail('Insufficient spin-output budget reached contraction or assembly')

    monkeypatch.setattr(module, 'transform_eri_blocks', forbidden)
    monkeypatch.setattr(jnp, 'zeros', forbidden)
    with pytest.raises(ValueError, match='budget|memory|resident'):
        _prepare(cs, fs, (4, 3), energy, eri=source, max_memory=.1)


@pytest.mark.parametrize('bad', ['restricted', 'occupations', 'complex', 'shape', 'two-sources'])
def test_unrestricted_input_validation_precedes_contraction(bad, monkeypatch):
    module = importlib.import_module('gradscf.cc.unrestricted_integrals')
    cs, fs, energy, source, _, _ = _problem()
    nocc, options = (2, 1), dict(eri=source)
    if bad == 'restricted':
        cs = cs[0]
    elif bad == 'occupations':
        nocc = (True, 1)
    elif bad == 'complex':
        fs = fs[0].astype(jnp.complex128), fs[1]
    elif bad == 'shape':
        cs = cs[0], cs[1][:, :3]
    else:
        options['df_factors'] = jnp.ones((1, 4, 4))
    monkeypatch.setattr(module, 'transform_eri_blocks',
        lambda *a, **kw: pytest.fail('Invalid input reached integral transformation'))
    with pytest.raises((ValueError, NotImplementedError)):
        _prepare(cs, fs, nocc, energy, **options)


@pytest.mark.parametrize('method', ['ccsd', 'ccd'])
def test_prepared_and_raw_ucc_share_the_same_solver_body(method):
    from gradscf.cc.types import CCConfig
    from gradscf.cc.uccsd import run_ucc_from_integrals

    cs, fs, energy, source, _, _ = _problem('df')
    source = source+source.swapaxes(1, 2)
    fs = tuple(jnp.diag(jnp.diag(f)) for f in fs)
    ao_eri = jnp.einsum('Lpq,Lrs->pqrs', source, source, precision='highest')
    prepared = _prepare(cs, fs, (2, 1), energy, df_factors=source, max_memory=2.)
    h, g, enuc = _raw_frame(cs, fs, energy, ao_eri, (2, 1))
    cfg = CCConfig(method=method, conv_tol=1e-12, residual_tol=1e-11)
    actual = jax.jit(lambda c, f, e, data: run_ucc_from_integrals(
        *_prepare(c, f, (2, 1), e, df_factors=data, max_memory=2.), config=cfg))(cs, fs, energy, source)
    expected = jax.jit(lambda h, g, en: run_ucc(h, g, nocc=(2, 1),
        nuclear_repulsion=en, config=cfg))(h, g, enuc)
    assert actual.converged and expected.converged
    _assert_trees(actual[:-4], expected[:-4], atol=3e-10, rtol=3e-9)
    assert prepared[1:] == ((2, 1), (2, 3))
