"""Preparation keeps every spin output and its caller in one MB budget."""
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import mp
from gradscf._memory import tree_storage_bytes
from _orbital_frame import ao_source, orbital_frame
from test_direct import legacy_jax_direct_rhf


def test_full_uhf_joint_outputs_preflight_before_first_transform(monkeypatch):
    from gradscf.ao2mo import reference, blocks
    c, h, eri = jnp.eye(4), jnp.eye(4), jnp.ones((4,)*4)
    def forbidden(*a, **k):
        pytest.fail('Joint UHF output exhaustion reached an ERI contraction')
    monkeypatch.setattr(reference, 'transform_eri_block', forbidden)
    monkeypatch.setattr(blocks, 'transform_eri_block', forbidden)
    with pytest.raises(ValueError, match='memory|budget|resident'):
        reference.transform_unrestricted_integrals(h, (c, c), eri=eri, max_memory=.02)


def test_full_uhf_batch_preserves_raw_mixed_dtype_and_joint_second_response():
    from gradscf.ao2mo.reference import transform_unrestricted_integrals
    rng = np.random.default_rng(4192)
    point = tuple(jnp.asarray(rng.normal(size=shape), dtype) for shape, dtype in (
        ((3, 3), jnp.float32), ((3, 2), jnp.float32),
        ((3, 2), jnp.float64), ((3,)*4, jnp.float32)))
    tangent = jax.tree.map(lambda a: jnp.full_like(a, .02), point)
    def integrals(values, blocked):
        h, ca, cb, eri = values
        if blocked:
            return transform_unrestricted_integrals(h, (ca, cb), eri=eri,
                max_memory=.05, reserved_bytes=4096)
        def block(left, right):
            return jnp.einsum('pqrs,pi,qj,rk,sl->ijkl', eri,
                left, left, right, right, precision='highest', optimize='optimal')
        return (ca.T@h@ca, cb.T@h@cb), (block(ca, ca), block(ca, cb), block(cb, cb))
    actual, expected = integrals(point, True), integrals(point, False)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        assert a.dtype == b.dtype
        np.testing.assert_allclose(a, b, atol=3e-6, rtol=3e-6)
    def loss(values, blocked):
        return sum(jnp.sum(jnp.sin(a)**2) for a in jax.tree.leaves(integrals(values, blocked)))
    def bundle(blocked):
        return jax.jit(lambda values: jax.jvp(jax.value_and_grad(
            lambda x: loss(x, blocked)), (values,), (tangent,)))(point)
    for a, b in zip(jax.tree.leaves(bundle(True)), jax.tree.leaves(bundle(False))):
        np.testing.assert_allclose(a, b, atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize('kind', ['array', 'df'])
def test_impossible_preparation_budget_precedes_any_transform(kind, monkeypatch):
    integrals = importlib.import_module('gradscf.mp.integrals')
    c, fs, e, factors, eri = orbital_frame(0., unrestricted=True)
    def forbidden(*a, **k):
        pytest.fail('An exhausted joint input budget reached AO2MO')
    monkeypatch.setattr(integrals, 'transform_eri_blocks', forbidden, raising=False)
    monkeypatch.setattr(integrals, 'transform_df_factors', forbidden, raising=False)
    with pytest.raises(ValueError, match='memory|budget|resident'):
        integrals.from_orbitals(c, fs, (2, 1), e, max_memory=.001,
            reserved_bytes=4096, **ao_source(kind, factors, eri))


def test_three_spin_blocks_use_one_request_batch_and_reservation(monkeypatch):
    integrals = importlib.import_module('gradscf.mp.integrals')
    from gradscf.ao2mo import transform_eri_blocks
    c, fs, e, _, eri = orbital_frame(0., unrestricted=True)
    fs = tuple(f+jnp.diag(jnp.arange(4)*.11) for f in fs)
    calls = []
    def observe(coefficients, requests, **kwargs):
        calls.append((requests, kwargs['max_memory'], kwargs['reserved_bytes']))
        return transform_eri_blocks(coefficients, requests, **kwargs)
    monkeypatch.setattr(integrals, 'transform_eri_blocks', observe, raising=False)
    data = integrals.from_orbitals(c, fs, (2, 1), e, frozen=([0], [3]), eri=eri,
                                   max_memory=2., reserved_bytes=4096)
    assert len(calls) == 1 and len(calls[0][0]) == 3
    assert calls[0][1] == 2. and calls[0][2] > 4096
    assert tuple(x.shape for x in data.blocks) == ((1, 2, 1, 2),)*3


@pytest.mark.parametrize('unrestricted,kind', [(False, 'array'), (True, 'df')])
def test_joint_preparation_and_evaluation_keep_second_response(unrestricted, kind):
    integrals = importlib.import_module('gradscf.mp.integrals')
    from gradscf.mp.mp2 import evaluate_mp2
    config = mp.MPConfig(orbital_basis='noncanonical', with_t2=True)
    nocc = (2, 1) if unrestricted else 2
    def energy(x, budgeted):
        c, fs, e, factors, eri = orbital_frame(x, unrestricted=unrestricted)
        source = ao_source(kind, factors, eri)
        options = dict(max_memory=.1, reserved_bytes=4096) if budgeted else {}
        data = integrals.from_orbitals(c, fs, nocc, e, **source, **options)
        caller = 4096+tree_storage_bytes(c, fs, e, source, exclude=(data,))
        evaluation = dict(max_memory=.1, reserved_bytes=caller) if budgeted else {}
        return evaluate_mp2(data, config, **evaluation).total_energy
    bundle = lambda budgeted: jax.jit(lambda x: jax.jvp(
        jax.value_and_grad(lambda y: energy(y, budgeted)), (x,), (1.,)))(0.)
    for a, b in zip(jax.tree.leaves(bundle(True)), jax.tree.leaves(bundle(False))):
        np.testing.assert_allclose(a, b, atol=3e-11, rtol=3e-10)


def test_legacy_source_caller_reservation_precedes_packed_eri(legacy_jax_direct_rhf, monkeypatch):
    from gradscf.ao2mo.scf import prepare_scf_source
    from gradscf.scf.inputs.types import RKSIntegralInputs
    source, _ = legacy_jax_direct_rhf
    monkeypatch.setattr(RKSIntegralInputs, 'response_eri_pair_matrix', lambda *a: pytest.fail(
        'Caller exhaustion reached legacy AO ERI construction'))
    with pytest.raises(ValueError, match='memory|budget|resident'):
        prepare_scf_source(source, max_memory=1., reserved_bytes=2_000_000)
