"""The streamed triples budget includes its own first/second response owners."""
import importlib

import jax.numpy as jnp
import pytest

from gradscf import cc
from gradscf.cc.integrals import CCIntegrals
from gradscf.cc.types import CCResult


def _restricted_case(integral_dtype=jnp.float64, amplitude_dtype=jnp.float64):
    o, v = 2, 16
    zeros = lambda shape: jnp.zeros(shape, dtype=integral_dtype)
    data = CCIntegrals(*map(zeros, ((o+v,)*2, (o+v,), (o,)*4, (o,v,o,o),
        (o,v,o,v), (o,o,v,v), (o,v,v,o), (o,v,v,v), (v,)*4, ())))
    zeros = lambda shape: jnp.zeros(shape, dtype=amplitude_dtype)
    state = CCResult(zeros(()), zeros(()), zeros(()), zeros((o,v)),
        zeros((o,o,v,v)), zeros(()), zeros(()), 1, True, zeros(()), 2)
    return data, state


@pytest.mark.parametrize('memory', [2.95, 64.])
def test_streamed_ad_budget_checks_before_residual(memory, monkeypatch):
    data, state = _restricted_case()
    module = importlib.import_module('gradscf.cc.triples')

    class PhysicalWorkReached(Exception):
        pass

    def reached(*args, **kwargs):
        raise PhysicalWorkReached

    monkeypatch.setattr(module, 'residual', reached)
    # The old metadata preflight admitted about 2.946849 MB, although its first
    # pullback alone retains 2,677,016 B beyond the original input owners.
    if memory < 3.:
        with pytest.raises(ValueError, match='memory|budget'):
            cc.evaluate_triples_from_integrals(data, state, algorithm='streamed',
                orbital_basis='semicanonical', max_memory=memory)
    else:
        with pytest.raises(PhysicalWorkReached):
            cc.evaluate_triples_from_integrals(data, state, algorithm='streamed',
                orbital_basis='semicanonical', max_memory=memory)


@pytest.mark.parametrize('unrestricted', [False, True])
def test_budgeted_streamed_public_jit_keeps_first_and_second_response(unrestricted):
    import jax
    import numpy as np
    from gradscf.cc.integrals import prepare_integrals
    from gradscf.cc.uccsd import prepare_ucc_integrals
    from test_prepared_properties import _problem, _config
    from gradscf.cc._memory import _streamed_response_workspace

    _streamed_response_workspace.cache_clear()
    h, g = _problem()
    probe = jnp.diag(jnp.array([.02, -.01, .03, -.02]))
    config = _config()

    def value(x, memory):
        if unrestricted:
            data, occupied, virtual = prepare_ucc_integrals(
                (h+x*probe, h-.4*x*probe), (g,g,g), nocc=(2,1))
            state = cc.run_ucc_from_integrals(data, occupied, virtual,
                                            config=config, max_memory=memory)
        else:
            data = prepare_integrals(h+x*probe, g, nocc=2)
            state = cc.run_cc_from_integrals(data, config=config, max_memory=memory)
        correction = cc.evaluate_triples_from_integrals(data, state,
            algorithm='streamed', orbital_basis='semicanonical', max_memory=memory)
        return state.total_energy+correction.energy, (state.converged,
                                                       correction.valid, correction.energy)

    def bundle(memory):
        function = lambda x: value(x, memory)
        return jax.jit(lambda x: jax.jvp(jax.value_and_grad(function, has_aux=True),
                                        (x,), (jnp.asarray(1.),)))(jnp.asarray(.02))

    actual, expected = bundle(32.), bundle(None)
    (_, (ground_ok, triples_ok, correction)), _ = actual[0]
    assert bool(ground_ok) and bool(triples_ok) and abs(float(correction)) > 1e-14
    for a, e in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        if getattr(a.dtype, 'names', None) is None:
            assert np.isfinite(np.asarray(a)).all() and np.isfinite(np.asarray(e)).all()
            np.testing.assert_allclose(a, e, atol=3e-10, rtol=3e-8, equal_nan=False)


@pytest.mark.parametrize('reason', ['retained_validation', 'amplitude_precision'])
def test_triples_validation_owners_and_precision_are_joint(reason, monkeypatch):
    import jax
    from gradscf._memory import array_storage_bytes, tree_storage_bytes
    from gradscf.cc._memory import _streamed_response_workspace
    from gradscf.cc.amplitudes import AmplitudeSpace

    if reason == 'retained_validation':
        data, state = _restricted_case()
        inputs = (data.fock, data.ovvv, data.ovoo, data.ovov, state.t1, state.t2)
        kernel = _streamed_response_workspace(tuple(a.shape for a in inputs),
                                              tuple(a.dtype.name for a in inputs), False)
        # The checked residual arrays have amplitude shapes but are different
        # owners; the packed coordinate length follows the existing space API.
        held = array_storage_bytes(state.t1, state.t2)+AmplitudeSpace(2, 16).size*8
        memory = (tree_storage_bytes(data, state)+kernel+held//2)*1.25e-6
        options = dict(algorithm='streamed', orbital_basis='semicanonical')
    else:
        data, state = _restricted_case(jnp.float32, jnp.float64)
        memory = 1.2
        options = dict(algorithm='auto', orbital_basis='canonical')

    class PhysicalWorkReached(Exception):
        pass

    def reached(*args, **kwargs):
        raise PhysicalWorkReached

    module = importlib.import_module('gradscf.cc.triples')
    monkeypatch.setattr(module, 'residual', reached)
    with pytest.raises(ValueError, match='memory|budget'):
        cc.evaluate_triples_from_integrals(data, state, max_memory=memory, **options)


def test_streamed_complete_lift_budget_is_checked(monkeypatch):
    data, state = _restricted_case()
    module = importlib.import_module('gradscf.cc.triples')

    class PhysicalWorkReached(Exception):
        pass

    def reached(*args, **kwargs):
        raise PhysicalWorkReached

    monkeypatch.setattr(module, 'residual', reached)
    # The previous kernel-only plan admitted this amount. The complete
    # caller lift's directional-second graph requires a larger working set.
    with pytest.raises(ValueError, match='memory|budget'):
        cc.evaluate_triples_from_integrals(data, state, algorithm='streamed',
            orbital_basis='semicanonical', max_memory=40.411891)
    with pytest.raises(PhysicalWorkReached):
        cc.evaluate_triples_from_integrals(data, state, algorithm='streamed',
            orbital_basis='semicanonical', max_memory=64.)
