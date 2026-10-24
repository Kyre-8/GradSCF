"""Public native CC selects a representation fitting preparation and ground."""
import importlib

import jax
import jax.numpy as jnp
import pytest

from gradscf import cc
from gradscf._memory import tree_storage_bytes
from gradscf.ao2mo import VirtualERI
from gradscf.ao2mo.scf import SCFSource
from gradscf.cc.integrals import from_orbitals
from test_native_virtual_action import _problem


@pytest.fixture
def frame():
    _, basis, _, _ = _problem()
    # A legal rectangular algebraic frame tests storage policy, not a
    # physical forty-MO calculation in four AOs or a scaling benchmark.
    coefficients = .01*jnp.cos(jnp.arange(basis.nao*40, dtype=jnp.float64)).reshape(basis.nao, 40)
    fock = jnp.diag(jnp.concatenate((jnp.linspace(-1.8, -1.1, 8),
                                    jnp.linspace(.2, 1.6, 32))))
    return SCFSource(coefficients, (fock,), 8, jnp.asarray(-2.),
                     jnp.zeros((basis.nao, basis.nao)), {'direct_basis': basis})


class NumericalSolverReached(Exception):
    pass


def _stop_at_solver(monkeypatch):
    owner = importlib.import_module('gradscf.cc.ground')

    def reached(residual, initial, **kwargs):
        # Also check the actual operator's native minimum and selected RHS
        # width; the outer solver plan alone must not hide an action failure.
        jax.make_jaxpr(residual)(initial)
        raise NumericalSolverReached

    monkeypatch.setattr(owner, 'solve_nonlinear', reached)


def test_native_auto_counterexample_has_distinct_phase_limits(frame, monkeypatch):
    from gradscf.cc.amplitudes import AmplitudeSpace
    from gradscf.cc.ground import METHODS

    for method in METHODS:
        space = AmplitudeSpace(2, 3, method)
        assert space.size == space.pack(jnp.ones((2, 3)), jnp.ones((2, 2, 3, 3))).size

    def prepare(mode):
        return jax.eval_shape(lambda: from_orbitals(
            frame.coefficients, frame.focks[0], frame.nocc, frame.reference_energy,
            max_memory=55., vvvv=mode, **frame.ao_kwargs))

    dense, direct = prepare('dense'), prepare('direct')
    assert not isinstance(dense.vvvv, VirtualERI)
    assert isinstance(direct.vvvv, VirtualERI)
    _stop_at_solver(monkeypatch)
    with pytest.raises(ValueError, match='memory|budget'):
        jax.eval_shape(lambda ints: cc.run_cc_from_integrals(
            ints, max_memory=55., reserved_bytes=tree_storage_bytes(frame)), dense)
    with pytest.raises(NumericalSolverReached):
        jax.eval_shape(lambda ints: cc.run_cc_from_integrals(
            ints, max_memory=55., reserved_bytes=tree_storage_bytes(frame)), direct)


@pytest.mark.parametrize('memory,compact', [(55., True), (96., False)])
def test_public_native_auto_checks_ground_before_actual_transformation(
        frame, memory, compact, monkeypatch):
    from gradscf.scf.facade import RKS

    # Source preparation has independent molecular tests. Here retain the
    # actual public orchestration while supplying only its already-built frame.
    source = RKS(None, xc='hf', max_memory=memory)
    monkeypatch.setattr(importlib.import_module('gradscf.ao2mo.scf'),
                        'prepare_scf_source', lambda *args, **kwargs: frame)
    owner = importlib.import_module('gradscf.cc.integrals')
    original = owner.transform_eri_blocks
    actual_requests = []

    def record(coefficients, requests, **kwargs):
        # Metadata traces may inspect a rejected candidate. Only concrete
        # producer calls can materialize its integral blocks.
        if not any(isinstance(value, jax.core.Tracer) for value in coefficients):
            has_vvvv = (1, 1, 1, 1) in requests
            actual_requests.append(has_vvvv)
            assert has_vvvv != compact, 'Auto materialized the wrong ground-state representation'
        return original(coefficients, requests, **kwargs)

    monkeypatch.setattr(owner, 'transform_eri_blocks', record)
    _stop_at_solver(monkeypatch)
    with pytest.raises(NumericalSolverReached):
        cc.CCSD(source, max_memory=memory).kernel()
    assert actual_requests == [not compact]


def test_native_auto_preview_uses_the_source_device(frame, monkeypatch):
    from gradscf.scf.facade import RKS

    owner = importlib.import_module('gradscf.cc.api')
    original = owner._prepare_ground
    calls = []

    def checked(*args, **kwargs):
        calls.append(jax.config.jax_default_device)
        assert calls[-1] == frame.coefficients.device
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, '_prepare_ground', checked)
    model = cc.CCSD(RKS(None, xc='hf', max_memory=96.))
    model.reference = frame
    model._restricted_integrals(frame, model._config(), None, None, 0)
    assert len(calls) == 1


def test_native_auto_preview_does_not_pack_coordinates(frame, monkeypatch):
    from gradscf.cc.amplitudes import AmplitudeSpace
    from gradscf.scf.facade import RKS

    phase = 'preview'
    packed = []
    original_pack = AmplitudeSpace.pack
    owner = importlib.import_module('gradscf.cc.integrals')
    original_transform = owner.transform_eri_blocks

    def pack(space, *args):
        assert phase == 'actual', 'Metadata preview constructed packed coordinate indices'
        packed.append(space.method)
        return original_pack(space, *args)

    def transform(coefficients, requests, **kwargs):
        nonlocal phase
        if not any(isinstance(value, jax.core.Tracer) for value in coefficients):
            phase = 'actual'
        return original_transform(coefficients, requests, **kwargs)

    monkeypatch.setattr(AmplitudeSpace, 'pack', pack)
    monkeypatch.setattr(owner, 'transform_eri_blocks', transform)
    monkeypatch.setattr(importlib.import_module('gradscf.ao2mo.scf'),
                        'prepare_scf_source', lambda *args, **kwargs: frame)
    _stop_at_solver(monkeypatch)
    with pytest.raises(NumericalSolverReached):
        cc.CCSD(RKS(None, xc='hf', max_memory=55.)).kernel()
    assert phase == 'actual' and len(packed) >= 3
