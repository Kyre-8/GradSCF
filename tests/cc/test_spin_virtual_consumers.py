"""Compact UCC vvvv through the original ground/left/property consumers."""
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import cc
from gradscf.ao2mo import unrestricted_frozen_indices
from gradscf.cc.uccsd import prepare_ucc_integrals
from gradscf.cc.unrestricted_integrals import _spin_terms
from test_prepared_properties import _problem, _config
from test_scf_unrestricted_prepared import source, _trees, _old_density


def _compact_raw(ints, occupied, virtual, eri, nocc=(2, 1), frozen=None):
    """Reuse producer terms; only the independent raw-MO chemical gathers differ."""
    module = importlib.import_module('gradscf.cc._spin_virtual')
    frozen = unrestricted_frozen_indices(eri[0].shape[0], nocc, frozen)
    indices = tuple(jnp.asarray([i for i in range(no, eri[0].shape[0]) if i not in fr],
                               dtype=jnp.int32) for no, fr in zip(nocc, frozen))
    terms = tuple(t for t in _spin_terms((occupied, virtual)) if t[0] == 5)
    values = {}
    for _, request, _, _, _ in terms:
        left, right = request[0]//2, request[2]//2
        a, b = indices[left], indices[right]
        values[request] = eri[left+right][jnp.ix_(a, a, b, b)]
    return ints._replace(vvvv=module._SpinVirtualERI.from_terms(values, terms, virtual,
                                                              ints.fock.dtype))


def test_compact_preparation_charges_actual_children_under_pressure(monkeypatch):
    from gradscf._memory import array_storage_bytes, tree_storage_bytes
    module = importlib.import_module('gradscf.cc.unrestricted_integrals')
    rng = np.random.default_rng(6556)
    factors = rng.normal(scale=.02, size=(3, 12, 12))
    factors = .5*(factors+factors.swapaxes(1, 2))
    eri = jnp.asarray(np.einsum('Lpq,Lrs->pqrs', factors, factors))
    coefficients = (jnp.eye(12),)*2
    focks = (jnp.diag(jnp.linspace(-1.4, .9, 12)),)*2
    # 3 MB also covers the shared AO planner's output versions and one-leg
    # scratch; dense spin-vvvv assembly buffers alone exceed this window.
    actual, _, _ = module.from_orbitals(coefficients, focks, (2, 1), jnp.asarray(-4.),
        eri=eri, max_memory=3., vvvv='blocks')
    assert tree_storage_bytes(actual.vvvv) == array_storage_bytes(*jax.tree.leaves(actual.vvvv))

    def forbidden(*args, **kwargs):
        pytest.fail('Dense spin vvvv passed an insufficient preparation budget')

    monkeypatch.setattr(module, 'transform_eri_blocks', forbidden)
    with pytest.raises(ValueError, match='memory|budget|resident'):
        module.from_orbitals(coefficients, focks, (2, 1), jnp.asarray(-4.),
            eri=eri, max_memory=3.)


@pytest.mark.parametrize('method,frozen', [('ccsd', None), ('ccd', ([0], [5]))])
def test_uhf_public_uses_compact_virtuals_and_preserves_original_consumers(
        source, method, frozen, monkeypatch):
    api = importlib.import_module('gradscf.cc.api')
    original = api.ucc_from_orbitals

    def checked(*args, **kwargs):
        assert kwargs.get('vvvv') == 'blocks', 'UHF facade retained dense spin vvvv'
        out = original(*args, **kwargs)
        assert not hasattr(out[0].vvvv, 'shape')
        return out

    monkeypatch.setattr(api, 'ucc_from_orbitals', checked)
    from test_scf_unrestricted_prepared import test_uhf_facade_ground_left_density_and_triples_share_selected_snapshot
    test_uhf_facade_ground_left_density_and_triples_share_selected_snapshot(
        source, method, frozen, monkeypatch)


@pytest.mark.parametrize('frame,algorithm', [('canonical', 'auto'),
    ('semicanonical', 'auto'), ('semicanonical', 'streamed')])
def test_compact_triples_diagnostics_and_bad_state_match_raw(frame, algorithm, monkeypatch):
    module = importlib.import_module('test_scf_unrestricted_prepared')
    original = module.prepare_ucc_integrals

    def compact(h, eri, *, nocc):
        ints, occupied, virtual = original(h, eri, nocc=nocc)
        return _compact_raw(ints, occupied, virtual, eri, nocc), occupied, virtual

    monkeypatch.setattr(module, 'prepare_ucc_integrals', compact)
    module.test_prepared_ucc_triples_all_diagnostics_and_bad_state_match_raw(frame, algorithm)


def test_compact_density_joint_raw_hg_response_keeps_both_state_responses():
    from gradscf.cc.properties import _density_ucc_from_integrals
    h, g = _problem()
    frozen, cfg = ([0], [3]), _config()
    direction = jnp.array([[0., .02, 0., 0.], [-.01, 0., .03, 0.],
                           [0., 0., 0., .01], [.02, 0., -.01, 0.]])
    probe = jnp.arange(16.).reshape(4, 4)/16

    def value(x, compact):
        hs = (h+x*direction, h.T+.7*x*direction.T)
        gs = (g*(1+.03*x), g.at[0, 0, 2, 3].add(.01*x), 1.05*g)
        ints, no, nv = prepare_ucc_integrals(hs, gs, nocc=(2, 1), frozen=frozen)
        if compact:
            ints = _compact_raw(ints, no, nv, gs, frozen=frozen)
        state = cc.run_ucc_from_integrals(ints, no, nv, config=cfg)
        left = cc.solve_lambda_from_integrals(ints, state, config=cfg)
        if compact:
            partial, _, _ = prepare_ucc_integrals(tuple(.5*(a+a.T) for a in hs), gs,
                                                 nocc=(2, 1), frozen=frozen)
            partial = _compact_raw(partial, no, nv, gs, frozen=frozen)
            density = _density_ucc_from_integrals(partial, state, left,
                nmo=4, nocc=(2, 1), frozen=frozen, config=cfg)
        else:
            density = _old_density(hs, gs, state, left, frozen, cfg)
        return jnp.sum(density[0]*probe)+.7*jnp.sum(density[1]*probe.T)

    operation = lambda compact: jax.jit(lambda x: jax.jvp(jax.value_and_grad(
        lambda t: value(t, compact)), (x,), (jnp.ones_like(x),)))(.02)
    _trees(operation(True), operation(False), atol=3e-9, rtol=3e-8)


@pytest.mark.parametrize('method', ['ccsd', 'ccsd(t)'])
def test_compact_uhf_relaxed_geometry_matches_full_mo_and_two_fd_steps(source, method, monkeypatch):
    workflow = importlib.import_module('gradscf_tools.correlated_response')
    original = workflow.ucc_from_orbitals

    def compact(*args, **kwargs):
        kwargs['vvvv'] = 'blocks'
        out = original(*args, **kwargs)
        assert not hasattr(out[0].vvvv, 'shape')
        return out

    monkeypatch.setattr(workflow, 'ucc_from_orbitals', compact)
    from test_scf_unrestricted_geometry import test_selected_uhf_relaxed_gradient_hvp_and_nonzero_triples
    test_selected_uhf_relaxed_gradient_hvp_and_nonzero_triples(source, method, monkeypatch)
