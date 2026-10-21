"""Prepared triples/EOM and metadata-only dense RDM2 consumers."""
from dataclasses import replace
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import cc
from gradscf.cc.integrals import prepare_integrals
from test_prepared_properties import _problem, _config


TRIPLES = [('ccsd+t(ccsd)', 'canonical', 'auto'),
           ('ccsd(t)', 'canonical', 'auto'), ('qcisd(t)', 'canonical', 'auto'),
           ('ccsd(t)', 'semicanonical', 'auto'),
           ('ccsd(t)', 'semicanonical', 'streamed'),
           ('ccsd(t)', 'canonical', 'streamed')]


def _entry(module, name):
    function = getattr(module, name, None)
    assert callable(function), 'Missing prepared consumer: '+name
    return function


def _canonical_problem():
    h, g = _problem()
    f = prepare_integrals(h, g, nocc=2).fock
    return h-f+jnp.diag(jnp.diag(f)), g


def _old_semicanonical(h, g, state, **kwargs):
    """Independent original restricted-to-UCC tensor reference."""
    from gradscf.cc._spin_triples import evaluate_ucc_triples
    same = state.t2-state.t2.swapaxes(2, 3)
    spin = state._replace(t1=(state.t1, state.t1), t2=(same, state.t2, same))
    return evaluate_ucc_triples((h, h), (g, g, g), spin, nocc=(2, 2),
                               orbital_basis='semicanonical', **kwargs)


@pytest.mark.parametrize('variant,frame,algorithm', TRIPLES)
def test_prepared_triples_values_and_diagnostics(variant, frame, algorithm):
    h, g = _canonical_problem()
    cfg = _config('qcisd' if variant == 'qcisd(t)' else 'ccsd')
    if frame == 'semicanonical':
        h = h.at[0, 1].add(.02).at[1, 0].add(.02)
    ints = prepare_integrals(h, g, nocc=2)
    state = cc.run_cc_from_integrals(ints, config=cfg)
    options = dict(variant=variant, orbital_basis=frame, algorithm=algorithm)
    expected = cc.evaluate_triples(h, g, state, nocc=2, **options)
    actual = _entry(cc, 'evaluate_triples_from_integrals')(ints, state, **options)
    assert state.converged and actual.valid and expected.valid
    for a, e in zip(actual, expected):
        assert a.dtype == e.dtype
        np.testing.assert_allclose(a, e, atol=2e-11, rtol=2e-9)
    scalar = _entry(cc, 'triples_correction_from_integrals')(ints, state, **options)
    np.testing.assert_allclose(scalar, actual.energy, atol=1e-15)
    if frame == 'semicanonical':
        old = _old_semicanonical(h, g, state)
        for a, e in zip(actual, old):
            np.testing.assert_allclose(a, e, atol=2e-11, rtol=2e-9)


@pytest.mark.parametrize('sector', ['ee', 'ip', 'ea'])
def test_prepared_eom_actions_transpose_and_roots(sector):
    from gradscf.cc import eom
    h, g = _problem()
    cfg = _config()
    ints = prepare_integrals(h, g, nocc=2)
    state = cc.run_cc_from_integrals(ints, config=cfg)
    options = dict(config=cc.EOMConfig(sector=sector), cc_config=cfg)
    raw, raw_space, valid = eom.build_eom_operator(h, g, state, nocc=2, **options)
    op, space, prepared_valid = _entry(eom, 'build_eom_operator_from_integrals')(
        ints, state, **options)
    assert valid and prepared_valid and space == raw_space
    eye = jnp.eye(space.size)
    np.testing.assert_allclose(op.apply(eye), raw.apply(eye), atol=2e-12)
    np.testing.assert_allclose(op.T.apply(eye), raw.T.apply(eye), atol=2e-12)
    expected = eom.run_eom(h, g, state, nocc=2, **options)
    actual = _entry(eom, 'run_eom_from_integrals')(ints, state, **options)
    for a, e in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(a, e, atol=2e-11, rtol=2e-9)
    assert np.all(actual.converged & actual.response_valid)


@pytest.mark.parametrize('frozen', [None, [0, 3]])
@pytest.mark.parametrize('mixed', [False, True])
def test_metadata_rdm2_matches_raw_frozen_and_dtype(frozen, mixed):
    from gradscf.cc import properties
    h, g = _problem()
    h = h.astype(jnp.float32) if mixed else h
    cfg = _config()
    state = cc.run_cc(h, g, nocc=2, frozen=frozen, config=cfg)
    left = cc.solve_lambda(h, g, state, nocc=2, frozen=frozen, config=cfg)
    expected = properties._rdm2_from_lambda(h, g, state, left,
                                          nocc=2, frozen=frozen, config=cfg)
    actual = _entry(properties, '_rdm2_from_states')(
        state, left, nmo=4, nocc=2, frozen=frozen, dtype=h.dtype, config=cfg)
    assert state.converged and left.converged and actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, atol=2e-12)
    dm1 = cc.make_rdm1(h, g, state, nocc=2, frozen=frozen, config=cfg)
    np.testing.assert_allclose(jnp.einsum('pqrr->pq', actual), 3*dm1, atol=3e-7 if mixed else 2e-10)


def test_prepared_consumers_do_not_prepare_or_materialize(monkeypatch):
    from gradscf.cc import eom, properties, _spin_triples
    h, g = _canonical_problem()
    cfg = _config()
    ints = prepare_integrals(h, g, nocc=2)
    state = cc.run_cc_from_integrals(ints, config=cfg)
    left = cc.solve_lambda_from_integrals(ints, state, config=cfg)

    def reject(*args, **kwargs):
        raise AssertionError('Prepared consumer re-prepared an integral snapshot')

    for name in ('gradscf.cc.integrals', 'gradscf.cc.triples', 'gradscf.cc.eom.operators',
                 'gradscf.cc.properties'):
        monkeypatch.setattr(importlib.import_module(name), 'prepare_integrals', reject)
    monkeypatch.setattr(_spin_triples, 'prepare_ucc_integrals', reject)
    for frame, algorithm in [('canonical', 'auto'), ('semicanonical', 'auto'),
                              ('semicanonical', 'streamed')]:
        result = _entry(cc, 'evaluate_triples_from_integrals')(
            ints, state, orbital_basis=frame, algorithm=algorithm)
        assert result.valid
    for sector in ('ee', 'ip', 'ea'):
        result = _entry(eom, 'run_eom_from_integrals')(
            ints, state, config=cc.EOMConfig(sector=sector), cc_config=cfg)
        assert np.all(result.converged)
    dm2 = _entry(properties, '_rdm2_from_states')(
        state, left, nmo=4, nocc=2, dtype=h.dtype, config=cfg)
    assert np.isfinite(dm2).all()


@pytest.mark.parametrize('variant,frame,algorithm', TRIPLES)
def test_prepared_triples_jvp_vjp_hvp(variant, frame, algorithm):
    h, g = _canonical_problem()
    levels = jnp.diag(prepare_integrals(h, g, nocc=2).fock)
    probe = jnp.asarray(np.random.default_rng(6521).normal(scale=.025, size=h.shape))
    probe = .5*(probe+probe.T)
    cfg = _config('qcisd' if variant == 'qcisd(t)' else 'ccsd')
    options = dict(variant=variant, orbital_basis=frame, algorithm=algorithm)

    def values(x):
        gs = g*(1+.13*x)
        if frame == 'canonical':
            mean = prepare_integrals(jnp.zeros_like(h), gs, nocc=2).fock
            hs = jnp.diag(levels+x*jnp.diag(probe))-mean
        else:
            hs = h+(.2+x)*probe
        ints = prepare_integrals(hs, gs, nocc=2)
        state = cc.run_cc_from_integrals(ints, config=cfg)
        actual = _entry(cc, 'evaluate_triples_from_integrals')(ints, state, **options).energy
        expected = (_old_semicanonical(hs, gs, state).energy if frame == 'semicanonical' else
                    cc.triples_correction(hs, gs, state, nocc=2, **options))
        return jnp.stack((actual, expected))

    value, tangent = jax.jit(lambda x: jax.jvp(values, (x,), (1.,)))(0.)
    reverse = jax.jit(jax.jacrev(values))(0.)
    curvature = jax.jit(lambda x: jax.jvp(jax.jacrev(values), (x,), (1.,))[1])(0.)
    for pair in (value, tangent, reverse, curvature):
        assert np.isfinite(pair).all()
        np.testing.assert_allclose(pair[0], pair[1], atol=3e-13, rtol=3e-7)
    np.testing.assert_allclose(tangent, reverse, atol=3e-13, rtol=3e-7)
    step = 2e-4
    np.testing.assert_allclose(tangent, (values(step)-values(-step))/(2*step),
                               atol=3e-12, rtol=3e-5)
    gradient = jax.jit(jax.jacrev(values))
    np.testing.assert_allclose(curvature, (gradient(step)-gradient(-step))/(2*step),
                               atol=3e-12, rtol=3e-5)


def test_prepared_triples_state_capacity_and_ad_order_guards():
    from gradscf.cc.ground import METHODS
    h, g = _canonical_problem()
    ints = prepare_integrals(h, g, nocc=2)
    state = cc.run_cc_from_integrals(ints, config=_config())
    evaluate = _entry(cc, 'evaluate_triples_from_integrals')
    for options, match in [({'variant': 'bad'}, 'variant'),
                            ({'orbital_basis': 'bad'}, 'orbital_basis'),
                            ({'algorithm': 'bad'}, 'algorithm'),
                            ({'denominator_tol': 0.}, 'tolerances'),
                            ({'max_virtual_triples': 1}, 'max_virtual_triples'),
                            ({'orbital_basis': 'semicanonical', 'max_triples_elements': 1},
                             'max_triples_elements'),
                            ({'algorithm': 'streamed', 'max_workspace_elements': 255},
                             'max_workspace_elements')]:
        with pytest.raises(ValueError, match=match):
            evaluate(ints, state, **options)
    with pytest.raises(NotImplementedError, match='only CCSD'):
        evaluate(ints, state, algorithm='streamed', variant='qcisd(t)')
    for bad in (state._replace(converged=jnp.asarray(False)),
                state._replace(method_id=jnp.asarray(METHODS.index('qcisd'))),
                state._replace(t2=state.t2.at[0, 0, 0, 1].add(.01))):
        for frame, algorithm in [('canonical', 'auto'), ('semicanonical', 'auto'),
                                  ('semicanonical', 'streamed')]:
            out = evaluate(ints, bad, orbital_basis=frame, algorithm=algorithm)
            assert not out.valid and np.isnan(out.energy)
    moved = ints._replace(fock=ints.fock.at[0, 1].add(.01))
    assert not evaluate(moved, state).valid
    with pytest.raises(ValueError, match='shape'):
        evaluate(ints, state._replace(t1=state.t1[:1]))

    def scalar(x):
        changed = state._replace(t2=state.t2*x)
        return evaluate(ints, changed, algorithm='streamed', residual_tol=1e6).energy

    with pytest.raises(NotImplementedError, match='second.order'):
        jax.jit(jax.grad(jax.grad(jax.grad(scalar))))(1.)
    singular = ints._replace(fock=jnp.zeros_like(ints.fock), mo_energy=jnp.zeros_like(ints.mo_energy))
    bad = lambda x: evaluate(singular, state._replace(t2=state.t2*x),
                            algorithm='streamed', residual_tol=1e6).energy
    assert np.isnan(bad(1.))
    assert np.isnan(jax.jit(jax.grad(bad))(1.))


@pytest.mark.parametrize('sector', ['ee', 'ip', 'ea'])
def test_prepared_eom_energy_and_left_right_vector_response(sector):
    from gradscf.cc import eom
    h, g = _problem()
    probe = jnp.asarray(np.random.default_rng(6522).normal(scale=.02, size=h.shape))
    probe = .5*(probe+probe.T)
    cfg = _config()
    eom_cfg = cc.EOMConfig(sector=sector, vector_response=True)

    def values(x):
        hs, gs = h+x*probe, g*(1+.11*x)
        ints = prepare_integrals(hs, gs, nocc=2)
        state = cc.run_cc_from_integrals(ints, config=cfg)
        actual = _entry(eom, 'run_eom_from_integrals')(ints, state, config=eom_cfg, cc_config=cfg)
        expected = eom.run_eom(hs, gs, state, nocc=2, config=eom_cfg, cc_config=cfg)

        def observable(out):
            weights = jnp.arange(out.right_vectors.shape[0])+1.
            return out.energies[0]+.03*jnp.sum(weights[:, None]*out.left_vectors*out.right_vectors)

        return jnp.stack((observable(actual), observable(expected)))

    value, tangent = jax.jit(lambda x: jax.jvp(values, (x,), (1.,)))(0.)
    reverse = jax.jit(jax.jacrev(values))(0.)
    for pair in (value, tangent, reverse):
        assert np.isfinite(pair).all()
        np.testing.assert_allclose(pair[0], pair[1], atol=2e-10, rtol=2e-8)
    np.testing.assert_allclose(tangent, reverse, atol=2e-10, rtol=2e-8)
    step = 1e-4
    np.testing.assert_allclose(tangent, (values(step)-values(-step))/(2*step),
                               atol=3e-8, rtol=3e-6)


@pytest.mark.parametrize('sector', ['ee', 'ip', 'ea'])
def test_prepared_eom_storage_state_and_provider_guards(sector):
    from gradscf.cc import eom
    h, g = _problem()
    cfg = _config()
    frozen = [0, 3]
    ints = prepare_integrals(h, g, nocc=2, frozen=frozen)
    state = cc.run_cc_from_integrals(ints, config=cfg)
    build = _entry(eom, 'build_eom_operator_from_integrals')
    stored = sum(x.size for x in jax.tree.leaves(ints))
    options = dict(config=cc.EOMConfig(sector=sector, max_intermediate_elements=stored), cc_config=cfg)
    _, _, valid = build(ints, state, **options)
    assert valid and stored < g.size
    with pytest.raises(ValueError, match='max_intermediate_elements'):
        eom.build_eom_operator(h, g, state, nocc=2, frozen=frozen, **options)
    with pytest.raises(ValueError, match='max_intermediate_elements'):
        build(ints, state, config=replace(options['config'], max_intermediate_elements=stored-1),
              cc_config=cfg)
    with pytest.raises(ValueError, match='max_dense'):
        build(ints, state, config=cc.EOMConfig(sector=sector, max_dense=1), cc_config=cfg)
    with pytest.raises(ValueError, match='CCSD'):
        build(ints, state, cc_config=_config('cc2'))

    class Provider:
        def __array__(self, *args, **kwargs):
            raise AssertionError('Matrix-free virtual ERI must not be materialized')

    with pytest.raises(NotImplementedError, match='dense.*virtual|virtual.*dense'):
        build(ints._replace(vvvv=Provider()), state, **options)
    for bad in (state._replace(converged=jnp.asarray(False)),
                state._replace(t2=state.t2+jnp.nan)):
        _, _, valid = build(ints, bad, **options)
        assert not valid


@pytest.mark.parametrize('frozen', [None, [0, 3]])
def test_metadata_rdm2_outer_response_and_model_guard(frozen):
    from gradscf.cc import properties
    h, g = _problem()
    cfg = _config()
    probe = jnp.asarray(np.random.default_rng(6523).normal(scale=.04, size=g.shape))
    direction = jnp.diag(jnp.array([.02, -.03, .04, -.01]))

    def values(x):
        hs, gs = h+x*direction, g*(1+.13*x)
        ints = prepare_integrals(hs, gs, nocc=2, frozen=frozen)
        state = cc.run_cc_from_integrals(ints, config=cfg)
        left = cc.solve_lambda_from_integrals(ints, state, config=cfg)
        actual = _entry(properties, '_rdm2_from_states')(
            state, left, nmo=4, nocc=2, frozen=frozen, dtype=h.dtype, config=cfg)
        expected = properties._rdm2_from_lambda(hs, gs, state, left,
                                               nocc=2, frozen=frozen, config=cfg)
        return jnp.stack((jnp.sum(actual*probe), jnp.sum(expected*probe)))

    value, tangent = jax.jit(lambda x: jax.jvp(values, (x,), (1.,)))(0.)
    reverse = jax.jit(jax.jacrev(values))(0.)
    for pair in (value, tangent, reverse):
        assert np.isfinite(pair).all()
        np.testing.assert_allclose(pair[0], pair[1], atol=2e-11, rtol=2e-9)
    np.testing.assert_allclose(tangent, reverse, atol=2e-11, rtol=2e-9)
    step = 1e-4
    np.testing.assert_allclose(tangent, (values(step)-values(-step))/(2*step),
                               atol=3e-9, rtol=3e-6)
    for method in ('cc2', 'qcisd'):
        with pytest.raises(NotImplementedError, match='2-RDM'):
            _entry(properties, '_rdm2_from_states')(
                None, None, nmo=4, nocc=2, dtype=h.dtype, config=_config(method))


def test_semicanonical_auto_corrupted_state_diagnostics_match_original_ucc():
    h, g = _canonical_problem()
    ints = prepare_integrals(h, g, nocc=2)
    state = cc.run_cc_from_integrals(ints, config=_config())
    anti = jnp.array([[0., 1.], [-1., 0.]])
    symmetric = jnp.array([[.4, .2], [.2, -.1]])
    defect = .01*jnp.einsum('ij,ab->ijab', anti, symmetric)
    changed = state._replace(t2=state.t2+defect)
    actual = cc.evaluate_triples(h, g, changed, nocc=2, orbital_basis='semicanonical')
    expected = _old_semicanonical(h, g, changed)
    print('corrupted general diagnostics:', actual, expected)
    # This threshold exposed the previous RCC-to-spin substitution: old UCC
    # accepts the state while that unsupported substitution rejected it.
    threshold = .022910434027031447
    boundary_actual = cc.evaluate_triples(h, g, changed, nocc=2,
        orbital_basis='semicanonical', residual_tol=threshold)
    boundary_expected = _old_semicanonical(h, g, changed, residual_tol=threshold)
    print('corrupted general residual boundary:', threshold, boundary_actual.valid, boundary_expected.valid)
    for a, e in zip(actual, expected):
        assert a.dtype == e.dtype
        np.testing.assert_allclose(a, e, atol=2e-11, rtol=2e-9)
    for a, e in zip(boundary_actual, boundary_expected):
        assert a.dtype == e.dtype
        np.testing.assert_allclose(a, e, atol=2e-11, rtol=2e-9)
    assert boundary_actual.valid and boundary_expected.valid
    prepared = _entry(cc, 'evaluate_triples_from_integrals')(
        ints, changed, orbital_basis='semicanonical', residual_tol=1.)
    assert not prepared.valid and np.isnan(prepared.energy) and np.isnan(prepared.cc_residual_norm)
    for bad in (state._replace(converged=jnp.asarray(False)),
                state._replace(t2=state.t2+jnp.nan)):
        actual = cc.evaluate_triples(h, g, bad, nocc=2, orbital_basis='semicanonical')
        expected = _old_semicanonical(h, g, bad)
        for a, e in zip(actual, expected):
            np.testing.assert_allclose(a, e, atol=2e-11, rtol=2e-9)

    def raw_energy(eri):
        return cc.evaluate_triples(h, eri, state, nocc=2, orbital_basis='semicanonical',
                                   residual_tol=1.).energy

    oracle = lambda eri: _old_semicanonical(h, eri, state, residual_tol=1.).energy
    value, gradient = jax.jit(jax.value_and_grad(raw_energy))(g)
    reference, expected_gradient = jax.jit(jax.value_and_grad(oracle))(g)
    np.testing.assert_allclose(value, reference, atol=3e-13, rtol=3e-9)
    np.testing.assert_allclose(gradient, expected_gradient, atol=3e-13, rtol=3e-9)
    assert abs(float(expected_gradient[0, 0, 2, 3])) > 1e-14
    print('raw off-manifold g[0,0,2,3] VJP:', gradient[0, 0, 2, 3])
