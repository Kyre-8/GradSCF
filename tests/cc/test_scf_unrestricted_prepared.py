"""Selected UHF snapshots feed the existing UCC ground/left/property kernels."""
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import cc, gto, scf
from gradscf.cc.uccsd import prepare_ucc_integrals
from gradscf.cc.spin_amplitudes import SpinAmplitudeSpace
from gradscf.cc import _spin_equations
from gradscf.scf.reference import unrestricted_reference_from_source
from test_prepared_properties import _problem, _config


@pytest.fixture(scope='module')
def source():
    mol = gto.M(atom='Li 0 0 0; H 0 0 3.1', basis='sto-3g',
                charge=1, spin=1, unit='Bohr', cart=True)
    mf = scf.UHF(mol, integral_backend='native', execution_device='cpu',
        max_memory=64., conv_tol=1e-12, conv_tol_density=1e-11,
        conv_tol_grad=1e-10, max_cycle=250).run()
    assert mf.converged
    return mf


def _trees(actual, expected, *, atol=3e-9, rtol=3e-8):
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        assert a.dtype == b.dtype
        np.testing.assert_allclose(a, b, atol=atol, rtol=rtol)


def _forbid_prepare(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Selected UCC consumer requested a full MO transform or re-preparation')
    targets = [('gradscf.cc.api', 'unrestricted_reference_from_source'),
               ('gradscf.ao2mo.reference', 'transform_unrestricted_integrals'),
               ('gradscf.cc.lambda_equations', 'prepare_ucc_integrals'),
               ('gradscf.cc.properties', 'prepare_ucc_integrals'),
               ('gradscf.cc._spin_triples', 'prepare_ucc_integrals')]
    for name, attribute in targets:
        monkeypatch.setattr(importlib.import_module(name), attribute, forbidden)


@pytest.mark.parametrize('method,frozen', [('ccsd', None), ('ccd', ([0], [5]))])
def test_uhf_facade_ground_left_density_and_triples_share_selected_snapshot(
        source, method, frozen, monkeypatch):
    raw = unrestricted_reference_from_source(source)
    options = dict(method=method, frozen=frozen, conv_tol=1e-12, residual_tol=1e-10)
    expected = cc.CC(raw, **options).run()
    expected.solve_lambda()
    density = expected.make_rdm1()
    correction = expected.ccsd_t() if method == 'ccsd' else None
    if correction is not None:
        assert abs(float(correction)) > 1e-12, 'Nonzero UCC(T) gate is vacuous'
    _forbid_prepare(monkeypatch)
    actual = cc.CC(source, **options).run()
    assert actual.converged and actual._integrals is not None
    snapshot = actual._integrals
    actual.solve_lambda()
    assert actual.converged_lambda
    _trees(actual.result, expected.result)
    _trees(actual.lambda_result, expected.lambda_result)
    _trees(actual.make_rdm1(), density)
    assert actual._integrals is snapshot
    for d, electrons in zip(actual.make_rdm1(), (2, 1)):
        np.testing.assert_allclose(jnp.trace(d), electrons, atol=3e-9)
    if correction is not None:
        np.testing.assert_allclose(actual.ccsd_t(), correction, atol=3e-11, rtol=3e-8)
    actual._ready()
    actual.frozen = ([0], [])
    with pytest.raises(RuntimeError, match='changed'):
        actual.solve_lambda()


def _old_density(h, g, state, left, frozen, cfg):
    _, occupied, virtual = prepare_ucc_integrals(h, g, nocc=(2, 1), frozen=frozen)
    space = SpinAmplitudeSpace(occupied, virtual, cfg.method)
    t1, t2 = space.from_blocks(state.t1, state.t2, h[0].dtype)

    def lagrangian(raw):
        ints, _, _ = prepare_ucc_integrals(tuple(.5*(a+a.T) for a in raw), g,
                                          nocc=(2, 1), frozen=frozen)
        residual = space.pack(*_spin_equations.residual(t1, t2, ints))
        return (ints.reference_energy+_spin_equations.correlation_energy(t1, t2, ints)
                +jnp.vdot(left.adjoint, residual).real)

    return jax.grad(lagrangian)(h)


@pytest.mark.parametrize('dtype', [jnp.float32, jnp.float64])
def test_prepared_uhf_left_and_local_density_keep_raw_chart_and_dual(dtype):
    from gradscf.cc import properties
    h, g = _problem(dtype)
    hs, gs = (h, h+.01*jnp.eye(4, dtype=dtype)), (g, .9*g, 1.05*g)
    frozen, cfg = ([0], [3]), _config(dtype=dtype)
    ints, occupied, virtual = prepare_ucc_integrals(hs, gs, nocc=(2, 1), frozen=frozen)
    state = cc.run_ucc(hs, gs, nocc=(2, 1), frozen=frozen, config=cfg)
    expected = cc.solve_lambda(hs, gs, state, nocc=(2, 1), frozen=frozen, config=cfg)
    actual = cc.solve_lambda_from_integrals(ints, state, config=cfg)
    assert state.converged and actual.converged and expected.converged
    _trees(actual, expected, atol=2e-5 if dtype == jnp.float32 else 3e-10)
    helper = getattr(properties, '_density_ucc_from_integrals', None)
    assert callable(helper), 'Missing prepared UHF local one-body partial'
    density = helper(ints, state, actual, nmo=4, nocc=(2, 1), frozen=frozen, config=cfg)
    _trees(density, _old_density(hs, gs, state, expected, frozen, cfg),
           atol=2e-5 if dtype == jnp.float32 else 3e-10)
    space = SpinAmplitudeSpace(occupied, virtual, cfg.method)
    d1, d2 = space.unpack(actual.adjoint)
    _trees((actual.l1, actual.l2), space.to_blocks(d1, 4*d2), atol=1e-12)


def test_prepared_uhf_density_outer_ad_matches_old_off_manifold_partial():
    from gradscf.cc import properties
    h, g = _problem()
    frozen, cfg = ([0], [3]), _config()
    antisymmetric = jnp.array([[0., .02, 0., 0.], [-.01, 0., .03, 0.],
                              [0., 0., 0., .01], [.02, 0., -.01, 0.]])
    probe = jnp.arange(16.).reshape(4, 4)/16
    helper = getattr(properties, '_density_ucc_from_integrals', None)
    assert callable(helper), 'Missing prepared UHF local one-body partial'

    def value(x, prepared):
        hs = (h+x*antisymmetric, h.T+x*antisymmetric.T*.7)
        gs = (g*(1+.03*x), g.at[0, 0, 2, 3].add(.01*x), 1.05*g)
        state = cc.run_ucc(hs, gs, nocc=(2, 1), frozen=frozen, config=cfg)
        ints, _, _ = prepare_ucc_integrals(hs, gs, nocc=(2, 1), frozen=frozen)
        left = (cc.solve_lambda_from_integrals(ints, state, config=cfg) if prepared else
                cc.solve_lambda(hs, gs, state, nocc=(2, 1), frozen=frozen, config=cfg))
        if prepared:
            partial, _, _ = prepare_ucc_integrals(tuple(.5*(a+a.T) for a in hs), gs,
                                                 nocc=(2, 1), frozen=frozen)
            density = helper(partial, state, left, nmo=4, nocc=(2, 1),
                             frozen=frozen, config=cfg)
        else:
            density = _old_density(hs, gs, state, left, frozen, cfg)
        return jnp.sum(density[0]*probe)+.7*jnp.sum(density[1]*probe.T)

    def bundle(x, prepared):
        function = lambda y: value(y, prepared)
        v, jvp = jax.jvp(function, (x,), (1.,))
        gradient = jax.grad(function)(x)
        hvp = jax.jvp(jax.grad(function), (x,), (1.,))[1]
        return v, jvp, gradient, hvp

    actual = jax.jit(lambda x: bundle(x, True))(.02)
    expected = jax.jit(lambda x: bundle(x, False))(.02)
    _trees(actual, expected, atol=3e-9, rtol=3e-8)


@pytest.mark.parametrize('frame,algorithm', [('canonical', 'auto'),
    ('semicanonical', 'auto'), ('semicanonical', 'streamed')])
def test_prepared_ucc_triples_all_diagnostics_and_bad_state_match_raw(frame, algorithm):
    h, g = _problem()
    hs, gs = (h, h+.01*jnp.eye(4)), (g, .9*g, 1.05*g)
    ints, _, _ = prepare_ucc_integrals(hs, gs, nocc=(2, 1))
    # Use the same complete diagonal physical Fock in both charts. Retain a
    # small off-diagonal term for general-reference/streamed response paths.
    no = 3
    indices = (jnp.array([0, 1, 3, 4]), jnp.array([2, 5, 6, 7]))
    fs = tuple(ints.fock[jnp.ix_(i, i)] for i in indices)
    diagonal = tuple(jnp.diag(jnp.diag(f)) for f in fs)
    shift = tuple(f-d for f, d in zip(fs, diagonal))
    hs = tuple(a-s for a, s in zip(hs, shift))
    if frame == 'semicanonical':
        hs = (hs[0].at[0, 1].add(.005).at[1, 0].add(.005), hs[1])
    ints, _, _ = prepare_ucc_integrals(hs, gs, nocc=(2, 1))
    cfg = _config()
    state = cc.run_ucc(hs, gs, nocc=(2, 1), config=cfg)
    assert state.converged
    options = dict(orbital_basis=frame, algorithm=algorithm,
                   residual_tol=cfg.residual_tol)
    actual = cc.evaluate_triples_from_integrals(ints, state, **options)
    expected = cc.evaluate_triples(hs, gs, state, nocc=(2, 1), **options)
    assert actual.valid and expected.valid and abs(float(actual.energy)) > 1e-15
    _trees(actual, expected, atol=3e-12, rtol=3e-9)
    same = state.t2[0].at[0, 1, 0, 0].add(.02)
    bad = state._replace(t2=(same, state.t2[1], state.t2[2]))
    actual = cc.evaluate_triples_from_integrals(ints, bad, **options)
    expected = cc.evaluate_triples(hs, gs, bad, nocc=(2, 1), **options)
    assert not actual.valid and not expected.valid
    _trees(actual, expected, atol=3e-12, rtol=3e-9)
