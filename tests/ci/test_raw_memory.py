"""Explicit MO inputs share the selected CI consumer's working budget."""
from dataclasses import dataclass
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import ci
from gradscf.ci.types import CIResult, CISResult, UCISResult


def _inputs(unrestricted=False):
    h = jnp.diag(jnp.asarray([-1., -.2, .7]))
    g = jnp.zeros((3,)*4)
    if not unrestricted:
        return h,g
    # Independent old full-spin Fock chart, including non-symmetric raw ab.
    from gradscf.ao2mo import spin_orbital_integrals
    rng = np.random.default_rng(518)
    gs = tuple(jnp.asarray(rng.normal(size=(3,)*4)*.01) for _ in range(3))
    hs,spin = spin_orbital_integrals((jnp.zeros_like(h),jnp.zeros_like(h)),gs)
    occupied = jnp.asarray([0,1,3])
    potential = spin[:,:,occupied,occupied].sum(-1)-spin[:,occupied,occupied,:].sum(1)
    return (h-potential[:3,:3],h-potential[3:,3:]),gs


@pytest.mark.parametrize('method', ['ci', 'uci', 'cis', 'ucis'])
def test_raw_solver_rejects_an_exhausted_budget_before_preparation(method, monkeypatch):
    solver = importlib.import_module('gradscf.ci.solver')
    direct = importlib.import_module('gradscf.ci.direct')
    h, g = _inputs(method in ('uci', 'ucis'))

    def forbidden(*args, **kwargs):
        pytest.fail('An exhausted raw CI budget reached numerical preparation')

    monkeypatch.setattr(direct, '_reference', forbidden)
    monkeypatch.setattr(solver, 'restricted_fock', forbidden)
    options = dict(max_memory=.001, reserved_bytes=1000)
    with pytest.raises(ValueError, match='memory|budget|resident'):
        if method in ('ci', 'uci'):
            space = (ci.make_uci_space(3, (2, 1)) if method == 'uci'
                     else ci.make_ci_space(3, 1))
            ci.solve_ci(h, g, space, **options)
        elif method == 'cis':
            ci.solve_cis(h, g, nocc=1, **options)
        else:
            ci.solve_ucis(h, g, nocc=(2, 1), **options)


@pytest.mark.parametrize('method', ['CISD', 'CIS', 'UCIS'])
def test_explicit_facade_forwards_budget_and_retained_reference(method, monkeypatch):
    api = importlib.import_module('gradscf.ci.api')
    unrestricted = method == 'UCIS'
    h, g = _inputs(unrestricted)
    eps = (jnp.diag(h[0]), jnp.diag(h[1])) if unrestricted else jnp.diag(h)
    reference = (ci.UnrestrictedReference(h, g, (2, 1), mo_energy=eps)
                 if unrestricted else ci.CIReference(h, g, 1, mo_energy=eps))
    calls = []

    def checked(h, g, *args, **kwargs):
        assert kwargs.get('max_memory') == 32.
        held = sum(x.nbytes for x in eps) if unrestricted else eps.nbytes
        assert kwargs.get('reserved_bytes', 0) >= held
        calls.append(kwargs)
        if method == 'CISD':
            space = args[0]
            return CIResult(jnp.zeros(1), jnp.zeros(1), jnp.asarray(0.),
                jnp.ones((space.size, 1)), jnp.zeros(1), jnp.ones(1, bool))
        if unrestricted:
            return UCISResult(jnp.ones(1), (jnp.ones((1, 2, 1)), jnp.ones((1, 1, 2))),
                              jnp.zeros(1), jnp.ones(1, bool))
        return CISResult(jnp.ones(1), jnp.ones((1, 1, 2)), jnp.zeros(1), jnp.ones(1, bool))

    def forbidden(*args, **kwargs):
        pytest.fail('UCIS stationarity checking materialized a full spin ERI')

    monkeypatch.setattr(api, {'CISD': 'solve_ci', 'CIS': 'solve_cis', 'UCIS': 'solve_ucis'}[method], checked)
    monkeypatch.setattr(api, 'spin_orbital_integrals', forbidden, raising=False)
    monkeypatch.setattr(importlib.import_module('gradscf.ao2mo'),'spin_orbital_integrals',forbidden)
    getattr(ci, method)(reference, solver='dense', max_memory=32.).run()
    assert len(calls) == 1


def test_raw_cis_d_rejects_an_exhausted_budget_before_the_formula(monkeypatch):
    corrections = importlib.import_module('gradscf.ci.corrections')
    _, g = _inputs()
    singles = CISResult(jnp.ones(1), jnp.ones((1, 1, 2))/jnp.sqrt(2.),
                        jnp.zeros(1), jnp.ones(1, bool), True)

    def forbidden(*args, **kwargs):
        pytest.fail('An exhausted CIS(D) budget reached the correction formula')

    monkeypatch.setattr(corrections, '_evaluate', forbidden)
    with pytest.raises(ValueError, match='memory|budget|resident'):
        ci.cis_d_correction(g, jnp.asarray([-1., -.2, .7]), singles, nocc=1,
                            max_memory=.001, reserved_bytes=1000)


@dataclass
class _AOReference:
    mo_coeff: object
    mo_occ: object
    mo_energy: object
    nocc: int
    h1e: object


def test_ao_cis_passes_its_budget_to_the_common_eigensolver(monkeypatch):
    solver = importlib.import_module('gradscf.ci.solver')
    response = importlib.import_module('gradscf.tddft.response')
    molecule = _AOReference(jnp.eye(3), jnp.asarray([2., 0., 0.]),
                            jnp.asarray([-1., -.2, .7]), 1, jnp.ones((3, 3)))
    monkeypatch.setattr(response, 'build_restricted_tda_operator',
        lambda *a, **kw: (lambda x: x, jnp.ones(2), None))
    calls = []

    def checked(apply, diagonal, config, response=None, **kwargs):
        assert kwargs.get('max_memory') == 32.
        assert kwargs.get('reserved_bytes', 0) >= molecule.h1e.nbytes
        calls.append(kwargs)
        return jnp.ones(1), jnp.ones((2, 1)), jnp.zeros(1), jnp.ones(1, bool)

    monkeypatch.setattr(solver, '_eigenpairs', checked)
    solver.solve_cis_response(molecule, config=ci.CIConfig(solver='dense'), max_memory=32.)
    assert len(calls) == 1


def test_response_plan_exposes_its_existing_action_share():
    from gradscf._memory import workspace_bytes
    from gradscf.tddft.response_options import plan_response_memory

    plan = plan_response_memory(3, 2, 1, np.dtype('float64'), 32.)
    # Preserve the original four decisions; only expose their existing share.
    assert tuple(plan[:4]) == (2, 32, 0, 0)
    assert getattr(plan, 'action_bytes', None) == workspace_bytes(32.)//2+8*(8*3**2+4*2)


@pytest.mark.parametrize('method', ['ci', 'uci', 'cis', 'ucis'])
def test_budgeted_raw_solver_preserves_value_and_joint_first_response(method):
    rng = np.random.default_rng(517)
    factors = rng.normal(size=(4, 3, 3))
    factors = .07*(factors+factors.transpose(0,2,1))
    g = jnp.asarray(np.einsum('Lpq,Lrs->pqrs', factors, factors))
    h = jnp.diag(jnp.asarray([-1., -.2, .7]))
    perturbation = jnp.asarray(rng.normal(size=(3,3))*.03)
    dh = .5*(perturbation+perturbation.T)
    dg = jnp.asarray(rng.normal(size=(3,)*4)*.001)
    config = ci.CIConfig(solver='dense', gradient_mode='implicit_eigenvector')
    unrestricted = method in ('uci', 'ucis')
    space = (ci.make_uci_space(3,(2,1),frozen=([0],[])) if unrestricted else
             ci.make_ci_space(3,1,frozen=[2]))

    def loss(t, budgeted):
        ht, gt = h+t*dh, g+t*dg
        if unrestricted:
            ht = (ht,ht+jnp.diag(jnp.asarray([.02,-.01,.03])))
            gt = (gt,.9*gt,1.1*gt)
        options = dict(config=config, max_memory=32. if budgeted else None)
        if method in ('ci','uci'):
            result = ci.solve_ci(ht,gt,space,nuclear_repulsion=.3,**options)
            energy, amplitudes = result.total_energies[0],result.coefficients[:,0]
        elif method == 'cis':
            result = ci.solve_cis(ht,gt,nocc=1,frozen=[2],**options)
            energy, amplitudes = result.excitation_energies[0],result.amplitudes.ravel()
        else:
            result = ci.solve_ucis(ht,gt,nocc=(2,1),frozen=([0],[]),**options)
            energy = result.excitation_energies[0]
            amplitudes = jnp.concatenate([a.ravel() for a in result.amplitudes])
        weights = jnp.arange(amplitudes.size,dtype=amplitudes.dtype)+1
        return energy+.013*jnp.sum(weights*amplitudes**2)

    def bundle(t,budgeted):
        return jax.value_and_grad(lambda x:loss(x,budgeted))(t)

    expected = jax.jit(lambda t:bundle(t,False))(0.)
    actual = jax.jit(lambda t:bundle(t,True))(0.)
    for a,e in zip(jax.tree.leaves(actual),jax.tree.leaves(expected)):
        np.testing.assert_allclose(a,e,atol=3e-10,rtol=2e-8)
