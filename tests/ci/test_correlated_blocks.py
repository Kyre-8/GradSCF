"""Selected CI/CAS energies keep complete relaxed native geometry response.

The independent composition uses the original full-MO transformation and raw
CI/FCI interfaces in the same transported HF frame. Units are Bohr and Hartree.
Full FCI necessarily retains its active-space four-index tensor; the direct
adapter must avoid a separate full-AO/full-MO preparation before that output.
"""
import importlib
from copy import copy
from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import cc, ci, fci, gto, scf
from gradscf.ao2mo import transform_integrals, transform_unrestricted_integrals
from gradscf.ao2mo.frame import project_frame
from gradscf.solvers import EigenSolverConfig
from gradscf_tools.correlated_response import CorrelatedEnergy
from gradscf_tools.molecular_response import MolecularEnergy


@pytest.fixture(scope='module')
def source(request):
    kind = request.param
    atoms = {'h2': 'H 0 0 0; H 0 0 1.5',
             'h3': 'H 0 0 0; H 0 0 1.6; H 1.8 .2 .1',
             'lih': 'Li 0 0 0; H 0 0 2.8'}
    molecule = gto.M(atom=atoms[kind], basis='6-31g' if kind == 'h2' else 'sto-3g',
                     spin=int(kind == 'h3'), unit='Bohr', cart=True)
    cls = scf.UHF if kind == 'h3' else scf.RHF
    mf = cls(molecule, integral_backend='native', execution_device='cpu',
        conv_tol=1e-13, conv_tol_grad=1e-11, conv_tol_density=1e-11,
        max_cycle=250).run()
    assert mf.converged
    return mf


def _bundle(function, point, direction):
    (value, gradient), (jvp, hvp) = jax.jvp(jax.value_and_grad(function),
                                         (point,), (direction,))
    return value, gradient, jvp, hvp


@pytest.mark.parametrize('source,method,frozen,core', [
    ('h2', 'cisd', None, 0), ('h2', 'cisd', [3], 0),
    ('h3', 'cisd', None, 0), ('h3', 'cisd', ([0], [2]), 0),
    ('h2', 'fci', None, 0), ('lih', 'fci', None, 1),
], indirect=['source'])
def test_selected_relaxed_ci_fci_matches_full_mo(source, method, frozen, core, monkeypatch):
    config = (ci.CIConfig(solver='dense', conv_tol=1e-11) if method == 'cisd' else
              EigenSolverConfig(method='dense', atol=1e-11))
    direct = CorrelatedEnergy(source, method, config=config, frozen=frozen, core=core,
                               jk_backend='direct', max_memory=32.)
    full = MolecularEnergy(source)
    point = direct.coordinates
    direction = jnp.asarray(np.random.default_rng(1917).normal(size=point.shape))
    direction /= jnp.linalg.norm(direction)
    nmo = source.mo_coeff.shape[-1]
    no = direct.nocc
    if method == 'cisd':
        build = ci.make_uci_space if full.unrestricted else ci.make_ci_space
        space = build(nmo, no, max_excitation=2, frozen=frozen)
    else:
        space = fci.make_fci_space(nmo-core, (no-core, no-core))

    def legacy(r):
        state, inputs = full.solve_with_inputs(r)
        coefficients = state.mo_coeff
        if direct._selected_frame:
            if full.unrestricted:
                coefficients = jnp.stack([project_frame(c,a,inputs['overlap'],n)
                    for c,a,n in zip(coefficients,full.coefficients,no)])
            else:
                coefficients = project_frame(coefficients,full.coefficients,inputs['overlap'],no)
        transform = transform_unrestricted_integrals if full.unrestricted else transform_integrals
        h, g = transform(inputs['hcore'], coefficients, eri=inputs['eri'], max_memory=32.)
        if method == 'cisd':
            result = ci.solve_ci(h, g, space, nuclear_repulsion=inputs['nuclear_repulsion'], config=config)
        else:
            h, g, constant = fci.fold_core(h, g, core=core, ecore=inputs['nuclear_repulsion'])
            result = fci.solve_fci(h, g, space, ecore=constant, config=config, max_memory=32.)
        return jnp.where(state.stationary & result.converged[0], result.total_energies[0], jnp.nan)

    print(f'{method} frozen={frozen} core={core}: full-MO response', flush=True)
    expected = jax.jit(lambda r: _bundle(legacy, r, direction))(point)
    jax.block_until_ready(expected)
    workflow = importlib.import_module('gradscf_tools.correlated_response')

    def forbidden(*args, **kwargs):
        pytest.fail('CI/FCI adapter prepared a complete MO Hamiltonian before its selected producer')

    monkeypatch.setattr(workflow, 'transform_integrals', forbidden)
    monkeypatch.setattr(workflow, 'transform_unrestricted_integrals', forbidden)
    from gradscf.integrals.plan import IntegralPlan
    original = IntegralPlan.evaluate

    def checked(self, operator, *args, **kwargs):
        if operator == 'eri':
            pytest.fail('Direct CI/FCI response materialized complete AO ERIs')
        return original(self, operator, *args, **kwargs)

    monkeypatch.setattr(IntegralPlan, 'evaluate', checked)
    print(f'{method} frozen={frozen} core={core}: selected response', flush=True)
    actual = jax.jit(lambda r: _bundle(direct, r, direction))(point)
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(actual))
    for a, e in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(a, e, atol=4e-7, rtol=4e-6)
    np.testing.assert_allclose(actual[2], jnp.vdot(actual[1], direction), atol=3e-9)
    np.testing.assert_allclose(actual[1].sum(axis=0), 0., atol=5e-8)
    derivative = jax.jit(jax.grad(direct))
    print(f'{method} frozen={frozen} core={core}: two Hessian finite differences', flush=True)
    for step in (3e-4, 1e-4):
        finite = (derivative(point+step*direction)-derivative(point-step*direction))/(2*step)
        np.testing.assert_allclose(actual[-1], finite, atol=4e-6, rtol=3e-5)


@pytest.mark.parametrize('source', ['h2'], indirect=True)
@pytest.mark.parametrize('method', ['cisd', 'fci'])
def test_configuration_source_is_checked_before_the_moving_snapshot(source, method, monkeypatch):
    stale = copy(source)
    stale.mol = replace(source.mol, charge=2)
    workflow = importlib.import_module('gradscf_tools.correlated_response')
    original = workflow.MolecularEnergy

    def forbidden(*args, **kwargs):
        pytest.fail('A changed SCF source reached moving integral preparation')

    monkeypatch.setattr(workflow, 'MolecularEnergy', forbidden)
    with pytest.raises(RuntimeError, match='SCF|source|changed'):
        CorrelatedEnergy(stale, method, jk_backend='direct', max_memory=32.)
    monkeypatch.setattr(workflow, 'MolecularEnergy', original)
    fresh = CorrelatedEnergy(source, method, jk_backend='direct', max_memory=32.)
    assert fresh.nocc == 1


def test_unrestricted_workflow_requests_the_verified_virtual_blocks(monkeypatch):
    workflow = importlib.import_module('gradscf_tools.correlated_response')
    state = SimpleNamespace(mo_coeff=jnp.stack([jnp.eye(2)]*2), stationary=jnp.asarray(True),
        fock_matrix=jnp.stack([jnp.diag(jnp.array([-1., 1.]))]*2), total_energy=jnp.asarray(-1.))
    energy = CorrelatedEnergy.__new__(CorrelatedEnergy)
    energy.method, energy.max_memory, energy.nocc = 'ccsd', 32., (1, 1)
    energy._selected_mp2, energy._selected_cc, energy.config = False, True, cc.CCConfig()
    energy.reference = SimpleNamespace(unrestricted=True,
        solve_with_inputs=lambda *args, **kwargs: (state, {'eri': jnp.zeros((2,)*4)}))

    class Prepared(Exception):
        pass

    def prepared(*args, **kwargs):
        assert kwargs.get('vvvv') == 'blocks'
        raise Prepared

    monkeypatch.setattr(workflow, 'ucc_from_orbitals', prepared)
    with pytest.raises(Prepared):
        energy(jnp.zeros((1, 3)))
