"""Existing workflow methods forward one MB budget and retained source owners."""
import importlib
from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from gradscf import cc, mp
from gradscf._memory import tree_storage_bytes


@pytest.mark.parametrize('method', ['mp2', 'ccsd', 'ccsd(t)'])
def test_existing_selected_workflow_forwards_producer_and_consumer_budget(method, monkeypatch):
    workflow = importlib.import_module('gradscf_tools.correlated_response')
    c = jnp.eye(2)
    fock = jnp.diag(jnp.array([-1., 1.]))
    state = SimpleNamespace(mo_coeff=c, fock_matrix=jnp.stack([fock]*2),
        stationary=jnp.asarray(True), total_energy=jnp.asarray(-1.))
    inputs = dict(eri=jnp.zeros((2,)*4), hcore=fock, overlap=c, nuclear_repulsion=jnp.asarray(0.))
    model = workflow.CorrelatedEnergy.__new__(workflow.CorrelatedEnergy)
    model.method, model.max_memory, model.nocc = method, 2., 1
    model._selected_mp2, model._selected_cc, model._selected_mp_series = method == 'mp2', method != 'mp2', False
    model.config = (mp.MPConfig(orbital_basis='noncanonical') if method == 'mp2' else cc.CCConfig())
    model.reference = SimpleNamespace(unrestricted=False, parameters={'centers': jnp.ones((2, 3))},
        coefficients=c, coordinates=jnp.zeros((2, 3)), occupations=jnp.ones((2, 2)),
        plan=SimpleNamespace(topology={'weights': jnp.ones(2)}),
        solve_with_inputs=lambda *a, **k: (state, inputs))
    prepared = SimpleNamespace(fock=fock, ovov=jnp.ones((1, 1, 1, 1)))
    result = SimpleNamespace(total_energy=jnp.asarray(-1.1), valid=jnp.asarray(True),
        converged=jnp.asarray(True), t1=jnp.zeros((1, 1)), t2=jnp.zeros((1,)*4))
    calls = []
    ao_focks, mo_focks = [], []
    transform_fock = workflow.transform_fock
    def observe_fock(fock, coefficients):
        value = transform_fock(fock, coefficients)
        ao_focks.append(fock)
        mo_focks.append(value)
        return value
    monkeypatch.setattr(workflow, 'transform_fock', observe_fock)
    def checked(label, value):
        def call(*a, **kwargs):
            assert kwargs['max_memory'] == 2. and kwargs['reserved_bytes'] > 0
            if label == 'triples':
                source = (state, inputs, c, model.reference.parameters,
                    model.reference.coefficients, model.reference.coordinates,
                    model.reference.occupations, model.reference.plan.topology)
                assert kwargs['reserved_bytes'] == tree_storage_bytes(
                    *source, ao_focks, mo_focks, exclude=(prepared, result))
            calls.append(label)
            return value
        return call
    monkeypatch.setattr(workflow, 'from_orbitals' if method == 'mp2' else 'cc_from_orbitals',
                        checked('prepare', prepared))
    if method == 'mp2':
        monkeypatch.setattr(workflow, 'evaluate_mp2', checked('solve', result))
    else:
        monkeypatch.setattr(cc, 'run_cc_from_integrals', checked('solve', result))
        monkeypatch.setattr(cc, 'evaluate_triples_from_integrals', checked('triples',
            SimpleNamespace(energy=jnp.asarray(0.), valid=jnp.asarray(True))))
    model(jnp.zeros((2, 3)))
    assert calls == (['prepare', 'solve', 'triples'] if method == 'ccsd(t)' else ['prepare', 'solve'])
