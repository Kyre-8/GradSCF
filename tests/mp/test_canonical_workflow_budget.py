"""Canonical orbital solves retain the correlated caller's original budget."""
import importlib
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import mp
from gradscf._memory import tree_storage_bytes


def _model(unrestricted):
    workflow = importlib.import_module('gradscf_tools.correlated_response')
    c = jnp.eye(3)
    fock = jnp.diag(jnp.array([-1.4, -.3, .8]))
    coefficients = jnp.stack((c, c)) if unrestricted else c
    state = SimpleNamespace(mo_coeff=coefficients, fock_matrix=jnp.stack((fock, fock)),
        stationary=jnp.asarray(True), total_energy=jnp.asarray(-2.),
        diagnostics=jnp.ones(17))
    inputs = dict(eri=jnp.zeros((3,)*4), hcore=fock, overlap=c,
                  nuclear_repulsion=jnp.asarray(0.), retained=jnp.ones(23))
    model = workflow.CorrelatedEnergy.__new__(workflow.CorrelatedEnergy)
    model.method, model.max_memory, model.nmo = 'mp2', 2., 3
    model.nocc = (2, 1) if unrestricted else 1
    model._selected_mp2, model._selected_cc, model._selected_mp_series = True, False, False
    model.config = mp.MPConfig(orbital_basis='canonical', with_t2=False)
    model.reference = SimpleNamespace(unrestricted=unrestricted,
        parameters={'centers': jnp.ones((2, 3))}, coefficients=coefficients,
        coordinates=jnp.zeros((2, 3)), occupations=jnp.ones((2, 3)),
        plan=SimpleNamespace(topology={'weights': jnp.ones(29)}),
        spec=SimpleNamespace(charges=jnp.ones(31), coords_bohr=jnp.ones((31, 3))),
        solve_with_inputs=lambda *a, **k: (state, inputs))
    return model, state, inputs


@pytest.mark.parametrize('unrestricted', [False, True])
def test_canonical_workflow_rejects_exhausted_budget_before_eigensolve(unrestricted, monkeypatch):
    model, _, _ = _model(unrestricted)
    model.max_memory = .0001
    eigen = importlib.import_module('gradscf.solvers.eigen.api')
    monkeypatch.setattr(eigen, 'solve_ritz', lambda *a, **k: pytest.fail(
        'Exhausted correlated caller budget reached the canonical eigensolve'))
    with pytest.raises(ValueError, match='memory|budget|resident'):
        model(jnp.zeros((2, 3)))


@pytest.mark.parametrize('unrestricted', [False, True])
def test_canonical_eigensolve_keeps_sources_and_prior_spin(unrestricted, monkeypatch):
    workflow = importlib.import_module('gradscf_tools.correlated_response')
    model, state, inputs = _model(unrestricted)
    reference = model.reference
    parent_bytes = tree_storage_bytes(state, inputs, reference.parameters,
        reference.coefficients, reference.coordinates, reference.occupations,
        reference.plan.topology, reference.spec)
    reservations = []

    def eigensolve(matrix, **kwargs):
        assert kwargs.get('max_memory') == model.max_memory
        assert kwargs.get('reserved_bytes', 0) >= parent_bytes
        assert kwargs['config'].method == 'dense'
        assert kwargs['config'].nroots == model.nmo and kwargs['config'].atol == 1e-10
        assert kwargs['response'].target == 'eigenpairs'
        reservations.append(kwargs['reserved_bytes'])
        return SimpleNamespace(vectors=jnp.eye(model.nmo), response_valid=jnp.ones(model.nmo, bool))

    class Prepared(Exception):
        pass

    def stop_at_preparation(*args, **kwargs):
        raise Prepared

    monkeypatch.setattr(workflow, 'solve_hermitian', eigensolve)
    monkeypatch.setattr(workflow, 'from_orbitals', stop_at_preparation)
    with pytest.raises(Prepared):
        model(jnp.zeros((2, 3)))
    assert len(reservations) == (2 if unrestricted else 1)
    if unrestricted:
        # Alpha's rotated coefficients and validity scalar survive the beta solve.
        assert reservations[1] >= reservations[0] + model.nmo**2*8 + 1


@pytest.mark.parametrize('unrestricted', [False, True])
def test_canonical_workflow_preserves_separated_value_and_gradient(unrestricted, monkeypatch):
    from gradscf import gto, scf
    workflow = importlib.import_module('gradscf_tools.correlated_response')
    atom = ('H 0 0 0; H 0 0 1.5; H 1.4 0 .2' if unrestricted
            else 'H 0 0 0; H 0 0 1.5')
    molecule = gto.M(atom=atom, basis='sto-3g' if unrestricted else '6-31g',
                     spin=int(unrestricted), unit='Bohr', cart=True)
    source = (scf.UHF if unrestricted else scf.RHF)(molecule,
        integral_backend='native', execution_device='cpu', conv_tol=1e-12,
        conv_tol_density=1e-11, conv_tol_grad=1e-10, max_cycle=250).run()
    assert source.converged
    assert np.min(np.abs(np.diff(np.asarray(source.mo_energy), axis=-1))) > 1e-5
    model = workflow.CorrelatedEnergy(source, 'mp2', max_memory=32.,
        config=mp.MPConfig(orbital_basis='canonical', with_t2=False))
    original = workflow.solve_hermitian

    def previous_eigensolve(*args, **kwargs):
        kwargs.pop('max_memory', None)
        kwargs.pop('reserved_bytes', None)
        return original(*args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(workflow, 'solve_hermitian', previous_eigensolve)
        expected = jax.jit(jax.value_and_grad(lambda r: model(r)))(model.coordinates)
    actual = jax.jit(jax.value_and_grad(lambda r: model(r)))(model.coordinates)
    for value, reference in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        assert np.isfinite(np.asarray(value)).all()
        np.testing.assert_allclose(value, reference, atol=2e-10, rtol=2e-9)
