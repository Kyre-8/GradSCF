"""LiH+/STO-3G UCCD: selected blocks retain relaxed nuclear response."""
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import ao2mo, cc
from gradscf.ao2mo import reference as ao_reference
from gradscf.cc.unrestricted_integrals import from_orbitals
from gradscf.integrals.plan import IntegralPlan
from gradscf_tools.molecular_response import MolecularEnergy
from test_scf_unrestricted_geometry import _bundle
from test_scf_unrestricted_prepared import source


def test_uccd_selected_relaxed_gradient_hvp_matches_dense(source, monkeypatch):
    full = MolecularEnergy(source, jk_backend='full', max_memory=64.)
    direct = MolecularEnergy(source, jk_backend='direct', max_memory=64.)
    config = cc.CCConfig(method='ccd', conv_tol=1e-11, residual_tol=1e-10)
    point = direct.coordinates
    direction = jnp.array([[.1, -.2, -.4], [-.1, .2, .4]])
    direction /= jnp.linalg.norm(direction)

    def calculate(coordinates, selected):
        state, inputs = (direct if selected else full).solve_with_inputs(coordinates)
        if selected:
            cs = tuple(state.mo_coeff)
            focks = tuple(ao_reference.transform_fock(f, c)
                          for f, c in zip(state.fock_matrix, cs))
            data, occupied, virtual = from_orbitals(cs, focks, (2, 1),
                state.total_energy, direct_basis=inputs['direct_basis'],
                vvvv='blocks', max_memory=64.)
            result = cc.run_ucc_from_integrals(data, occupied, virtual,
                                               config=config, max_memory=64.)
        else:
            h, g = ao2mo.transform_unrestricted_integrals(inputs['hcore'],
                state.mo_coeff, eri=inputs['eri'], max_memory=64.)
            result = cc.run_ucc(h, g, nocc=(2, 1),
                nuclear_repulsion=inputs['nuclear_repulsion'], config=config,
                max_memory=64.)
        value = jnp.where(state.stationary & result.converged, result.total_energy, jnp.nan)
        return value, result.correlation_energy, result.residual_norm

    dense_energy = lambda r: calculate(r, False)[0]
    selected_energy = lambda r: calculate(r, True)[0]
    expected = jax.block_until_ready(jax.jit(lambda r: _bundle(dense_energy, r, direction))(point))
    assert all(np.isfinite(np.asarray(a)).all() for a in expected)

    def forbidden(*args, **kwargs):
        pytest.fail('Selected UCCD geometry requested complete AO/MO integrals')

    original = IntegralPlan.evaluate

    def checked(self, operator, *args, **kwargs):
        if operator == 'eri':
            forbidden()
        return original(self, operator, *args, **kwargs)

    # The dense oracle has finished; only the selected route sees these traps.
    monkeypatch.setattr(IntegralPlan, 'evaluate', checked)
    for module in (ao2mo, ao_reference):
        monkeypatch.setattr(module, 'transform_integrals', forbidden)
        monkeypatch.setattr(module, 'transform_unrestricted_integrals', forbidden)
    actual = jax.block_until_ready(jax.jit(lambda r: _bundle(selected_energy, r, direction))(point))
    diagnostic = jax.jit(lambda r: calculate(r, True))(point)
    assert all(np.isfinite(np.asarray(a)).all() for a in (*actual, *diagnostic))
    assert abs(float(diagnostic[1])) > 1e-10, 'The UCCD correlation gate is vacuous'
    np.testing.assert_allclose(actual[0], expected[0], atol=3e-9, rtol=0.)
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, atol=3e-7, rtol=3e-6)
    np.testing.assert_allclose(actual[2], jnp.vdot(actual[1], direction), atol=2e-10)
    np.testing.assert_allclose(actual[1].sum(axis=0), 0., atol=2e-8)
    gradient = jax.jit(jax.grad(selected_energy))
    fd_errors = {}
    for step in (3e-4, 1e-4):
        finite = (gradient(point+step*direction)-gradient(point-step*direction))/(2*step)
        assert np.isfinite(np.asarray(finite)).all()
        np.testing.assert_allclose(actual[-1], finite, atol=3e-6, rtol=2e-5)
        fd_errors[str(step)] = float(jnp.max(jnp.abs(actual[-1]-finite)))
    print(json.dumps(dict(method='UCCD',geometry_unit='Bohr',energy_hartree=float(actual[0]),
        correlation_hartree=float(diagnostic[1]),residual_norm=float(diagnostic[2]),
        dense_errors=[float(jnp.max(jnp.abs(a-b))) for a,b in zip(actual,expected)],
        translation_residual=float(jnp.max(jnp.abs(actual[1].sum(axis=0)))),
        hvp_norm=float(jnp.linalg.norm(actual[-1])),fd_hvp_errors=fd_errors)))
