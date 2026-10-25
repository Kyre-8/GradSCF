"""Native selected MP4 retains the existing nonfrozen relaxed HF response.

The raw oracle uses the same transported frame and the original full-MO
Taylor entry. Coordinates are Bohr, energies Hartree. The shared HF fixtures
fix H2/6-31G (RHF) and H3/STO-3G (UHF), without changing their SCF thresholds.
"""
import importlib
from pathlib import Path
import runpy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import mp
from gradscf.ao2mo import transform_integrals, transform_unrestricted_integrals
from gradscf.mp.series import run_series
from gradscf_tools.correlated_response import CorrelatedEnergy
from gradscf_tools.molecular_response import MolecularEnergy


source = runpy.run_path(str(Path(__file__).parents[1]/'ci'/'test_correlated_blocks.py'))['source']


def _problem(source, algorithm):
    config = mp.MPConfig(order=4, algorithm=algorithm, orbital_basis='noncanonical',
                         with_t2=False)
    selected = CorrelatedEnergy(source, 'mp', config=config,
                                jk_backend='direct', max_memory=32.)
    reference = MolecularEnergy(source)
    direction = jnp.asarray(np.random.default_rng(1917).normal(size=selected.coordinates.shape))
    direction /= jnp.linalg.norm(direction)

    def legacy(coordinates):
        state, inputs = reference.solve_with_inputs(coordinates)
        transform = transform_unrestricted_integrals if reference.unrestricted else transform_integrals
        h, g = transform(inputs['hcore'], state.mo_coeff, eri=inputs['eri'], max_memory=32.)
        result = run_series(h, g, nocc=selected.nocc,
            nuclear_repulsion=inputs['nuclear_repulsion'], config=config, max_memory=32.)
        return jnp.where(state.stationary & result.valid, result.total_energy, jnp.nan)

    return selected, legacy, selected.coordinates, direction


def _bundle(function, point, direction):
    (value, gradient), (jvp, hvp) = jax.jvp(jax.value_and_grad(function),
                                         (point,), (direction,))
    return value, gradient, jvp, hvp


def _evaluate(function, point, direction):
    return jax.jit(lambda coordinates: _bundle(function, coordinates, direction))(point)


def _forbid_full_inputs(monkeypatch):
    workflow = importlib.import_module('gradscf_tools.correlated_response')

    def forbidden(*args, **kwargs):
        pytest.fail('Selected MP4 rebuilt a complete AO/MO Hamiltonian')

    monkeypatch.setattr(workflow, 'transform_integrals', forbidden)
    monkeypatch.setattr(workflow, 'transform_unrestricted_integrals', forbidden)
    from gradscf.integrals.plan import IntegralPlan
    original = IntegralPlan.evaluate

    def checked(self, operator, *args, **kwargs):
        if operator == 'eri':
            forbidden()
        return original(self, operator, *args, **kwargs)

    monkeypatch.setattr(IntegralPlan, 'evaluate', checked)


@pytest.mark.parametrize('source,algorithm', [('h2', 'series'), ('h3', 'tensor')], indirect=['source'])
def test_selected_series_relaxed_geometry_matches_raw(source, algorithm, monkeypatch):
    selected, legacy, point, direction = _problem(source, algorithm)
    expected = _evaluate(legacy, point, direction)
    _forbid_full_inputs(monkeypatch)
    actual = _evaluate(selected, point, direction)
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(actual))
    for a, e in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(a, e, atol=4e-7, rtol=4e-6)
    np.testing.assert_allclose(actual[2], jnp.vdot(actual[1], direction), atol=3e-9)
    np.testing.assert_allclose(actual[1].sum(axis=0), 0., atol=5e-8)
    derivative = jax.jit(jax.grad(selected))
    for step in (3e-4, 1e-4):
        finite = (derivative(point+step*direction)-derivative(point-step*direction))/(2*step)
        np.testing.assert_allclose(actual[-1], finite, atol=4e-6, rtol=3e-5)
