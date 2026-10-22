"""Diagnose the already fixed H3 UHF unequal-frozen geometry point."""
import json
import importlib
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.linalg import solve_triangular

from gradscf import ci
from gradscf_tools.correlated_response import CorrelatedEnergy
from gradscf_tools.molecular_response import MolecularEnergy
from gradscf.ao2mo import transform_unrestricted_integrals
from gradscf.ao2mo.frame import project_frame
from test_correlated_blocks import source, _bundle


def _projected_initial_frame(state, inputs, seed, nocc):
    """A test-only, explicit gauge from stationary O/V projectors and A0."""
    metric = inputs['overlap']
    framed = []
    for current, initial, no in zip(state.mo_coeff, seed, nocc):
        blocks = []
        for selection in (slice(None, no), slice(no, None)):
            c, a0 = current[:, selection], initial[:, selection]
            projected = c@(c.T@metric@a0)
            gram = projected.T@metric@projected
            lower = jnp.linalg.cholesky(.5*(gram+gram.T))
            blocks.append(solve_triangular(lower, projected.T, lower=True).T)
        framed.append(jnp.concatenate(blocks, axis=1))
    return replace(state, mo_coeff=jnp.stack(framed))


@pytest.mark.parametrize('source', ['h3'], indirect=True)
def test_projected_initial_frame_closes_the_same_frozen_point(source, monkeypatch):
    energy = CorrelatedEnergy(source, 'cisd', frozen=([0], [2]), jk_backend='direct',
        max_memory=32., config=ci.CIConfig(solver='dense', conv_tol=1e-11))
    point = energy.coordinates
    direction = jnp.asarray(np.random.default_rng(1917).normal(size=point.shape))
    direction /= jnp.linalg.norm(direction)
    original = energy.reference.solve_with_inputs
    workflow = importlib.import_module('gradscf_tools.correlated_response')
    original_frame = workflow.project_frame
    monkeypatch.setattr(workflow, 'project_frame', lambda c, *args: c)
    old_value, old_gradient = jax.jit(jax.value_and_grad(energy))(point)
    monkeypatch.setattr(workflow, 'project_frame', original_frame)
    initial, inputs = original(point)
    framed = _projected_initial_frame(initial, inputs, energy.reference.coefficients, energy.nocc)
    occ = jnp.asarray(energy.reference.occupations)
    density = jnp.einsum('spi,si,sqi->spq', framed.mo_coeff, occ, framed.mo_coeff)
    np.testing.assert_allclose(density, initial.density_matrix, atol=1e-12, rtol=0.)
    np.testing.assert_array_equal(framed.total_energy, initial.total_energy)

    actual = jax.jit(lambda r: _bundle(energy, r, direction))(point)
    np.testing.assert_allclose(actual[0], old_value, atol=1e-12, rtol=0.)
    np.testing.assert_allclose(actual[1], old_gradient, atol=4e-7, rtol=4e-6)
    full = MolecularEnergy(source)
    space = ci.make_uci_space(3, energy.nocc, frozen=([0], [2]))

    def legacy(r):
        state, data = full.solve_with_inputs(r)
        state = _projected_initial_frame(state, data, full.coefficients, energy.nocc)
        h, g = transform_unrestricted_integrals(data['hcore'], state.mo_coeff,
                                                eri=data['eri'], max_memory=32.)
        result = ci.solve_ci(h, g, space, nuclear_repulsion=data['nuclear_repulsion'],
                             config=energy.config)
        return jnp.where(state.stationary & result.converged[0], result.total_energies[0], jnp.nan)

    expected = jax.jit(lambda r: _bundle(legacy, r, direction))(point)
    for a, e in zip(actual, expected):
        np.testing.assert_allclose(a, e, atol=4e-7, rtol=4e-6)
    derivative = jax.jit(jax.grad(energy))
    errors = []
    for step in (3e-4, 1e-4):
        finite = (derivative(point+step*direction)-derivative(point-step*direction))/(2*step)
        errors.append(float(jnp.max(jnp.abs(actual[-1]-finite))))
        np.testing.assert_allclose(actual[-1], finite, atol=4e-6, rtol=3e-5)
    print(json.dumps(dict(projected_initial_frame_hvp_errors=errors)), flush=True)


@pytest.mark.parametrize('dtype', [jnp.float32, jnp.float64])
def test_frame_preserves_dtype_and_reports_anchor_rank_loss(dtype):
    anchor = jnp.eye(2, dtype=dtype)
    frame = project_frame(anchor, anchor, anchor, 1)
    assert frame.dtype == dtype
    np.testing.assert_allclose(frame, anchor, atol=0., rtol=0.)
    invalid = project_frame(anchor[:,::-1], anchor, anchor, 1)
    assert not np.isfinite(invalid).all()


