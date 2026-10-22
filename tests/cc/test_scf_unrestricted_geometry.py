"""LiH+ selected UCCSD/(T): full relaxed geometry response in Hartree/Bohr."""
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import cc
from gradscf.ao2mo import transform_unrestricted_integrals
from gradscf_tools.correlated_response import CorrelatedEnergy
from test_scf_unrestricted_prepared import source


def _bundle(function, point, direction):
    (value, gradient), (jvp, hvp) = jax.jvp(jax.value_and_grad(function),
                                          (point,), (direction,))
    return value, gradient, jvp, hvp


@pytest.mark.parametrize('method', ['ccsd', 'ccsd(t)'])
def test_selected_uhf_relaxed_gradient_hvp_and_nonzero_triples(source, method, monkeypatch):
    full = CorrelatedEnergy(source, method, max_memory=64.)
    direct = CorrelatedEnergy(source, method, jk_backend='direct', max_memory=64.)
    point = direct.coordinates
    direction = jnp.array([[.1, -.2, -.4], [-.1, .2, .4]])
    direction /= jnp.linalg.norm(direction)

    def legacy(coordinates, triples=True):
        state, inputs = full.reference.solve_with_inputs(coordinates)
        h, g = transform_unrestricted_integrals(inputs['hcore'], state.mo_coeff,
                                                 eri=inputs['eri'], max_memory=64.)
        right = cc.run_ucc(h, g, nocc=(2, 1),
            nuclear_repulsion=inputs['nuclear_repulsion'], config=full.config)
        total, valid = right.total_energy, state.stationary & right.converged
        if method == 'ccsd(t)' and triples:
            correction = cc.evaluate_triples(h, g, right, nocc=(2, 1),
                orbital_basis='semicanonical', algorithm='streamed',
                denominator_tol=full.config.denominator_tol,
                residual_tol=full.config.residual_tol)
            total, valid = total+correction.energy, valid & correction.valid
        return jnp.where(valid, total, jnp.nan)

    expected = jax.jit(lambda r: _bundle(legacy, r, direction))(point)
    jax.block_until_ready(expected)
    if method == 'ccsd(t)':
        ground = jax.jit(lambda r: legacy(r, False))(point)
        assert abs(float(expected[0]-ground)) > 1e-12, 'Relaxed triples gate is vacuous'

    def forbidden(*args, **kwargs):
        pytest.fail('Selected UHF geometry built complete AO/MO integral tensors')

    workflow = importlib.import_module('gradscf_tools.correlated_response')
    monkeypatch.setattr(workflow, 'transform_unrestricted_integrals', forbidden)
    from gradscf.integrals.plan import IntegralPlan
    original = IntegralPlan.evaluate

    def checked(self, operator, *args, **kwargs):
        if operator == 'eri':
            forbidden()
        return original(self, operator, *args, **kwargs)

    monkeypatch.setattr(IntegralPlan, 'evaluate', checked)
    actual = jax.jit(lambda r: _bundle(direct, r, direction))(point)
    assert all(np.isfinite(np.asarray(a)).all() for a in actual)
    np.testing.assert_allclose(actual[0], expected[0], atol=3e-9, rtol=0.)
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, atol=3e-7, rtol=3e-6)
    np.testing.assert_allclose(actual[2], jnp.vdot(actual[1], direction), atol=2e-10)
    np.testing.assert_allclose(actual[1].sum(axis=0), 0., atol=2e-8)
    gradient = jax.jit(jax.grad(direct))
    for step in (3e-4, 1e-4):
        finite = (gradient(point+step*direction)-gradient(point-step*direction))/(2*step)
        np.testing.assert_allclose(actual[-1], finite, atol=3e-6, rtol=2e-5)
