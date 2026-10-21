"""Selected restricted CC blocks retain relaxed geometry response.

Four hydrogens/STO-3G exercise a nonzero triples correction. The legacy oracle
is a composition of unchanged src interfaces, with a complete MO transform.
Geometry and gradients use Bohr and Hartree/Bohr, respectively.
"""
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import cc, gto, scf
from gradscf.ao2mo import transform_integrals
from gradscf_tools.correlated_response import CorrelatedEnergy


@pytest.fixture(scope='module')
def source():
    molecule = gto.M(atom='H 0 0 0; H 0 0 1.4; H 1.5 0 .1; H 1.5 0 1.5',
                     basis='sto-3g', unit='Bohr', cart=True)
    mf = scf.RHF(molecule, integral_backend='native', execution_device='cpu',
        conv_tol=1e-12, conv_tol_density=1e-11, conv_tol_grad=1e-10,
        max_cycle=250).run()
    assert mf.converged
    return mf


def _bundle(function, point, direction):
    value, gradient = jax.value_and_grad(function)(point)
    jvp = jax.jvp(function, (point,), (direction,))[1]
    hvp = jax.jvp(jax.grad(function), (point,), (direction,))[1]
    return value, gradient, jvp, hvp


@pytest.mark.parametrize('method', ['ccsd', 'ccsd(t)'])
def test_selected_cc_relaxed_gradient_and_hvp_match_full_mo(source, method, monkeypatch):
    full = CorrelatedEnergy(source, method, max_memory=32.)
    direct = CorrelatedEnergy(source, method, jk_backend='direct', max_memory=32.)
    point = direct.coordinates
    rng = np.random.default_rng(6418)
    direction = jnp.asarray(rng.normal(size=point.shape))
    direction /= jnp.linalg.norm(direction)

    def legacy(r):
        state, inputs = full.reference.solve_with_inputs(r)
        h, g = transform_integrals(inputs['hcore'], state.mo_coeff, eri=inputs['eri'],
                                    max_memory=32.)
        result = cc.run_cc(h, g, nocc=2, nuclear_repulsion=inputs['nuclear_repulsion'],
                           config=full.config)
        total, valid = result.total_energy, state.stationary & result.converged
        if method == 'ccsd(t)':
            correction = cc.evaluate_triples(h, g, result, nocc=2,
                orbital_basis='semicanonical', algorithm='streamed',
                denominator_tol=full.config.denominator_tol,
                residual_tol=full.config.residual_tol)
            total, valid = total+correction.energy, valid & correction.valid
        return jnp.where(valid, total, jnp.nan)

    expected = jax.jit(lambda r: _bundle(legacy, r, direction))(point)
    jax.block_until_ready(expected)
    if method == 'ccsd(t)':
        ground = CorrelatedEnergy(source, 'ccsd', max_memory=32.)
        correction = expected[0]-jax.jit(ground)(point)
        assert abs(float(correction)) > 1e-12, 'The triples geometry gate is vacuous'

    def forbidden(*args, **kwargs):
        pytest.fail('Selected CC response transformed a complete MO Hamiltonian')
    workflow = importlib.import_module('gradscf_tools.correlated_response')
    monkeypatch.setattr(workflow, 'transform_integrals', forbidden)
    from gradscf.integrals.plan import IntegralPlan
    original = IntegralPlan.evaluate

    def checked(self, operator, *args, **kwargs):
        if operator == 'eri':
            pytest.fail('Direct selected CC response materialized complete AO ERIs')
        return original(self, operator, *args, **kwargs)
    monkeypatch.setattr(IntegralPlan, 'evaluate', checked)
    actual = jax.jit(lambda r: _bundle(direct, r, direction))(point)
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(actual))
    np.testing.assert_allclose(actual[0], expected[0], atol=3e-9, rtol=0.)
    for a, e in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(a, e, atol=3e-7, rtol=3e-6)
    np.testing.assert_allclose(actual[2], jnp.vdot(actual[1], direction), atol=2e-10)
    np.testing.assert_allclose(actual[1].sum(axis=0), 0., atol=2e-8)
    derivative = jax.jit(jax.grad(direct))
    for step in (3e-4, 1e-4):
        finite = (derivative(point+step*direction)-derivative(point-step*direction))/(2*step)
        np.testing.assert_allclose(actual[-1], finite, atol=3e-6, rtol=2e-5)
