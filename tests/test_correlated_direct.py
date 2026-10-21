"""Relaxed direct MP2 geometry response; float64, Hartree and Bohr.

The extra scalar scales a fixed AO potential (Hartree). Its mixed response
tests the same SCF/amplitude dependencies as coordinate-only differentiation.
"""
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture(scope='module', params=[False, True], ids=['rhf', 'uhf'])
def hf_source(request):
    from gradscf import gto, scf
    atom = ('H 0 0 0; H 0 0 1.5; H 1.4 0 .2' if request.param
            else 'H 0 0 0; H 0 0 1.5')
    molecule = gto.M(atom=atom, basis='sto-3g' if request.param else '3-21g',
                     spin=int(request.param), unit='Bohr', cart=True)
    source = (scf.UHF if request.param else scf.RHF)(molecule,
        integral_backend='native', execution_device='cpu', max_memory=128.,
        conv_tol=1e-12, conv_tol_density=1e-11, conv_tol_grad=1e-10,
        max_cycle=250).run()
    assert source.converged
    return source


def _forbid_full_integrals(monkeypatch):
    from gradscf.integrals.plan import IntegralPlan
    original = IntegralPlan.evaluate

    def checked(self, operator, *args, **kwargs):
        if operator == 'eri':
            pytest.fail('Direct correlated response evaluated a full AO ERI')
        return original(self, operator, *args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail('Direct MP2 response transformed a full MO ERI')

    monkeypatch.setattr(IntegralPlan, 'evaluate', checked)
    workflow = importlib.import_module('gradscf_tools.correlated_response')
    monkeypatch.setattr(workflow, 'transform_integrals', forbidden)
    monkeypatch.setattr(workflow, 'transform_unrestricted_integrals', forbidden)


def test_direct_mp2_relaxed_response_matches_dense(hf_source, monkeypatch):
    from gradscf_tools.correlated_response import CorrelatedEnergy
    dense = CorrelatedEnergy(hf_source, 'mp2', gradient_tolerance=1e-10)
    direct = CorrelatedEnergy(hf_source, 'mp2', jk_backend='direct',
                              max_memory=32., gradient_tolerance=1e-10)
    coordinates = direct.coordinates
    rng = np.random.default_rng(507)
    nao = hf_source.mo_coeff.shape[-2]
    potential = jnp.asarray(rng.normal(scale=.03, size=(nao, nao)))
    potential = .5*(potential + potential.T)
    direction = jnp.asarray(rng.normal(size=coordinates.shape))
    direction /= jnp.linalg.norm(direction)
    directions = direction, jnp.asarray(.02)
    point = coordinates, jnp.asarray(0.)

    def energy(model):
        return lambda r, strength: model(r, potential=strength*potential)

    full, blocked = energy(dense), energy(direct)
    full_grad = jax.grad(full, argnums=(0, 1))
    expected = jax.jit(jax.value_and_grad(full, argnums=(0, 1)))(*point)
    expected_hvp = jax.jit(lambda r, s: jax.jvp(full_grad, (r, s), directions)[1])(*point)
    jax.block_until_ready((expected, expected_hvp))
    _forbid_full_integrals(monkeypatch)
    derivative = jax.jit(jax.grad(blocked, argnums=(0, 1)))
    actual = jax.jit(jax.value_and_grad(blocked, argnums=(0, 1)))(*point)
    actual_hvp = jax.jit(lambda r, s: jax.jvp(
        jax.grad(blocked, argnums=(0, 1)), (r, s), directions)[1])(*point)
    assert np.isfinite(actual[0])
    for a, e in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(a, e, atol=2e-8, rtol=2e-7)
    for a, e in zip(jax.tree.leaves(actual_hvp), jax.tree.leaves(expected_hvp)):
        np.testing.assert_allclose(a, e, atol=2e-7, rtol=2e-6)
    for step in (3e-4, 1e-4):
        plus = jax.tree.map(lambda x, v: x + step*v, point, directions)
        minus = jax.tree.map(lambda x, v: x - step*v, point, directions)
        finite = jax.tree.map(lambda a, b: (a-b)/(2*step), derivative(*plus), derivative(*minus))
        for a, e in zip(jax.tree.leaves(actual_hvp), jax.tree.leaves(finite)):
            np.testing.assert_allclose(a, e, atol=3e-6, rtol=2e-5)
    np.testing.assert_allclose(actual[1][0].sum(axis=0), 0., atol=2e-8, rtol=0)


def test_direct_mp2_nitrogen_internal_degeneracy_has_finite_geometry_response(monkeypatch):
    from gradscf import gto, scf
    from gradscf_tools.correlated_response import CorrelatedEnergy
    mol = gto.M(atom='N 0 0 -1.04; N 0 0 1.04', basis='sto-3g', unit='Bohr', cart=True)
    mf = scf.RHF(mol, integral_backend='native', conv_tol=1e-13,
        conv_tol_density=1e-11, conv_tol_grad=1e-10, max_cycle=250,
        init_guess=1.4*np.eye(10)).run()
    assert mf.converged
    assert np.min(abs(np.diff(np.asarray(mf.mo_energy)))) < 1e-9
    dense = CorrelatedEnergy(mf, 'mp2', gradient_tolerance=1e-10)
    direct = CorrelatedEnergy(mf, 'mp2', jk_backend='direct', max_memory=32.,
                              gradient_tolerance=1e-10)
    expected = jax.jit(jax.value_and_grad(dense))(dense.coordinates)
    jax.block_until_ready(expected)
    _forbid_full_integrals(monkeypatch)
    value, gradient = jax.jit(jax.value_and_grad(direct))(direct.coordinates)
    assert np.isfinite(value) and np.isfinite(np.asarray(gradient)).all()
    np.testing.assert_allclose(value, expected[0], atol=3e-9, rtol=0)
    np.testing.assert_allclose(gradient, expected[1], atol=3e-7, rtol=0)
    direction = jnp.array([[0., 0., -.5], [0., 0., .5]])
    hvp = jax.jit(lambda r: jax.jvp(jax.grad(direct), (r,), (direction,))[1])(direct.coordinates)
    derivative = jax.jit(jax.grad(direct))
    for step in (3e-4, 1e-4):
        finite = (derivative(direct.coordinates+step*direction)
                  - derivative(direct.coordinates-step*direction))/(2*step)
        np.testing.assert_allclose(hvp, finite, atol=3e-6, rtol=2e-5)
