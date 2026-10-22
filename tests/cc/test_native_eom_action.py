"""Prepared native virtual actions through the existing EE/IP/EA equations."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import cc
from gradscf.cc.integrals import from_orbitals
from gradscf.cc.eom import build_eom_operator_from_integrals
from test_native_virtual_action import _problem, _assert_trees


def _prepared(coefficients, basis, mode):
    return from_orbitals(coefficients, jnp.diag(jnp.asarray([-1.2, .4, .7])),
        1, jnp.asarray(-1.4), direct_basis=basis, max_memory=8., vvvv=mode)


@pytest.mark.parametrize('sector', ['ee', 'ip', 'ea'])
def test_native_eom_right_left_actions_and_roots_match_dense(sector):
    _, basis, coefficients, _ = _problem()
    coefficients = .7*coefficients
    config = cc.CCConfig(conv_tol=1e-12, residual_tol=1e-11, adjoint_tol=1e-11)
    eom_config = cc.EOMConfig(sector=sector, solver='dense', conv_tol=1e-10)
    outputs = []
    for mode in ('dense', 'direct'):
        ints = _prepared(coefficients, basis, mode)
        ground = cc.run_cc_from_integrals(ints, config=config)
        assert ground.converged
        operator, space, valid = build_eom_operator_from_integrals(ints, ground,
            config=eom_config, cc_config=config, max_memory=8.)
        assert valid
        vector = jnp.asarray(np.random.default_rng(6563).normal(size=space.size))
        action = operator.matvec(vector)
        transpose = jax.linear_transpose(operator.matvec, vector)(vector)[0]
        roots = cc.run_eom_from_integrals(ints, ground, config=eom_config,
            cc_config=config, max_memory=8.)
        assert jnp.all(roots.converged) and jnp.all(roots.response_valid)
        outputs.append((action, transpose, roots.energies))
    _assert_trees(outputs[1], outputs[0], atol=3e-10, rtol=3e-9)


def test_native_ea_root_keeps_ground_and_orbital_response():
    _, basis, coefficients, _ = _problem()
    coefficients = .7*coefficients
    config = cc.CCConfig(conv_tol=1e-12, residual_tol=1e-11, adjoint_tol=1e-11)
    eom_config = cc.EOMConfig(sector='ea', solver='dense', conv_tol=1e-10)

    def energy(scale, mode):
        ints = _prepared(coefficients*(1+.03*scale), basis, mode)
        ground = cc.run_cc_from_integrals(ints, config=config)
        return cc.run_eom_from_integrals(ints, ground, config=eom_config,
            cc_config=config, max_memory=8.).energies[0]

    actual = jax.jit(jax.value_and_grad(lambda x: energy(x, 'direct')))(0.)
    expected = jax.jit(jax.value_and_grad(lambda x: energy(x, 'dense')))(0.)
    _assert_trees(actual, expected, atol=3e-9, rtol=3e-8)
    finite = (energy(1e-4, 'direct')-energy(-1e-4, 'direct'))/2e-4
    np.testing.assert_allclose(actual[1], finite, atol=3e-8, rtol=3e-6)
