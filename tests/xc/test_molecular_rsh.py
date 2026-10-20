"""Complete range-separated recipes retain full J, short- and long-range K."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.scf.rsh import rsh_preset, validate_rsh_inputs
from gradscf.tddft.response import _restricted_ao_response_action
from gradscf.tddft.unrestricted import _unrestricted_ao_response_action


@pytest.mark.parametrize('name', ['lc-wpbe', 'hse06'])
def test_complete_standard_rsh_recipe_is_available_without_dispersion(name):
    eri = jnp.zeros((2, 2, 2, 2))
    preset = rsh_preset(name)
    assert preset is not None, name
    assert validate_rsh_inputs(name, eri, eri, None) == preset


def test_unregistered_rsh_expression_cannot_lose_its_long_range_exchange():
    eri = jnp.zeros((2, 2, 2, 2))
    with pytest.raises((NotImplementedError, ValueError), match='(?i)named|range.separated|eri_lr'):
        validate_rsh_inputs('RSH(0.4,1.0,-1.0) + pbe', eri, None, None)


def test_custom_complete_rsh_expression_uses_its_full_sr_lr_metadata():
    spec = '0.2*HF + RSH(0.3,0.6,-0.2) + 0.7*pbe'
    preset = rsh_preset(spec)
    assert preset is not None
    np.testing.assert_allclose(preset.to_range_separated_hybrid_coefficients(), [.3, .8, .6],
                               rtol=0, atol=2e-15)
    eri = jnp.zeros((2, 2, 2, 2))
    assert validate_rsh_inputs(spec, eri, eri, None) == preset


def test_actual_unrestricted_reference_is_a_valid_jit_and_jvp_input():
    from dataclasses import replace
    from gradscf.scf.molecules import QuadratureGrid, UnrestrictedMolecule

    eri = jnp.arange(16., dtype=jnp.float64).reshape((2,)*4)/31
    eri_lr = jnp.flip(eri, axis=0)/3
    reference = UnrestrictedMolecule(
        ao=jnp.ones((1, 2)), grid=QuadratureGrid(jnp.ones(1)),
        dipole_integrals=jnp.zeros((3, 2, 2)), rep_tensor=eri, eri_lr=eri_lr,
        mo_coeff=jnp.stack([jnp.eye(2)]*2), mo_occ=jnp.asarray([[1., 0.]]*2),
        mo_energy=jnp.asarray([[-1., .5]]*2), rdm1=jnp.stack([jnp.eye(2)/2]*2),
        h1e=jnp.eye(2), nuclear_repulsion=0.,
        range_separated_coefficients=(.11, 0., .25), xc_spec='hse06')
    da = jnp.asarray([[[.2, .03], [-.07, .1]]])
    db = .7*da

    def action(state):
        return jnp.stack(_unrestricted_ao_response_action(
            state, .25, include_exchange=True, dtype=jnp.float64)(da, db))

    j = jnp.einsum('pqrs,nrs->npq', eri, da+db)
    k = lambda tensor, d: jnp.einsum('prqs,nrs->npq', tensor, d)
    expected = jnp.stack([j-.25*k(eri-eri_lr, da), j-.25*k(eri-eri_lr, db)])
    np.testing.assert_allclose(jax.jit(action)(reference), expected, rtol=0, atol=2e-15)
    tangent = jax.tree_util.tree_map(jnp.zeros_like, reference)
    tangent = replace(tangent, eri_lr=eri_lr)
    actual_jvp = jax.jvp(jax.jit(action), (reference,), (tangent,))[1]
    expected_jvp = .25*jnp.stack([k(eri_lr, da), k(eri_lr, db)])
    np.testing.assert_allclose(actual_jvp, expected_jvp, rtol=0, atol=2e-15)


@pytest.mark.parametrize('spin', ['restricted', 'unrestricted'])
@pytest.mark.parametrize('coefficients', [(.11, 0., .25), (.4, 1., 0.)],
                         ids=['hse06', 'lc-wpbe'])
def test_general_rsh_ao_response_and_lr_derivative(spin, coefficients):
    rng = np.random.default_rng(734)
    eri = jnp.asarray(rng.normal(size=(2, 2, 2, 2)))
    eri_lr = jnp.asarray(rng.normal(size=eri.shape))
    da, db = jnp.asarray(rng.normal(size=(2, 3, 2, 2)))
    omega, lr, sr = coefficients

    def action(scale):
        molecule = SimpleNamespace(rep_tensor=eri, eri_lr=scale*eri_lr,
            dispersion_energy=None, range_separated_coefficients=(omega, lr, sr))
        if spin == 'restricted':
            return _restricted_ao_response_action(molecule, sr, include_exchange=sr != 0.,
                                                  dtype=jnp.float64)(da)
        return jnp.stack(_unrestricted_ao_response_action(
            molecule, sr, include_exchange=sr != 0., dtype=jnp.float64)(da, db))

    k = lambda tensor, d: jnp.einsum('prqs,nrs->npq', tensor, d)
    exchange = sr*eri+(lr-sr)*eri_lr
    if spin == 'restricted':
        expected = jnp.einsum('pqrs,nrs->npq', eri, da)-.5*k(exchange, da)
        derivative = -.5*(lr-sr)*k(eri_lr, da)
    else:
        j = jnp.einsum('pqrs,nrs->npq', eri, da+db)
        expected = jnp.stack([j-k(exchange, da), j-k(exchange, db)])
        derivative = -(lr-sr)*jnp.stack([k(eri_lr, da), k(eri_lr, db)])
    np.testing.assert_allclose(action(1.), expected, rtol=2e-14, atol=2e-14)
    actual_derivative = jax.jvp(action, (jnp.asarray(1.),), (jnp.asarray(1.),))[1]
    np.testing.assert_allclose(actual_derivative, derivative, rtol=2e-14, atol=2e-14)
