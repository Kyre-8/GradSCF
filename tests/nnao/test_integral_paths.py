"""Fixed primitive projection and native contraction share numerical conventions."""
import jax
import jax.numpy as jnp
import numpy as np

from gradscf.model.nnao import prepare_direct_basis
from gradscf.model.nnao.integrals.context import prepare_fixed_context, prepare_native_context
from gradscf.model.nnao.integrals.fixed_exponents import physical_inputs
from gradscf.model.nnao.integrals.variable_exponents import native_projection


def test_fixed_and_native_paths_match_at_nonzero_exponents_and_coefficients():
    atom = [('H', (0., 0., 0.)), ('H', (0., 0., .74))]
    layout = prepare_direct_basis(atom, cart=False, basis_family='szp663_direct')
    keys = layout.exponent_scale_keys
    beta = jnp.array([.03, -.02])
    scaled = layout.with_log_exponent_scales(dict(zip(keys, beta)))
    fixed = prepare_fixed_context(scaled, ('H', 'H'), np.array([p for _, p in atom]),
        basis_family='szp663_direct', jk_backend='df', auxbasis='def2-universal-jkfit')
    context = prepare_native_context(layout, atom, 'def2-universal-jkfit')
    forward, backward = native_projection(context, keys)
    coefficients = layout.reference_outputs() + .015*jnp.sin(
        jnp.arange(layout.reference_outputs().size).reshape(layout.reference_outputs().shape))
    project = lambda c: physical_inputs(scaled, c, fixed.overlap, fixed.hcore, fixed.factors)
    expected, pullback = jax.vjp(project, coefficients)
    actual = forward(coefficients, beta)
    for left, right in zip(expected, actual):
        np.testing.assert_allclose(left, right, rtol=2e-10, atol=2e-10)
    cotangent = tuple(jnp.cos(jnp.arange(x.size).reshape(x.shape)) for x in expected)
    native_dc, native_db = backward(coefficients, beta, cotangent)
    np.testing.assert_allclose(native_dc, pullback(cotangent)[0], rtol=2e-9, atol=2e-9)
    assert np.isfinite(native_db).all()
