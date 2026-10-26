"""Diagonal mode vertices avoid allocating a full mode-pair tensor."""

import jax
import jax.numpy as jnp
import numpy as np

from gradscf.gw.ep_coupling import PhononModel, validate_model, debye_waller


def test_compact_quadratic_equals_full_modes_for_values_and_gradients():
    omega = jnp.array([0.02, 0.07])
    diagonal = jnp.array([[[0.3, 0.1], [0.1, 0.2]], [[-0.2, 0.04], [0.04, 0.15]]])
    g = jnp.zeros((2, 2, 2))
    full = jnp.zeros((2, 2, 2, 2)).at[jnp.arange(2), jnp.arange(2)].set(diagonal)
    a = PhononModel(omega, g, diagonal)
    b = PhononModel(omega, g, full)
    validate_model(a)
    validate_model(b)
    np.testing.assert_allclose(debye_waller(a, 60.0), debye_waller(b, 60.0), atol=1e-14)
    loss = lambda x: debye_waller(PhononModel(omega, g, x), 60.0).sum()
    grad = jax.jit(jax.grad(loss))(diagonal)
    expected = (
        0.5
        * (1 + 2 / np.expm1(60.0 * np.asarray(omega)))[:, None, None]
        * np.ones((2, 2, 2))
    )
    np.testing.assert_allclose(grad, expected, atol=1e-14)


def test_compact_scgw_static_response_matches_full_layout():
    from gradscf.gw import scgw_matsubara_restricted
    from gradscf.scf.autodiff import SCFDifferentiationConfig
    from test_scgw_ep import inputs

    kw = inputs()
    kw["df_factors"] = jnp.zeros((1, 2, 2))
    kw["hcore_matrix"] = jnp.diag(jnp.array([-0.5, 0.7]))
    config = SCFDifferentiationConfig(tolerance=1e-10, max_iter=60, restart=30)

    def observable(t, compact):
        diagonal = t * jnp.array([[[0.006, 0.0], [0.0, -0.004]]])
        model = PhononModel(
            jnp.array([0.2]),
            jnp.zeros((1, 2, 2)),
            diagonal if compact else diagonal[None],
        )
        out = scgw_matsubara_restricted(**kw, phonons=model, differentiation=config)
        return out.chemical_potential + out.fock_mo[0, 0] + out.density_mo[0, 0]

    for derivative in (False, True):
        a = lambda t: observable(t, True)
        b = lambda t: observable(t, False)
        if derivative:
            a, b = jax.grad(a), jax.grad(b)
        np.testing.assert_allclose(jax.jit(a)(1.0), jax.jit(b)(1.0), atol=3e-10)
