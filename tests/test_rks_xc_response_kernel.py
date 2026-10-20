from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from gradscf.dft.xc import point_energy
from gradscf.scf import rks


def test_array_xc_value_and_grad_kernel_matches_pointwise_pbe0_gga():
    variables = jnp.asarray(
        [
            [0.25, 0.01, -0.02, 0.03],
            [0.70, -0.04, 0.02, 0.01],
            [1.20, 0.03, 0.05, -0.02],
        ],
        dtype=jnp.float64,
    )

    # The restricted projection divides both rho and grad between equal spins.
    # Differentiate that map independently of the RKS vectorized adapter.
    spin_energy = point_energy("pbe0", "GGA")

    def restricted_energy(fields):
        spin_fields = jnp.concatenate((fields[:1]/2, fields[:1]/2,
                                       fields[1:]/2, fields[1:]/2))
        return spin_energy(spin_fields)

    point_exc, point_grad = jax.vmap(jax.value_and_grad(restricted_energy))(variables)
    array_exc, array_grad = rks._array_xc_value_and_grad_kernel("pbe0", "GGA", 1e-12)(
        variables
    )

    np.testing.assert_allclose(np.asarray(array_exc), np.asarray(point_exc), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(array_grad), np.asarray(point_grad), rtol=1e-12, atol=1e-12)


def test_array_xc_value_and_grad_kernel_can_run_with_jit_disabled():
    variables = jnp.asarray([[0.70, -0.04, 0.02, 0.01]], dtype=jnp.float64)
    evaluate = rks._array_xc_value_and_grad_kernel("pbe0", "GGA", 1e-12)
    expected = evaluate(variables)
    with jax.disable_jit():
        point_exc, point_grad = evaluate(variables)

    np.testing.assert_allclose(point_exc, expected[0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(point_grad, expected[1], rtol=1e-12, atol=1e-12)
    assert point_exc.shape == (1,)
    assert point_grad.shape == (1, 4)
