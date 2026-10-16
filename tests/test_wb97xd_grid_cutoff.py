"""Explicit SCF quadrature cutoff guards generated-XC derivatives at tails.

The six positive-x rows and their negative-x mirrors are the twelve nonfinite
local-Hessian points observed for water/STO-3G/PySCF level-0 quadrature.
This tests a declared density cutoff, not replacement of failed derivatives.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.scf.rks import _xc_energy_and_potential_on_grid
from gradscf.scf.uks import _unrestricted_xc_energy_and_potential_on_grid


TAIL = np.array([
    [1.3451905393905827e-14, 3.9671081864271133e-14, 4.75373638936749e-14, 4.962939110535463e-14],
    [8.648327087690942e-15, 5.0286917783178985e-14, 8.386404372221945e-15, 6.402268916543459e-15],
    [6.336116681679465e-15, 3.684224627225555e-14, -8.728231943113147e-30, -7.78677145442289e-15],
    [8.526587190664028e-14, 3.701361796352363e-13, -1.0826807124294221e-28, -3.070147206756252e-13],
    [8.648327087690925e-15, 5.028691778317889e-14, -8.38640437222193e-15, 6.402268916543447e-15],
    [1.3451905393905805e-14, 3.967108186427107e-14, -4.7537363893674815e-14, 4.962939110535455e-14],
])
TAIL = np.concatenate([TAIL, TAIL*np.array([1., -1., 1., 1.])])


@pytest.mark.parametrize("unrestricted", [False, True], ids=["rks", "uks"])
def test_inactive_wb97xd_tails_have_consistent_energy_potential_and_derivatives(unrestricted):
    rho, gradient = jnp.asarray(TAIL[:, 0]), jnp.asarray(TAIL[:, 1:])
    common = dict(weights=jnp.ones(12), xc_spec="hyb_gga_xc_wb97x_d",
                  density_floor=1e-12, potential_clip=None, xc_kind="GGA")
    if unrestricted:
        ao = jnp.diag(jnp.sqrt(rho))
        dao = jnp.stack([ao, *[jnp.diag(gradient[:, k]/(2*jnp.sqrt(rho))) for k in range(3)]])

        def components(scale):
            return _unrestricted_xc_energy_and_potential_on_grid(ao=ao, ao_deriv1=dao,
                density_a=scale*jnp.eye(12), density_b=scale*jnp.eye(12), **common)
    else:
        def components(scale):
            return _xc_energy_and_potential_on_grid(rho=2*scale*rho, grad=2*scale*gradient,
                                                    tau=jnp.zeros(12), **common)
    for value in components(jnp.asarray(1.)):
        np.testing.assert_array_equal(value, jnp.zeros_like(value))
    energy = lambda scale: components(scale)[0]
    np.testing.assert_array_equal(jax.jit(jax.grad(energy))(jnp.asarray(1.)), 0.)
    derivative = energy
    for _ in range(3):
        previous = derivative
        derivative = lambda scale, previous=previous: jax.jvp(
            previous, (scale,), (jnp.ones_like(scale),))[1]
        actual = jax.jit(derivative)(jnp.asarray(1.))
        assert bool(jnp.isfinite(actual))
        np.testing.assert_array_equal(actual, 0.)
