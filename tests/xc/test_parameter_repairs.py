"""Independent regressions for confirmed legacy jax_xc array-reader defects."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.dft.xc import (eval_xc_energy_density_unrestricted_from_density_gradients,
                            load_jax_xc, point_energy)
from test_libxc_reference import _libxc_energy_gradient


@pytest.mark.parametrize("spec", ["m11l", "m05", "m05_2x", "b97m_v"])
def test_confirmed_parameter_reader_repairs_against_libxc(spec):
    y = jnp.asarray([.31, .23, .12, -.04, .03, -.02, .11, .05, .25, .20])

    def energy(fields):
        # Explicit opt-in diagnoses the repaired local graph while the standard
        # recipe remains disabled until this independent comparison passes.
        return eval_xc_energy_density_unrestricted_from_density_gradients(
            spec, fields[0], fields[1], fields[2:5], fields[5:8],
            tau_a=fields[8], tau_b=fields[9], allow_experimental_jax_xc=True,
        )

    actual_energy, actual_gradient = jax.jit(jax.value_and_grad(energy))(y)
    expected_energy, expected_gradient = _libxc_energy_gradient(spec, y, "MGGA")
    np.testing.assert_allclose(actual_energy, expected_energy, rtol=2e-10, atol=2e-12)
    np.testing.assert_allclose(actual_gradient, expected_gradient, rtol=2e-8, atol=2e-10)


@pytest.mark.parametrize("parameter,center", [("_mu", .2195149727645171), ("_kappa", .804)])
def test_current_factory_scalar_parameter_ad_matches_energy_finite_difference(parameter, center):
    module, _ = load_jax_xc()
    origin = jnp.zeros(3)

    def rho_fn(r):
        return jnp.asarray([.31+jnp.dot(jnp.asarray([.12, -.04, .03]), r),
                            .23+jnp.dot(jnp.asarray([-.02, .11, .05]), r)])

    def energy(value):
        functional = module.gga_x_pbe(polarized=True, **{parameter: value})
        return .54*functional(rho_fn, origin)

    actual = jax.jit(jax.grad(energy))(jnp.asarray(center))
    step = 1e-5
    expected = (energy(center+step)-energy(center-step))/(2*step)
    np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=1e-10)


def test_positive_runtime_omega_ad_and_explicit_zero_boundary():
    y = jnp.asarray([.31, .23, .12, -.04, .03, -.02, .11, .05])
    point = point_energy("lc-wpbe", "GGA")
    actual = jax.jit(jax.grad(lambda omega: point(y, omega=omega)))(jnp.asarray(.4))
    step = 1e-5
    expected = (point(y, omega=.4+step)-point(y, omega=.4-step))/(2*step)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-9)
    with pytest.raises(ValueError, match="positive"):
        point(y, omega=0.)
