"""Higher derivatives at active water/STO-3G level-0 quadrature points."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.scf.uks import _point_unrestricted_xc_energy


# Spin density and its Cartesian gradient from converged PySCF water. These
# total densities exceed the SCF default cutoff of 1e-12 electron/bohr^3.
ACTIVE_TAIL = np.array([
    [8.636919859349086e-12, 2.5471182671807996e-11, 3.052180093400116e-11, -1.9077358434937415e-11],
    [9.175368832242917e-13, 4.133349692481134e-12, 1.8082467708918142e-12, -2.0266692971380378e-12],
])


@pytest.mark.parametrize("point", ACTIVE_TAIL)
def test_active_wb97xd_tail_third_derivative_is_finite_and_matches_fd(point):
    pytest.importorskip("jax_xc")
    x = jnp.asarray(np.r_[point[:1], point[:1], point[1:], point[1:]])
    direction = x*jnp.asarray([.3, -.2, -.4, .1, .2, .2, -.3, .4])
    energy = _point_unrestricted_xc_energy("hyb_gga_xc_wb97x_d", "GGA")
    line = lambda t: energy(x+t*direction)
    second = jax.grad(jax.grad(line))
    third = jax.jvp(second, (jnp.asarray(0.),), (jnp.asarray(1.),))[1]
    assert bool(jnp.isfinite(third))
    reverse_third = jax.grad(second)(jnp.asarray(0.))
    np.testing.assert_allclose(reverse_third, third, rtol=2e-11, atol=1e-28)
    for step in (2e-3, 1e-3):
        fd = (second(step)-second(-step))/(2*step)
        np.testing.assert_allclose(third, fd, rtol=2e-5, atol=1e-27)


@pytest.mark.parametrize("density", [1e-2, 1e-5, 1e-8, 1e-11])
def test_factored_local_matches_libxc_across_sr_attenuation_branches(density):
    from gradscf.dft.libxc_jax.jax_xc_adapter import (
        eval_jax_xc_energy_density_from_unrestricted_density_gradients,
    )
    libxc = pytest.importorskip("pyscf.dft.libxc")
    point = density*jnp.asarray([[.6, .9, -.2, .4], [.4, -.3, .7, .2]])
    # The existing unrestricted adapter uses these one-sided offsets.
    effective = np.asarray(point).copy()
    effective[:, :2] += 1e-15

    def energy(x):
        return eval_jax_xc_energy_density_from_unrestricted_density_gradients(
            "hyb_gga_xc_wb97x_d", x[0, 0], x[1, 0], x[0, 1:], x[1, 1:])

    value, potential = jax.value_and_grad(energy)(point)
    kernel = jax.hessian(energy)(point)
    expected = [libxc.eval_xc_eff("WB97X_D", effective[..., None], deriv=i, spin=1)
                for i in range(3)]
    np.testing.assert_allclose(value, effective[:, 0].sum()*expected[0][0],
                               rtol=2e-10, atol=1e-25)
    np.testing.assert_allclose(potential, expected[1][..., 0], rtol=2e-8, atol=1e-14)
    if density == 1e-11:
        # LibXC 7's expanded double-precision kernel loses 6.68e-6 in this
        # small element, although its largest element is about 5e7. The
        # original expanded jax_xc pol expression evaluated/differentiated
        # independently with mpmath at 80 decimal digits gives this value;
        # comparisons/wb97xd_tail_high_precision.py records source hash and
        # corrected B97 coefficients for its reproduction.
        np.testing.assert_allclose(kernel[0, 1, 0, 1], .279048653119796962875595,
                                   rtol=2e-11, atol=1e-13)
        reference_kernel = expected[2][..., 0]
        scale = np.max(np.abs(reference_kernel))
        np.testing.assert_allclose(kernel/scale, reference_kernel/scale,
                                   rtol=2e-7, atol=2e-12)
    else:
        np.testing.assert_allclose(kernel, expected[2][..., 0], rtol=2e-7, atol=1e-9)


@pytest.mark.parametrize("minority", [1e-8, 0.])
def test_factored_local_high_spin_and_absent_spin_are_finite(minority):
    from gradscf.dft.libxc_jax.jax_xc_adapter import (
        eval_jax_xc_energy_density_from_unrestricted_density_gradients,
    )
    libxc = pytest.importorskip("pyscf.dft.libxc")
    point = jnp.asarray([[.3, .04, -.02, .01], [minority, 0., 0., 0.]])
    effective = np.asarray(point).copy()
    effective[:, :2] += 1e-15
    energy = lambda x: eval_jax_xc_energy_density_from_unrestricted_density_gradients(
        "hyb_gga_xc_wb97x_d", x[0, 0], x[1, 0], x[0, 1:], x[1, 1:])
    value, gradient = jax.value_and_grad(energy)(point)
    kernel = jax.hessian(energy)(point)
    expected = libxc.eval_xc_eff("WB97X_D", effective[..., None], deriv=0, spin=1)
    np.testing.assert_allclose(value, effective[:, 0].sum()*expected[0], rtol=2e-11, atol=2e-13)
    assert np.isfinite(gradient).all()
    assert np.isfinite(kernel).all()


@pytest.mark.parametrize("side", [-1, 1])
def test_sr_attenuation_derivative_near_existing_branch_matches_fd(side):
    from gradscf.dft.libxc_jax.wb97_local import _sr_attenuation

    point = jnp.asarray(1.35 + side*1e-4)
    derivative = jax.grad(_sr_attenuation)
    second = jax.grad(derivative)
    step = 1e-5
    np.testing.assert_allclose(derivative(point),
        (_sr_attenuation(point+step)-_sr_attenuation(point-step))/(2*step),
        rtol=2e-6, atol=2e-10)
    np.testing.assert_allclose(second(point),
        (derivative(point+step)-derivative(point-step))/(2*step),
        rtol=2e-6, atol=2e-10)
    # The eight-term LibXC asymptotic branch is an approximation, so require
    # its existing small jump, not artificial C-infinity matching at 1.35.
    np.testing.assert_allclose(_sr_attenuation(1.35-1e-9), _sr_attenuation(1.35+1e-9),
                               rtol=0, atol=1e-9)


def test_sr_attenuation_exact_boundary_uses_selected_series_derivative():
    from gradscf.dft.libxc_jax.wb97_local import _sr_attenuation

    point, step = jnp.asarray(1.35), 1e-5
    derivative = jax.grad(_sr_attenuation)(point)
    right_difference = (-3*_sr_attenuation(point)+4*_sr_attenuation(point+step)
                        - _sr_attenuation(point+2*step))/(2*step)
    assert abs(float(derivative)) > 1e-4
    np.testing.assert_allclose(derivative, right_difference, rtol=2e-7, atol=2e-10)
