"""Independent small-gradient curvature of the separate HSE components."""

import json

import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.dft.xc import FunctionalRecipe, LocalXCTermSpec, grid_hessian
from test_libxc_reference import _libxc_energy_gradient


def _wpbe_small_gradient_coefficient(w):
    """Differentiate terms 2--5 of Libxc 6.0.0's Maple at s²=0.

    term1 starts at s⁴. This factored coefficient of s² avoids subtraction
    at large screening and is independent of the generated kernel and VXC.
    https://gitlab.com/libxc/libxc/-/raw/6.0.0/maple/gga_exc/gga_x_wpbeh.mpl
    """
    d, c, f0, g0 = .57786348, -.077215461, .47965830, -.02628417880
    root = np.sqrt(d+w*w)
    ratio = w/root
    delta = d/(root*(root+w))
    return (-2*c*f0/(9*d*d)*delta**2*(ratio+2)
            -g0/(9*d**3)*delta**3*(3*ratio**2+9*ratio+8))


def _wpbe_zero_gradient_curvature(density, omega):
    w = omega/(np.cbrt(6*np.pi**2)*np.cbrt(density))
    coefficient = _wpbe_small_gradient_coefficient(w)
    cx, k = .75*np.cbrt(6/np.pi), 2*np.cbrt(6*np.pi**2)
    return -2*cx*coefficient/(k*k*density**(4/3))


@pytest.mark.parametrize("name,omega", [
    ("gga_x_wpbeh", 0.), ("gga_x_wpbeh", .11),
    ("gga_x_wpbeh", .18898815748423098), ("gga_c_pbe", None),
])
def test_hse_component_zero_gradient_curvature_matches_energy_fd(name, omega, record_property):
    total = 7.196856730011529
    fields = np.asarray([total/2, total/2, 0., 0., 0., 0., 0., 0.])
    recipe = FunctionalRecipe(name, (LocalXCTermSpec(name, 1., "fixed" if omega is not None else "none", omega),), kind="GGA")
    hessian = np.asarray(grid_hessian(recipe, "GGA")(jnp.asarray(fields[None])))[0]
    center = _libxc_energy_gradient(name, fields, "GGA", omega=omega)[0]
    steps, differences = [], []
    for fraction in (1e-3, 3e-3, 1e-2):
        step = total**(4/3)*fraction
        plus, minus = fields.copy(), fields.copy()
        plus[3], minus[3] = step, -step
        ep = _libxc_energy_gradient(name, plus, "GGA", omega=omega)[0]
        em = _libxc_energy_gradient(name, minus, "GGA", omega=omega)[0]
        steps.append(step)
        differences.append((ep-2*center+em)/step**2)
    record_property("component_curvature", json.dumps({"name": name, "omega": omega,
                    "density": total, "steps": steps, "libxc_energy_fd": differences,
                    "jax_hessian_gy_gy": float(hessian[3, 3])}))
    if name == "gga_x_wpbeh":
        analytic = _wpbe_zero_gradient_curvature(total/2, omega)
        record_property("maple_analytic_hessian_gy_gy", float(analytic))
        np.testing.assert_allclose(hessian[3, 3], analytic, rtol=3e-8, atol=3e-12)
    np.testing.assert_allclose(hessian[3, 3], differences[1], rtol=3e-5, atol=3e-6)
