"""Positive-density/zero-gradient domain checks, without non-finite cleanup."""

import jax.numpy as jnp
import json
import numpy as np
import pytest

from gradscf.dft.xc import grid_hessian, grid_value_and_grad, parse_recipe
from test_wpbe_small_gradient import _wpbe_zero_gradient_curvature

libxc = pytest.importorskip("pyscf.dft.libxc")


def test_lc_wpbe_actual_lih_active_grid_point_has_finite_derivatives():
    # LiH/STO-3G level-0 quadrature, restricted PySCF-reference density.
    # rho_total=7.068e-7 bohr^-3, well above the declared 1e-12 grid cutoff.
    y = jnp.asarray([[3.5340213743232704e-7, 3.5340213743232704e-7,
                      2.434638222827676e-7, 3.1408003247697477e-19, 1.3249044679466902e-7,
                      2.434638222827676e-7, 3.1408003247697477e-19, 1.3249044679466902e-7]])
    energy, gradient = grid_value_and_grad("lc-wpbe", "GGA")(y)
    hessian = grid_hessian("lc-wpbe", "GGA")(y)
    assert np.all(np.isfinite(energy))
    assert np.all(np.isfinite(gradient))
    assert np.all(np.isfinite(hessian))


def _domain_fields(kind):
    points = []
    for total in np.logspace(-8, 2, 8):
        for fraction in (.5, .7):
            for nonzero_gradient in (False, True):
                ra, rb = fraction*total, (1-fraction)*total
                ga = ra*np.asarray([.7, -.2, .1]) if nonzero_gradient else np.zeros(3)
                gb = rb*np.asarray([-.1, .6, .3]) if nonzero_gradient else np.zeros(3)
                point = [ra, rb, *ga, *gb]
                if kind == "MGGA":
                    # tau_sigma >= tau_W,sigma, with the convention tau=1/2|grad psi|².
                    point += [5*ra**(5/3)+np.dot(ga, ga)/(8*ra),
                              5*rb**(5/3)+np.dot(gb, gb)/(8*rb)]
                points.append(point)
    return np.asarray(points, dtype=np.float64)


def _reference_values(reference, variables, kind):
    npoint = len(variables)
    fields = np.zeros((2, 6 if kind == "MGGA" else 4, npoint))
    fields[:, 0] = variables[:, :2].T + 1e-15
    offset = np.asarray([1e-15, 0., 0.])
    ga, gb = variables[:, 2:5]+offset, variables[:, 5:8]+offset
    fields[0, 1:4], fields[1, 1:4] = ga.T, gb.T
    if kind == "MGGA":
        fields[:, 5] = variables[:, 8:10].T
    epsilon, vxc, _, _ = libxc.eval_xc(reference, fields, spin=1, deriv=1)
    gradient = np.zeros_like(variables)
    gradient[:, :2] = vxc[0]
    aa, ab, bb = np.asarray(vxc[1]).T
    gradient[:, 2:5] = 2*aa[:, None]*ga + ab[:, None]*gb
    gradient[:, 5:8] = 2*bb[:, None]*gb + ab[:, None]*ga
    if kind == "MGGA":
        gradient[:, 8:10] = vxc[3]
    return fields[:, 0].sum(axis=0)*epsilon, gradient


def _zero_gradient_energy_fd(reference, variables):
    """Independent energy differences at a predeclared zero-gradient subset.

    HSE's unsimplified Libxc analytic VXC loses cancellation at sigma~1e-30.
    Steps scale as rho^(4/3), so all steps probe a small reduced gradient at
    every density. This is selected by the input domain, never by its error.
    """
    center = _reference_values(reference, variables, "GGA")[0]
    scale = variables[:, :2].sum(axis=1)**(4/3)
    gradients, diagonals, all_steps = [], [], []
    for fraction in (1e-3, 2e-3, 3e-3):
        steps = fraction*scale
        gradient, diagonal = np.zeros((len(variables), 6)), np.zeros((len(variables), 6))
        for component in range(6):
            plus, minus = variables.copy(), variables.copy()
            plus[:, component+2] += steps
            minus[:, component+2] -= steps
            ep = _reference_values(reference, plus, "GGA")[0]
            em = _reference_values(reference, minus, "GGA")[0]
            gradient[:, component] = (ep-em)/(2*steps)
            diagonal[:, component] = (ep-2*center+em)/steps**2
        gradients.append(gradient)
        diagonals.append(diagonal)
        all_steps.append(steps)
    return np.asarray(gradients), np.asarray(diagonals), np.asarray(all_steps)


def _hse_zero_gradient_hessian(reference, variables):
    """Maple WPBE s² coefficient plus the regular PBE-C sigma potential."""
    fields = np.zeros((2, 4, len(variables)))
    fields[:, 0] = variables[:, :2].T+1e-15
    _, vxc, _, _ = libxc.eval_xc("GGA_C_PBE", fields, spin=1, deriv=1)
    correlation = 2*np.asarray(vxc[1])[:, [0, 2]]
    omega = .18898815748423098 if reference == "HSE03" else .11
    exchange = (_wpbe_zero_gradient_curvature(fields[:, 0].T, 0.)
                -.25*_wpbe_zero_gradient_curvature(fields[:, 0].T, omega))
    return np.repeat(exchange+correlation, 3, axis=1)


@pytest.mark.parametrize("spec,reference", [
    ("lc-wpbe", "LC_WPBE"), ("hse03", "HSE03"), ("hse06", "HSE06"),
    ("scan", "SCAN"), ("r2scan", "R2SCAN"), ("tpss", "TPSS"),
])
def test_broad_positive_density_domain_energy_potential_and_hessian(spec, reference, record_property):
    kind = parse_recipe(spec).kind
    variables = _domain_fields(kind)
    energy, gradient = grid_value_and_grad(spec, kind)(jnp.asarray(variables))
    hessian = grid_hessian(spec, kind)(jnp.asarray(variables))
    assert np.all(np.isfinite(energy))
    assert np.all(np.isfinite(gradient))
    assert np.all(np.isfinite(hessian))
    expected_energy, expected_gradient = _reference_values(reference, variables, kind)
    zero = np.all(variables[:, 2:8] == 0., axis=1)
    if reference in {"HSE03", "HSE06"}:
        # The analytic oracle remains unchanged for every nonzero-gradient
        # point; only the predefined zero-gradient limit uses energy FD.
        fd_gradient, fd_diagonal, steps = _zero_gradient_energy_fd(reference, variables[zero])
        actual_diagonal = np.diagonal(np.asarray(hessian), axis1=1, axis2=2)[zero, 2:8]
        analytic_diagonal = _hse_zero_gradient_hessian(reference, variables[zero])
        # Keep the predefined ordinary Richardson pairs as diagnostics.
        # WPBE term1 contains s⁴ log(s), hence energy-FD curvature includes
        # h² log(h); eliminating h² alone does not give an O(h⁴) oracle.
        # Native double-precision E1/log cancellation further affects these
        # differences. The independently verified analytic B is the oracle
        # for every point in this predefined zero-gradient subset.
        richardson12 = (4*fd_diagonal[0]-fd_diagonal[1])/3
        richardson23 = (9*fd_diagonal[1]-4*fd_diagonal[2])/5
        record_property("zero_gradient_oracle", json.dumps({
            "density": variables[zero, :2].sum(axis=1).tolist(), "steps": steps.tolist(),
            "libxc_analytic_gradient": expected_gradient[zero, 2:8].tolist(),
            "libxc_energy_fd_gradient": fd_gradient.tolist(),
            "jax_gradient": np.asarray(gradient)[zero, 2:8].tolist(),
            "libxc_energy_fd_hessian_diagonal": fd_diagonal.tolist(),
            "richardson_hessian_diagonal": [richardson12.tolist(), richardson23.tolist()],
            "maple_exchange_plus_libxc_pbec_hessian_diagonal": analytic_diagonal.tolist(),
            "jax_hessian_diagonal": actual_diagonal.tolist(),
        }))
        np.testing.assert_allclose(fd_gradient[0], fd_gradient[1], rtol=2e-5, atol=2e-8)
        expected_gradient[zero, 2:8] = fd_gradient[0]
        np.testing.assert_allclose(actual_diagonal, analytic_diagonal, rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(energy, expected_energy, rtol=3e-8, atol=2e-12)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=2e-5, atol=2e-8)
