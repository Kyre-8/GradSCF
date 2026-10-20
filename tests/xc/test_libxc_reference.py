"""Local e, potential and Cartesian Hessian against independent Libxc 7.x.

Atomic units; polarized rho=(.31,.23), nonparallel gradients, tau=(.25,.20).
Finite differences use analytic Libxc potentials, not JAX energy differences.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.dft.xc import (FunctionalRecipe, LocalXCTermSpec, grid_hessian,
                            grid_value_and_grad, load_jax_xc, parse_recipe)

libxc = pytest.importorskip("pyscf.dft.libxc")

CASES = [
    ("lda", "LDA_X,LDA_C_PW"), ("pbe", "PBE"), ("blyp", "BLYP"),
    ("pbe0", "PBE0"), ("b3lyp", "B3LYP"), ("r2scan", "R2SCAN"),
    ("scan", "SCAN"), ("tpss", "TPSS"), ("lc-wpbe", "LC_WPBE"),
    ("hse03", "HSE03"), ("hse06", "HSE06"),
    ("cam-b3lyp", "CAM_B3LYP"), ("m06-2x", "M06_2X"),
    ("b97", "B97"), ("b97_1", "B97_1"),
    ("wb97", "WB97"), ("wb97x", "WB97X"),
]


def _libxc_energy_gradient(reference, variables, kind, *, omega=None):
    y = np.asarray(variables, dtype=np.float64)
    if kind == "LDA":
        fields = y[:2, None]
    else:
        fields = np.zeros((2, 6 if kind == "MGGA" else 4, 1))
        fields[:, 0, 0] = y[:2]
        fields[0, 1:4, 0], fields[1, 1:4, 0] = y[2:5], y[5:8]
        if kind == "MGGA":
            fields[:, 5, 0] = y[8:10]
    epsilon, vxc, _, _ = libxc.eval_xc(reference, fields, spin=1, deriv=1, omega=omega)
    gradient = np.zeros_like(y)
    gradient[:2] = np.asarray(vxc[0])[0]
    if kind != "LDA":
        aa, ab, bb = np.asarray(vxc[1])[0]
        gradient[2:5] = 2*aa*y[2:5] + ab*y[5:8]
        gradient[5:8] = 2*bb*y[5:8] + ab*y[2:5]
    if kind == "MGGA":
        gradient[8:10] = np.asarray(vxc[3])[0]
    return np.sum(y[:2])*epsilon[0], gradient


@pytest.mark.parametrize("spec,reference", CASES)
def test_local_energy_potential_hessian_against_libxc(spec, reference):
    jax.config.update("jax_enable_x64", True)
    kind = parse_recipe(spec).kind
    size = {"LDA": 2, "GGA": 8, "MGGA": 10}[kind]
    y = np.asarray([.31, .23, .12, -.04, .03, -.02, .11, .05, .25, .20])[:size]
    expected_energy, expected_gradient = _libxc_energy_gradient(reference, y, kind)
    h = 2e-5
    columns = []
    for index in range(size):
        delta = np.eye(size)[index]*h
        plus = _libxc_energy_gradient(reference, y+delta, kind)[1]
        minus = _libxc_energy_gradient(reference, y-delta, kind)[1]
        columns.append((plus-minus)/(2*h))
    expected_hessian = np.stack(columns, axis=1)
    energy, gradient = grid_value_and_grad(spec, kind)(jnp.asarray(y[None, :]))
    hessian = grid_hessian(spec, kind)(jnp.asarray(y[None, :]))
    np.testing.assert_allclose(energy[0], expected_energy, rtol=2e-10, atol=2e-12)
    np.testing.assert_allclose(gradient[0], expected_gradient, rtol=2e-8, atol=2e-10)
    np.testing.assert_allclose(hessian[0], expected_hessian, rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize("name,omega", [("gga_x_wpbeh", 0.), ("gga_x_pbe", None)])
def test_zero_omega_wpbeh_and_pbe_are_independent_reference_channels(name, omega):
    y = np.asarray([.31, .23, .12, -.04, .03, -.02, .11, .05])
    recipe = FunctionalRecipe(
        name+"-probe", (LocalXCTermSpec(name, 1., "fixed" if omega is not None else "none", omega),),
        kind="GGA",
    )
    expected_energy, expected_gradient = _libxc_energy_gradient(name, y, "GGA", omega=omega)
    energy, gradient = grid_value_and_grad(recipe, "GGA")(jnp.asarray(y[None]))
    np.testing.assert_allclose(energy[0], expected_energy, rtol=2e-10, atol=2e-12)
    np.testing.assert_allclose(gradient[0], expected_gradient, rtol=2e-8, atol=2e-10)
    if omega == 0.:
        module, _ = load_jax_xc()
        functional = module.gga_x_wpbeh(polarized=True, _omega=0.)
        parameters = [cell.cell_contents for cell in functional.__closure__
                      if hasattr(cell.cell_contents, "cam_omega")]
        assert len(parameters) == 1
        assert parameters[0].cam_omega == 0.
