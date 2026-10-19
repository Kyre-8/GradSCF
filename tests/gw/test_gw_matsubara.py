"""Finite-temperature matrix GW kernels: analytic limits, tails, and AD."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_noninteracting_density_is_exact_on_a_coarse_grid():
    from gradscf.gw.matsubara import matsubara_grid, dyson_green_and_density

    grid = matsubara_grid(nw=8, beta=30.0)
    rotation = jnp.array([[0.8, -0.6], [0.6, 0.8]])
    energy = jnp.array([-0.7, 0.9])
    fock = rotation @ jnp.diag(energy) @ rotation.T
    sigma = jnp.zeros((16, 2, 2), dtype=jnp.complex128)
    green, density = dyson_green_and_density(fock, sigma, 0.1, grid)
    occupation = jax.nn.sigmoid(-grid.beta * (energy - 0.1))
    ref = rotation @ jnp.diag(occupation) @ rotation.T
    np.testing.assert_allclose(density, ref, rtol=0, atol=1e-13)
    np.testing.assert_allclose(green, green[::-1].conj(), rtol=0, atol=1e-13)
    inverse = (1j * grid.fermion[:, None, None] + 0.1) * jnp.eye(2) - fock
    np.testing.assert_allclose(inverse @ green, jnp.broadcast_to(jnp.eye(2), green.shape), rtol=0, atol=1e-13)


def test_degenerate_static_reference_density_has_smooth_matrix_response():
    from gradscf.gw.matsubara import matsubara_grid, dyson_green_and_density

    grid = matsubara_grid(nw=4, beta=4.0)
    sigma = jnp.zeros((8, 3, 3), dtype=jnp.complex128)
    direction = jnp.array([[0.2, 0.1, 0.0], [0.1, -0.2, 0.0], [0.0, 0.0, 0.0]])
    mu = -jnp.log(2.0) / grid.beta
    density = lambda scale: dyson_green_and_density(scale * direction, sigma, mu, grid)[1]
    value, tangent = jax.jvp(density, (jnp.array(0.0),), (jnp.array(1.0),))
    np.testing.assert_allclose(value, np.eye(3) / 3, rtol=0, atol=1e-13)
    expected = -grid.beta * (1 / 3) * (2 / 3) * direction
    np.testing.assert_allclose(tangent, expected, rtol=0, atol=1e-12)


def test_single_level_thermal_zero_mode_and_self_energy_tail():
    from gradscf.gw.matsubara import matsubara_grid, dyson_green_and_density, gw_matsubara_step

    grid = matsubara_grid(nw=8, beta=4.0)
    b = jnp.array([[[0.3]]])
    fock = jnp.zeros((1, 1))
    green, _ = dyson_green_and_density(fock, jnp.zeros((16, 1, 1), dtype=complex), 0.0, grid)
    state = gw_matsubara_step(green, fock, 0.0, b, grid)
    pi0 = -grid.beta * float(b[0, 0, 0])**2 / 2
    pi_ref = np.zeros((17, 1, 1))
    pi_ref[8, 0, 0] = pi0
    np.testing.assert_allclose(state["polarizability_inu"], pi_ref, rtol=0, atol=1e-13)
    w0 = pi0 / (1 - pi0)
    moment = -float(b[0, 0, 0])**2 * w0 / grid.beta
    np.testing.assert_allclose(state["screened_tau"], w0 / grid.beta, rtol=0, atol=1e-13)
    np.testing.assert_allclose(state["sigma_moment"], moment, rtol=0, atol=1e-13)
    np.testing.assert_allclose(state["sigma_iw"][:, 0, 0], moment / (1j * grid.fermion), rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(state["correlation_energy"], float(b[0, 0, 0])**2 * w0 / 4, rtol=1e-12)


def test_reference_bubble_matches_two_level_rpa_on_a_coarse_grid():
    from gradscf.gw.matsubara import matsubara_grid, dyson_green_and_density, gw_matsubara_step

    grid = matsubara_grid(nw=10, beta=40.0)
    fock = jnp.diag(jnp.array([-0.6, 0.6]))
    b = jnp.array([[[0.0, 0.2], [0.2, 0.0]]])
    green, _ = dyson_green_and_density(fock, jnp.zeros((20, 2, 2), dtype=complex), 0.0, grid)
    state = gw_matsubara_step(green, fock, 0.0, b, grid)
    fdiff = jnp.tanh(grid.beta * 0.6 / 2)
    ref = -4 * 0.2**2 * fdiff * 1.2 / (grid.boson**2 + 1.2**2)
    np.testing.assert_allclose(state["polarizability_inu"][:, 0, 0], ref, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(state["sigma_iw"], state["sigma_iw"][::-1].conj(), rtol=0, atol=1e-12)
    # Analytic two-level RPA reference: Ec_GM = -V^4/[4*gap*Omega*(gap+Omega)]
    # at this effectively zero temperature. It must not depend on a coarse
    # midpoint approximation of the Sigma(tau) G(-tau) integral.
    vertex_squared = 4 * 0.2**2 * 1.2 * fdiff
    omega = jnp.sqrt(1.2**2 + vertex_squared)
    ec_ref = -vertex_squared**2 / (4 * 1.2 * omega * (1.2 + omega))
    np.testing.assert_allclose(state["correlation_energy"], ec_ref, rtol=1e-9, atol=1e-13)
    f_virtual = jax.nn.sigmoid(-grid.beta * 0.6)
    n_bose = 1 / jnp.expm1(grid.beta * omega)
    sigma_ref = 0.2**2 * vertex_squared / (2 * omega) * (
        (1 - f_virtual + n_bose) / (1j * grid.fermion - 0.6 - omega)
        + (f_virtual + n_bose) / (1j * grid.fermion - 0.6 + omega)
    )
    np.testing.assert_allclose(state["sigma_iw"][:, 0, 0], sigma_ref, rtol=1e-11, atol=1e-13)


def test_kernel_uses_dressed_green_function_and_has_correct_weak_coupling_order():
    from gradscf.gw.matsubara import matsubara_grid, dyson_green_and_density, gw_matsubara_step

    grid = matsubara_grid(nw=32, beta=30.0)
    fock = jnp.diag(jnp.array([-0.5, 0.8]))
    b = jnp.array([[[0.1, 0.2], [0.2, -0.1]]])
    bare, _ = dyson_green_and_density(fock, jnp.zeros((64, 2, 2), dtype=complex), 0.0, grid)
    first = gw_matsubara_step(bare, fock, 0.0, b, grid)
    dressed, _ = dyson_green_and_density(fock, first["sigma_iw"], 0.0, grid)
    second = gw_matsubara_step(dressed, fock, 0.0, b, grid)
    assert np.max(np.abs(np.asarray(second["polarizability_inu"] - first["polarizability_inu"]))) > 1e-7
    ec1 = gw_matsubara_step(bare, fock, 0.0, 0.02 * b, grid)["correlation_energy"]
    ec2 = gw_matsubara_step(bare, fock, 0.0, 0.04 * b, grid)["correlation_energy"]
    np.testing.assert_allclose(ec2 / ec1, 16.0, rtol=1e-3)


def test_matsubara_kernel_jit_gradient_matches_finite_difference():
    from gradscf.gw.matsubara import matsubara_grid, dyson_green_and_density, gw_matsubara_step

    grid = matsubara_grid(nw=8, beta=4.0)
    fock = jnp.zeros((1, 1))
    green, _ = dyson_green_and_density(fock, jnp.zeros((16, 1, 1), dtype=complex), 0.0, grid)
    loss = lambda scale: gw_matsubara_step(green, fock, 0.0, jnp.ones((1, 1, 1)) * scale, grid)["correlation_energy"]
    step = 1e-5
    fd = (loss(0.3 + step) - loss(0.3 - step)) / (2 * step)
    np.testing.assert_allclose(jax.jit(jax.grad(loss))(0.3), fd, rtol=1e-6, atol=1e-10)


def _matrix_case():
    from gradscf.gw.matsubara import matsubara_grid, dyson_green_and_density, gw_matsubara_step
    rng = np.random.default_rng(614)
    b = rng.normal(size=(2, 3, 3)) * 0.08
    b = jnp.asarray(0.5 * (b + b.transpose(0, 2, 1)))
    fock = jnp.diag(jnp.array([-0.7, 0.2, 1.2]))
    grid = matsubara_grid(nw=24, beta=20.0)
    green, _ = dyson_green_and_density(fock, jnp.zeros((48, 3, 3), dtype=complex), 0.0, grid)
    first = gw_matsubara_step(green, fock, 0.0, b, grid)
    return fock, b, grid, first


@pytest.mark.parametrize("variable", ["vertices", "dressed_green"])
def test_general_matrix_kernel_gradient(variable):
    from gradscf.gw.matsubara import dyson_green_and_density, gw_matsubara_step
    fock, b, grid, first = _matrix_case()
    def loss(scale):
        factor = scale if variable == "dressed_green" else 1.0
        green, _ = dyson_green_and_density(fock, factor * first["sigma_iw"], 0.0, grid)
        vertices = scale * b if variable == "vertices" else b
        return gw_matsubara_step(green, fock, 0.0, vertices, grid, factor * first["sigma_moment"])["correlation_energy"]
    step = 1e-5
    fd = (loss(1.0 + step) - loss(1.0 - step)) / (2 * step)
    np.testing.assert_allclose(jax.jit(jax.grad(loss))(1.0), fd, rtol=2e-5, atol=1e-10)


def test_matrix_kernel_is_covariant_under_basis_rotations():
    from gradscf.gw.matsubara import dyson_green_and_density, gw_matsubara_step
    fock, b, grid, first = _matrix_case()
    green, _ = dyson_green_and_density(fock, first["sigma_iw"], 0.0, grid)
    ref = gw_matsubara_step(green, fock, 0.0, b, grid, first["sigma_moment"])
    rng = np.random.default_rng(311)
    orbital, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    auxiliary, _ = np.linalg.qr(rng.normal(size=(2, 2)))
    rotated_b = jnp.einsum("QP,Pij->Qij", jnp.asarray(auxiliary), orbital.T @ b @ orbital)
    result = gw_matsubara_step(orbital.T @ green @ orbital, orbital.T @ fock @ orbital, 0.0,
                                rotated_b, grid, orbital.T @ first["sigma_moment"] @ orbital)
    np.testing.assert_allclose(result["sigma_iw"], orbital.T @ ref["sigma_iw"] @ orbital, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(result["correlation_energy"], ref["correlation_energy"], rtol=1e-9, atol=1e-12)


def test_distinct_soft_bosonic_modes_are_not_treated_as_equal():
    from gradscf.gw.matsubara import _boson_product_sum
    a, b, beta = 1e-6, 2e-6, 4.0
    # The nu=0 term dominates by more than twenty orders of magnitude.
    expected = 1 / (beta * a**2 * b**2)
    np.testing.assert_allclose(_boson_product_sum(a, b, beta), expected, rtol=1e-12)


def test_soft_two_level_reference_energy_preserves_the_thermal_zero_mode():
    from gradscf.gw.matsubara import matsubara_grid, dyson_green_and_density, gw_matsubara_step
    grid = matsubara_grid(nw=8, beta=4.0)
    fock = jnp.diag(jnp.array([-5e-7, 5e-7]))
    vertices = jnp.array([[[0.0, 1.0], [1.0, 0.0]]])
    green, _ = dyson_green_and_density(fock, jnp.zeros((16, 2, 2), dtype=complex), 0.0, grid)
    state = gw_matsubara_step(green, fock, 0.0, vertices, grid)
    pi0 = state["polarizability_inu"][grid.nw, 0, 0]
    expected = -pi0 * (pi0 / (1 - pi0)) / (2 * grid.beta)
    np.testing.assert_allclose(state["correlation_energy"], expected, rtol=1e-10, atol=1e-12)
