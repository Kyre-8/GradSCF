"""Complete molecular electric/geometry response, including the nuclear field."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture(scope="module")
def molecule():
    from gradscf import gto, scf
    from gradscf_tools.molecular_response import MolecularEnergy

    mol = gto.M(atom="H 0 0 -.7; H 0 0 .7", basis="sto-3g", unit="Bohr")
    mf = scf.RHF(mol, conv_tol=1e-12, conv_tol_density=1e-11,
                 integral_backend="native").run()
    return mf, MolecularEnergy(mf)


def test_dipole_polarizability_and_dfpt_agree(molecule):
    from gradscf import dfpt

    mf, energy = molecule
    coordinates = energy.coordinates
    zero = jnp.zeros(3)
    dipole = jax.jit(jax.grad(energy, argnums=1))
    alpha = -jax.jit(jax.hessian(energy, argnums=1))(coordinates, zero)
    np.testing.assert_allclose(dipole(coordinates, zero), 0., atol=1e-9)
    reference = mf.to_reference()
    z = reference.dipole_integrals[2]
    response = dfpt.DFPT(mf).response(potential=z)
    assert response.converged
    np.testing.assert_allclose(alpha[2, 2], -jnp.sum(response.density*z.T), atol=1e-7)
    step = 1e-4
    fd = -(dipole(coordinates, zero.at[2].set(step))
           - dipole(coordinates, zero.at[2].set(-step))) / (2*step)
    np.testing.assert_allclose(alpha[:, 2], fd, atol=1e-6)
    # Neutral molecule: shifting origin cannot create a dipole or field force.
    field = jnp.array([.001, -.002, .003])
    shifted = coordinates + jnp.array([.2, -.1, .3])
    np.testing.assert_allclose(energy(shifted, field), energy(coordinates, field), atol=1e-10)


def test_mixed_raman_derivative_keeps_reference_response(molecule):
    _, energy = molecule
    coordinates, field = energy.coordinates, jnp.zeros(3)
    direction = jnp.array([[0., 0., -.5], [0., 0., .5]])
    alpha = lambda r: -jax.hessian(energy, argnums=1)(r, field)[2, 2]
    mixed = jax.jit(lambda r: jax.jvp(alpha, (r,), (direction,))[1])(coordinates)
    calculate = jax.jit(alpha)
    step = 1e-4
    fd = (calculate(coordinates+step*direction)-calculate(coordinates-step*direction))/(2*step)
    assert abs(float(mixed)) > .1
    np.testing.assert_allclose(mixed, fd, atol=3e-6, rtol=1e-5)


def test_pbe_moving_grid_hvp_matches_gradient_differences():
    from gradscf import dft, gto
    from gradscf_tools.molecular_response import MolecularEnergy

    mol = gto.M(atom="H 0 0 -.7; H 0 0 .7", basis="sto-3g", unit="Bohr", cart=True)
    mf = dft.RKS(mol, xc="pbe", grids_level=0, conv_tol=1e-12,
                 conv_tol_density=1e-11, conv_tol_grad=1e-10,
                 integral_backend="native").run()
    energy = MolecularEnergy(mf)
    coordinates = energy.coordinates
    direction = jnp.array([[0., 0., -.5], [0., 0., .5]])
    gradient = jax.jit(jax.grad(energy))
    product = jax.jit(lambda r: jax.jvp(jax.grad(energy), (r,), (direction,))[1])(coordinates)
    np.testing.assert_allclose(energy(coordinates), mf.e_tot, atol=1e-9, rtol=0)
    assert np.isfinite(np.asarray(product)).all()
    assert np.max(np.abs(np.asarray(product))) > .1
    for step in (3e-4, 1e-4):
        difference = (gradient(coordinates + step*direction)
                      - gradient(coordinates - step*direction)) / (2*step)
        np.testing.assert_allclose(product, difference, atol=2e-6, rtol=0)
    np.testing.assert_allclose(product.sum(axis=0), 0., atol=1e-8, rtol=0)


def test_pbe_mixed_raman_derivative_keeps_reference_response():
    from gradscf import dft, gto
    from gradscf_tools.molecular_response import MolecularEnergy

    mol = gto.M(atom="H 0 0 -.7; H 0 0 .7", basis="sto-3g", unit="Bohr", cart=True)
    mf = dft.RKS(mol, xc="pbe", grids_level=0, conv_tol=1e-12,
                 conv_tol_density=1e-11, conv_tol_grad=1e-10,
                 integral_backend="native").run()
    energy = MolecularEnergy(mf)
    coordinates, field = energy.coordinates, jnp.zeros(3)
    direction = jnp.array([[0., 0., -.5], [0., 0., .5]])
    alpha = lambda r: -jax.hessian(energy, argnums=1)(r, field)[2, 2]
    mixed = jax.jit(lambda r: jax.jvp(alpha, (r,), (direction,))[1])(coordinates)
    calculate = jax.jit(alpha)
    assert np.isfinite(float(mixed)) and abs(float(mixed)) > .1
    for step in (3e-4, 1e-4):
        fd = (calculate(coordinates+step*direction)-calculate(coordinates-step*direction))/(2*step)
        np.testing.assert_allclose(mixed, fd, atol=3e-6, rtol=1e-5)
