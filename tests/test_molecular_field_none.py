"""No-field assembly avoids unused position AD; explicit zero fields stay live."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture(scope="module")
def hydrogen_energy():
    from gradscf import gto, scf
    from gradscf_tools.molecular_response import MolecularEnergy

    mol = gto.M(atom="H 0 0 -.7; H 0 0 .7", basis="sto-3g", unit="Bohr", cart=True)
    mf = scf.RHF(mol, integral_backend="native", conv_tol=1e-12,
                 conv_tol_density=1e-11, conv_tol_grad=1e-10).run()
    return MolecularEnergy(mf)


def test_none_field_supports_pure_geometry_cubic_without_dipole(hydrogen_energy):
    energy = hydrogen_energy
    direction = jnp.array([[0., 0., -.5], [0., 0., .5]])
    scalar = lambda q: energy(energy.coordinates+q*direction)
    first = lambda q: jax.jvp(scalar, (q,), (jnp.ones_like(q),))[1]
    second = lambda q: jax.jvp(first, (q,), (jnp.ones_like(q),))[1]
    third = jax.jit(lambda q: jax.jvp(second, (q,), (jnp.ones_like(q),))[1])(jnp.asarray(0.))
    assert np.isfinite(third) and abs(float(third)) > 1e-3
    curvature = jax.jit(second)
    step = 3e-4
    numerical = (curvature(step)-curvature(-step))/(2*step)
    np.testing.assert_allclose(third, numerical, atol=3e-5, rtol=1e-5)


def test_none_and_explicit_zero_field_energy_gradient_hvp_agree(hydrogen_energy):
    energy = hydrogen_energy
    zero = jnp.zeros(3)
    default = lambda r: energy(r)
    explicit = lambda r: energy(r, zero)
    default_value, default_gradient = jax.jit(jax.value_and_grad(default))(energy.coordinates)
    explicit_value, explicit_gradient = jax.jit(jax.value_and_grad(explicit))(energy.coordinates)
    np.testing.assert_allclose(default_value, explicit_value, atol=1e-12, rtol=0)
    np.testing.assert_allclose(default_gradient, explicit_gradient, atol=1e-10, rtol=0)
    direction = jnp.array([[.1, -.2, -.5], [-.1, .2, .5]])
    left = jax.jit(lambda r: jax.jvp(jax.grad(default), (r,), (direction,))[1])(energy.coordinates)
    right = jax.jit(lambda r: jax.jvp(jax.grad(explicit), (r,), (direction,))[1])(energy.coordinates)
    np.testing.assert_allclose(left, right, atol=1e-9, rtol=0)


def test_explicit_zero_field_retains_real_electric_response(hydrogen_energy):
    energy = hydrogen_energy
    coordinates = energy.coordinates
    field_energy = lambda field: energy(coordinates, field)
    alpha = -jax.jit(jax.hessian(field_energy))(jnp.zeros(3))
    density_identity = jax.jit(jax.jacfwd(lambda field: energy.dipole(coordinates, field)))(jnp.zeros(3))
    assert np.isfinite(alpha).all() and float(alpha[2, 2]) > .5
    np.testing.assert_allclose(alpha, density_identity, atol=1e-8, rtol=1e-8)
    field = jnp.array([0., 0., 1e-3])
    dipole = -jax.jit(jax.grad(field_energy))(field)
    assert np.linalg.norm(dipole) > 1e-4
    np.testing.assert_allclose(dipole, energy.dipole(coordinates, field), atol=1e-9, rtol=0)
