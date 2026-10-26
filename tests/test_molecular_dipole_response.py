"""Stationary dipole identity retains relaxed field and geometry response."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("method", ["HF", "PBE"])
def test_stationary_dipole_and_its_relaxed_derivatives(method):
    from gradscf import dft, gto, scf
    from gradscf_tools.molecular_response import MolecularEnergy

    mol = gto.M(atom="H 0 0 -.7; H 0 0 .7", basis="sto-3g", unit="Bohr", cart=True)
    controls = dict(conv_tol=1e-12, conv_tol_density=1e-11, conv_tol_grad=1e-10,
                    integral_backend="native")
    mf = (scf.RHF(mol, **controls) if method == "HF"
          else dft.RKS(mol, xc="pbe", grids_level=0, **controls)).run()
    energy = MolecularEnergy(mf)
    coordinates, field = energy.coordinates, jnp.array([0., 0., .001])
    dipole = jax.jit(energy.dipole)
    actual = dipole(coordinates, field)
    expected = -jax.jit(jax.grad(energy, argnums=1))(coordinates, field)
    np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=1e-7)
    alpha = jax.jit(jax.jacfwd(energy.dipole, argnums=1))
    expected_alpha = -jax.jit(jax.hessian(energy, argnums=1))(coordinates, field)
    np.testing.assert_allclose(alpha(coordinates, field), expected_alpha, atol=2e-7, rtol=1e-7)
    direction = jnp.array([[0., 0., -.5], [0., 0., .5]])
    component = lambda r: alpha(r, field)[2, 2]
    mixed = jax.jit(lambda r: jax.jvp(component, (r,), (direction,))[1])(coordinates)
    assert np.isfinite(float(mixed)) and abs(float(mixed)) > .1
    for step in (3e-4, 1e-4):
        numerical = (component(coordinates+step*direction)-component(coordinates-step*direction))/(2*step)
        np.testing.assert_allclose(mixed, numerical, atol=3e-6, rtol=1e-5)
    np.testing.assert_allclose(dipole(coordinates+jnp.array([.3, -.2, .4]), field), actual,
                               atol=1e-8, rtol=0)
