"""The molecular workflow keeps full/LR exchange and CHG as explicit leaves."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("unrestricted", [False, True])
def test_complete_wb97xd_energy_geometry_and_dipole(unrestricted):
    from gradscf import dft, gto
    from gradscf_tools.molecular_response import MolecularEnergy

    atom = "N 0 0 0; H 0 0 1.95" if unrestricted else "O 0 0 0; H 0 1.43 1.11; H 0 -1.43 1.11"
    mol = gto.M(atom=atom, basis="sto-3g", unit="Bohr", cart=True, spin=2 if unrestricted else 0)
    cls = dft.UKS if unrestricted else dft.RKS
    mf = cls(mol, xc="wb97xd", grids_level=0, conv_tol=1e-12,
             conv_tol_density=1e-11, conv_tol_grad=1e-10,
             max_cycle=200, integral_backend="native").run()
    energy = MolecularEnergy(mf, gradient_tolerance=1e-10)
    r = energy.coordinates
    inputs = energy.inputs(r)
    assert inputs["eri"].ndim == 4 and inputs["eri_lr"].shape == inputs["eri"].shape
    assert np.isfinite(float(inputs["dispersion_energy"]))
    calculate = jax.jit(jax.value_and_grad(energy))
    value, gradient = calculate(r)
    np.testing.assert_allclose(value, mf.e_tot, atol=2e-9, rtol=0)
    assert np.isfinite(np.asarray(gradient)).all()
    np.testing.assert_allclose(gradient.sum(axis=0), 0., atol=2e-8, rtol=0)
    direction = np.asarray(r).copy()
    direction -= direction.mean(axis=0)
    direction /= np.linalg.norm(direction)
    for step in (3e-4, 1e-4):
        finite = (calculate(r+step*direction)[0]-calculate(r-step*direction)[0])/(2*step)
        np.testing.assert_allclose(finite, jnp.sum(gradient*direction), atol=3e-6, rtol=2e-5)
    field = jnp.zeros(3)
    mu = jax.jit(energy.dipole)(r, field)
    expected = -jax.jit(jax.grad(energy, argnums=1))(r, field)
    np.testing.assert_allclose(mu, expected, atol=1e-7, rtol=1e-7)
