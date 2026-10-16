"""Unrestricted molecular adapter: fixed-spin NH triplet geometry response."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("method", ["HF", "PBE"])
def test_unrestricted_geometry_gradient_and_longitudinal_dipole(method):
    from gradscf import dft, gto, scf
    from gradscf_tools.molecular_response import MolecularEnergy

    mol = gto.M(atom="N 0 0 0; H 0 0 1.95", spin=2, basis="sto-3g", unit="Bohr", cart=True)
    controls = dict(conv_tol=1e-12, conv_tol_density=1e-11, conv_tol_grad=1e-10,
                    max_cycle=200, integral_backend="native")
    mf = (scf.UHF(mol, **controls) if method == "HF"
          else dft.UKS(mol, xc="pbe", grids_level=0, **controls)).run()
    energy = MolecularEnergy(mf, gradient_tolerance=1e-10)
    r = energy.coordinates
    value_gradient = jax.jit(jax.value_and_grad(energy))
    value, gradient = value_gradient(r)
    np.testing.assert_allclose(value, mf.e_tot, atol=1e-9, rtol=0)
    np.testing.assert_allclose(gradient.sum(axis=0), 0., atol=1e-8, rtol=0)
    direction = jnp.array([[0., 0., -.5], [0., 0., .5]])
    for step in (3e-4, 1e-4):
        plus = value_gradient(r+step*direction)[0]
        minus = value_gradient(r-step*direction)[0]
        np.testing.assert_allclose((plus-minus)/(2*step), jnp.sum(gradient*direction),
                                   atol=2e-6, rtol=1e-5)
    field = jnp.zeros(3)
    mu = jax.jit(energy.dipole)(r, field)
    field_derivative = -jax.jit(jax.grad(lambda f: energy(r, f)))(field)
    np.testing.assert_allclose(mu, field_derivative, atol=1e-7, rtol=1e-7)
    result = jax.jit(energy.solve)(r)
    assert result.density_matrix.shape[0] == 2
    metric = energy.inputs(r)["overlap"]
    electrons = jnp.einsum("spq,qp->s", result.density_matrix, metric)
    np.testing.assert_allclose(electrons, [5., 3.], atol=1e-10, rtol=0)
