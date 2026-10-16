"""Slab ion-ion energy, independent PySCF and geometry response."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.integrals.periodic.truncated import slab_energy


@pytest.mark.parametrize('height', [16., 24.])
def test_slab_energy_and_forces_match_pyscf(height):
    from pyscf.pbc import gto
    lattice = np.array([[5., 0., 0.], [1., 6., 0.], [0., 0., height]])
    coords = np.array([[.2, .3, .4], [2., 1., 1.3]])
    charges = jnp.array([1., 2.])
    def reference(positions):
        cell = gto.M(atom=list(zip(['H', 'He'], positions)), a=lattice,
                     unit='Bohr', dimension=2, basis='gth-szv', pseudo='gth-pade',
                     precision=1e-11, spin=1, verbose=0)
        return cell.energy_nuc()
    energy = lambda positions: slab_energy(jnp.asarray(lattice), positions, charges,
                                           precision=1e-11, reference_lattice=lattice)
    np.testing.assert_allclose(jax.jit(energy)(jnp.asarray(coords)), reference(coords), atol=2e-10)
    actual = np.asarray(jax.jit(jax.grad(energy))(jnp.asarray(coords)))
    direction = np.array([[.2, -.3, .1], [-.1, .2, -.15]])
    h = 1e-4
    expected = (reference(coords+h*direction)-reference(coords-h*direction))/(2*h)
    np.testing.assert_allclose(np.sum(actual*direction), expected, atol=2e-9, rtol=1e-7)
    np.testing.assert_allclose(actual.sum(axis=0), 0., atol=2e-12)
    # Only the two periodic axes may be wrapped; translating z is physical.
    shifted = coords.copy(); shifted[0] += lattice[0]-2*lattice[1]
    np.testing.assert_allclose(energy(jnp.asarray(shifted)), energy(jnp.asarray(coords)), atol=2e-11)


def test_slab_energy_second_strain_response_and_rotation():
    lattice = jnp.diag(jnp.array([5., 6., 18.]))
    coords = jnp.array([[.2, .3, .4], [2., 1., 1.3]])
    charges = jnp.array([1., 2.])
    def energy(scale):
        deformation = jnp.diag(jnp.array([scale, scale, 1.]))
        return slab_energy(lattice@deformation, coords@deformation, charges,
                           reference_lattice=np.asarray(lattice))
    derivative = jax.jit(jax.grad(energy))
    second = jax.jit(jax.grad(jax.grad(energy)))(1.)
    h = 1e-4
    np.testing.assert_allclose(second, (derivative(1+h)-derivative(1-h))/(2*h), rtol=2e-7)
    rotation, _ = np.linalg.qr(np.random.default_rng(4).normal(size=(3, 3)))
    actual = slab_energy(lattice@rotation, coords@rotation, charges,
                         reference_lattice=np.asarray(lattice)@rotation)
    np.testing.assert_allclose(actual, energy(1.), atol=2e-12)
