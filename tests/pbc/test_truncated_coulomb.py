"""Independent finite-wavevector and strain checks for low-D electrostatics."""
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import dblquad
from scipy.special import k0

from gradscf.integrals.periodic.truncated import slab_kernel, wire_kernel


def test_slab_matches_independent_pyscf_and_rotates():
    from pyscf.pbc import gto, tools
    lattice = np.array([[5., 0., 0.], [1., 6., 0.], [0., 0., 18.]])
    cell = gto.M(atom='He 0 0 0', a=lattice, unit='Bohr', dimension=2,
                 basis='gth-szv', pseudo='gth-pade', mesh=(7, 7, 9), verbose=0)
    vectors = cell.get_Gv()
    for shift in (np.zeros(3), np.array([.13, -.07, 0.])):
        expected = tools.get_coulG(cell, k=shift, Gv=vectors, wrap_around=False)
        actual = slab_kernel(jnp.asarray(vectors+shift), jnp.asarray(lattice))
        np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-11)
    rotation, _ = np.linalg.qr(np.random.default_rng(7).normal(size=(3, 3)))
    np.testing.assert_allclose(slab_kernel(jnp.asarray(vectors@rotation),
        jnp.asarray(lattice@rotation)), slab_kernel(jnp.asarray(vectors),
        jnp.asarray(lattice)), atol=2e-10, rtol=2e-13)


def _wire_reference(lengths, indices, wave):
    """Direct real-space Wigner-Seitz integral, independent of Ewald splitting."""
    ly, lz = lengths
    gy, gz = 2*np.pi*np.array(indices)/np.array(lengths)
    def integrand(z, y):
        radius = np.hypot(y, z)
        value = 2*k0(abs(wave)*radius) if wave else -2*np.log(radius)
        return 4*value*np.cos(gy*y)*np.cos(gz*z)
    value, error = dblquad(integrand, 0., ly/2, lambda _:0., lambda _:lz/2,
                          epsabs=2e-7, epsrel=2e-9)
    assert error < 2e-6
    return value


@pytest.mark.parametrize('wave', [0., .19, .0001])
def test_wire_matches_direct_bessel_integral(wave):
    lattice = jnp.diag(jnp.array([5., 12., 14.]))
    actual = np.asarray(wire_kernel(lattice, (3, 5, 5), wave=wave))
    for iy, iz in [(0, 0), (1, 0), (0, 1), (1, 1), (2, 1)]:
        expected = _wire_reference((12., 14.), (iy, iz), wave)
        np.testing.assert_allclose(actual[0, iy, iz], expected, atol=2e-6, rtol=2e-8)


def test_cutoff_strain_and_finite_wavevector_derivatives():
    lattice = jnp.diag(jnp.array([5., 12., 14.]))
    indices = jnp.array(list(itertools.product(range(-2, 3), repeat=3)), dtype=float)
    def loss(scale):
        a = lattice*jnp.array([1., scale, scale])[:, None]
        vectors = indices@(2*jnp.pi*jnp.linalg.inv(a).T)
        # Slab normal is the last row; strain preserves orthogonality.
        slab = slab_kernel(vectors, a)
        wire = wire_kernel(a, (3, 5, 5), wave=.19)
        return jnp.sum(slab**2)*1e-5+jnp.sum(wire**2)*1e-5
    first = jax.jit(jax.grad(loss))(1.)
    second = jax.jit(jax.grad(jax.grad(loss)))(1.)
    for h in [1e-4, 3e-5]:
        np.testing.assert_allclose(first, (loss(1+h)-loss(1-h))/(2*h), rtol=2e-7)
        np.testing.assert_allclose(second, (jax.grad(loss)(1+h)-jax.grad(loss)(1-h))/(2*h), rtol=2e-7)
    energy = lambda q: wire_kernel(lattice, (3, 5, 5), wave=q)[0, 1, 1]
    h = 1e-5
    np.testing.assert_allclose(jax.grad(energy)(.19), (energy(.19+h)-energy(.19-h))/(2*h), rtol=2e-7)


def test_wire_quadrature_refinement_and_shape_errors():
    lattice = jnp.diag(jnp.array([5., 12., 14.]))
    fine = wire_kernel(lattice, (3, 9, 11), wave=.13, quadrature=128)
    coarse = wire_kernel(lattice, (3, 9, 11), wave=.13, quadrature=96)
    np.testing.assert_allclose(coarse, fine, atol=1e-8, rtol=1e-9)
    with pytest.raises(ValueError, match='odd'):
        wire_kernel(lattice, (3, 4, 5))
