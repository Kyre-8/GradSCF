"""One-dimensional ionic electrostatics against real-space lattice sums."""
import jax
import jax.numpy as jnp
import numpy as np

from gradscf.integrals.periodic.truncated import wire_energy


def test_alternating_wire_has_exact_madelung_energy():
    lattice = jnp.diag(jnp.array([5., 12., 14.]))
    coords = jnp.array([[0., 1., 1.], [2.5, 1., 1.]])
    charges = jnp.array([1., -1.])
    value = wire_energy(lattice, coords, charges, reference_lattice=np.asarray(lattice))
    np.testing.assert_allclose(value, -4*np.log(2)/5, atol=2e-11, rtol=0.)
    expanded = lattice.at[1, 1].set(18.).at[2, 2].set(22.)
    np.testing.assert_allclose(wire_energy(expanded, coords, charges,
        reference_lattice=np.asarray(expanded)), value, atol=2e-12)


def test_wire_energy_and_response_match_direct_neutral_sum():
    lattice = jnp.diag(jnp.array([5., 12., 14.]))
    coords = jnp.array([[.2, .3, .4], [1.9, -.3, .9]])
    charges = jnp.array([1., -1.])
    direction = jnp.array([[.2, -.1, .15], [-.1, .1, -.2]])
    def direct(positions):
        images = np.arange(-10000, 10001)[:, None]*np.asarray(lattice)[0]
        delta = positions[:, None, None]-positions[None, :, None]+images
        r2 = np.sum(delta**2, axis=-1)
        kernel = np.where(r2 > 1e-24, 1/np.sqrt(np.where(r2 > 1e-24, r2, 1.)), 0.)
        return .5*np.sum(np.asarray(charges)[:, None, None]*np.asarray(charges)[None, :, None]*kernel)
    energy = lambda t: wire_energy(lattice, coords+t*direction, charges,
                                    reference_lattice=np.asarray(lattice))
    np.testing.assert_allclose(jax.jit(energy)(0.), direct(np.asarray(coords)), atol=2e-9)
    h = 1e-4
    expected = (direct(np.asarray(coords+h*direction))-direct(np.asarray(coords-h*direction)))/(2*h)
    np.testing.assert_allclose(jax.jit(jax.grad(energy))(0.), expected, atol=2e-9)
    second = jax.jit(jax.grad(jax.grad(energy)))(0.)
    np.testing.assert_allclose(second, (jax.grad(energy)(h)-jax.grad(energy)(-h))/(2*h), atol=2e-9)
    rotation, _ = np.linalg.qr(np.random.default_rng(6).normal(size=(3, 3)))
    np.testing.assert_allclose(wire_energy(lattice@rotation, coords@rotation, charges,
        reference_lattice=np.asarray(lattice)@rotation), energy(0.), atol=2e-12)
