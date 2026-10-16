"""Original CHG damping and Grimme 2006 coefficients, in atomic units."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.constants import Avogadro, physical_constants


def _energy(symbols, coordinates):
    from gradscf.dft.dispersion import wb97xd_dispersion_energy
    return wb97xd_dispersion_energy(symbols, coordinates)


@pytest.mark.parametrize("symbols,c6,radii", [
    (("H", "H"), (0.14, 0.14), (1.001, 1.001)),
    (("C", "N"), (1.75, 1.23), (1.452, 1.397)),
    (("O", "F"), (0.70, 0.75), (1.342, 1.287)),
])
def test_original_pair_formula_and_coefficient_units(symbols, c6, radii):
    bohr_m = physical_constants["Bohr radius"][0]
    hartree_j = physical_constants["Hartree energy"][0]
    distance_bohr = 7.3
    distance_nm = distance_bohr * bohr_m / 1e-9
    radius_nm = sum(radii) * 0.1
    # Independent SI/mol oracle: table C6 is J nm^6/mol, radii are Angstrom.
    reference = -np.sqrt(c6[0]*c6[1]) / distance_nm**6
    reference /= 1 + 6*(distance_nm/radius_nm)**-12
    reference /= Avogadro * hartree_j
    coordinates = jnp.asarray([[0., 0., 0.], [distance_bohr, 0., 0.]])
    np.testing.assert_allclose(_energy(symbols, coordinates), reference, rtol=1e-8, atol=1e-14)


def test_geometry_gradient_hvp_and_rigid_motions():
    symbols = ("C", "N", "O", "H", "F")
    coordinates = jnp.asarray([[0., 0., 0.], [2.7, 0.3, -0.1], [0.4, 3.5, 0.2],
                               [-1.4, 0.7, 1.1], [3.2, 2.8, -0.7]])
    direction = jnp.asarray([[0.2, -0.1, 0.3], [-0.1, 0.2, 0.1], [0.3, 0.1, -0.2],
                            [-0.3, 0.2, 0.1], [0.1, -0.2, 0.3]])
    direction /= jnp.linalg.norm(direction)
    energy = jax.jit(lambda r: _energy(symbols, r))
    gradient = jax.jit(jax.grad(energy))
    g = gradient(coordinates)
    hvp = jax.jvp(gradient, (coordinates,), (direction,))[1]
    step = 1e-4
    finite_g = np.zeros_like(coordinates)
    for index in np.ndindex(coordinates.shape):
        shift = jnp.zeros_like(coordinates).at[index].set(step)
        finite_g[index] = (energy(coordinates+shift)-energy(coordinates-shift))/(2*step)
    finite_hvp = (gradient(coordinates+step*direction)-gradient(coordinates-step*direction))/(2*step)
    np.testing.assert_allclose(g, finite_g, rtol=2e-7, atol=2e-11)
    np.testing.assert_allclose(hvp, finite_hvp, rtol=2e-7, atol=2e-11)
    np.testing.assert_allclose(g.sum(axis=0), 0., atol=2e-14)
    np.testing.assert_allclose(jnp.cross(coordinates, g).sum(axis=0), 0., atol=2e-14)
    rotation = jnp.asarray([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    moved = coordinates @ rotation.T + jnp.asarray([0.3, -1.2, 0.7])
    np.testing.assert_allclose(energy(moved), energy(coordinates), rtol=2e-14, atol=2e-15)
    np.testing.assert_allclose(gradient(moved), g @ rotation.T, rtol=2e-13, atol=2e-14)
    print("CHG gradient/HVP FD maxabs:", float(np.max(np.abs(g-finite_g))),
          float(np.max(np.abs(hvp-finite_hvp))))


def test_short_range_limit_and_single_atom_are_finite():
    for symbols, coordinates in [(("H",), jnp.zeros((1, 3))),
                                  (("H", "H"), jnp.zeros((2, 3)))]:
        energy = lambda r: _energy(symbols, r)
        assert float(energy(coordinates)) == 0.0
        np.testing.assert_array_equal(jax.grad(energy)(coordinates), 0.0)
        np.testing.assert_array_equal(jax.hessian(energy)(coordinates), 0.0)


def test_unsupported_element_and_coordinate_shape_fail_explicitly():
    with pytest.raises(ValueError, match="Unsupported.*He"):
        _energy(("H", "He"), jnp.zeros((2, 3)))
    with pytest.raises(ValueError, match="shape"):
        _energy(("H", "H"), jnp.zeros((3, 3)))


@pytest.mark.parametrize("compiled", [False, True])
def test_complex_coordinates_are_rejected(compiled):
    coordinates = jnp.asarray([[0., 0., 0.], [7.3+0.1j, 0., 0.]])
    energy = lambda r: _energy(("H", "H"), r)
    if compiled:
        energy = jax.jit(energy)
    with pytest.raises(TypeError, match="coords_bohr.*real"):
        energy(coordinates)
