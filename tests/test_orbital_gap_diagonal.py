"""Frozen-Fock curvature fixes the private orbital-gap units and pair order."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.scf import orbital_optimization as orbital


@pytest.mark.parametrize('method,occupations,expected_dimension', [
    ('roks', ((1., 1., 0., 0.), (1., 1., 0., 0.)), 4),
    ('uks', ((1., 1., 0., 0.), (1., 0., 0., 0.)), 7),
    ('roks', ((1., 1., 0., 0.), (1., 0., 0., 0.)), 5),
    ('uks', ((0., 1., 0., 1.), (1., 0., 0., 0.)), 7),
    ('gks', ((1., 0., 1., 0.),), 8),
], ids=['rhf', 'uhf', 'common_rohf', 'non_aufbau', 'gks_real_imag'])
def test_gap_diagonal_matches_frozen_fock_energy_hessian(method, occupations, expected_dimension):
    gap_diagonal = getattr(orbital, '_orbital_gap_diagonal', None)
    assert callable(gap_diagonal), 'The shared private orbital-gap helper is missing'
    dimension, rotate, density_from = orbital._rotation_functions(method, occupations)
    assert dimension == expected_dimension
    # A nontrivial AO frame forces an actual C^H F C projection. Equal
    # virtual energies are internal degeneracy, hence introduce no angle.
    frame = jnp.asarray([[1., 1., 1., 1.], [1., -1., 1., -1.],
                         [1., 1., -1., -1.], [1., -1., -1., 1.]])/2
    common = method == 'roks'
    spinors = method == 'gks'
    if spinors:
        frame = frame*jnp.exp(1j*jnp.asarray([.1, .3, -.2, .4]))
    coeff = jnp.stack([frame]*(1 if common or spinors else 2))
    eps = jnp.asarray([[-.9, -.4, .6, .6], [-.7, -.2, .8, .8]])[:len(occupations)]
    fock = jnp.stack([frame @ jnp.diag(row) @ frame.conj().T for row in eps])

    def frozen_energy(angles):
        density = density_from(rotate(angles, coeff))
        return jnp.einsum('spq,sqp->', density, fock).real

    hessian = jax.hessian(frozen_energy)(jnp.zeros(dimension))
    actual = gap_diagonal(method, occupations, coeff, fock)
    assert actual.shape == (dimension,)
    assert not jnp.iscomplexobj(actual)
    np.testing.assert_allclose(actual, jnp.diag(hessian), atol=3e-14, rtol=3e-14)
    np.testing.assert_allclose(hessian, jnp.diag(actual), atol=3e-14, rtol=3e-14)
    if method == 'roks' and occupations[0] == occupations[1]:
        # Restricted full-energy angle Hessian is 4*(eps_a-eps_i) Eh.
        np.testing.assert_allclose(actual, [6., 6., 4., 4.], atol=3e-14, rtol=3e-14)
    if occupations[0][0] == 0.:
        assert jnp.any(actual < 0.), 'Non-Aufbau occupation differences must retain their sign'
    if spinors:
        half = dimension//2
        np.testing.assert_allclose(actual[:half], actual[half:], atol=0., rtol=0.)
        assert abs(actual[half-1]) < 3e-14, 'A physical occupied-virtual zero gap must remain zero'
