"""Projected phonons must retain the full Cartesian force response and metric."""
import numpy as np
import pytest
from scipy.linalg import eigh
from gradscf.pbc import gto, scf
from gradscf.pbc.phonon import Phonons


@pytest.fixture(scope='module')
def reference():
    cell = gto.M(atom='H .4 .5 .6; H 1.8 .8 .9', a=np.eye(3)*6.,
                 unit='Bohr', mesh=(7,)*3)
    mf = scf.RHF(cell, temperature=.15).run(
        conv_tol=1e-12, conv_tol_density=1e-11, conv_tol_grad=1e-10)
    assert mf.converged
    return mf


def test_projection_matches_full_hessian_and_generalized_mass(reference):
    from scipy.constants import physical_constants
    driver = Phonons(reference, masses=[1., 2.])
    full = driver.kernel()
    directions = np.array([[1., .2, 0., -1., 0., 0.],
                           [0., 1., .1, 0., -1., 0.]])
    projected = driver.project(directions.reshape(2, 2, 3))
    phi = directions @ np.asarray(full.force_constants) @ directions.T
    mass_ratio = physical_constants['atomic mass constant'][0]/physical_constants['electron mass'][0]
    metric = (directions*np.repeat([1., 2.], 3)*mass_ratio) @ directions.T
    expected, _ = eigh(phi, metric)
    assert projected.converged
    np.testing.assert_allclose(projected.force_constants, phi, atol=2e-10)
    np.testing.assert_allclose(projected.mass_matrix, metric, atol=1e-10)
    np.testing.assert_allclose(projected.frequencies,
                               np.sign(expected)*np.sqrt(abs(expected)), atol=1e-9)
    np.testing.assert_allclose(projected.modes.conj().T @ projected.modes,
                               np.eye(2), atol=1e-10)
    assert projected.subspace_residual_norm >= 0.
    full_residual = (np.asarray(full.dynamical_matrix) @ projected.modes
                     - projected.modes*(np.sign(projected.frequencies)*projected.frequencies**2))
    np.testing.assert_allclose(projected.subspace_residual_norm,
                               np.linalg.norm(full_residual), atol=1e-10)
    # A nonsingular change of displacement coordinates cannot alter frequencies.
    changed = np.array([[2., .3], [.1, .8]]) @ directions
    other = driver.project(changed.reshape(2, 2, 3))
    np.testing.assert_allclose(other.frequencies, projected.frequencies, atol=1e-9)


def test_complete_projection_recovers_full_modes(reference):
    driver = Phonons(reference, masses=[1., 2.])
    full = driver.kernel()
    projected = driver.project(np.eye(6).reshape(6, 2, 3))
    np.testing.assert_allclose(projected.frequencies, full.frequencies, atol=1e-9)
    assert projected.subspace_residual_norm < 1e-10


def test_integer_direction_input_matches_floating_directions(reference):
    driver = Phonons(reference, masses=[1., 2.])
    direction = np.array([[[1, 0, 0], [-1, 0, 0]]])
    actual = driver.project(direction)
    expected = driver.project(direction.astype(float))
    np.testing.assert_allclose(actual.force_constants, expected.force_constants, atol=1e-12)
    np.testing.assert_allclose(actual.frequencies, expected.frequencies, atol=1e-12)


def test_direction_scale_does_not_change_response_accuracy(reference):
    driver = Phonons(reference, masses=[1., 2.])
    direction = np.array([[[1., .2, 0.], [-1., 0., 0.]]])
    full = driver.project(direction)
    tiny = driver.project(direction*1e-8)
    np.testing.assert_allclose(tiny.frequencies, full.frequencies, atol=1e-10, rtol=1e-8)
    np.testing.assert_allclose(tiny.force_constants/1e-16, full.force_constants, atol=1e-9)


@pytest.mark.parametrize('scale', [1., 1e-6])
def test_nonhermitian_action_is_rejected_independent_of_basis_scale(reference, monkeypatch, scale):
    # Independent bad linear response: scaling the coordinates must not hide it.
    matrix = np.eye(6)
    matrix[0, 1] = .1
    monkeypatch.setattr(Phonons, '_force_constant_action',
        lambda self, q, config, method: lambda direction: (matrix @ direction.ravel(), 0.))
    driver = Phonons(reference, masses=[1., 2.])
    with pytest.raises(RuntimeError, match='Hermiticity'):
        driver.project((scale*np.eye(6)).reshape(6, 2, 3))


@pytest.mark.parametrize('directions', [np.zeros((1, 2, 3)), np.ones((2, 2, 3)),
                                      np.ones((1, 3, 3)), np.full((1, 2, 3), np.nan)])
def test_invalid_subspace_is_rejected(reference, directions):
    with pytest.raises(ValueError, match='directions|independent|finite'):
        Phonons(reference, masses=[1., 2.]).project(directions)


def test_unrepresentable_mass_metric_is_rejected(reference):
    direction = np.eye(6)[:1].reshape(1, 2, 3)*1e-200
    with pytest.raises(ValueError, match='metric|directions'):
        Phonons(reference, masses=[1., 2.]).project(direction)


def test_mutated_masses_are_revalidated(reference):
    driver = Phonons(reference, masses=[1., 2.])
    driver.masses[0] = np.nan
    with pytest.raises(ValueError, match='mass'):
        driver.project(np.eye(6)[:1].reshape(1, 2, 3))


def test_stale_reference_is_rejected(reference):
    from copy import copy
    stale = copy(reference)
    stale.temperature = .2
    with pytest.raises(ValueError, match='temperature'):
        Phonons(stale, masses=[1., 2.]).project(np.eye(6)[:1].reshape(1, 2, 3))


@pytest.mark.parametrize('entry', ['kernel', 'project'])
def test_failed_call_does_not_retain_previous_response(reference, entry):
    driver = Phonons(reference, masses=[1., 2.])
    driver.super_reference = object()
    driver.reference_forces = object()
    with pytest.raises(ValueError):
        if entry == 'kernel':
            driver.kernel(q=[.1, 0., 0.], asr=True)
        else:
            driver.project(np.zeros((1, 2, 3)))
    assert driver.result is None and driver.super_reference is None
    assert driver.reference_forces is None


def test_finite_q_projection_recovers_full_complex_response():
    cell = gto.M(atom='H .4 .5 .6; H 1.8 .8 .9', a=np.eye(3)*6.,
                 unit='Bohr', mesh=(7,)*3)
    mf = scf.KRHF(cell, kpts=cell.make_kpts((3, 1, 1)), temperature=.15).run(
        conv_tol=1e-12, conv_tol_density=1e-11, conv_tol_grad=1e-10)
    assert mf.converged
    driver = Phonons(mf, masses=[1., 2.])
    full = driver.kernel([1/3, 0., 0.], method='primitive')
    projected = driver.project(np.eye(6, dtype=int).reshape(6, 2, 3), [1/3, 0., 0.])
    assert np.max(abs(np.asarray(full.force_constants).imag)) > 1e-7
    np.testing.assert_allclose(projected.frequencies, full.frequencies, atol=1e-9)
    assert projected.subspace_residual_norm < 1e-10
