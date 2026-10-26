"""Saved periodic matrix states in equivalent reference MO gauges."""
from types import SimpleNamespace

import numpy as np
import pytest


def _frames():
    energy = np.array([[-.6, -.6, .4], [-.8, -.3, .7]])
    overlap = np.tile(np.diag([2., 3., 4.]), (2, 1, 1))
    coeff = np.tile(np.diag(1/np.sqrt([2., 3., 4.])), (2, 1, 1)).astype(complex)
    unitary = np.array([[[1., 1j, 0.], [1j, 1., 0.], [0., 0., np.sqrt(2.)*1j]],
                        [[np.sqrt(2.), 0., 0.], [0., np.sqrt(2.)*1j, 0.],
                         [0., 0., -np.sqrt(2.)]]])/np.sqrt(2.)
    return SimpleNamespace(energy=energy, coeff=coeff), overlap, coeff@unitary, unitary


def test_alignment_preserves_complex_frequency_matrices_under_phase_and_degenerate_rotation():
    from gradscf.gw.pbc import interpolation
    model, overlap, saved_coeff, unitary = _frames()
    rng = np.random.default_rng(48)
    # A Matsubara self-energy is generally not Hermitian at one frequency.
    expected = rng.normal(size=(2, 4, 3, 3))+1j*rng.normal(size=(2, 4, 3, 3))
    moment = rng.normal(size=(2, 3, 3))+1j*rng.normal(size=(2, 3, 3))
    old = np.einsum('kia,kwij,kjb->kwab', unitary.conj(), expected, unitary)
    old_moment = unitary.conj().swapaxes(-1, -2)@moment@unitary
    actual, actual_moment = interpolation.align_source(model, overlap, saved_coeff,
        model.energy, old, old_moment, tolerance=1e-10)
    np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=0)
    np.testing.assert_allclose(actual_moment, moment, atol=2e-14, rtol=0)
    assert np.max(abs(actual-actual.conj().swapaxes(-1, -2))) > 1.


@pytest.mark.parametrize('invalid', ['nonunitary', 'different_energy', 'matrix_shape', 'nonfinite'])
def test_alignment_rejects_unmatched_reference_frames(invalid):
    from gradscf.gw.pbc import interpolation
    model, overlap, saved_coeff, _ = _frames()
    energy = model.energy.copy()
    matrix = np.tile(np.eye(3), (2, 1, 1)).astype(complex)
    if invalid == 'nonunitary':
        saved_coeff *= 1.1
    elif invalid == 'different_energy':
        energy[0, 0] -= .02
    elif invalid == 'matrix_shape':
        matrix = matrix[:, :2, :2]
    else:
        matrix[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        interpolation.align_source(model, overlap, saved_coeff, energy, matrix)
