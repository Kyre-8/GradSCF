"""Gauge-invariant transition-density matching for geometry examples."""
import numpy as np
import pytest

from gradscf_tools.excited import transition_density_overlaps, select_tracked_root


def test_transition_density_matching_is_invariant_to_orbital_rotations_and_state_sign():
    occupied = np.array([[np.cos(.3), -np.sin(.3)], [np.sin(.3), np.cos(.3)]])
    virtual = np.array([[np.cos(.7), -np.sin(.7)], [np.sin(.7), np.cos(.7)]])
    rotated_coefficients = np.zeros((4, 4))
    rotated_coefficients[:2, :2] = occupied
    rotated_coefficients[2:, 2:] = virtual
    amplitudes = np.array([[[1., 0.], [0., 0.]], [[0., 0.], [0., 1.]]])
    rotated_amplitudes = np.array([occupied.T @ x @ virtual for x in amplitudes])
    rotated_amplitudes[0] *= -1
    scores = transition_density_overlaps(np.eye(4), amplitudes[0],
                                        rotated_coefficients, rotated_amplitudes,
                                        np.eye(4), nocc=2)
    np.testing.assert_allclose(scores, [1., 0.], atol=1e-14, rtol=0)
    match = select_tracked_root(scores, [.2, .4])
    assert match["root"] == 0
    np.testing.assert_allclose(match["overlap"], 1., atol=1e-14)


@pytest.mark.parametrize("scores,energies,message", [
    ([.6, .1], [.2, .4], "overlap"),
    ([.9, .85], [.2, .4], "ambiguous"),
    ([.99, .01], [.2, .200001], "gap"),
])
def test_tracking_rejects_low_overlap_ambiguity_and_near_degeneracy(scores, energies, message):
    with pytest.raises(RuntimeError, match=message):
        select_tracked_root(scores, energies)
