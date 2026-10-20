"""Canonical reciprocal-cell boundaries must tolerate roundoff from skew cells."""
import numpy as np

from gradscf.pbc.momentum import momentum_transfer_table, wrap_to_kmesh


def test_half_mesh_roundoff_uses_the_negative_boundary_and_is_idempotent():
    edges = np.array([np.nextafter(-.5, -np.inf), -.5,
                      np.nextafter(.5, -np.inf), .5])
    points = np.column_stack((edges, np.zeros((4, 2))))
    expected = np.column_stack((np.full(4, -.5), np.zeros((4, 2))))
    wrapped = wrap_to_kmesh(points)
    np.testing.assert_array_equal(wrapped, expected)
    np.testing.assert_array_equal(wrap_to_kmesh(wrapped), expected)


def test_source_mesh_roundoff_retains_its_momentum_partners():
    # A skew reciprocal inverse produced this last-row boundary value in the
    # saved graphene reference. It is the canonical 2x2x1 source mesh.
    points = np.array([[0., 0., 0.], [0., -.5, 0.],
                       [-.5, -1.25e-17, 0.], [-.5, -.5000000000000001, 0.]])
    canonical = wrap_to_kmesh(points)
    np.testing.assert_allclose(canonical, points, atol=1e-12, rtol=0)
    exact = np.array([[0., 0., 0.], [0., -.5, 0.], [-.5, 0., 0.], [-.5, -.5, 0.]])
    np.testing.assert_array_equal(momentum_transfer_table(canonical),
                                  momentum_transfer_table(exact))
