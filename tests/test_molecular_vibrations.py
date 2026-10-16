"""Mass metrics and rigid-motion diagnostics for molecular normal modes."""
import numpy as np
import pytest


def test_float32_configuration_is_rejected():
    import jax
    from gradscf_tools.vibrations import normal_modes

    enabled = jax.config.x64_enabled
    try:
        jax.config.update("jax_enable_x64", False)
        with pytest.raises(ValueError, match="float64"):
            normal_modes(np.eye(6), [[0., 0., 0.], [0., 0., 1.]], [1., 1.])
    finally:
        jax.config.update("jax_enable_x64", enabled)


def test_diatomic_frequency_and_isotope_scaling():
    from gradscf_tools.vibrations import normal_modes, AMU_TO_ELECTRON_MASS

    coordinates = np.array([[0., 0., -.7], [0., 0., .7]])
    hessian = np.zeros((6, 6))
    hessian[np.ix_([2, 5], [2, 5])] = .4 * np.array([[1., -1.], [-1., 1.]])
    hydrogen = normal_modes(hessian, coordinates, [1., 1.])
    deuterium = normal_modes(hessian, coordinates, [2., 2.])
    assert hydrogen.rigid_rank == 5
    np.testing.assert_allclose(hydrogen.eigenvalues, [.8 / AMU_TO_ELECTRON_MASS])
    np.testing.assert_allclose(deuterium.frequencies_cm1,
                               hydrogen.frequencies_cm1 / np.sqrt(2))
    vectors = hydrogen.displacements.reshape(6, 1)
    np.testing.assert_allclose(vectors.T @ (AMU_TO_ELECTRON_MASS * vectors), [[1.]])
    assert hydrogen.rigid_residual < 1e-14


def test_nonlinear_modes_and_raw_translation_error_are_reported():
    from gradscf_tools.vibrations import normal_modes

    coordinates = np.array([[0., 0., 0.], [0., 1., 1.], [0., -1., 1.]])
    # Deliberately unphysical: projection must retain evidence of the defect.
    result = normal_modes(np.eye(9), coordinates, [16., 1., 1.])
    assert result.rigid_rank == 6
    assert result.displacements.shape == (3, 3, 3)
    assert result.rigid_residual > 1e-5
    assert result.hessian_asymmetry == 0


def test_imaginary_mode_is_not_clipped():
    from gradscf_tools.vibrations import normal_modes

    coordinates = np.array([[0., 0., -.7], [0., 0., .7]])
    hessian = np.zeros((6, 6))
    hessian[np.ix_([2, 5], [2, 5])] = -.4 * np.array([[1., -1.], [-1., 1.]])
    result = normal_modes(hessian, coordinates, [1., 1.])
    assert result.frequencies_cm1[0] < 0


def test_raw_rigid_residual_is_not_symmetrized_away():
    from gradscf_tools.vibrations import normal_modes, AMU_TO_ELECTRON_MASS

    coordinates = np.array([[0., 0., -.7], [0., 0., .7]])
    hessian = np.zeros((6, 6))
    hessian[0, 3], hessian[3, 0] = 1e-8, -1e-8
    result = normal_modes(hessian, coordinates, [1., 1.])
    np.testing.assert_allclose(result.rigid_residual,
                               1e-8 / (np.sqrt(2) * AMU_TO_ELECTRON_MASS))


@pytest.mark.parametrize("defect", ["asymmetric", "mass", "complex"])
def test_invalid_input_rejected(defect):
    from gradscf_tools.vibrations import normal_modes

    hessian = np.eye(6)
    masses = [1., 1.]
    if defect == "asymmetric":
        hessian[0, 1] = .1
    elif defect == "mass":
        masses[0] = 0.
    else:
        hessian = hessian.astype(complex) * (1 + .01j)
    with pytest.raises(ValueError):
        normal_modes(hessian, [[0., 0., 0.], [0., 0., 1.]], masses)
