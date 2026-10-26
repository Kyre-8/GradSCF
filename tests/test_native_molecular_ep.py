"""Native RHF stationary gradients (Ha/bohr) and orbital-frame transport."""
import numpy as np
import pytest


def test_native_h2_stationary_gradient_and_rigid_motion():
    from gradscf_tools.molecular_ep import NativeRHF

    model = NativeRHF("H .1 -.2 .3; H .4 .2 1.8", unit="Bohr")
    r = model.coordinates
    result, gradient = model.evaluate(r)
    assert bool(result.converged)
    step = 2e-4
    direction = np.array([[.3, -.2, .1], [-.1, .4, -.5]])
    plus, _ = model.evaluate(r + step * direction, gradient=False)
    minus, _ = model.evaluate(r - step * direction, gradient=False)
    fd = (float(plus.total_energy) - float(minus.total_energy)) / (2 * step)
    np.testing.assert_allclose(np.sum(gradient * direction), fd, atol=2e-8)
    np.testing.assert_allclose(gradient.sum(axis=0), 0., atol=1e-10)
    np.testing.assert_allclose(np.cross(r, gradient).sum(axis=0), 0., atol=1e-10)
    translated, shifted_gradient = model.evaluate(r + np.array([.2, -.1, .4]))
    np.testing.assert_allclose(translated.total_energy, result.total_energy, atol=1e-11)
    np.testing.assert_allclose(shifted_gradient, gradient, atol=1e-10)
    angle = .37
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                         [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    rotated, rotated_gradient = model.evaluate(r @ rotation.T)
    np.testing.assert_allclose(rotated.total_energy, result.total_energy, atol=1e-11)
    np.testing.assert_allclose(rotated_gradient, gradient @ rotation.T, atol=1e-10)


def test_native_cross_overlap_and_transport_at_reference():
    from gradscf_tools.molecular_ep import NativeRHF

    model = NativeRHF("H 0 0 0; H 0 0 1.4", unit="Bohr")
    r = model.coordinates
    result, _ = model.evaluate(r, gradient=False)
    cross = model.cross_overlap(r, r)
    np.testing.assert_allclose(cross, result.overlap_matrix, atol=1e-13)
    transported, singular = model.transport(result, result.mo_coeff, r, r)
    np.testing.assert_allclose(transported, np.diag(result.mo_energy), atol=1e-12)
    assert singular > .999999999
    displaced = r.copy()
    displaced[1, 2] += .01
    other, _ = model.evaluate(displaced, init_density=result.density_matrix, gradient=False)
    _, singular = model.transport(other, result.mo_coeff, r, displaced)
    assert singular > .95
    np.testing.assert_allclose(model.cross_overlap(r, displaced),
                               model.cross_overlap(displaced, r).T, atol=1e-13)


def test_transport_gauge_invariance():
    from gradscf_tools.molecular_ep import align_fock

    rng = np.random.default_rng(13)
    q, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    overlap = q @ np.diag([1., .99, .98, .97, .96])
    energies = np.array([-1., -.3, -.3, .2, .6])
    expected, singular = align_fock(overlap, energies)
    permutation = np.array([2, 4, 0, 1, 3])
    signs = np.array([-1., 1., -1., 1., -1.])
    actual, _ = align_fock(overlap[:, permutation] * signs, energies[permutation])
    np.testing.assert_allclose(actual, expected, atol=1e-13)
    rotation = np.eye(5)
    angle = .7
    rotation[1:3, 1:3] = [[np.cos(angle), -np.sin(angle)],
                          [np.sin(angle), np.cos(angle)]]
    actual, _ = align_fock(overlap @ rotation, energies)
    np.testing.assert_allclose(actual, expected, atol=1e-13)
    assert singular == pytest.approx(.96)
    with pytest.raises(ValueError, match="overlap"):
        align_fock(np.eye(5) * .9, energies)


@pytest.mark.parametrize('atom', [
    'H .1 -.2 .3; H .4 .2 1.8',
    'O .1 -.2 .3; H .2 1.3 1.3; H -.1 -1.7 1.4',
])
def test_native_analytic_response_matches_scf_displacements(atom):
    """Full screened Hessian/Fock response, Ha/bohr^2 and Ha/bohr."""
    from gradscf_tools.molecular_ep import NativeRHF

    model = NativeRHF(atom, unit='Bohr')
    r = model.coordinates
    reference, _ = model.evaluate(r)
    response = model.response(r, reference)
    rng = np.random.default_rng(41)
    u, v = rng.normal(size=(2,) + r.shape)
    u /= np.linalg.norm(u)
    v /= np.linalg.norm(v)
    hu, fu = response(u)
    hv, _ = response(v)
    step = 2e-4
    plus, gp = model.evaluate(r + step*u, init_density=reference.density_matrix)
    minus, gm = model.evaluate(r - step*u, init_density=reference.density_matrix)
    fp, _ = model.transport(plus, reference.mo_coeff, r, r + step*u)
    fm, _ = model.transport(minus, reference.mo_coeff, r, r - step*u)
    np.testing.assert_allclose(hu, (gp-gm)/(2*step), atol=2e-6, rtol=2e-5)
    np.testing.assert_allclose(fu, (fp-fm)/(2*step), atol=2e-6, rtol=2e-5)
    np.testing.assert_allclose(np.sum(v*hu), np.sum(u*hv), atol=2e-8)
    np.testing.assert_allclose(fu, fu.T, atol=1e-11)
    translation = np.broadcast_to([.3, -.1, .4], r.shape)
    ht, ft = response(translation)
    np.testing.assert_allclose(ht, 0., atol=2e-9)
    # The reference AO frame is fixed in space: translated molecular orbitals
    # have nonzero transport connection, even though the energies are invariant.
    ftp, _ = model.transport(reference, reference.mo_coeff, r, r + step*translation)
    ftm, _ = model.transport(reference, reference.mo_coeff, r, r - step*translation)
    np.testing.assert_allclose(ft, (ftp-ftm)/(2*step), atol=2e-8)


def test_native_response_rejects_unconverged_reference():
    from types import SimpleNamespace
    from gradscf_tools.molecular_ep import NativeRHF

    model = NativeRHF('H 0 0 0; H 0 0 1.4', unit='Bohr')
    with pytest.raises(ValueError, match='converged'):
        model.response(model.coordinates, SimpleNamespace(converged=False))


def test_native_response_rejects_reference_at_other_geometry():
    from gradscf_tools.molecular_ep import NativeRHF

    model = NativeRHF('H 0 0 0; H 0 0 1.4', unit='Bohr')
    reference, _ = model.evaluate(model.coordinates, gradient=False)
    moved = model.coordinates.copy()
    moved[1, 2] += .1
    response = model.response(moved, reference)
    with pytest.raises(RuntimeError, match='reference|stationarity'):
        response(np.ones_like(moved))


def test_native_response_degenerate_occupied_gauge_covariance():
    """Linear water has a degenerate occupied pi pair; never AD its eigenvectors."""
    from types import SimpleNamespace
    from gradscf_tools.molecular_ep import NativeRHF

    model = NativeRHF('O 0 0 0; H 0 0 1.8; H 0 0 -1.8', unit='Bohr')
    r = model.coordinates
    reference, _ = model.evaluate(r, gradient=False)
    energies = np.asarray(reference.mo_energy)
    occupied = np.flatnonzero(np.asarray(reference.mo_occ) == 2)
    pairs = [(i, j) for i in occupied for j in occupied if i < j and abs(energies[i]-energies[j]) < 1e-9]
    assert pairs, 'Test molecule must have an exactly degenerate occupied pair.'
    i, j = pairs[0]
    gauge = np.eye(len(energies))
    angle = .63
    gauge[np.ix_([i, j], [i, j])] = [[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle), np.cos(angle)]]
    rotated = SimpleNamespace(converged=True, mo_coeff=reference.mo_coeff @ gauge,
                              mo_occ=reference.mo_occ)
    direction = np.random.default_rng(17).normal(size=r.shape)
    h, f = model.response(r, reference)(direction)
    hg, fg = model.response(r, rotated)(direction)
    np.testing.assert_allclose(hg, h, atol=2e-8)
    np.testing.assert_allclose(fg, gauge.T @ f @ gauge, atol=2e-8)
    step = 2e-4
    plus, _ = model.evaluate(r+step*direction, init_density=reference.density_matrix, gradient=False)
    minus, _ = model.evaluate(r-step*direction, init_density=reference.density_matrix, gradient=False)
    fp, _ = model.transport(plus, reference.mo_coeff, r, r+step*direction)
    fm, _ = model.transport(minus, reference.mo_coeff, r, r-step*direction)
    np.testing.assert_allclose(f, (fp-fm)/(2*step), atol=2e-6, rtol=2e-5)


def test_native_response_rejects_failed_linear_solve():
    from gradscf.scf.autodiff import SCFDifferentiationConfig
    from gradscf_tools.molecular_ep import NativeRHF

    model = NativeRHF('O .1 -.2 .3; H .2 1.3 1.3; H -.1 -1.7 1.4', unit='Bohr')
    reference, _ = model.evaluate(model.coordinates, gradient=False)
    response = model.response(model.coordinates, reference,
        differentiation=SCFDifferentiationConfig(tolerance=1e-12, max_iter=1, restart=1))
    direction = np.random.default_rng(41).normal(size=model.coordinates.shape)
    direction /= np.linalg.norm(direction)
    with pytest.raises(RuntimeError, match='linear convergence'):
        response(direction)


def test_native_response_rejects_regularized_implicit_solve():
    from types import SimpleNamespace
    from gradscf.scf.autodiff import SCFDifferentiationConfig
    from gradscf_tools.molecular_ep import NativeRHF

    model = NativeRHF('H 0 0 0; H 0 0 1.4', unit='Bohr')
    reference = SimpleNamespace(converged=True, mo_coeff=np.eye(2), mo_occ=np.array([2., 0.]))
    with pytest.raises(ValueError, match='regularization'):
        model.response(model.coordinates, reference,
            differentiation=SCFDifferentiationConfig(regularization=1e-3))


@pytest.mark.parametrize('dtype', [np.float32, np.complex128])
def test_native_response_rejects_non_real_float64_reference(dtype):
    from types import SimpleNamespace
    from gradscf_tools.molecular_ep import NativeRHF

    model = NativeRHF('H 0 0 0; H 0 0 1.4', unit='Bohr')
    reference = SimpleNamespace(converged=True, mo_coeff=np.eye(2, dtype=dtype),
                                mo_occ=np.array([2., 0.]))
    with pytest.raises(ValueError, match='real float64'):
        model.response(model.coordinates, reference)
