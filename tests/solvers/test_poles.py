"""Causal matrix spectra and an independent 14-orbital finite measure."""
import numpy as np


def _data():
    poles = np.array([-1.1, .8])
    vertex = np.array([[.5, .12+.19j], [.08-.17j, .4]])
    residues = np.einsum('li,lj->lij', vertex, vertex.conj())
    z = 1j*np.geomspace(.03, 30., 48)
    values = np.einsum('wl,lij->wij', 1/(z[:, None]-poles), residues)
    return poles, residues, z, values


def test_complex_matrix_realization_has_positive_retarded_spectrum_and_exact_moment():
    from gradscf.solvers.poles import interpolate_causal, evaluate_poles
    _, residues, z, values = _data()
    fit = interpolate_causal(z[::2], values[::2],
        moment=residues.sum(axis=0), tolerance=1e-9)
    assert fit.converged and fit.iterations == 0
    assert fit.relative_error < 1e-9
    assert fit.moment_error < 1e-10
    np.testing.assert_allclose(evaluate_poles(fit, z[1::2]), values[1::2], atol=2e-10)
    retarded = evaluate_poles(fit, np.linspace(-2., 2., 101)+.02j)
    spectral = -(retarded-retarded.conj().swapaxes(-1, -2))/(2j*np.pi)
    assert np.linalg.eigvalsh(spectral).min() > -1e-12
    np.testing.assert_allclose(fit.residues.sum(axis=0), residues.sum(axis=0), atol=1e-10)


def test_auxiliary_dyson_poles_and_spectral_weights_are_basis_covariant():
    from gradscf.solvers.poles import interpolate_causal, auxiliary_spectrum
    poles, residues, z, values = _data()
    unitary = np.array([[1, 1j], [1j, 1]])/np.sqrt(2.)
    transform = lambda a: unitary.conj().T@a@unitary
    fock = np.array([[-.4, .04j], [-.04j, .3]])
    left = interpolate_causal(z, values, moment=residues.sum(axis=0), tolerance=1e-9)
    right = interpolate_causal(z, transform(values),
        moment=transform(residues.sum(axis=0)), tolerance=1e-9)
    e, weight = auxiliary_spectrum(fock, left, mu=.1)
    er, wr = auxiliary_spectrum(transform(fock), right, mu=.1)
    np.testing.assert_allclose(e, er, atol=1e-10)
    np.testing.assert_allclose(weight, wr, atol=1e-10)
    np.testing.assert_allclose(weight.sum(), 2., atol=1e-12)
    assert weight.min() >= 0.
    zz = .2+.07j
    sigma = np.einsum('l,lij->ij', 1/(zz-poles), residues)
    expected = np.trace(np.linalg.inv((zz+.1)*np.eye(2)-fock-sigma))
    np.testing.assert_allclose(np.sum(weight/(zz+.1-e)), expected, atol=1e-10)


def test_known_feasible_14_orbital_noncommuting_six_pole_measure():
    from gradscf.solvers.poles import interpolate_causal, evaluate_poles
    positive = np.geomspace(.001, 20., 64)
    poles = np.r_[-positive[::-1], 0., positive][[8, 29, 54, 64, 91, 119]]
    rng = np.random.default_rng(211)
    amplitude = rng.normal(size=(6, 14, 14))+1j*rng.normal(size=(6, 14, 14))
    total = np.einsum('lij,lkj->ik', amplitude, amplitude.conj())
    factor = np.linalg.cholesky(total)
    normalized = np.linalg.solve(factor, amplitude.transpose(1, 0, 2).reshape(14, -1))
    normalized = normalized.reshape(14, 6, 14).transpose(1, 0, 2)
    moment = np.diag(np.linspace(.04, .15, 14))
    root = np.sqrt(moment)
    residues = root@(normalized@normalized.conj().swapaxes(-1, -2))@root
    np.testing.assert_allclose(residues.sum(axis=0), moment, atol=2e-15, rtol=2e-14)
    assert np.linalg.eigvalsh(residues).min() > 0
    assert np.linalg.norm(residues[0]@residues[1]-residues[1]@residues[0]) > 1e-6
    z = 1j*(2*np.arange(128)+1)*np.pi/400.
    values = np.einsum('wl,lij->wij', 1/(z[:, None]-poles), residues)
    fit = interpolate_causal(z[::2], values[::2], moment=moment, tolerance=8e-5)
    assert fit.converged and fit.iterations == 0
    training = np.linalg.norm(evaluate_poles(fit, z[::2])-values[::2])/np.linalg.norm(values[::2])
    holdout = np.linalg.norm(evaluate_poles(fit, z[1::2])-values[1::2])/np.linalg.norm(values[1::2])
    assert training <= 8e-5
    assert holdout <= 1e-4
    assert fit.moment_error <= 1e-9
    assert np.linalg.eigvalsh(fit.residues).min() >= -1e-12*np.linalg.eigvalsh(moment).max()
