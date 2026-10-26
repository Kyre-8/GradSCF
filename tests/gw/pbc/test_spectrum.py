"""No scGW bands may be inferred from a static Fock diagonalization."""
from types import SimpleNamespace
import numpy as np
import pytest


def _result():
    frequency = np.geomspace(.03, 20., 32)
    poles = np.array([-1.1, .8])
    residues = np.array([np.diag([.08, .03]), np.diag([.02, .06])])
    sigma = np.einsum('wl,lij->wij', 1/(1j*frequency[:, None]-poles), residues)
    result = SimpleNamespace(converged=True, sigma_c_iw=sigma[None],
        sigma_moment=residues.sum(axis=0)[None],
        grid=SimpleNamespace(fermion=frequency), chemical_potential=0.)
    return result, poles


def test_scgw_continuation_checks_unseen_frequencies_and_full_matrix_moment():
    from gradscf.gw.pbc.spectrum import continue_self_energy
    result, poles = _result()
    fits, reports = continue_self_energy(result, tolerance=1e-8)
    assert len(fits) == len(reports) == 1
    assert reports[0]['holdout_relative_error'] < 1e-8
    assert reports[0]['moment_relative_error'] < 1e-10
    np.testing.assert_allclose(fits[0].residues.sum(axis=0), result.sigma_moment[0], atol=1e-10)
    saved = result.sigma_c_iw
    result.sigma_c_iw = -saved
    with pytest.raises(ArithmeticError, match='realization|PSD'):
        continue_self_energy(result, tolerance=1e-6)
    result.sigma_c_iw = saved
    result.converged = False
    with pytest.raises(ValueError, match='converged'):
        continue_self_energy(result)


def test_source_realizations_have_an_exact_common_pole_representation(monkeypatch):
    from gradscf.gw.pbc import spectrum
    from gradscf.solvers.poles import PoleFit, evaluate_poles
    frequency = np.geomspace(.03, 20., 32)
    residues = np.array([np.diag([.08, .03]), np.diag([.02, .06])])
    original = (
        PoleFit(np.array([-1.1, .8]), residues, 0., 0., 1, True),
        PoleFit(np.array([-.9, -.9, 1.2]),
                np.array([residues[0]/3, 2*residues[0]/3, residues[1]]),
                0., 0., 1, True))
    values = np.asarray([evaluate_poles(fit, 1j*frequency) for fit in original])
    result = SimpleNamespace(converged=True, sigma_c_iw=values,
        sigma_moment=np.asarray([fit.residues.sum(axis=0) for fit in original]),
        grid=SimpleNamespace(fermion=frequency))
    calls = []
    def fitted(z, values, **kwargs):
        assert kwargs['rank_tolerance'] == 1e-12
        fit = original[len(calls)]
        np.testing.assert_allclose(evaluate_poles(fit, z), values, atol=1e-15)
        calls.append(kwargs)
        return fit
    monkeypatch.setattr(spectrum, 'interpolate_causal', fitted)
    fits, reports = spectrum.continue_self_energy(result, tolerance=1e-10)
    expected = np.array([-1.1, -.9, .8, 1.2])
    for before, after, report in zip(original, fits, reports):
        np.testing.assert_array_equal(after.poles, expected)
        z = np.array([.17+.05j, -.65+.1j, 2.+.4j])
        np.testing.assert_allclose(evaluate_poles(after, z),
                                   evaluate_poles(before, z), atol=1e-15)
        np.testing.assert_allclose(after.residues.sum(axis=0),
                                   before.residues.sum(axis=0), atol=1e-15)
        assert np.linalg.eigvalsh(after.residues).min() >= 0.
        assert report['pole_count'] == len(expected)
        assert report['source_pole_count'] == len(before.poles)


def test_explicit_continuation_window_keeps_high_frequency_error_visible():
    from gradscf.gw.pbc.spectrum import continue_self_energy
    result, poles = _result()
    frequency = result.grid.fermion
    result.sigma_c_iw[:, frequency > 1.] += .04*np.eye(2)
    saved = result.sigma_c_iw.copy()
    fits, reports = continue_self_energy(result, tolerance=1e-8,
                                         frequency_cutoff=1.)
    assert fits[0].converged
    report = reports[0]
    assert report['fit_frequency_count'] == np.count_nonzero(frequency <= 1.)
    assert report['total_positive_frequency_count'] == len(frequency)
    assert report['frequency_cutoff_hartree'] == 1.
    assert report['fitted_frequency_max_hartree'] <= 1.
    assert report['excluded_frequency_relative_error'] > .1
    np.testing.assert_array_equal(result.sigma_c_iw, saved)
    for cutoff in (-1., 0., np.nan, np.inf):
        with pytest.raises(ValueError, match='cutoff'):
            continue_self_energy(result, frequency_cutoff=cutoff)
    with pytest.raises(ValueError, match='four positive'):
        continue_self_energy(result, frequency_cutoff=.031)


def test_static_query_reproduces_source_and_refuses_incomplete_qps():
    from gradscf.gw.pbc.spectrum import interpolate_quasiparticles
    energy = np.array([[-.4, .3], [-.3, .5]])
    coefficients = np.tile(np.eye(2, dtype=complex), (2, 1, 1))
    kpoints = np.array([[0., 0., 0.], [np.pi, 0., 0.]])
    model = SimpleNamespace(energy=energy, coeff=coefficients, nocc=1,
        lattice=np.eye(3), kpoints=kpoints)
    mf = SimpleNamespace(inputs=SimpleNamespace(overlap=coefficients),
        get_bands=lambda points, **kw: (energy, coefficients))
    qp = energy+np.array([[-.1, .1], [-.12, .15]])
    result = SimpleNamespace(mo_energy=qp, mo_coeff=coefficients,
        method='periodic_evgw_full_cd_diagonal', converged=True,
        converged_mask=np.ones_like(energy, dtype=bool),
        qp_computed_mask=np.ones_like(energy, dtype=bool))
    actual, reference = interpolate_quasiparticles(mf, model, result, kpoints)
    np.testing.assert_allclose(actual, qp, atol=1e-14)
    np.testing.assert_allclose(reference, energy, atol=0)
    def no_repeated_query(*args, **kwargs):
        raise AssertionError('The same mean-field query must not be recomputed')
    mf.get_bands = no_repeated_query
    cached, _ = interpolate_quasiparticles(mf, model, result, kpoints,
                                           query_bands=(energy, coefficients))
    np.testing.assert_allclose(cached, actual, atol=1e-14)
    result.converged_mask[1, 1] = False
    with pytest.raises(ValueError, match='converged|complete'):
        interpolate_quasiparticles(mf, model, result, kpoints)
    result.converged_mask[:] = True
    result.qp_computed_mask[1, 1] = False
    with pytest.raises(ValueError, match='computed|complete'):
        interpolate_quasiparticles(mf, model, result, kpoints,
                                   query_bands=(energy, coefficients))
    result.qp_computed_mask[:] = True
    result.mo_energy = np.array([[.4, .3], [-.42, .65]])
    with pytest.raises(ValueError, match='occupied|cross'):
        interpolate_quasiparticles(mf, model, result, kpoints,
                                   query_bands=(energy, coefficients))


def test_continued_qp_poles_agree_with_independent_auxiliary_eigenvalues():
    from gradscf.gw.pbc.spectrum import continued_quasiparticles
    fock = np.diag([-.4, .3])[None]
    poles = np.array([-1.1, .8])
    residue = np.array([np.diag([.03, .01]), np.diag([.01, .04])])[None]
    data = dict(fock=fock, residues=residue, poles=poles,
        reference_energy=np.array([[-.4, .3]]))
    result = continued_quasiparticles(data, mu=0., nocc=1, eta=0.)
    assert result['converged_mask'].all()
    expected = []
    for p in range(2):
        vertex = np.sqrt(residue[0, :, p, p])
        matrix = np.diag(np.r_[fock[0, p, p], poles])
        matrix[0, 1:] = matrix[1:, 0] = vertex
        energy, coeff = np.linalg.eigh(matrix)
        expected.append(energy[np.argmax(abs(coeff[0])**2)])
    np.testing.assert_allclose(result['energy'][0], expected, atol=2e-9)
    assert np.max(abs(result['energy']-np.diagonal(fock, axis1=-2, axis2=-1))) > 1e-3


def test_scgw_trace_only_query_matches_full_spectral_matrix_without_retaining_it():
    from gradscf.gw.pbc.spectrum import interpolate_scgw, continue_self_energy
    result, poles = _result()
    result.full_fock = np.diag([-.4, .3])[None]
    fits, _ = continue_self_energy(result, tolerance=1e-8)
    energy = np.array([[-.4, .3]])
    coeff = np.eye(2)[None]
    model = SimpleNamespace(energy=energy, coeff=coeff, lattice=np.eye(3),
                            kpoints=np.zeros((1, 3)))
    mf = SimpleNamespace(inputs=SimpleNamespace(overlap=coeff),
        get_bands=lambda points, **kw: (energy, coeff))
    axis = np.linspace(-2., 2., 101)
    full = interpolate_scgw(mf, model, result, fits, model.kpoints, axis)
    trace = interpolate_scgw(mf, model, result, fits, model.kpoints, axis,
                              trace_only=True, query_bands=(energy, coeff))
    assert trace['spectral_matrix'] is None
    np.testing.assert_allclose(trace['spectral_trace'],
        np.trace(full['spectral_matrix'], axis1=-2, axis2=-1).real, atol=1e-13)
    assert np.min(trace['spectral_trace']) >= 0.


def test_continued_particle_number_uses_thermal_satellite_weights():
    from gradscf.gw.pbc.spectrum import continue_self_energy, continued_particle_number
    result, poles = _result()
    result.full_fock = np.array([[-.4, .03j], [-.03j, .3]])[None]
    result.grid.beta = 20.
    fits, _ = continue_self_energy(result, tolerance=1e-8)
    # Independent Hermitian auxiliary Hamiltonian for the two matrix poles.
    residues = np.array([np.diag([.08, .03]), np.diag([.02, .06])])
    coupling = np.concatenate([np.diag(np.sqrt(np.diag(r))) for r in residues], axis=1)
    matrix = np.block([[result.full_fock[0], coupling],
                       [coupling.conj().T, np.diag(np.repeat(poles, 2))]])
    energy, vector = np.linalg.eigh(matrix)
    occupation = np.exp(-np.logaddexp(0., result.grid.beta*energy))
    expected = 2*np.sum(abs(vector[:2])**2@occupation)
    actual = continued_particle_number(result, fits)
    np.testing.assert_allclose(actual, expected, atol=1e-9)
    assert abs(actual-2.) > 1e-3
    result.converged = False
    with pytest.raises(ValueError, match='converged'):
        continued_particle_number(result, fits)
