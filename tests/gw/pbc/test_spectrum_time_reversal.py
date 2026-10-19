"""A causal Matsubara fit need not preserve time reversal on the real axis.

These synthetic source arrays use complex MO gauges and a non-TRIM k pair.
No SCF/GW calculation is needed: the stub realization has PSD residues, an
exact first moment and an accepted imaginary-axis error, while nearly equal
poles hide a substantial time-reversal-odd residue component.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from gradscf.solvers.poles import PoleFit, evaluate_poles


def _source():
    frequency = np.geomspace(.03, 20., 24)
    poles = np.array([-1.2, .35-1e-6, .35+1e-6])
    # Independent complex MO phase gauges at Gamma, +1/3 and -1/3.
    coeff = np.asarray([np.diag(np.exp(1j*np.array(phases)))
        for phases in ((.2, -.7), (.9, -.3), (-.4, 1.1))])
    table = np.array([[0, 2, 1], [1, 0, 2], [2, 1, 0]])
    model = SimpleNamespace(coeff=coeff, table=table, zero_q=0,
        energy=np.tile([-.6, .4], (3, 1)), lattice=np.eye(3), nocc=1,
        kpoints=np.array([[0., 0., 0.], [2*np.pi/3, 0., 0.], [-2*np.pi/3, 0., 0.]]))
    base = np.array([[[.09, .012], [.012, .07]],
                     [[.08, -.014], [-.014, .06]],
                     [[.07, .009], [.009, .09]]], complex)
    partner = base.copy()
    partner[:, 0, 1] += np.array([.008j, -.006j, .011j])
    partner[:, 1, 0] = partner[:, 0, 1].conj()
    residues_ao = np.array([base, partner, partner.conj()])
    fock_ao = np.array([[[ -.6, .04], [.04, .4]],
                        [[-.55, .03+.02j], [.03-.02j, .45]],
                        [[-.55, .03-.02j], [.03+.02j, .45]]])
    to_mo = lambda k, value: coeff[k].conj().T@value@coeff[k]
    residues = np.asarray([to_mo(k, value) for k, value in enumerate(residues_ao)])
    values = np.asarray([evaluate_poles(PoleFit(poles, r, 0., 0., 0, True),
                                      1j*frequency) for r in residues])
    result = SimpleNamespace(converged=True, sigma_c_iw=values,
        sigma_moment=residues.sum(axis=1),
        full_fock=np.asarray([to_mo(k, value) for k, value in enumerate(fock_ao)]),
        grid=SimpleNamespace(fermion=frequency, beta=40.), chemical_potential=0.)
    # The same TR-odd perturbation at both partner k points violates their
    # relation. Opposite signs at adjacent poles cancel its first moment.
    odd = np.array([[0., .018j], [-.018j, 0.]])
    bad = residues_ao.copy()
    bad[:, 1] += odd
    bad[:, 2] -= odd
    fits = []
    for k in range(3):
        r = to_mo(k, bad[k])
        p = poles.copy()
        if k == 1:
            p, r = np.r_[p, 2.], np.concatenate((r, np.zeros((1, 2, 2))))
        fits.append(PoleFit(p, r, 0., 0., 0, True))
    return model, result, fits


def test_continuation_restores_same_pole_tr_after_complex_gauge_and_exact_union(monkeypatch):
    from gradscf.gw.pbc import spectrum
    model, result, original = _source()
    saved = {name: np.array(getattr(result, name), copy=True)
             for name in ('full_fock', 'sigma_moment', 'sigma_c_iw')}
    calls = []
    def realization(z, values, **options):
        fit = original[len(calls)]
        error = np.linalg.norm(evaluate_poles(fit, z)-values)/np.linalg.norm(values)
        assert error < options['tolerance']
        np.testing.assert_allclose(fit.residues.sum(axis=0), options['moment'], atol=1e-15)
        assert np.linalg.eigvalsh(fit.residues).min() >= -1e-15
        calls.append(error)
        return fit
    monkeypatch.setattr(spectrum, 'interpolate_causal', realization)
    z = np.array([.35-1e-6+1e-7j, .35+1e-6+1e-7j, -.4+.02j])
    # AO transport is independent of the MO gauge. This table row gives an
    # independent expected k/-k pairing for the declared physical k points.
    partners = model.table[model.zero_q]
    legacy, legacy_reports = spectrum.continue_self_energy(result, tolerance=1e-4)
    assert all(report['holdout_relative_error'] < 1e-4 for report in legacy_reports)
    old_ao = [c@evaluate_poles(fit, z)@c.conj().T
              for c, fit in zip(model.coeff, legacy)]
    assert max(np.linalg.norm(old_ao[k]-old_ao[bar].swapaxes(-1, -2))
               for k, bar in enumerate(partners)) > 1.
    calls.clear()
    fits, reports = spectrum.continue_self_energy(result, model=model, tolerance=1e-4)
    assert len(calls) == len(fits) == 3
    ao = [c@fit.residues@c.conj().T for c, fit in zip(model.coeff, fits)]
    # A non-TRIM partner is allowed a genuine complex Hermitian AO residue.
    assert np.max(abs(ao[1].imag)) > .005
    for k, bar in enumerate(partners):
        np.testing.assert_array_equal(fits[k].poles, fits[bar].poles)
        np.testing.assert_allclose(ao[k], ao[bar].conj(), atol=2e-14)
        np.testing.assert_allclose(fits[k].residues.sum(axis=0), result.sigma_moment[k], atol=2e-14)
        assert np.linalg.eigvalsh(fits[k].residues).min() >= -1e-14
        a = model.coeff[k]@evaluate_poles(fits[k], z)@model.coeff[k].conj().T
        b = model.coeff[bar]@evaluate_poles(fits[bar], z)@model.coeff[bar].conj().T
        np.testing.assert_allclose(a, b.swapaxes(-1, -2), atol=2e-8, rtol=2e-12)
        predicted = evaluate_poles(fits[k], 1j*result.grid.fermion)
        assert np.linalg.norm(predicted-result.sigma_c_iw[k])/np.linalg.norm(result.sigma_c_iw[k]) < 1e-4
        assert reports[k]['holdout_relative_error'] < 1e-4
        errors = [np.linalg.norm(predicted[take]-result.sigma_c_iw[k, take])
                  /np.linalg.norm(result.sigma_c_iw[k, take])
                  for take in (slice(None, None, 2), slice(1, None, 2))]
        np.testing.assert_allclose([fits[k].relative_error, reports[k]['training_relative_error']],
                                   [errors[0]]*2, atol=1e-16, rtol=1e-12)
        np.testing.assert_allclose(reports[k]['holdout_relative_error'], errors[1], atol=1e-16)
        assert reports[k]['realization']['time_reversal']['partner'] == bar
    for name, value in saved.items():
        np.testing.assert_array_equal(getattr(result, name), value)


@pytest.mark.parametrize('quantity', ['full_fock', 'sigma_moment', 'sigma_c_iw'])
def test_genuine_raw_tr_breaking_is_rejected_before_realization(monkeypatch, quantity):
    from gradscf.gw.pbc import spectrum
    model, result, _ = _source()
    # Break only +1/3, so a Gamma-only or identity-column check is insufficient.
    odd = np.array([[0., .005j], [-.005j, 0.]])
    change = model.coeff[1].conj().T@odd@model.coeff[1]
    if quantity == 'sigma_c_iw':
        change = change[None]/(1j*result.grid.fermion[:, None, None]-.9)
    getattr(result, quantity)[1] += change
    def no_fit(*args, **kwargs):
        pytest.fail('Genuinely TR-breaking source reached causal realization')
    monkeypatch.setattr(spectrum, 'interpolate_causal', no_fit)
    with pytest.raises((ValueError, ArithmeticError), match='[Tt]ime.?reversal|TR'):
        spectrum.continue_self_energy(result, model=model, tolerance=1e-4)


def test_nonclosed_time_reversed_mo_subspaces_are_rejected(monkeypatch):
    from gradscf.gw.pbc import spectrum
    model, result, _ = _source()
    coeff = np.zeros((3, 3, 2), complex)
    coeff[:, :2] = model.coeff
    coeff[2] = np.array([[1., 0.], [0., 0.], [0., 1.]])
    model.coeff = coeff
    def no_fit(*args, **kwargs):
        pytest.fail('Nonclosed TR subspaces reached causal realization')
    monkeypatch.setattr(spectrum, 'interpolate_causal', no_fit)
    with pytest.raises((ValueError, ArithmeticError), match='[Tt]ime.?reversal|TR|subspace|unitar|clos'):
        spectrum.continue_self_energy(result, model=model, tolerance=1e-4)


def test_zero_self_energy_accepts_time_reversal_context_and_empty_bath():
    from gradscf.gw.pbc.spectrum import continue_self_energy
    model, result, _ = _source()
    result.sigma_c_iw[:] = 0.
    result.sigma_moment[:] = 0.
    fits, reports = continue_self_energy(result, model=model, tolerance=1e-8)
    for fit, report in zip(fits, reports):
        assert fit.converged and fit.residues.shape == (0, 2, 2)
        assert fit.poles.shape == (0,)
        np.testing.assert_array_equal(evaluate_poles(fit, [.2+.03j, -.7+.1j]),
                                      np.zeros((2, 2, 2)))
        assert report['training_relative_error'] == 0.
        assert report['holdout_relative_error'] == 0.
        assert report['moment_relative_error'] == 0.
