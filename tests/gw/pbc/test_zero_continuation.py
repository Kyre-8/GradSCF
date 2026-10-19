"""An empty causal bath remains a usable zero dynamical self-energy."""
from types import SimpleNamespace

import numpy as np


def test_empty_bath_auxiliary_spectrum_returns_only_physical_states():
    from gradscf.solvers.poles import PoleFit, auxiliary_spectrum
    fit = PoleFit(np.empty(0), np.zeros((0, 2, 2), complex), 0., 0., 0, True)
    energy, weights = auxiliary_spectrum(np.diag([-.4, .3]), fit, mu=.07)
    np.testing.assert_allclose(energy, [-.4, .3], atol=1e-14)
    np.testing.assert_allclose(weights, [1., 1.], atol=1e-14)


def test_empty_bath_spectral_query_and_continued_roots_reduce_to_full_static_fock():
    from gradscf.solvers.poles import PoleFit
    from gradscf.gw.pbc.spectrum import interpolate_scgw, continued_quasiparticles
    energy = np.array([[-.4, .3]])
    coeff = np.eye(2)[None]
    fock = np.array([[[-.38, .02j], [-.02j, .33]]])
    fit = PoleFit(np.empty(0), np.zeros((0, 2, 2), complex), 0., 0., 0, True)
    result = SimpleNamespace(converged=True, full_fock=fock, chemical_potential=0.)
    model = SimpleNamespace(energy=energy, coeff=coeff, lattice=np.eye(3), kpoints=np.zeros((1, 3)))
    mf = SimpleNamespace(inputs=SimpleNamespace(overlap=coeff))
    axis, eta = np.linspace(-1., 1., 21), .03
    data = interpolate_scgw(mf, model, result, (fit,), model.kpoints, axis,
        eta=eta, query_bands=(energy, coeff))
    green = np.linalg.inv((axis+1j*eta)[:, None, None]*np.eye(2)-fock[0])
    expected = -(green-green.conj().swapaxes(-1, -2))/(2j*np.pi)
    np.testing.assert_allclose(data['spectral_matrix'][0], expected, atol=1e-13)
    roots = continued_quasiparticles(data, mu=0., nocc=1, eta=eta)
    assert roots['converged_mask'].all()
    np.testing.assert_allclose(roots['energy'][0], np.linalg.eigvalsh(fock[0]), atol=2e-9)
