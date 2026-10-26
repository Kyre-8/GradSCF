"""A certified retarded spectrum is independent of real-part QP selection."""
from types import SimpleNamespace
import importlib
import json

import numpy as np
import pytest


@pytest.fixture
def saved_example(tmp_path, monkeypatch):
    example = importlib.import_module('examples.pbc.band_structures.gaas_gw_spectra')
    energy = np.array([[-.4, .3]])
    coeff = np.eye(2, dtype=complex)[None]
    density = np.tile(np.diag([1., 0.]), (2, 1, 1, 1))
    coordinates = np.zeros((1, 3))
    model = SimpleNamespace(energy=energy, coeff=coeff, nocc=1,
        kpoints=np.zeros((1, 3)), lattice=np.eye(3))
    cell = SimpleNamespace(coords=coordinates, charges=np.array([2.]),
        reciprocal_vectors=lambda: np.eye(3),
        make_kpts=lambda mesh: np.array([[.25, 0., 0.]]))
    mf = SimpleNamespace(cell=cell, inputs=SimpleNamespace(overlap=coeff),
        result=SimpleNamespace(density_spin=density),
        get_bands=lambda points, **kwargs: (np.repeat(energy, len(points), axis=0),
                                           np.repeat(coeff, len(points), axis=0)))
    path = SimpleNamespace(kpts=np.array([[0., 0., 0.], [.5, 0., 0.]]),
        get_linear_kpoint_axis=lambda: (np.array([0., 1.]), np.array([0., 1.]), ['G', 'X']))
    atoms = SimpleNamespace(get_atomic_numbers=lambda: np.array([2]), pbc=np.ones(3, bool),
        cell=SimpleNamespace(bandpath=lambda *args, **kwargs: path))
    common = dict(source_kmesh=[1, 1, 1], fft_mesh=[61, 61, 61], basis='gth-szv',
        pseudo='gth-pbe', xc='pbe', soc=False, fc=False, exxdiv=None,
        screening_cutoff_rydberg=16., source_model=dict(compression_tolerance=1e-12),
        converged=True, nw=64, eta_hartree=.001)
    source = dict(reference_energy=energy, reference_coeff=coeff, reference_density=density,
        source_kpoints=model.kpoints, lattice_bohr=model.lattice, coordinates_bohr=coordinates,
        atomic_numbers=np.array([2]), valence_charges=cell.charges, overlap=coeff, nocc=np.array(1))
    for method in example.METHODS:
        report = dict(common, method='periodic_'+method+'_full_cd_diagonal')
        data = dict(source)
        if method == 'scgw':
            report.update(method='periodic_scgw_matsubara', eta_hartree=None, nw=1024,
                beta_hartree_inverse=400., time_quadrature='gauss', time_oversampling=4, ntau=8192)
            data.update(sigma_c_iw=np.zeros((1, 2048, 2, 2), complex),
                sigma_moment=np.zeros((1, 2, 2)), full_fock=np.array([np.diag(energy[0])]),
                chemical_potential=np.array(0.), fermion=(2*np.arange(-1024, 1024)+1)*np.pi/400.)
        else:
            report['method'] = ('periodic_qsgw_full_cd_matrix_mode_a' if method == 'qsgw'
                                else report['method'])
            data.update(qp_energy=energy, qp_coeff=coeff, converged_mask=np.ones_like(energy, bool),
                qp_computed_mask=np.ones_like(energy, bool), qp_weight=np.ones_like(energy),
                qp_branch_valid=np.ones_like(energy, bool), qp_candidate_count=np.ones_like(energy, int))
        np.savez_compressed(tmp_path/(method+'.npz'), **data)
        (tmp_path/(method+'.json')).write_text(json.dumps(report))
    monkeypatch.setattr(example, 'reference', lambda **kwargs: mf)
    monkeypatch.setattr(example, 'build_periodic_model', lambda *args, **kwargs: model)
    monkeypatch.setattr(example, 'structure', lambda name: (atoms, 'GX', None))
    fit = SimpleNamespace(poles=np.array([-.7, .9]), residues=np.zeros((2, 2, 2)))
    monkeypatch.setattr(example, 'continue_self_energy', lambda *args, **kwargs: ((fit,), [{}]))
    monkeypatch.setattr(example, 'continued_particle_number', lambda *args: 2.)
    monkeypatch.setattr(example, 'interpolate_quasiparticles',
        lambda mf, model, result, points, **kwargs: (np.repeat(result.mo_energy, len(points), axis=0), None))
    def spectral(mf, model, state, fits, points, frequencies, **kwargs):
        assert kwargs['trace_only']
        return dict(spectral_trace=np.full((len(points), len(frequencies)), .125),
            spectral_matrix=None, fock=np.repeat(state.full_fock, len(points), axis=0),
            residues=np.zeros((len(points), 2, 2, 2)), poles=fit.poles,
            reference_energy=np.repeat(energy, len(points), axis=0))
    monkeypatch.setattr(example, 'interpolate_scgw', spectral)
    qp = dict(energy=np.tile([-.3, .5], (4, 1)), converged_mask=np.ones((4, 2), bool),
        branch_valid=np.ones((4, 2), bool), weight=np.full((4, 2), .7),
        residual=np.full((4, 2), 1e-12), candidate_count=np.ones((4, 2), int))
    monkeypatch.setattr(example, 'continued_quasiparticles',
        lambda *args, **kwargs: {name: value.copy() for name, value in qp.items()})
    plotted = []
    monkeypatch.setattr(example, 'plot_comparison', lambda directory: plotted.append(directory) or {'png': 'toy.png'})
    return SimpleNamespace(example=example, directory=tmp_path, qp=qp, plotted=plotted)


@pytest.mark.parametrize('ambiguous', ['dos', 'gamma'])
def test_valid_spectrum_survives_uncertified_scgw_roots_without_filling_them(saved_example, ambiguous):
    case = saved_example
    # query order: two path points, one independent DOS point, explicit Gamma.
    indices = [2] if ambiguous == 'dos' else [0, 3]
    case.qp['branch_valid'][indices, 1] = False
    case.qp['candidate_count'][indices, 1] = 2
    raw_energy = case.qp['energy'].copy()
    case.example.analyze(case.directory, path_points=2, dos_kmesh=(1, 1, 1),
        axis_ev=np.linspace(-50., 50., 101), frequency_cutoff=2.01)
    assert case.plotted == [case.directory]
    report = json.loads((case.directory/'comparison.json').read_text())
    gap = report['gaps']['scgw']
    if ambiguous == 'dos':
        assert gap['gamma_direct_ev'] == pytest.approx(.8*case.example.EV_PER_HARTREE)
        assert gap['gamma_gap_change_ev'] is not None
        for name in ('sampled_gap_ev', 'minimum_direct_ev', 'sampled_gap_change_ev',
                     'minimum_direct_gap_change_ev'):
            assert gap[name] is None
    else:
        assert gap['gamma_direct_ev'] is None
        assert gap['gamma_gap_change_ev'] is None
        assert gap['sampled_gap_ev'] == pytest.approx(.8*case.example.EV_PER_HARTREE)
        assert gap['minimum_direct_ev'] is not None
    with np.load(case.directory/'comparison.npz', allow_pickle=False) as data:
        np.testing.assert_array_equal(data['scgw_qp_raw_energy'], raw_energy)
        np.testing.assert_array_equal(data['scgw_qp_certified_mask'], case.qp['branch_valid'])
        np.testing.assert_array_equal(data['scgw_qp_branch_valid'], case.qp['branch_valid'])
        np.testing.assert_array_equal(data['scgw_qp_converged_mask'], case.qp['converged_mask'])
        np.testing.assert_array_equal(data['scgw_qp_weight'], case.qp['weight'])
        np.testing.assert_array_equal(data['scgw_qp_residual'], case.qp['residual'])
        np.testing.assert_array_equal(data['scgw_qp_candidate_count'], case.qp['candidate_count'])
        certified = case.qp['branch_valid']
        assert np.isnan(data['scgw_qp_energy'][~certified]).all()
        np.testing.assert_array_equal(data['scgw_qp_energy'][certified], raw_energy[certified])
        np.testing.assert_allclose(data['scgw_dos'], 2*.125/case.example.EV_PER_HARTREE)
        assert np.isfinite(data['scgw_spectral']).all()
        np.testing.assert_array_equal(np.isnan(data['scgw_bands']), ~certified[:2])


@pytest.mark.parametrize('method', ['g0w0', 'evgw', 'qsgw'])
def test_static_source_qp_coverage_remains_strict(saved_example, method):
    case = saved_example
    path = case.directory/(method+'.npz')
    with np.load(path, allow_pickle=False) as saved:
        data = {name: saved[name].copy() for name in saved.files}
    data['converged_mask'][0, 0] = False
    np.savez_compressed(path, **data)
    with pytest.raises(ArithmeticError, match='incomplete source QP'):
        case.example.analyze(case.directory, path_points=2, dos_kmesh=(1, 1, 1),
            axis_ev=np.linspace(-50., 50., 101), frequency_cutoff=2.01)
    assert not case.plotted
    assert not (case.directory/'comparison.npz').exists()
