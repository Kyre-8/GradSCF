"""BerkeleyGW's binary WFN records and plane-wave normalization contract."""
import struct
from itertools import product

import numpy as np
import pytest


def _records(path):
    records = []
    with open(path, 'rb') as stream:
        while marker := stream.read(4):
            length, = struct.unpack('<i', marker)
            body = stream.read(length)
            assert stream.read(4) == marker
            records.append(body)
    return records


def _input():
    mesh = (3, 3, 3)
    frac = np.stack(np.meshgrid(*[np.arange(n)/n for n in mesh], indexing='ij'), axis=-1).reshape(-1, 3)
    values = np.stack([np.ones(len(frac)), np.exp(2j*np.pi*frac[:, 0])], axis=-1)[None]/np.sqrt(8.)
    return dict(lattice_bohr=np.eye(3)*2., mesh=mesh,
        kpoints_fractional=np.zeros((1, 3)), kgrid=(1, 1, 1),
        energies_hartree=np.array([[-.4, .2]]), occupations=np.array([[1., 0.]]),
        periodic_values=values, atom_numbers=[2], atom_positions_bohr=np.zeros((1, 3)))


def test_wfn_binary_uses_rydberg_headers_and_normalized_fourier_coefficients(tmp_path):
    from gradscf_tools.berkeleygw import write_wfn

    path = tmp_path/'WFN'
    report = write_wfn(path, **_input())
    records = _records(path)
    assert records[0][:11] == b'WFN-Complex'
    # Header layout follows BGW Common/wfn_rho_vxc_io_inc.f90, not an
    # application-specific read-back helper.
    ns, ng, ntran, symmetry, nat, ecutrho, nk, nb, ngmax, ecut = struct.unpack('<5id3id', records[1])
    assert (ns, ntran, symmetry, nat, nk, nb, ngmax) == (1, 1, 0, 1, 1, 2, 27)
    assert ng >= 27 and ecutrho >= ecut > 0
    np.testing.assert_allclose(np.frombuffer(records[13], '<f8'), [-.8, .4], atol=0, rtol=0)
    np.testing.assert_array_equal(np.frombuffer(records[14], '<f8'), [1., 0.])
    # Each G/data block consists of nrecord=1, length, then payload.
    assert struct.unpack('<i', records[15]) == (1,)
    assert struct.unpack('<i', records[16]) == (ng,)
    gvecs = np.frombuffer(records[20], '<i4').reshape(-1, 3)
    first = np.frombuffer(records[23], '<c16')
    second = np.frombuffer(records[26], '<c16')
    np.testing.assert_allclose(np.sum(abs(first)**2), 1., atol=1e-13)
    np.testing.assert_allclose(np.sum(abs(second)**2), 1., atol=1e-13)
    np.testing.assert_allclose(first[gvecs.tolist().index([0, 0, 0])], 1., atol=1e-13)
    np.testing.assert_allclose(second[gvecs.tolist().index([1, 0, 0])], 1., atol=1e-13)
    assert report['max_orthonormality_error'] < 1e-13


def test_wfn_rejects_nonorthonormal_states_instead_of_renormalizing(tmp_path):
    from gradscf_tools.berkeleygw import write_wfn

    data = _input()
    data['periodic_values'] *= 2
    with pytest.raises(ValueError, match='orthonormal'):
        write_wfn(tmp_path/'WFN', **data)
    assert not (tmp_path/'WFN').exists()


def test_shifted_and_source_wfn_can_share_one_declared_cutoff(tmp_path):
    from gradscf_tools.berkeleygw import write_wfn

    data = _input()
    write_wfn(tmp_path/'WFN', **data, wavefunction_cutoff_rydberg=40.)
    data['kpoints_fractional'] = np.array([[0., 0., .001]])
    write_wfn(tmp_path/'WFNq', **data, shift=(0., 0., .001),
              wavefunction_cutoff_rydberg=40.)
    source, shifted = _records(tmp_path/'WFN'), _records(tmp_path/'WFNq')
    assert struct.unpack('<5id3id', source[1])[-1] == 40.
    assert struct.unpack('<5id3id', shifted[1])[-1] == 40.
    # Changing only the common header cutoff must preserve every coefficient.
    assert source[23:] == shifted[23:]


def test_wfn_pads_a_complete_kinetic_sphere_with_exact_zero_coefficients(tmp_path):
    from gradscf_tools.berkeleygw import write_wfn

    write_wfn(tmp_path/'WFN', **_input(), wavefunction_cutoff_rydberg=40.)
    records = _records(tmp_path/'WFN')
    vectors = np.frombuffer(records[20], '<i4').reshape(-1, 3)
    expected = {g for g in product(range(-2, 3), repeat=3) if np.pi**2*sum(n*n for n in g) <= 40.}
    assert set(map(tuple, vectors)) == expected
    assert len(vectors) == 33
    first = np.frombuffer(records[23], '<c16')
    second = np.frombuffer(records[26], '<c16')
    added = np.any(abs(vectors) > 1, axis=1)
    np.testing.assert_array_equal(first[added], 0.)
    np.testing.assert_array_equal(second[added], 0.)
    np.testing.assert_allclose(np.vdot(first, first), 1., atol=1e-13)
    np.testing.assert_allclose(np.vdot(second, second), 1., atol=1e-13)


def test_vxc_dat_uses_ev_and_one_based_band_spin_indices(tmp_path):
    from gradscf_tools.berkeleygw import write_vxc

    path = tmp_path/'vxc.dat'
    write_vxc(path, np.array([[.25, 0., 0.]]), np.array([[-.2, -.1]]))
    rows = [line.split() for line in path.read_text().splitlines()]
    assert rows[0] == ['0.250000000000', '0.000000000000', '0.000000000000', '2', '0']
    assert rows[1][:2] == ['1', '1'] and rows[2][:2] == ['1', '2']
    np.testing.assert_allclose(float(rows[1][2]), -.2*27.211386245988, atol=1e-11)
    assert float(rows[1][3]) == 0.


def test_export_scf_retains_same_source_orbitals_and_xc(tmp_path):
    from gradscf_tools.berkeleygw import export_scf
    from examples.pbc.band_structures.gw_bands import reference

    mf = reference('gaas')
    report = export_scf(mf, tmp_path)
    assert report['nmo'] == 14 and report['nocc'] == 9
    assert report['max_orthonormality_error'] < 1e-8
    assert report['kgrid'] == [2, 1, 1]
    assert report['orbital_fft_mesh'] == [21, 21, 21]
    assert (tmp_path/'WFN').is_file() and (tmp_path/'vxc.dat').is_file()
    assert report['max_vxc_imaginary_hartree'] < 1e-10
