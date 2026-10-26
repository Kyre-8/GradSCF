"""Independent BerkeleyGW text-format fixtures; no electronic calculation."""
import numpy as np
import pytest


def _spectrum(path, *, duplicate=False):
    # CD spectrum: k(3), spin, one-based band, iw, energy, then four complex
    # channels: X+Cor, Res, Cor, Int. All energies are eV.
    path.write_text('# CD spectrum\n'
        '0 0 0 1 9 1 0 27.211386245988 2.7211386245988 0 0 13.605693122994 0 13.605693122994 0\n'
        + ('0 0 0 1 9 2 0' if duplicate else '0 0 0 1 9 2 27.211386245988')
        + ' 54.422772491976 0 0 0 40.817079368982 0 40.817079368982 0\n')


def test_read_sigma_preserves_complex_channels_and_converts_ev_to_hartree(tmp_path):
    from gradscf_tools.berkeleygw import read_sigma

    path = tmp_path/'spectrum.dat'
    _spectrum(path)
    records = read_sigma(path)
    assert len(records) == 1
    row = records[0]
    assert row['band'] == 8 and row['spin'] == 1
    np.testing.assert_array_equal(row['kpoint_fractional'], [0., 0., 0.])
    np.testing.assert_allclose(row['frequency_hartree'], [0., 1.], atol=1e-14)
    np.testing.assert_allclose(row['sigma_total_hartree'], [1.+.1j, 2.], atol=1e-14)
    np.testing.assert_allclose(row['sigma_correlation_hartree'], [.5, 1.5], atol=1e-14)
    np.testing.assert_allclose(row['sigma_integral_hartree'], [.5, 1.5], atol=1e-14)


def test_read_sigma_directory_attaches_actual_exchange_and_vxc(tmp_path):
    from gradscf_tools.berkeleygw import read_sigma

    _spectrum(tmp_path/'spectrum.dat')
    (tmp_path/'x.dat').write_text('0 0 0 1 0\n1 9 -27.211386245988 0\n')
    (tmp_path/'vxc.dat').write_text('0 0 0 1 0\n1 9 -54.422772491976 0\n')
    row, = read_sigma(tmp_path)
    assert row['exchange_hartree'] == -1.
    assert row['vxc_hartree'] == -2.


def test_read_sigma_preserves_reported_multiple_root_flag(tmp_path):
    from gradscf_tools.berkeleygw import read_sigma

    _spectrum(tmp_path/'spectrum.dat')
    (tmp_path/'x.dat').write_text('0 0 0 1 0\n1 9 -27.211386245988 0\n')
    (tmp_path/'vxc.dat').write_text('0 0 0 1 0\n1 9 -54.422772491976 0\n')
    (tmp_path/'sigma_hp.log').write_text('k = 0 0 0 ik = 1 spin = 1\n'
        '9 0 0 0 0 0 0 0 0 0 0 MULT:3\n')
    row, = read_sigma(tmp_path)
    assert row['reported_qp_flag'] == 'MULT:3'


def test_read_sigma_rejects_duplicate_frequency_samples(tmp_path):
    from gradscf_tools.berkeleygw import read_sigma

    path = tmp_path/'spectrum.dat'
    _spectrum(path, duplicate=True)
    with pytest.raises(ValueError, match='increasing'):
        read_sigma(path)


def test_read_eqp_preserves_external_reference_and_raw_complex_qp(tmp_path):
    from gradscf_tools.berkeleygw import read_eqp

    path = tmp_path/'eqp1.dat'
    path.write_text('0.5 0 0 2\n'
        '1 9 27.211386245988 54.422772491976 2.7211386245988\n'
        '1 10 54.422772491976 81.634158737964 0\n')
    rows = read_eqp(path)
    assert [row['band'] for row in rows] == [8, 9]
    assert rows[0]['reference_hartree'] == 1.
    np.testing.assert_allclose(rows[0]['qp_hartree'], 2.+.1j, atol=1e-14)
    assert rows[1]['reference_hartree'] == 2.


def test_read_eqp_rejects_truncated_blocks(tmp_path):
    from gradscf_tools.berkeleygw import read_eqp

    path = tmp_path/'eqp1.dat'
    path.write_text('0 0 0 2\n1 9 0 0 0\n')
    with pytest.raises(ValueError, match='truncated'):
        read_eqp(path)


def test_readers_honor_an_explicit_build_energy_constant(tmp_path):
    from gradscf_tools.berkeleygw import read_eqp, read_sigma

    eqp = tmp_path/'eqp.dat'
    eqp.write_text('0 0 0 1\n1 9 10 20 1\n')
    row, = read_eqp(eqp, hartree_to_ev=10.)
    assert row['reference_hartree'] == 1.
    assert row['qp_hartree'] == 2.+.1j
    spectrum = tmp_path/'spectrum.dat'
    _spectrum(spectrum)
    row, = read_sigma(spectrum, hartree_to_ev=10.)
    np.testing.assert_allclose(row['frequency_hartree'][1], 2.7211386245988, atol=1e-14)


@pytest.mark.parametrize('band', [9, 12])
def test_read_sigma_accepts_full_width_adjacent_band_and_frequency_fields(tmp_path, band):
    from gradscf_tools.berkeleygw import read_sigma

    # Independent BGW shiftenergy_dyn.f90 format3000:
    # 3F12.5,2x,3I4,2x,9F12.5. Iw1000 fills I4 and joins the band.
    prefix = ''.join(f'{x:12.5f}' for x in [0., 0., 0.])+'  '
    rows = []
    for iw, energy, real, imaginary in [(999, 1., 3., .25), (1000, 2., 4., -.5)]:
        rows.append(prefix+f'{1:4d}{band:4d}{iw:4d}'+'  '
            +''.join(f'{x:12.5f}' for x in [energy, real, imaginary, 5., 6., 7., 8., 9., 10.]))
    assert len(rows[1].split()) == 14
    path = tmp_path/'spectrum.dat'
    path.write_text('# CD fixed-width fixture\n'+'\n'.join(rows)+'\n')
    record, = read_sigma(path, hartree_to_ev=1.)
    assert record['band'] == band-1 and record['spin'] == 1
    np.testing.assert_array_equal(record['frequency_hartree'], [1., 2.])
    np.testing.assert_array_equal(record['sigma_total_hartree'], [3.+.25j, 4.-.5j])
    np.testing.assert_array_equal(record['sigma_correlation_hartree'], [7.+8j, 7.+8j])
