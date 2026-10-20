"""Public GaAs workflow rejects a coarse reciprocal-inequivalent reference."""
import pytest


def test_gaas_calculate_rejects_coarse_fft_before_building_gw(monkeypatch, tmp_path):
    from examples.pbc.band_structures import gaas_self_consistent_gw as example

    def forbidden_model(*args, **kwargs):
        pytest.fail('coarse reference reached GW')

    monkeypatch.setattr(example, 'build_periodic_model', forbidden_model)
    # SCF and query bands are real native calculations; only the GW boundary
    # is a sentinel. No self-energy or dielectric model may be built here.
    with pytest.raises(ArithmeticError, match=r'(?i)FFT|reciprocal|mesh'):
        example.calculate('g0w0', tmp_path, source_kmesh=(2, 1, 1),
                          mesh=(21, 21, 21))
