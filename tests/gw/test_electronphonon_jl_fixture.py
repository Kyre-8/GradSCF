"""Offline numerical data from the pinned original Julia linewidth routine."""

import hashlib
import json
from pathlib import Path

import numpy as np

from gradscf.gw.ep_coupling import fan_linewidth


def test_electronphonon_jl_gaussian_linewidth_fixture():
    folder = Path(__file__).with_name('data') / 'electronphonon_jl'
    data = json.loads((folder / 'inputs.json').read_text())
    expected = json.loads((folder / 'results.json').read_text())
    provenance = json.loads((folder / 'provenance.json').read_text())
    assert provenance['upstream_commit'] == 'a0ecee01b7413af5566134d94cad5b15431d6565'
    for name in ('inputs', 'results'):
        assert hashlib.sha256((folder / f'{name}.json').read_bytes()).hexdigest() == provenance[f'{name}_sha256']
    computed = []
    for q, weight in enumerate(data['q_weights']):
        vertex = np.sqrt(data['g_squared_ha2_q_internal_external_mode'][q]).transpose(2, 1, 0)
        rows = []
        for temperature in data['temperature_kbt_ha']:
            # This fixture has no level exactly at mu; finite beta underflows
            # to the exact zero-temperature occupations used by Julia.
            beta = 1e8 if temperature == 0 else 1 / temperature
            rows.append(weight * fan_linewidth(data['external_energy_ha'],
                data['internal_energy_ha'][q], data['phonon_energy_ha'][q], vertex,
                mu=data['chemical_potential_ha'], beta=beta,
                eta=data['gaussian_width_ha'], smearing='gaussian'))
        computed.append(rows)
    computed = np.asarray(computed)
    np.testing.assert_allclose(computed, expected['fullwidth_ha_q_temperature_external'], atol=1e-13, rtol=0)
    np.testing.assert_allclose(computed.sum(axis=0), expected['fullwidth_ha_temperature_external'], atol=1e-13, rtol=0)
