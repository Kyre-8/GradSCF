"""Dataset-selection regressions without optional numerical dependencies."""
import hashlib
import json
from pathlib import Path
import runpy

import pytest


def selector():
    from gradscf.model.nnao.workflows.select import select_structures
    return select_structures


def structure(identifier, symbols):
    coordinates = [[i * 1.1, 0., 0.] for i in range(len(symbols))]
    return dict(structure_id=identifier, subset='fixture', symbols=list(symbols),
        natoms=len(symbols), charge=0, spin=0,
        nelectron=sum({'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}[s] for s in symbols),
        coords_angstrom=coordinates, referenced_by_retained_reaction=True,
        coordinate_signature_sha256=hashlib.sha256(
            json.dumps([symbols, coordinates]).encode()).hexdigest())


def conformers():
    rows = []
    for nh in range(2, 12, 2):
        for conformer in range(3):
            row = structure(f'C2H{nh}/{conformer}', ['C', 'C'] + ['H'] * nh)
            row['coords_angstrom'][1][1] = conformer * .1
            row['coordinate_signature_sha256'] = hashlib.sha256(
                json.dumps(row['coords_angstrom']).encode()).hexdigest()
            rows.append(row)
    return rows


def test_stage2_group_split_is_reproducible_and_excludes_seen_formulas():
    rows = conformers(); pilot = [rows[0]]
    quotas = {'2-12': {'train': 4, 'validation': 2}}
    result = selector()(rows, pilot, quotas=quotas, seed=0)
    repeated = selector()(list(reversed(rows)), pilot, quotas=quotas, seed=0)
    assert result == repeated
    train = result['train']; validation = result['validation']
    assert len(train) == 4 and len(validation) == 2
    formula = lambda row: tuple(sorted(row['symbols']))
    assert not {formula(row) for row in train} & {formula(row) for row in validation}
    assert all(formula(row) != formula(pilot[0]) for row in validation)
    assert max(sum(formula(other) == formula(row) for other in train) for row in train) <= 2
    changed = selector()(rows, pilot, quotas=quotas, seed=1)
    assert changed['validation'] != validation


def test_stage2_deduplicates_coordinates_and_applies_df_budget():
    row = structure('original', ['C', 'C', 'H', 'H'])
    duplicate = dict(row, structure_id='duplicate')
    large = structure('oversized', ['C'] * 26 + ['H'] * 12)
    result = selector()([row, duplicate, large], [],
        quotas={'2-12': {'train': 1, 'validation': 0}}, max_cache_gib=6.)
    assert len(result['train']) == 1
    assert result['stats']['excluded']['duplicate_coordinate'] == 1
    assert result['stats']['excluded']['df_cache_budget'] == 1
    estimate = result['train'][0]['size_estimate']
    assert estimate['nao'] == 28 and estimate['primitive_nao'] == 114
    assert estimate['naux'] == 186
    assert estimate['df_bytes'] == 8 * 186 * 114 * 115 // 2


def test_stage2_rejects_unqualified_structures_and_fails_insufficient_quotas():
    good = structure('good', ['H', 'H'])
    odd = structure('odd', ['C', 'H'])
    charged = dict(good, structure_id='charged', charge=1)
    unreferenced = dict(good, structure_id='unreferenced', referenced_by_retained_reaction=False)
    overlap = dict(good, structure_id='overlap', coords_angstrom=[[0., 0., 0.], [0., 0., 0.]])
    nonfinite = dict(good, structure_id='nonfinite', coords_angstrom=[[0., 0., 0.], [float('nan'), 0., 0.]])
    result = selector()([odd, charged, unreferenced, overlap, nonfinite], [], quotas={})
    assert result['stats']['excluded'] == dict(odd_electron_count=1,
        charge_or_spin=1, not_referenced=1, invalid_coordinates=2)
    with pytest.raises(ValueError, match='Insufficient.*2-12'):
        selector()([good], [], quotas={'2-12': {'train': 1, 'validation': 1}})
