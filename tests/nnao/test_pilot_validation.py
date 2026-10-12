"""Pilot orchestration tests deliberately avoid optional numerical imports."""
from importlib import import_module
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest


def module():
    return vars(import_module("gradscf.model.nnao.training.validation"))


def structure(identifier='W4-11/h2'):
    return dict(structure_id=identifier, symbols=['H', 'H'], charge=0, spin=0,
                coords_angstrom=[[0., 0., 0.], [0., 0., .74]])


def test_pilot_rejects_unknown_ids_and_open_shell(tmp_path):
    read = import_module("gradscf.model.nnao.runtime").read_structures
    source = tmp_path / 'structures.jsonl'
    source.write_text(json.dumps(structure()) + '\n')
    with pytest.raises(ValueError, match='Unknown structure IDs'):
        read(source, ids=['unknown'])
    record = structure(); record['spin'] = 2
    source.write_text(json.dumps(record) + '\n')
    with pytest.raises(ValueError, match='neutral spin=0'):
        read(source)


def test_pilot_preserves_dataset_order_and_limits(tmp_path):
    read = import_module("gradscf.model.nnao.runtime").read_structures
    source = tmp_path / 'structures.jsonl'
    source.write_text('\n'.join(json.dumps(structure(x)) for x in ['b', 'a', 'c']))
    rows = read(source, ids=['a', 'b'], limit=1)
    assert [r['structure_id'] for r in rows] == ['b']
    assert rows[0]['name'] == 'b'


def test_pilot_records_failure_and_resumes_completed_structure(tmp_path):
    run = module()['run_suite']
    calls = []

    def validate(row, path):
        calls.append(row['structure_id'])
        if row['structure_id'] == 'bad':
            raise RuntimeError('SCF did not converge')
        return dict(converged=True, energy_hartree=-1.)

    rows = [structure('good'), structure('bad')]
    summary = run(rows, tmp_path, dict(seed=0), validate, resume=True)
    assert summary['exit_code'] == 1
    assert summary['passed'] == 1 and summary['failed'] == 1
    assert summary['results'][1]['error'] == 'SCF did not converge'
    assert (tmp_path / 'summary.json').is_file()
    run(rows, tmp_path, dict(seed=0), validate, resume=True)
    assert calls == ['good', 'bad', 'bad']


def test_pilot_invalidates_resume_when_seed_or_structure_changes(tmp_path):
    run = module()['run_suite']; calls = []

    def validate(row, path):
        calls.append(row['structure_id'])
        return dict(converged=True)

    row = structure()
    run([row], tmp_path, dict(seed=0), validate, resume=True)
    run([row], tmp_path, dict(seed=1), validate, resume=True)
    row['coords_angstrom'][1][2] = .75
    run([row], tmp_path, dict(seed=1), validate, resume=True)
    assert len(calls) == 3


def test_pilot_summary_rejects_nonfinite_success(tmp_path):
    run = module()['run_suite']
    summary = run([structure()], tmp_path, dict(seed=0),
                  lambda row, path: dict(energy_hartree=float('nan')), resume=False)
    assert summary['exit_code'] == 1
    assert summary['failed'] == 1
    assert 'Out of range float' in summary['results'][0]['error']


def test_pilot_does_not_resume_missing_or_modified_cache(tmp_path):
    run = module()['run_suite']; cache_hash = import_module("gradscf.model.nnao.runtime").file_hash; calls = []
    cache = tmp_path / 'integrals.npz'

    def prepare(row, path):
        calls.append(row['structure_id']); cache.write_bytes(b'valid cache')
        return dict(cache_path=str(cache), cache_sha256=cache_hash(cache))

    row = structure()
    run([row], tmp_path / 'results', {}, prepare, resume=True)
    cache.write_bytes(b'modified cache')
    run([row], tmp_path / 'results', {}, prepare, resume=True)
    cache.unlink()
    run([row], tmp_path / 'results', {}, prepare, resume=True)
    assert len(calls) == 3


def test_directional_check_detects_incorrect_gradient():
    np = pytest.importorskip('numpy')
    check = module()['directional_check']
    point = np.array([.2, -.3]); direction = np.array([.6, .8])
    energy = lambda x: (float(np.sum(x**4)),)
    correct = check(energy, point, 4 * point**3, direction,
                    step=1e-4, atol=2e-6, rtol=2e-5)
    incorrect = check(energy, point, np.zeros(2), direction,
                      step=1e-4, atol=2e-6, rtol=2e-5)
    assert correct['passed'] and not incorrect['passed']
    assert len(correct['perturbed_energies_hartree']) == 4


def test_scientific_source_signature_changes_with_kernel_or_basis(tmp_path):
    signature = import_module("gradscf.model.nnao.runtime").scientific_source_hash
    package = tmp_path / 'src' / 'gradscf'
    package.mkdir(parents=True)
    source = package / 'scf.py'; source.write_text('energy = 1\n')
    basis = package / 'basis.dat'; basis.write_text('1.0 0.5\n')
    initial = signature(tmp_path)
    source.write_text('energy = 2\n')
    changed_kernel = signature(tmp_path)
    basis.write_text('2.0 0.5\n')
    assert len({initial, changed_kernel, signature(tmp_path)}) == 3


def test_pilot_rejects_installed_copy_even_when_other_modules_are_current(tmp_path):
    verify = import_module("gradscf.model.nnao.runtime").verify_source_imports
    paths = {'gradscf': '__init__.py', 'gradscf.scf.rks': 'scf/rks.py',
             'gradscf.solvers.nonlinear.fixed_point': 'solvers/nonlinear/fixed_point.py'}
    imported = {}
    for name, relative in paths.items():
        path = tmp_path / 'src' / 'gradscf' / relative
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text('# current source\n')
        imported[name] = SimpleNamespace(__file__=str(path))
    confirmed = verify(tmp_path, importer=imported.__getitem__)
    assert confirmed['gradscf.scf.rks']['path'] == str((tmp_path / 'src/gradscf/scf/rks.py').resolve())
    installed = tmp_path / 'installed' / 'gradscf' / 'scf' / 'rks.py'
    installed.parent.mkdir(parents=True); installed.write_text('# old copy\n')
    imported['gradscf.scf.rks'] = SimpleNamespace(__file__=str(installed))
    with pytest.raises(RuntimeError, match='Source mismatch for gradscf.scf.rks'):
        verify(tmp_path, importer=imported.__getitem__)


def test_pilot_cli_forwards_and_records_implicit_tolerance(tmp_path, monkeypatch):
    main = import_module('gradscf.model.nnao.workflows.validate').main; namespace = main.__globals__; calls = []
    source = tmp_path / 'structures.jsonl'
    source.write_text(json.dumps(structure()) + '\n')
    monkeypatch.setitem(namespace, 'numerical_runtime', lambda platform: (None, None))
    monkeypatch.setitem(namespace, 'prepare_cache', lambda row, path, **kwargs:
                        calls.append(kwargs) or {})
    output = tmp_path / 'results'
    assert main(['prepare-cache', '--structures', str(source), '--cache-dir',
                 str(tmp_path / 'cache'), '--output-dir', str(output),
                 '--implicit-tolerance', '1e-8']) == 0
    assert calls[0]['implicit_tolerance'] == 1e-8
    summary = json.loads((output / 'summary.json').read_text())
    assert summary['config']['implicit_tolerance'] == 1e-8
    main(['prepare-cache', '--structures', str(source), '--cache-dir',
          str(tmp_path / 'cache'), '--output-dir', str(output)])
    assert calls[1]['implicit_tolerance'] == 1e-9
    assert len(calls) == 2  # A changed tolerance must invalidate resume.


@pytest.mark.parametrize('tolerance', ['0', '-1', 'nan', 'inf'])
def test_pilot_cli_rejects_invalid_implicit_tolerance(tmp_path, tolerance, capsys):
    with pytest.raises(SystemExit) as error:
        import_module('gradscf.model.nnao.workflows.validate').main(['validate', '--structures', str(tmp_path / 'missing'),
            '--cache-dir', str(tmp_path), '--output-dir', str(tmp_path),
            '--implicit-tolerance', tolerance])
    assert error.value.code == 2
    assert 'implicit tolerance must be finite and positive' in capsys.readouterr().err.lower()


def test_pilot_cli_forwards_mmap_for_validation(tmp_path,monkeypatch):
    main=import_module('gradscf.model.nnao.workflows.validate').main;namespace=main.__globals__;calls=[]
    source=tmp_path/'structures.jsonl';source.write_text(json.dumps(structure())+'\n')
    monkeypatch.setitem(namespace,'numerical_runtime',lambda platform:(None,None))
    monkeypatch.setitem(namespace,'validate',lambda row,path,**kwargs:calls.append(kwargs) or {})
    output=tmp_path/'results'
    assert main(['validate','--structures',str(source),'--cache-dir',str(tmp_path),
        '--output-dir',str(output),'--cache-storage','mmap','--scf-rescue-level-shift','0.2'])==0
    assert calls[0]['cache_storage']=='mmap'
    assert calls[0]['scf_rescue_level_shift']==.2
    assert json.loads((output/'summary.json').read_text())['config']['cache_storage']=='mmap'
    assert json.loads((output/'summary.json').read_text())['config']['scf_rescue_level_shift']==.2


def test_pilot_cache_preparation_rejects_host_or_mmap(tmp_path,capsys):
    with pytest.raises(SystemExit):
        import_module('gradscf.model.nnao.workflows.validate').main(['prepare-cache','--structures',str(tmp_path/'missing'),
            '--cache-dir',str(tmp_path),'--output-dir',str(tmp_path),'--cache-storage','mmap'])
    assert 'requires --cache-storage device' in capsys.readouterr().err
