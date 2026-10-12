"""Shared-training control tests use analytical energies and a NumPy Adam."""
from importlib import import_module
import json
from pathlib import Path
import runpy

import numpy as np
import pytest


def module():
    return vars(import_module("gradscf.model.nnao.training.trainer"))


def info():
    return dict(converged=True, orbital_residual=1e-10,
                min_overlap_eigenvalue=.1, reconstruction_error=1e-13,
                scf_cycles=8,fixed_point_residual=1e-12)


def test_training_rejects_non_aufbau_stationary_scf():
    details=info();details['fixed_point_residual']=.5
    assert not module()['valid_scf'](details)


def evaluators():
    def energy(center):
        def evaluate(vector):
            delta = vector - center
            return float(np.dot(delta, delta)), 2 * delta, info()
        return evaluate
    return [('a', energy(np.array([1., 2.]))), ('b', energy(np.array([-2., 0.])))]


class Adam:
    """Serializes moments and counter, rather than restarting at checkpoints."""
    def init(self, vector):
        return dict(count=0, m=np.zeros_like(vector), v=np.zeros_like(vector))

    def update(self, vector, gradient, state):
        gradient = gradient / max(1., np.linalg.norm(gradient))
        count = state['count'] + 1
        m = .9 * state['m'] + .1 * gradient
        v = .999 * state['v'] + .001 * gradient**2
        updated = vector - .03 * (m / (1 - .9**count)) / (np.sqrt(v / (1 - .999**count)) + 1e-8)
        return updated, dict(count=count, m=m, v=v)

    def encode(self, state):
        return json.dumps({key: value.tolist() if isinstance(value, np.ndarray) else value
                           for key, value in state.items()}).encode()

    def decode(self, data, template):
        state = json.loads(data)
        return dict(count=state['count'], m=np.array(state['m']), v=np.array(state['v']))


def test_shared_objective_uses_same_parameters_and_equal_mean_gradient():
    batch = module()['evaluate_batch']
    vector = np.array([.4, -.2])
    values = [evaluate(vector) for _, evaluate in evaluators()]
    loss, gradient, records = batch(vector, evaluators())
    assert loss == pytest.approx(np.mean([result[0] for result in values]))
    np.testing.assert_allclose(gradient, np.mean([result[1] for result in values], axis=0))
    assert [row['structure_id'] for row in records] == ['a', 'b']


@pytest.mark.parametrize('invalid', ['nan', 'unconverged', 'residual'])
def test_invalid_sample_stops_batch_without_silent_filtering(invalid):
    batch = module()['evaluate_batch']
    calls = []
    def bad(vector):
        details = info()
        gradient = np.ones_like(vector)
        if invalid == 'nan': gradient[0] = np.nan
        if invalid == 'unconverged': details['converged'] = False
        if invalid == 'residual': details['orbital_residual'] = 1e-4
        return -1., gradient, details
    def later(vector):
        calls.append(True)
        return 0., vector, info()
    with pytest.raises(RuntimeError, match='bad'):
        batch(np.zeros(2), [('bad', bad), ('later', later)])
    assert not calls


def test_checkpoint_loss_and_optimizer_state_match_accepted_parameters(tmp_path):
    api = module(); batch = api['evaluate_batch']; run = api['run_training']
    initial = np.zeros(2)
    run(initial, lambda vector: batch(vector, evaluators()), Adam(), tmp_path,
        identity={'dataset': 'same'}, epochs=3)
    checkpoint = api['load_checkpoint'](tmp_path / 'last.npz', {'dataset': 'same'}, Adam(), initial)
    expected_loss, expected_gradient, records = batch(checkpoint['parameters'], evaluators())
    assert checkpoint['epoch'] == 3 and checkpoint['optimizer']['count'] == 3
    assert checkpoint['loss_hartree'] == pytest.approx(expected_loss)
    np.testing.assert_allclose(checkpoint['gradient'], expected_gradient)
    assert checkpoint['molecules'] == records
    assert checkpoint['history'][-1]['epoch'] == 3
    progress = json.loads((tmp_path / 'progress.json').read_text())
    assert progress['status'] == 'complete' and progress['accepted_epoch'] == 3
    best = api['load_checkpoint'](tmp_path / 'best.npz', {'dataset': 'same'}, Adam(), initial)
    assert best['loss_hartree'] == min(row['loss_hartree'] for row in checkpoint['history'])


def test_resume_preserves_adam_moments_and_update_order(tmp_path):
    api = module(); evaluate = lambda vector: api['evaluate_batch'](vector, evaluators())
    full = tmp_path / 'full'; interrupted = tmp_path / 'interrupted'
    identity = {'dataset': 'same', 'seed': 0}
    api['run_training'](np.zeros(2), evaluate, Adam(), full, identity=identity, epochs=8)
    api['run_training'](np.zeros(2), evaluate, Adam(), interrupted, identity=identity, epochs=3)
    api['run_training'](np.zeros(2), evaluate, Adam(), interrupted,
                        identity=identity, epochs=8, resume=True)
    first = api['load_checkpoint'](full / 'last.npz', identity, Adam(), np.zeros(2))
    second = api['load_checkpoint'](interrupted / 'last.npz', identity, Adam(), np.zeros(2))
    np.testing.assert_array_equal(first['parameters'], second['parameters'])
    np.testing.assert_array_equal(first['optimizer']['m'], second['optimizer']['m'])
    np.testing.assert_array_equal(first['optimizer']['v'], second['optimizer']['v'])
    assert first['optimizer']['count'] == second['optimizer']['count'] == 8
    assert first['history'] == second['history']
    with pytest.raises(ValueError, match='identity'):
        api['load_checkpoint'](interrupted / 'last.npz', {'dataset': 'changed'}, Adam(), np.zeros(2))


def test_failed_update_keeps_last_valid_checkpoint_and_records_structure(tmp_path):
    api = module(); calls = []
    def evaluate(vector):
        calls.append(vector.copy())
        if len(calls) == 3: raise RuntimeError('bad: SCF failed')
        return api['evaluate_batch'](vector, evaluators())
    result = api['run_training'](np.zeros(2), evaluate, Adam(), tmp_path, identity={}, epochs=8)
    assert result['status'] == 'failed' and result['accepted_epoch'] == 1
    assert 'bad: SCF failed' in result['error']
    checkpoint = api['load_checkpoint'](tmp_path / 'last.npz', {}, Adam(), np.zeros(2))
    assert checkpoint['epoch'] == 1 and checkpoint['optimizer']['count'] == 1


@pytest.mark.parametrize('fail_at', [1, 3])
def test_failed_candidate_snapshot_survives_batch_wrapper_without_accepting_update(tmp_path, fail_at):
    api = module(); initial = np.zeros(2); calls = []; before = {}
    outputs = np.array([[[.2, .3]]]); coefficient_gradient = np.array([[[np.nan, np.inf]]])
    def evaluate(vector):
        calls.append(np.array(vector))
        if len(calls) == fail_at:
            for name in ('last.npz', 'best.npz', 'best_validation.npz'):
                path = tmp_path / name
                if path.exists(): before[name] = path.read_bytes()
            error = RuntimeError('Nonfinite energy or gradient')
            error.failure_arrays = dict(basis_outputs=outputs, coefficient_gradient=coefficient_gradient)
            error.failure_details = dict(energy_hartree=-1., scf=info())
            raise error
        delta = vector - 1.
        return float(np.dot(delta, delta)), 2 * delta, info()
    batch = lambda vector: api['evaluate_batch'](vector, [('ADIM6/AD6', evaluate)])
    heldout = lambda vector: (1., [dict(structure_id='held', energy_hartree=1., **info())])
    result = api['run_training'](initial, batch, Adam(), tmp_path, identity={'dataset': 'same'},
                                 epochs=8, heldout=heldout)
    assert result['status'] == 'failed' and result['accepted_epoch'] == (None if fail_at == 1 else 1)
    assert 'failure_snapshot' in result, 'The failed trial parameters and raw backward were not saved.'
    with np.load(tmp_path / result['failure_snapshot'], allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['parameters'], calls[-1])
        np.testing.assert_array_equal(saved['basis_outputs'], outputs)
        np.testing.assert_array_equal(saved['coefficient_gradient'], coefficient_gradient)
        metadata = json.loads(str(saved['metadata'].item()))
    assert metadata['structure_id'] == 'ADIM6/AD6'
    assert metadata['details'] == dict(energy_hartree=-1., scf=info())
    assert metadata['attempted_epoch'] == (0 if fail_at == 1 else 2)
    assert metadata['accepted_epoch'] == result['accepted_epoch']
    assert metadata['identity'] == {'dataset': 'same'}
    if fail_at == 1:
        assert not (tmp_path / 'last.npz').exists()
    else:
        for name, data in before.items(): assert (tmp_path / name).read_bytes() == data
        accepted = api['load_checkpoint'](tmp_path / 'last.npz', {'dataset': 'same'}, Adam(), initial)
        assert accepted['epoch'] == 1 and accepted['optimizer']['count'] == 1
        assert not np.array_equal(accepted['parameters'], calls[-1])
    assert not list(tmp_path.glob('*.tmp'))


def test_training_gate_requires_every_selected_record_and_same_tolerance(tmp_path):
    verify = module()['verify_validation']; source = tmp_path / 'structures.jsonl'
    source.write_text('{"structure_id": "a"}\n')
    import hashlib
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    summary = dict(passed=1, failed=0, exit_code=0,
        config=dict(structures_sha256=digest, implicit_tolerance=1e-8),
        results=[dict(structure_id='a', status='passed',
                      implicit_config=dict(tolerance=1e-8),
                      coefficient_directional_check={'passed': True},
                      parameter_directional_check={'passed': True},
                      accepted_adam_steps=5, implicit_gradient_finite=True,
                      initial_scf=info(),
                      adam_history=[dict(step=step, energy_hartree=-1., **info())
                                    for step in range(6)])])
    path = tmp_path / 'summary.json'; path.write_text(json.dumps(summary))
    verify(path, source, ['a'], tolerance=1e-8)
    with pytest.raises(ValueError, match='structures'):
        verify(path, source, ['a', 'b'], tolerance=1e-8)
    with pytest.raises(ValueError, match='tolerance'):
        verify(path, source, ['a'], tolerance=1e-9)
    summary['failed'] = 1; path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='validation'):
        verify(path, source, ['a'], tolerance=1e-8)
    summary['failed'] = 0
    summary['results'][0]['parameter_directional_check']['passed'] = False
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='gradient'):
        verify(path, source, ['a'], tolerance=1e-8)
    summary['results'][0]['parameter_directional_check']['passed'] = True
    summary['results'][0]['adam_history'][3]['orbital_residual'] = 1e-4
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='SCF'):
        verify(path, source, ['a'], tolerance=1e-8)
    summary['results'][0]['adam_history'][3]['orbital_residual'] = 1e-10
    summary['results'][0]['accepted_adam_steps'] = 4
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='Adam'):
        verify(path, source, ['a'], tolerance=1e-8)


def test_real_optax_checkpoint_round_trip_preserves_next_update(tmp_path):
    pytest.importorskip('optax'); pytest.importorskip('flax')
    api = module(); optimizer = api['Adam'](1e-4); initial = np.array([.2, -.4])
    state = optimizer.init(initial)
    vector, state = optimizer.update(initial, np.array([.1, .3]), state)
    checkpoint = dict(identity={}, parameters=vector, optimizer=state,
        gradient=np.array([-.2, .5]), epoch=1, loss_hartree=-1., molecules=[], history=[])
    api['save_checkpoint'](tmp_path / 'state.npz', checkpoint, optimizer)
    restored = api['load_checkpoint'](tmp_path / 'state.npz', {}, optimizer, initial)
    expected, _ = optimizer.update(vector, checkpoint['gradient'], state)
    actual, _ = optimizer.update(restored['parameters'], restored['gradient'], restored['optimizer'])
    np.testing.assert_array_equal(np.asarray(expected), np.asarray(actual))


def test_heldout_is_value_only_and_evaluated_at_saved_parameters(tmp_path):
    api = module(); points = []
    def heldout(vector):
        points.append(vector.copy())
        value = float(np.dot(vector - 3., vector - 3.))
        return value, [dict(structure_id='heldout', energy_hartree=value, **info())]
    evaluate = lambda vector: api['evaluate_batch'](vector, evaluators())
    baseline = tmp_path / 'baseline'; held = tmp_path / 'held'
    api['run_training'](np.zeros(2), evaluate, Adam(), baseline, {}, epochs=5)
    api['run_training'](np.zeros(2), evaluate, Adam(), held, {}, epochs=5,
                        heldout=heldout, validation_interval=2)
    first = api['load_checkpoint'](baseline / 'last.npz', {}, Adam(), np.zeros(2))
    second = api['load_checkpoint'](held / 'last.npz', {}, Adam(), np.zeros(2))
    np.testing.assert_array_equal(first['parameters'], second['parameters'])
    assert [row['epoch'] for row in second['validation_history']] == [0, 2, 4, 5]
    assert second['validation_history'][-1]['loss_hartree'] == pytest.approx(heldout(second['parameters'])[0])
    assert len(points) == 5  # Four scheduled evaluations plus the assertion above.
    validation_best = api['load_checkpoint'](held / 'best_validation.npz', {}, Adam(), np.zeros(2))
    assert validation_best['validation_history'][-1]['loss_hartree'] == min(
        row['loss_hartree'] for row in second['validation_history'])


def test_bad_heldout_stops_without_overwriting_accepted_checkpoint(tmp_path):
    api = module(); calls = []
    def heldout(vector):
        calls.append(True)
        if len(calls) > 1: raise RuntimeError('heldout: SCF failed')
        return 1., [dict(structure_id='heldout', energy_hartree=1., **info())]
    result = api['run_training'](np.zeros(2), lambda vector: api['evaluate_batch'](vector, evaluators()),
        Adam(), tmp_path, {}, epochs=5, heldout=heldout, validation_interval=2)
    assert result['status'] == 'failed' and result['accepted_epoch'] == 1
    saved = api['load_checkpoint'](tmp_path / 'last.npz', {}, Adam(), np.zeros(2))
    assert saved['epoch'] == 1 and saved['optimizer']['count'] == 1


def test_value_batch_never_requires_a_gradient_and_rejects_bad_scf():
    api = module()
    values = [('a', lambda vector: (2., info())), ('b', lambda vector: (4., info()))]
    loss, rows = api['evaluate_values'](np.zeros(2), values)
    assert loss == 3. and len(rows) == 2
    details = info(); details['converged'] = False
    with pytest.raises(RuntimeError, match='bad'):
        api['evaluate_values'](np.zeros(2), [('bad', lambda vector: (1., details))])


def test_cosine_schedule_resume_keeps_global_update_count(tmp_path):
    pytest.importorskip('optax'); pytest.importorskip('flax')
    api = module(); optimizer = api['Adam'](1e-4, final_learning_rate=1e-5, epochs=8)
    assert optimizer.learning_rate_at(0) == pytest.approx(1e-4)
    assert optimizer.learning_rate_at(8) == pytest.approx(1e-5)
    assert 1e-5 < optimizer.learning_rate_at(4) < 1e-4
    evaluate = lambda vector: api['evaluate_batch'](vector, evaluators())
    full = tmp_path / 'full'; split = tmp_path / 'split'
    api['run_training'](np.zeros(2), evaluate, optimizer, full, {}, epochs=8)
    api['run_training'](np.zeros(2), evaluate, optimizer, split, {}, epochs=3)
    api['run_training'](np.zeros(2), evaluate, optimizer, split, {}, epochs=8, resume=True)
    expected = api['load_checkpoint'](full / 'last.npz', {}, optimizer, np.zeros(2))
    actual = api['load_checkpoint'](split / 'last.npz', {}, optimizer, np.zeros(2))
    np.testing.assert_array_equal(actual['parameters'], expected['parameters'])


def test_host_fixed_integrals_match_device_storage(tmp_path):
    pytest.importorskip('jax')
    numerical = vars(import_module("gradscf.model.nnao.training.objective"))
    experiment_class = numerical['BasisRHF']
    geometry = dict(name='H2', symbols=['H', 'H'], charge=0, spin=0,
                    coords_angstrom=[[0., 0., 0.], [0., 0., .74]])
    kwargs = dict(geometry=geometry, basis_family='szp663_direct', core_primitives=6,
                  jk_backend='df', implicit_tolerance=1e-8)
    device = experiment_class(**kwargs)
    path = device.write_integral_cache(tmp_path / 'integrals.npz')
    host = experiment_class(**kwargs, integral_cache=path, cache_storage='host')
    assert all(isinstance(array, np.ndarray) for array in (host.ps, host.ph, host.rep))
    outputs = device.layout.reference_outputs()
    expected = device.evaluate(outputs); actual = host.evaluate(outputs)
    assert actual[0] == pytest.approx(expected[0], abs=1e-12)
    np.testing.assert_allclose(actual[1], expected[1], atol=1e-12, rtol=1e-12)
    assert actual[2] == expected[2]
    with pytest.raises(ValueError, match='cache_storage'):
        experiment_class(**kwargs, integral_cache=path, cache_storage='unknown')


def test_stage2_requires_both_heldout_inputs_before_starting(capsys):
    api = module()
    args = ['--structures', 'missing', '--cache-dir', 'missing',
            '--validation-summary', 'missing', '--output-dir', 'missing',
            '--heldout-structures', 'missing-heldout', '--cache-storage', 'host',
            '--final-learning-rate', '1e-5']
    with pytest.raises(SystemExit) as error:
        import_module('gradscf.model.nnao.workflows.train').main(args)
    assert error.value.code == 2
    assert 'together' in capsys.readouterr().err


def test_every_split_gate_checks_same_scientific_and_helper_sources(tmp_path):
    api = module(); tools = tmp_path / 'src/gradscf/model/nnao'; tools.mkdir(parents=True)
    (tools/'training').mkdir(); (tools/'workflows').mkdir()
    filenames = [('experiment_source_sha256', 'training/objective.py'),
                 ('source_sha256', 'workflows/validate.py')]
    import hashlib
    config = dict(scientific_source_sha256='scientific')
    for field, name in filenames:
        path = tools / name; path.write_text(name)
        config[field] = hashlib.sha256(path.read_bytes()).hexdigest()
    api['verify_validation_sources']({'config': config}, tmp_path, scientific_hash='scientific')
    (tools / 'workflows/validate.py').write_text('changed')
    with pytest.raises(ValueError, match='changed since validation'):
        api['verify_validation_sources']({'config': config}, tmp_path, scientific_hash='scientific')


def test_initial_heldout_failure_does_not_claim_an_accepted_checkpoint(tmp_path):
    api = module()
    def heldout(vector): raise RuntimeError('heldout failed initially')
    result = api['run_training'](np.zeros(2), lambda vector: api['evaluate_batch'](vector, evaluators()),
                                Adam(), tmp_path, {}, epochs=2, heldout=heldout)
    assert result['accepted_epoch'] is None and not (tmp_path / 'last.npz').exists()


@pytest.mark.parametrize('order', ['C', 'F'])
def test_npz_numeric_mapping_preserves_order_bytes_and_readonly(tmp_path, order):
    numerical = vars(import_module("gradscf.model.nnao.training.objective"))
    expected = np.array(np.arange(30, dtype=np.float64).reshape(5, 6), order=order)
    path = tmp_path / 'integrals.npz'; np.savez(path, factors=expected)
    actual = import_module('gradscf.model.nnao.integrals.cache').map_npz_numeric(path, 'factors')
    assert isinstance(actual, np.memmap) and actual.nbytes == expected.nbytes
    assert Path(actual.filename) == path
    assert not actual.flags.writeable
    assert actual.flags.f_contiguous if order == 'F' else actual.flags.c_contiguous
    np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError, match='read-only'):
        actual[0, 0] = 1.


def test_npz_mapping_rejects_compressed_and_object_members(tmp_path):
    numerical = vars(import_module("gradscf.model.nnao.training.objective"))
    compressed = tmp_path / 'compressed.npz'
    np.savez_compressed(compressed, factors=np.arange(10.))
    with pytest.raises(ValueError, match='ZIP_STORED'):
        import_module('gradscf.model.nnao.integrals.cache').map_npz_numeric(compressed, 'factors')
    objects = tmp_path / 'objects.npz'; np.savez(objects, factors=np.array([{}], dtype=object))
    with pytest.raises(ValueError, match='numeric'):
        import_module('gradscf.model.nnao.integrals.cache').map_npz_numeric(objects, 'factors')


def test_mmap_integrals_match_device_energy_and_gradient(tmp_path):
    numerical = vars(import_module("gradscf.model.nnao.training.objective"))
    geometry = dict(name='H2', symbols=['H', 'H'], charge=0, spin=0,
                    coords_angstrom=[[0., 0., 0.], [0., 0., .74]])
    kwargs = dict(geometry=geometry, basis_family='szp663_direct', core_primitives=6,
                  jk_backend='df', implicit_tolerance=1e-8)
    device = numerical['BasisRHF'](**kwargs)
    path = device.write_integral_cache(tmp_path / 'integrals.npz')
    mapped = numerical['BasisRHF'](**kwargs, integral_cache=path, cache_storage='mmap')
    assert all(isinstance(array, np.memmap) and not array.flags.writeable
               for array in (mapped.ps, mapped.ph, mapped.rep))
    outputs = device.layout.reference_outputs()
    expected = device.evaluate(outputs); actual = mapped.evaluate(outputs)
    assert actual[0] == pytest.approx(expected[0], abs=1e-12)
    np.testing.assert_allclose(actual[1], expected[1], atol=1e-12, rtol=1e-12)
    assert actual[2] == expected[2]
