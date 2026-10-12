"""Alternating optimization must preserve accepted states and element locality."""
from importlib import import_module
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def driver():
    return import_module("gradscf.model.nnao.training.alternating")


ROWS = [dict(structure_id='ch4', symbols=['C', 'H']),
        dict(structure_id='nh3', symbols=['N', 'H'])]


def diagnostics():
    return dict(converged=True, orbital_residual=0., fixed_point_residual=0.,
                min_overlap_eigenvalue=1., reconstruction_error=0.)


class CounterOptimizer:
    def init(self, theta): return 0
    def update(self, theta, gradient, state): return theta - .1 * gradient, state + 1


def step_once(fun, point, **kwargs):
    fun(point)
    fun(point + .1)
    return SimpleNamespace(success=False, status=1, nfev=2,
                           message='Maximum number of function evaluations exceeded.', x=point + .1)


def test_powell_uses_only_affected_molecules_and_best_observed_valid_point():
    module = driver(); beta = np.zeros(14); called = []
    def objective(identifier, candidate):
        called.append(identifier)
        c = candidate[module.EXPONENT_KEYS.index(('C', 0))]
        if c > .15: raise RuntimeError('invalid SCF')
        return 1. - c, diagnostics()
    def budget(fun, point, **kwargs):
        fun(point); fun(point + .1); assert np.isinf(fun(point + .2))
        return SimpleNamespace(success=False, status=1, nfev=3, message='maxfev', x=point + .2)
    proposal = module.powell_element(beta, 'C', ROWS, {'ch4': 1., 'nh3': 2.},
        objective, log_bound=.25, maxfev=3, minimize=budget)
    assert set(called) == {'ch4'}
    assert proposal['improved'] and proposal['status'] == 'budget-limited'
    assert proposal['energies']['nh3'] == 2.
    np.testing.assert_allclose(proposal['beta'][2:5], .1)
    np.testing.assert_array_equal(beta, np.zeros(14))
    assert proposal['history'][-1]['valid'] is False


def test_out_of_bounds_candidates_do_not_run_scf():
    module = driver(); called = []
    def minimize(fun, point, **kwargs):
        assert np.isinf(fun(point + .5))
        return SimpleNamespace(success=False, status=1, nfev=1, message='maxfev')
    proposal = module.powell_element(np.zeros(14), 'C', ROWS, {'ch4': 1., 'nh3': 2.},
        lambda identifier, beta: called.append(identifier), log_bound=.25, maxfev=1, minimize=minimize)
    assert called == [] and not proposal['improved']


def test_failed_coefficient_backward_preserves_theta_beta_optimizer_and_cache():
    module = driver(); saved = []
    def prepare(beta, affected, previous): return {'marker': 'initial'}
    def evaluate(theta, beta, cache):
        if theta[0] < 0.: raise RuntimeError('nonfinite implicit gradient')
        records = [dict(structure_id=row['structure_id'], energy_hartree=1., **diagnostics()) for row in ROWS]
        return 1., np.ones_like(theta), records
    state, result = module.run_joint(np.zeros(1), np.zeros(14), ROWS,
        CounterOptimizer(), prepare, evaluate, None, save=lambda s: saved.append(s.copy()),
        coefficient_steps=1, cycles=1, maxfev=2, log_bound=.25, minimize=step_once)
    assert result['status'] == 'failed' and len(saved) == 1
    np.testing.assert_array_equal(state['theta'], [0.]); np.testing.assert_array_equal(state['beta'], np.zeros(14))
    assert state['optimizer'] == 0 and state['cache']['marker'] == 'initial'


@pytest.mark.parametrize('fail_refresh', [False, True])
def test_alpha_commit_rebuilds_only_affected_caches_and_requires_full_backward(fail_refresh):
    module = driver(); preparations = []; backwards = []; saved = []
    def prepare(beta, affected, previous):
        preparations.append(affected)
        return {'marker': 'initial' if affected is None else 'new-C'}
    def energy(identifier, beta):
        return 1. - beta[2] if identifier == 'ch4' else 2.
    def evaluate(theta, beta, cache):
        backwards.append(tuple(row['structure_id'] for row in ROWS))
        if fail_refresh and beta[2] > 0.: raise RuntimeError('failed refreshed implicit gradient')
        records = [dict(structure_id=row['structure_id'], energy_hartree=energy(row['structure_id'], beta),
                        **diagnostics()) for row in ROWS]
        return float(np.mean([r['energy_hartree'] for r in records])), np.zeros_like(theta), records
    state, result = module.run_joint(np.zeros(1), np.zeros(14), ROWS,
        CounterOptimizer(), prepare, evaluate, lambda identifier, beta: (energy(identifier, beta), diagnostics()),
        save=lambda s: saved.append(s.copy()), coefficient_steps=0, cycles=1, maxfev=2,
        log_bound=.25, minimize=step_once)
    assert preparations == [None, ['ch4']]
    assert backwards == [('ch4', 'nh3'), ('ch4', 'nh3')]
    assert state['optimizer'] == 0
    if fail_refresh:
        assert result['status'] == 'failed' and len(saved) == 1
        assert state['beta'][2] == 0. and state['cache']['marker'] == 'initial'
    else:
        assert result['status'] == 'complete' and len(saved) == 2
        assert state['beta'][2] == .1 and state['cache']['marker'] == 'new-C'
        assert state['energies']['nh3'] == 2. and state['alpha_steps'] == 1


def test_refreshed_primitive_and_candidate_contracted_energies_must_match():
    module = driver()
    with pytest.raises(RuntimeError, match='parity'):
        module.check_energy_parity({'x': -1.}, {'x': -1. + 2e-8})
    module.check_energy_parity({'x': -1.}, {'x': -1. + 1e-10})


def test_one_cycle_has_coefficient_updates_before_and_after_powell():
    module = driver()
    def evaluate(theta, beta, cache):
        records = [dict(structure_id=row['structure_id'], energy_hartree=1., **diagnostics()) for row in ROWS]
        return 1., np.zeros_like(theta), records
    state, result = module.run_joint(np.zeros(1), np.zeros(14), ROWS,
        CounterOptimizer(), lambda *args: {}, evaluate, lambda *args: (1., diagnostics()),
        save=lambda state: None, coefficient_steps=2, cycles=1, maxfev=2,
        log_bound=.25, minimize=step_once)
    assert result['status'] == 'complete' and state['theta_steps'] == state['optimizer'] == 4
    phases = [item['phase'] for item in result['history']]
    assert phases[:3] == ['initial', 'coeff-before', 'coeff-before']
    assert phases[-2:] == ['coeff-after', 'coeff-after']
    assert state['history'][-1]['phase'] == 'coeff-after'


@pytest.mark.parametrize('bad', ['nonfinite', 'duplicate', 'wrong_mean'])
def test_full_batch_records_cannot_hide_invalid_or_inconsistent_energies(bad):
    module = driver()
    def evaluate(theta, beta, cache):
        records = [dict(structure_id=row['structure_id'], energy_hartree=1., **diagnostics()) for row in ROWS]
        if bad == 'nonfinite': records[0]['energy_hartree'] = float('nan')
        if bad == 'duplicate': records.append(dict(records[0]))
        return 2. if bad == 'wrong_mean' else 1., np.zeros_like(theta), records
    state, result = module.run_joint(np.zeros(1), np.zeros(14), ROWS,
        CounterOptimizer(), lambda *args: {}, evaluate, None, save=lambda state: None,
        coefficient_steps=0, cycles=1, maxfev=2, log_bound=.25, minimize=step_once)
    assert state is None and result['status'] == 'failed'


def test_parent_parameters_cannot_be_changed_under_original_metadata(tmp_path):
    module = driver()
    original = Path(__file__).resolve().parents[2] / 'artifacts/nnao-gmtkn55-first-training-20261007/training-93108/best.npz'
    if not original.is_file(): pytest.skip('The original archived best975 checkpoint is not present.')
    with np.load(original, allow_pickle=False) as saved:
        data = {key: np.array(saved[key]) for key in saved.files}
    data['parameters'][-1] += .25
    changed = tmp_path / 'tampered.npz'; np.savez(changed, **data)
    with pytest.raises(ValueError, match='SHA|bytes|checkpoint hash'):
        module.read_parent(changed, [])
