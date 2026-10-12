"""NNAO batch evaluation and transactional training state."""
from __future__ import annotations
import json
from pathlib import Path
import time
import traceback
import numpy as np

from .. import runtime as _pilot
from .checkpoints import save_checkpoint, load_checkpoint, save_failure_snapshot

def valid_scf(details):
    keys = ('orbital_residual', 'min_overlap_eigenvalue', 'reconstruction_error', 'fixed_point_residual')
    return (details.get('converged') is True and details.get('stable',True) is True
        and all(key in details for key in keys)
        and np.isfinite([details[key] for key in keys]).all()
        and details['orbital_residual'] <= 1e-7
        and details['fixed_point_residual'] <= 1e-7
        and details['min_overlap_eigenvalue'] >= 1e-8
        and details['reconstruction_error'] <= 1e-9)

def verify_validation(path, structures, identifiers, *, tolerance, rescue_level_shift=0.0):
    """Require a complete successful gate for exactly the training structures."""
    summary = json.loads(Path(path).read_text())
    results = summary.get('results', [])
    if (summary.get('config', {}).get('structures_sha256') != _pilot.file_hash(structures)
            or sorted(row['structure_id'] for row in results) != sorted(identifiers)):
        raise ValueError('Validation structures differ from the training input.')
    if (summary.get('exit_code') != 0 or summary.get('failed') != 0
            or summary.get('passed') != len(identifiers)
            or any(row.get('status') != 'passed' for row in results)):
        raise ValueError('A complete successful validation is required before training.')
    if (summary.get('config', {}).get('implicit_tolerance') != tolerance
            or any(row.get('implicit_config', {}).get('tolerance') != tolerance for row in results)):
        raise ValueError('Validation implicit tolerance differs from the training configuration.')
    if (summary.get('config', {}).get('scf_rescue_level_shift', 0.) != rescue_level_shift
            or any(row.get('scf_rescue_level_shift', 0.) != rescue_level_shift for row in results)):
        raise ValueError('Validation SCF rescue policy differs from the training configuration.')
    if any(row.get(key, {}).get('passed') is not True for row in results
           for key in ('coefficient_directional_check', 'parameter_directional_check')):
        raise ValueError('Validation must include passing coefficient and parameter gradient checks.')
    if any(row.get('implicit_gradient_finite') is not True for row in results):
        raise ValueError('Validation requires finite implicit gradients for every molecule.')
    for row in results:
        history = row.get('adam_history', [])
        if row.get('accepted_adam_steps') != 5 or [item.get('step') for item in history] != list(range(6)):
            raise ValueError('Validation requires five accepted Adam steps for every molecule.')
        if not valid_scf(row.get('initial_scf', {})) or any(not valid_scf(item) for item in history):
            raise ValueError('Validation contains missing or invalid SCF diagnostics.')
    return summary

def verify_validation_sources(summary, root, *, scientific_hash):
    if summary['config'].get('scientific_source_sha256') != scientific_hash:
        raise ValueError('Scientific sources changed since validation; rerun the gate.')
    for field, filename in (('experiment_source_sha256', 'training/objective.py'),
                            ('source_sha256', 'workflows/validate.py')):
        if summary['config'].get(field) != _pilot.file_hash(Path(root) / 'src/gradscf/model/nnao' / filename):
            raise ValueError(f'{filename} changed since validation; rerun the gate.')

def evaluate_batch(vector, evaluators, *, progress=None):
    """Evaluate every molecule at the same vector; never skip invalid samples."""
    records = []; gradient_sum = np.zeros_like(np.asarray(vector)); energies = []
    if not evaluators:
        raise ValueError('The training batch is empty.')
    for identifier, evaluate in evaluators:
        try:
            value, gradient, details = evaluate(vector)
            gradient = np.asarray(gradient)
            valid = (np.isfinite(value) and gradient.shape == gradient_sum.shape
                and np.isfinite(gradient).all() and valid_scf(details))
            if not valid:
                raise RuntimeError(f'Invalid SCF/gradient diagnostics: {details}')
        except Exception as error:
            failure = RuntimeError(f'{identifier}: {error}')
            failure.structure_id = identifier
            raise failure from error
        energies.append(float(value)); gradient_sum += gradient
        records.append(dict(structure_id=identifier, energy_hartree=float(value),
                            parameter_gradient_norm=float(np.linalg.norm(gradient)), **details))
        if progress is not None:
            progress(records)
    return float(np.mean(energies)), gradient_sum / len(evaluators), records

def evaluate_values(vector, evaluators, *, progress=None):
    """Held-out diagnostics use the same SCF primal, without a gradient or update."""
    if not evaluators:
        raise ValueError('The held-out batch is empty.')
    records = []
    for identifier, evaluate in evaluators:
        try:
            value, details = evaluate(vector)
            if not np.isfinite(value) or not valid_scf(details):
                raise RuntimeError(f'Invalid held-out SCF diagnostics: {details}')
        except Exception as error:
            failure = RuntimeError(f'{identifier}: {error}')
            failure.structure_id = identifier
            raise failure from error
        records.append(dict(structure_id=identifier, energy_hartree=float(value), **details))
        if progress is not None:
            progress(records)
    return float(np.mean([row['energy_hartree'] for row in records])), records

def run_training(initial, evaluate, optimizer, output_dir, identity, *, epochs, resume=False,
                 heldout=None, validation_interval=20):
    """One complete batch and one Adam update per epoch, with accepted checkpoints."""
    directory = Path(output_dir); directory.mkdir(parents=True, exist_ok=True)
    last = directory / 'last.npz'; best = directory / 'best.npz'
    validation_best = directory / 'best_validation.npz'
    if validation_interval < 1:
        raise ValueError('validation_interval must be positive.')
    if last.exists() and not resume:
        raise ValueError('Output already contains a checkpoint; use --resume or a new output directory.')
    if resume and not last.exists():
        raise ValueError('Resume requires an existing last.npz checkpoint.')
    checkpoint = None; vector = None; started = time.perf_counter(); attempted_epoch = 0
    try:
        if resume:
            checkpoint = load_checkpoint(last, identity, optimizer, initial)
            vector = checkpoint['parameters']
        else:
            vector = initial
            loss, gradient, records = evaluate(vector)
            candidate = dict(identity=identity, parameters=np.asarray(initial),
                optimizer=optimizer.init(initial), gradient=np.asarray(gradient), epoch=0,
                loss_hartree=loss, molecules=records, history=[], validation_history=[])
            candidate['history'].append(dict(epoch=0, loss_hartree=loss, molecules=records))
            if heldout is not None:
                value, heldout_records = heldout(initial)
                candidate['validation_history'].append(dict(epoch=0, loss_hartree=value, molecules=heldout_records))
                save_checkpoint(validation_best, candidate, optimizer)
            checkpoint = candidate
            save_checkpoint(last, checkpoint, optimizer); save_checkpoint(best, checkpoint, optimizer)
        checkpoint.setdefault('validation_history', [])
        best_loss = load_checkpoint(best, identity, optimizer, initial)['loss_hartree']
        best_validation_loss = (load_checkpoint(validation_best, identity, optimizer, initial)['validation_history'][-1]['loss_hartree']
                                if heldout is not None else None)
        _pilot.atomic_json(directory / 'history.json', checkpoint['history'])
        if heldout is not None:
            _pilot.atomic_json(directory / 'validation_history.json', checkpoint['validation_history'])
        initial_loss = checkpoint['history'][0]['loss_hartree']
        _pilot.atomic_json(directory / 'progress.json', dict(status='running',
            accepted_epoch=checkpoint['epoch'], target_epochs=epochs,
            loss_hartree=checkpoint['loss_hartree'], best_loss_hartree=best_loss,
            elapsed_seconds=time.perf_counter() - started))
        for epoch in range(checkpoint['epoch'] + 1, epochs + 1):
            attempted_epoch = epoch; tick = time.perf_counter()
            vector, state = optimizer.update(checkpoint['parameters'], checkpoint['gradient'],
                                             checkpoint['optimizer'])
            loss, gradient, records = evaluate(vector)
            record = dict(epoch=epoch, loss_hartree=loss, molecules=records)
            validation_history = checkpoint['validation_history']
            heldout_value = None
            if heldout is not None and (epoch % validation_interval == 0 or epoch == epochs):
                heldout_value, heldout_records = heldout(vector)
                validation_history = validation_history + [dict(epoch=epoch, loss_hartree=heldout_value, molecules=heldout_records)]
            # Save parameters only after measuring their own converged energy and gradient.
            checkpoint = dict(identity=identity, parameters=np.asarray(vector), optimizer=state,
                gradient=np.asarray(gradient), epoch=epoch, loss_hartree=loss, molecules=records,
                history=checkpoint['history'] + [record], validation_history=validation_history)
            save_checkpoint(last, checkpoint, optimizer)
            if loss < best_loss:
                save_checkpoint(best, checkpoint, optimizer); best_loss = loss
            if heldout_value is not None and heldout_value < best_validation_loss:
                save_checkpoint(validation_best, checkpoint, optimizer); best_validation_loss = heldout_value
            _pilot.atomic_json(directory / 'history.json', checkpoint['history'])
            if heldout is not None:
                _pilot.atomic_json(directory / 'validation_history.json', checkpoint['validation_history'])
            progress = dict(status='running', accepted_epoch=epoch, target_epochs=epochs,
                loss_hartree=loss, best_loss_hartree=best_loss,
                mean_energy_decrease_millihartree=1000 * (initial_loss - loss),
                epoch_elapsed_seconds=time.perf_counter() - tick,
                elapsed_seconds=time.perf_counter() - started)
            if heldout_value is not None:
                progress.update(validation_epoch=epoch, validation_loss_hartree=heldout_value,
                                best_validation_loss_hartree=best_validation_loss)
            if hasattr(optimizer, 'learning_rate_at'):
                progress['learning_rate'] = optimizer.learning_rate_at(epoch - 1)
            _pilot.atomic_json(directory / 'progress.json', progress)
            print(f"Epoch {epoch}/{epochs}: mean E {loss:.12f} Ha; "
                  f"decrease {progress['mean_energy_decrease_millihartree']:.6f} mHa; "
                  f"{progress['epoch_elapsed_seconds']:.2f} s", flush=True)
        summary = dict(status='complete', accepted_epoch=checkpoint['epoch'], target_epochs=epochs,
            loss_hartree=checkpoint['loss_hartree'], best_loss_hartree=best_loss,
            initial_loss_hartree=initial_loss, elapsed_seconds=time.perf_counter() - started,
            mean_energy_decrease_millihartree=1000 * (initial_loss - checkpoint['loss_hartree']),
            molecules=checkpoint['molecules'], identity=identity)
        if heldout is not None:
            summary.update(validation_history=checkpoint['validation_history'],
                           best_validation_loss_hartree=best_validation_loss)
    except Exception as error:
        summary = dict(status='failed', accepted_epoch=checkpoint['epoch'] if checkpoint else None,
            attempted_epoch=attempted_epoch, target_epochs=epochs, error=str(error),
            traceback=traceback.format_exc(), elapsed_seconds=time.perf_counter() - started,
            identity=identity)
        if vector is not None:
            snapshot = directory / f'failed-candidate-epoch-{attempted_epoch:06d}.npz'
            try:
                save_failure_snapshot(snapshot, vector, error, summary)
                summary['failure_snapshot'] = snapshot.name
            except Exception as snapshot_error:
                summary['failure_snapshot_error'] = str(snapshot_error)
    _pilot.atomic_json(directory / 'summary.json', summary)
    _pilot.atomic_json(directory / 'progress.json', summary)
    return summary

class Adam:
    def __init__(self, learning_rate, *, final_learning_rate=None, epochs=1000):
        import jax
        import optax
        from flax import serialization
        self.jax = jax; self.serialization = serialization
        if final_learning_rate is not None:
            if not 0 < final_learning_rate <= learning_rate or epochs < 1:
                raise ValueError('Cosine learning rates require 0 < final <= initial and positive epochs.')
            self.schedule = optax.cosine_decay_schedule(learning_rate, epochs,
                                                       alpha=final_learning_rate / learning_rate)
        else:
            self.schedule = lambda count: learning_rate
        self.transform = optax.chain(optax.clip_by_global_norm(1.), optax.adam(
            self.schedule if final_learning_rate is not None else learning_rate))
        self.step = jax.jit(self._step)

    def learning_rate_at(self, count):
        return float(self.schedule(count))

    def init(self, vector):
        return self.transform.init(self.jax.numpy.asarray(vector))

    def _step(self, vector, gradient, state):
        import optax
        updates, state = self.transform.update(gradient, state, vector)
        return optax.apply_updates(vector, updates), state

    def update(self, vector, gradient, state):
        return self.step(self.jax.numpy.asarray(vector), self.jax.numpy.asarray(gradient), state)

    def encode(self, state):
        return self.serialization.to_bytes(self.jax.device_get(state))

    def decode(self, data, template):
        return self.serialization.from_bytes(template, data)

def make_evaluators(rows, args, jax, experiment_class):
    from flax import nnx
    from jax.flatten_util import ravel_pytree
    from gradscf.data.molecule import atomic_number
    from gradscf.model.nnao import MACEBasisModel, build_graph
    model_config = dict(elements=(1, 6, 7, 8, 9), channels=8, num_interactions=2,
        max_ell=1, correlation=2, zero_init=True, basis_family='szp663_direct')
    model = MACEBasisModel(**model_config, rngs=nnx.Rngs(args.seed))
    import mace_jax
    vendor = _pilot.vendor_directory()
    if not Path(mace_jax.__file__).resolve().is_relative_to(vendor):
        raise RuntimeError(f'MACE source mismatch: imported {mace_jax.__file__}; expected {vendor}.')
    definition, parameters, other = nnx.split(model, nnx.Param, ...)
    initial, unravel = ravel_pytree(parameters)
    if str(initial.dtype) != 'float64':
        raise RuntimeError('MACE parameter vector must use float64.')

    @jax.jit
    def predict(vector, graph):
        return nnx.merge(definition, unravel(vector), other)(graph)

    @jax.jit
    def backward(vector, graph, cotangent):
        return jax.vjp(lambda value: predict(value, graph), vector)[1](cotangent)[0]

    evaluators = []; value_evaluators = []; manifests = {}
    for row in rows:
        path = _pilot.cache_location(args.cache_dir, row)
        manifest = json.loads(path.with_suffix('.json').read_text())
        if _pilot.file_hash(path) != manifest['cache_sha256']:
            raise ValueError(f"{row['structure_id']}: native DF cache hash differs from its manifest.")
        experiment = experiment_class(geometry=row, basis_family='szp663_direct',
            core_primitives=6, jk_backend='df', auxbasis=args.auxbasis, integral_cache=path,
            implicit_tolerance=args.implicit_tolerance, cache_storage=args.cache_storage,
            scf_rescue_level_shift=args.scf_rescue_level_shift)
        graph = build_graph([atomic_number(symbol) for symbol in experiment.symbols],
                            experiment.coords, element_order=model.elements)
        if str(experiment.ps.dtype) != 'float64':
            raise RuntimeError('RHF integral arrays must use float64.')
        def evaluate(vector, graph=graph, experiment=experiment):
            vector = jax.numpy.asarray(vector)
            prediction = predict(vector, graph)
            value, output_gradient, details = experiment.evaluate(prediction)
            gradient = np.asarray(backward(vector, graph, output_gradient))
            return value, gradient, details
        forward = jax.jit(experiment._implicit_value)
        def value_only(vector, graph=graph, experiment=experiment, forward=forward):
            prediction = predict(jax.numpy.asarray(vector), graph)
            value, info = forward(prediction, experiment.ps, experiment.ph, experiment.rep)
            details = {key: bool(item) if key == 'converged' else int(item) if key == 'scf_cycles' else float(item)
                       for key, item in info.items()}
            return float(value), details
        evaluators.append((row['structure_id'], evaluate))
        value_evaluators.append((row['structure_id'], value_only))
        manifests[row['structure_id']] = dict(cache_sha256=manifest['cache_sha256'],
            integral_signature=experiment.integral_signature, nao=int(experiment.layout.topology.nao),
            primitive_nao=int(experiment.primitive_topology.nao), df_shape=list(experiment.rep.shape))
    return initial, evaluators, value_evaluators, manifests, model_config
