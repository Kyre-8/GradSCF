"""Numerical acceptance checks for NNAO caches and training."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import platform as host_platform
import time
import traceback

from ..runtime import (file_hash, atomic_json, record_directory, experiment_metadata, cache_location)

def run_suite(rows, output_dir, config, operation, *, resume):
    """Persist each outcome before continuing; failures yield a nonzero status."""
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    summary = dict(config=config, passed=0, failed=0, exit_code=0, results=results)
    for row in rows:
        directory = record_directory(output_dir, row['structure_id'])
        directory.mkdir(parents=True, exist_ok=True)
        signature = hashlib.sha256(json.dumps(dict(structure=row, config=config),
            sort_keys=True, allow_nan=False).encode()).hexdigest()
        path = directory / 'result.json'
        previous = json.loads(path.read_text()) if resume and path.is_file() else None
        cache_valid = True
        if previous and previous.get('cache_path'):
            cached_path = Path(previous['cache_path'])
            cache_valid = (cached_path.is_file()
                           and file_hash(cached_path) == previous.get('cache_sha256'))
        if (previous and previous.get('signature') == signature
                and previous.get('status') == 'passed' and cache_valid):
            result = previous
            print(f"Resume {row['structure_id']}: passed", flush=True)
        else:
            start = time.perf_counter()
            result = dict(structure_id=row['structure_id'], signature=signature,
                          status='running', output_dir=str(directory))
            atomic_json(path, result)
            # A stale partial report must not leak into a rerun with changed configuration.
            atomic_json(directory / 'progress.json', {})
            try:
                completed = operation(row, directory)
                json.dumps(completed, allow_nan=False)
                result.update(completed)
                result['status'] = 'passed'
            except Exception as error:
                progress = json.loads((directory / 'progress.json').read_text())
                result.update(progress, status='failed', error=str(error),
                              error_type=type(error).__name__, traceback=traceback.format_exc())
            result['elapsed_seconds'] = time.perf_counter() - start
            atomic_json(path, result)
            print(f"{row['structure_id']}: {result['status']} ({result['elapsed_seconds']:.1f} s)", flush=True)
        results.append(result)
        summary['passed'] = sum(value['status'] == 'passed' for value in results)
        summary['failed'] = sum(value['status'] == 'failed' for value in results)
        summary['exit_code'] = int(summary['failed'] > 0)
        atomic_json(output_dir / 'summary.json', summary)
    return summary

def prepare_cache(row, directory, *, cache_dir, auxbasis, jax, experiment_class,
                  implicit_tolerance=1e-9, scf_rescue_level_shift=0.0):
    experiment = experiment_class(geometry=row, basis_family='szp663_direct',
        core_primitives=6, jk_backend='df', auxbasis=auxbasis,
        implicit_tolerance=implicit_tolerance, scf_rescue_level_shift=scf_rescue_level_shift)
    path = cache_location(cache_dir, row); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('integrals.tmp.npz')
    experiment.write_integral_cache(temporary)
    os.replace(temporary, path)
    result = experiment_metadata(experiment, jax)
    from gradscf.integrals import _native
    suffix = '.dylib' if host_platform.system() == 'Darwin' else '.so'
    library = Path(_native.__file__).parent / 'lib' / ('libgradscf_integrals' + suffix)
    result.update(cache_path=str(path), cache_sha256=file_hash(path),
                  structure_id=row['structure_id'], mace_ran=False, adam_ran=False,
                  native_library_sha256=file_hash(library))
    atomic_json(path.with_suffix('.json'), result)
    jax.clear_caches()
    return result

def directional_check(evaluate, point, gradient, direction, *, step, atol, rtol):
    """Fourth-order finite difference; every evaluation reconverges SCF."""
    import numpy as np
    point = np.asarray(point); direction = np.asarray(direction)
    values = {str(scale): float(evaluate(point + scale * step * direction)[0])
              for scale in (1, -1, 2, -2)}
    finite_difference = (8 * (values['1'] - values['-1'])
                         - values['2'] + values['-2']) / (12 * step)
    analytic = float(np.sum(np.asarray(gradient) * direction))
    result = dict(analytic=analytic, finite_difference=finite_difference,
        absolute_error=abs(analytic - finite_difference), step=step, atol=atol, rtol=rtol,
        perturbed_energies_hartree=values,
        passed=bool(np.isclose(analytic, finite_difference, atol=atol, rtol=rtol)))
    return result

def validate(row, directory, *, cache_dir, auxbasis, jax, experiment_class,
             seed, adam_steps, learning_rate, fd_step, fd_atol, fd_rtol,
             implicit_tolerance=1e-9, cache_storage='device', scf_rescue_level_shift=0.0):
    import numpy as np
    import jax.numpy as jnp
    path = cache_location(cache_dir, row)
    cache_record = json.loads(path.with_suffix('.json').read_text())
    if file_hash(path) != cache_record['cache_sha256']:
        raise ValueError('DF cache SHA-256 differs from the CPU-prepared manifest.')
    experiment = experiment_class(geometry=row, basis_family='szp663_direct',
        core_primitives=6, jk_backend='df', auxbasis=auxbasis, integral_cache=path,
        implicit_tolerance=implicit_tolerance, cache_storage=cache_storage,
        scf_rescue_level_shift=scf_rescue_level_shift)
    report = experiment_metadata(experiment, jax)
    report.update(seed=seed, cache_path=str(path), cache_sha256=cache_record['cache_sha256'],
        cache_preparation=cache_record, mace_ran=False, adam_ran=False,
        training_scope='independent model per structure; no transferable training')

    def persist():
        atomic_json(directory / 'progress.json', report)

    persist()
    outputs = experiment.layout.reference_outputs()
    energy, gradient, info = experiment.evaluate(outputs)
    report.update(initial_energy_hartree=energy, initial_scf=info,
        coefficient_gradient_norm=float(np.linalg.norm(gradient)),
        implicit_gradient_finite=bool(np.isfinite(gradient).all()))
    mask = np.zeros(outputs.shape)
    for atom, slot, exponents in zip(experiment.layout.shell_atoms,
                                    experiment.layout.slots, experiment.layout.parameters.exponents):
        if slot >= 0:
            mask[atom, slot, :len(exponents)] = 1.
    direction = np.random.default_rng(seed).normal(size=outputs.shape) * mask
    direction /= np.linalg.norm(direction)
    coefficient_fd = directional_check(experiment.evaluate_value, outputs, gradient, direction,
        step=fd_step, atol=fd_atol, rtol=fd_rtol)
    report['coefficient_directional_check'] = coefficient_fd
    np.savez(directory / 'coefficient_check.npz', outputs=np.asarray(outputs),
             gradient=np.asarray(gradient), direction=direction)
    persist()
    if not coefficient_fd['passed']:
        raise RuntimeError('Implicit coefficient gradient fails reconverged SCF finite difference.')

    from flax import nnx
    import optax
    from jax.flatten_util import ravel_pytree
    from gradscf.data.molecule import atomic_number
    from gradscf.model.nnao import MACEBasisModel, build_graph
    # Fixed CHNOF vocabulary preserves head meaning across every pilot structure.
    model_config = dict(elements=(1, 6, 7, 8, 9), channels=8, num_interactions=2,
        max_ell=1, correlation=2, zero_init=True, basis_family='szp663_direct')
    model = MACEBasisModel(**model_config, rngs=nnx.Rngs(seed))
    graph = build_graph([atomic_number(symbol) for symbol in experiment.symbols],
                        experiment.coords, element_order=model.elements)
    definition, parameters, other = nnx.split(model, nnx.Param, ...)
    initial, unravel = ravel_pytree(parameters)

    @jax.jit
    def predict(vector):
        return nnx.merge(definition, unravel(vector), other)(graph)

    @jax.jit
    def backward(vector, cotangent):
        return jax.vjp(predict, vector)[1](cotangent)[0]

    def evaluate(vector):
        prediction = predict(jnp.asarray(vector))
        value, output_gradient, details = experiment.evaluate(prediction)
        parameter_gradient = np.asarray(backward(jnp.asarray(vector), output_gradient))
        if not np.isfinite(parameter_gradient).all():
            raise RuntimeError('MACE parameter gradient is not finite.')
        return value, parameter_gradient, details

    def value_only(vector):
        return experiment.evaluate_value(predict(jnp.asarray(vector)))

    initial_prediction = np.asarray(predict(initial))
    np.testing.assert_allclose(initial_prediction, outputs, atol=1e-12, rtol=1e-12)
    model_energy, parameter_gradient, model_info = evaluate(initial)
    np.testing.assert_allclose(model_energy, energy, atol=1e-10, rtol=0)
    norm = float(np.linalg.norm(parameter_gradient))
    parameter_direction = parameter_gradient / norm if norm > 1e-12 else (
        np.ones(parameter_gradient.shape) / np.sqrt(parameter_gradient.size))
    parameter_fd = directional_check(value_only, initial, parameter_gradient, parameter_direction,
        step=fd_step, atol=fd_atol, rtol=fd_rtol)
    report.update(mace_ran=True, model_config=model_config, parameter_count=int(initial.size),
        parameter_gradient_norm=norm, mace_initial_scf=model_info,
        parameter_directional_check=parameter_fd)
    persist()
    if not parameter_fd['passed']:
        raise RuntimeError('MACE parameter gradient fails reconverged SCF finite difference.')

    transform = optax.chain(optax.clip_by_global_norm(1.), optax.scale_by_adam())
    vector = initial; state = transform.init(vector)
    history = [dict(step=0, energy_hartree=model_energy, **model_info)]
    for step in range(1, adam_steps + 1):
        updates, state = transform.update(jnp.asarray(parameter_gradient), state, vector)
        vector = vector - learning_rate * updates
        model_energy, parameter_gradient, model_info = evaluate(vector)
        history.append(dict(step=step, energy_hartree=model_energy, **model_info))
        atomic_json(directory / 'adam_history.json', history)
        # Only valid, accepted parameter vectors are checkpointed.
        temporary = directory / 'checkpoint.tmp.npz'
        np.savez(temporary, parameters=np.asarray(vector),
                 adam_state=np.asarray(ravel_pytree(state)[0]), step=step,
                 initial_parameters=np.asarray(initial), outputs=np.asarray(predict(vector)))
        os.replace(temporary, directory / 'checkpoint.npz')
        report.update(adam_ran=True, accepted_adam_steps=step, adam_history=history)
        persist()
    decrease = energy - model_energy
    report.update(final_energy_hartree=model_energy, energy_decrease_hartree=decrease,
        accepted_adam_steps=adam_steps, adam_learning_rate=learning_rate,
        adam_gradient_clip_norm=1., adam_history=history,
        final_parameter_gradient_norm=float(np.linalg.norm(parameter_gradient)))
    persist()
    if adam_steps and (decrease < -1e-10 or (norm > 1e-9 and decrease <= 1e-10)):
        raise RuntimeError('Pilot Adam did not demonstrate energy reduction.')
    jax.clear_caches()
    return report
