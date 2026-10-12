"""Alternating coefficient/exponent training and molecular evaluation."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time
import traceback
import numpy as np

from . import trainer
from .. import runtime as pilot
from ..config import EXPONENT_KEYS, MODEL_CONFIG, PARENT_SOURCE, PARENT_CHECKPOINT_SHA

def trainable_parameter_mask(parameters):
    """Flatten NNX mutability metadata in the same order as ravel_pytree."""
    import jax
    return np.concatenate([np.full(leaf.size, variable.get_metadata().get('is_mutable', True), dtype=bool)
        for _, variable in parameters.flat_state() for leaf in jax.tree.leaves(variable)])

def filtered_scales(beta, keys):
    return {key: float(beta[EXPONENT_KEYS.index(key)]) for key in keys}

def check_energy_parity(primitive, contracted, tolerance=1e-8):
    if (set(primitive) != set(contracted) or any(not np.isfinite([primitive[key], contracted[key]]).all()
            or abs(primitive[key] - contracted[key]) > tolerance for key in primitive)):
        raise RuntimeError('Primitive/contracted DF energy parity failed (tolerance 1e-8 Ha).')

def powell_element(beta, element, rows, energies, evaluate, *, log_bound, maxfev, minimize=None):
    """Keep the best observed valid trial, including a budget-limited solve."""
    if minimize is None:
        from scipy.optimize import minimize
    indices = [i for i, key in enumerate(EXPONENT_KEYS) if key[0] == element]
    affected = [row['structure_id'] for row in rows if element in row['symbols']]
    initial = float(np.mean(list(energies.values())))
    best = dict(beta=np.array(beta, copy=True), energies=dict(energies), loss_hartree=initial)
    history = []; seen = {tuple(np.asarray(beta)[indices]): (initial, dict(energies))}

    def objective(point):
        candidate = np.array(beta, copy=True); candidate[indices] = point
        record = dict(element=element, beta=candidate.tolist(), affected=affected,
                      valid=False, objective_hartree='inf')
        try:
            if not np.isfinite(candidate).all() or np.any(np.abs(candidate) > log_bound + 1e-12):
                raise ValueError('Exponent candidate is outside its finite log bounds.')
            key = tuple(np.asarray(point))
            if key in seen:
                value, trial = seen[key]; record['cache_hit'] = True
            else:
                trial = dict(energies)
                for identifier in affected:
                    energy, details = evaluate(identifier, candidate)
                    if not np.isfinite(energy) or not trainer.valid_scf(details):
                        raise RuntimeError(f'{identifier}: invalid exponent-candidate SCF.')
                    trial[identifier] = float(energy)
                value = float(np.mean(list(trial.values()))); seen[key] = value, trial
            record.update(valid=True, objective_hartree=value)
            if value < best['loss_hartree'] - 1e-10:
                best.update(beta=candidate.copy(), energies=dict(trial), loss_hartree=value)
        except Exception as error:
            value = math.inf; record.update(error_type=type(error).__name__, error=str(error))
        history.append(record)
        return value

    result = minimize(objective, np.asarray(beta)[indices].copy(), method='Powell',
        bounds=[(-log_bound, log_bound)] * len(indices),
        options=dict(maxfev=maxfev, xtol=1e-3, ftol=1e-10))
    status = ('converged' if result.success else 'budget-limited'
              if getattr(result, 'status', None) == 1 or result.nfev >= maxfev else 'not-converged')
    return dict(**best, affected=affected, improved=best['loss_hartree'] < initial - 1e-10,
                status=status, optimizer_success=bool(result.success), nfev=int(result.nfev),
                message=str(result.message), history=history)

def run_joint(theta, beta, rows, optimizer, prepare, evaluate, forward, *, save,
              coefficient_steps, cycles, maxfev, log_bound, minimize=None, on_failure=None):
    """Transaction boundary includes valid full backward and rebuilt cache identity."""
    state = None; history = []; phase = 'initial'; started = time.perf_counter()
    trial_theta = np.array(theta, copy=True); trial_beta = np.array(beta, copy=True)
    def measured(parameters, scales, cache):
        loss, gradient, records = evaluate(parameters, scales, cache)
        if (not np.isfinite(loss) or np.asarray(gradient).shape != np.asarray(parameters).shape
                or not np.isfinite(gradient).all()
                or len(records) != len(rows)
                or {r['structure_id'] for r in records} != {r['structure_id'] for r in rows}
                or not all(np.isfinite(r.get('energy_hartree', np.nan)) for r in records)
                or abs(float(np.mean([r['energy_hartree'] for r in records])) - loss) > 1e-10
                or not all(trainer.valid_scf(row) for row in records)):
            raise RuntimeError('Invalid full-batch coefficient SCF/implicit gradient.')
        return float(loss), np.asarray(gradient), records
    def commit(candidate):
        candidate['history'] = deepcopy(history)
        try:
            save(candidate)
        except Exception:
            if history: history[-1]['accepted'] = False
            raise
        return candidate
    def coefficient_block(cycle, label):
        nonlocal state, phase, trial_theta, trial_beta
        for _ in range(coefficient_steps):
            phase = label; tick = time.perf_counter(); previous_loss = state['loss_hartree']
            trial_theta, trial_optimizer = optimizer.update(state['theta'], state['gradient'], state['optimizer'])
            trial_theta = np.asarray(trial_theta); trial_beta = state['beta']
            loss, gradient, records = measured(trial_theta, trial_beta, state['cache'])
            candidate = dict(state, theta=trial_theta.copy(), optimizer=trial_optimizer,
                loss_hartree=loss, gradient=gradient, molecules=records,
                energies={r['structure_id']: r['energy_hartree'] for r in records},
                theta_steps=state['theta_steps'] + 1)
            history.append(dict(phase=phase, cycle=cycle, accepted=True,
                previous_loss_hartree=previous_loss, loss_hartree=loss,
                elapsed_seconds=time.perf_counter() - tick))
            state = commit(candidate)
    try:
        cache = prepare(trial_beta, None, None)
        loss, gradient, records = measured(trial_theta, trial_beta, cache)
        history.append(dict(phase='initial', accepted=True, loss_hartree=loss,
                            elapsed_seconds=time.perf_counter() - started))
        state = commit(dict(theta=trial_theta, beta=trial_beta, optimizer=optimizer.init(trial_theta),
            gradient=gradient, loss_hartree=loss, molecules=records,
            energies={r['structure_id']: r['energy_hartree'] for r in records},
            cache=cache, theta_steps=0, alpha_steps=0))
        for cycle in range(cycles):
            coefficient_block(cycle, 'coeff-before')
            for element in dict.fromkeys(key[0] for key in EXPONENT_KEYS):
                if not any(element in row['symbols'] for row in rows): continue
                phase = 'alpha-by-element'; tick = time.perf_counter()
                proposal = powell_element(state['beta'], element, rows, state['energies'], forward,
                    log_bound=log_bound, maxfev=maxfev, minimize=minimize)
                record = dict(phase=phase, cycle=cycle, element=element,
                    optimizer_status=proposal['status'], optimizer_success=proposal['optimizer_success'],
                    previous_loss_hartree=state['loss_hartree'], nfev=proposal['nfev'],
                    message=proposal['message'], candidates=proposal['history'], accepted=False)
                history.append(record)
                if proposal['improved']:
                    phase = 'alpha-commit'; trial_theta = state['theta']; trial_beta = proposal['beta']
                    proposed_cache = prepare(trial_beta, proposal['affected'], state['cache'])
                    loss, gradient, records = measured(trial_theta, trial_beta, proposed_cache)
                    refreshed = {r['structure_id']: r['energy_hartree'] for r in records}
                    check_energy_parity(refreshed, proposal['energies'])
                    if loss >= state['loss_hartree'] - 1e-10:
                        raise RuntimeError('Refreshed exponent proposal is not a valid energy improvement.')
                    candidate = dict(state, beta=trial_beta.copy(), cache=proposed_cache,
                        gradient=gradient, loss_hartree=loss, molecules=records, energies=refreshed,
                        alpha_steps=state['alpha_steps'] + 1)
                    record.update(accepted=True, loss_hartree=loss, elapsed_seconds=time.perf_counter() - tick)
                    state = commit(candidate)
                record['elapsed_seconds'] = time.perf_counter() - tick
            coefficient_block(cycle, 'coeff-after')
        summary = dict(status='complete', history=history, elapsed_seconds=time.perf_counter() - started)
    except Exception as error:
        summary = dict(status='failed', phase=phase, error_type=type(error).__name__, error=str(error),
            traceback=traceback.format_exc(), history=history, elapsed_seconds=time.perf_counter() - started,
            failed_theta=np.asarray(trial_theta).tolist(), failed_beta=np.asarray(trial_beta).tolist())
        if on_failure is not None:
            try:
                on_failure(error, trial_theta, trial_beta, summary)
            except Exception as snapshot_error:
                summary['failure_snapshot_error'] = str(snapshot_error)
    summary.update(accepted_theta_steps=state['theta_steps'] if state else 0,
        accepted_alpha_steps=state['alpha_steps'] if state else 0,
        loss_hartree=state['loss_hartree'] if state else None)
    return state, summary

def read_parent(path, rows):
    if pilot.file_hash(path) != PARENT_CHECKPOINT_SHA:
        raise ValueError('Original best975 checkpoint SHA-256 bytes differ.')
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved['metadata'].item())); theta = np.array(saved['parameters'])
    identity = metadata['identity']
    if (metadata['epoch'] != 975 or theta.shape != (7171,) or theta.dtype != np.float64
            or not np.isfinite(theta).all() or identity.get('scientific_source_sha256') != PARENT_SOURCE
            or identity.get('seed') != 0 or identity.get('dtype') != 'float64'
            or identity.get('model_config') != json.loads(json.dumps(MODEL_CONFIG))
            or len(identity.get('structure_ids', [])) != 24):
        raise ValueError('Expected the original finite float64 24-structure best975 MACE checkpoint.')
    if any(row['structure_id'] not in identity['structure_ids'] for row in rows):
        raise ValueError('Joint pilot may use only structures from the original 24 training IDs.')
    return theta, identity

class JointEngine:
    """CPU native preparation, explicit GPU operands, shared contracted SCF JIT."""
    def __init__(self, rows, args, parent_identity):
        import jax
        import jax.numpy as jnp
        from flax import nnx
        from jax.flatten_util import ravel_pytree
        from gradscf import scf
        from gradscf.data.molecule import atomic_number
        from gradscf.model.nnao import MACEBasisModel, build_graph, prepare_direct_basis
        from gradscf.scf.core import _orthogonalizer, _diagonalize_fock, _build_density_from_occ
        from gradscf.scf.rks import RKSConfig, run_rks_from_integrals_traceable
        from gradscf.integrals.molecular.jk import build_jk_from_df
        from . import objective as experiment
        self.experiment = experiment.BasisRHF; self.jax = jax; self.jnp = jnp
        self.implicit_solver = experiment._df_rhf_solver
        self.rows = rows; self.args = args; self.parent_identity = parent_identity
        self.cpu = jax.devices('cpu')[0]
        self.device = jax.devices('gpu' if args.platform == 'cuda' else 'cpu')[0]
        self.contexts = {}; self.timings = []; self.generation = 0; self.theta = None
        self.prediction_theta = None; self.prediction_cache = {}
        with jax.default_device(self.device):
            model = MACEBasisModel(**MODEL_CONFIG, rngs=nnx.Rngs(0))
            definition, parameters, other = nnx.split(model, nnx.Param, ...)
            initial, unravel = ravel_pytree(parameters)
        self.trainable_mask = trainable_parameter_mask(parameters)
        if initial.shape != (7171,) or str(initial.dtype) != 'float64':
            raise RuntimeError('MACE tree differs from the parent float64 7171-parameter model.')
        pilot.verify_mace_source()
        self.predict = jax.jit(lambda vector, graph: nnx.merge(definition, unravel(vector), other)(graph))
        self.backward = jax.jit(lambda vector, graph, cotangent:
            jax.vjp(lambda value: self.predict(value, graph), vector)[1](cotangent)[0])
        cfg = RKSConfig(xc_spec='hf', jk_backend='df', max_cycle=150,
            conv_tol=1e-12, conv_tol_density=1e-10, conv_tol_grad=1e-9)
        def solve(s, h, factors, enuc, nelectron):
            n = s.shape[0]
            result, rescued, cycles = experiment._run_scf_with_rescue(run_rks_from_integrals_traceable,
                dict(overlap=s, hcore=h, eri=None, df_factors=factors, nelectron=nelectron,
                    nuclear_repulsion=enuc, ao=jnp.zeros((0, n)), ao_deriv1=jnp.zeros((4, 0, n)),
                    grid_weights=jnp.zeros(0)), cfg, rescue_level_shift=args.scf_rescue_level_shift)
            d = result.density_matrix; j, k = build_jk_from_df(factors, d); f = h + j - .5 * k
            energy = jnp.sum(d * h) + .5 * jnp.sum(d * j) - .25 * jnp.sum(d * k) + enuc
            _, c = _diagonalize_fock(f, _orthogonalizer(s, cfg.orthogonalization_eps))
            return energy, dict(converged=result.converged, scf_cycles=result.cycles,
                rescue_used=rescued, total_scf_cycles=cycles,
                orbital_residual=jnp.linalg.norm(f @ d @ s - s @ d @ f),
                fixed_point_residual=jnp.linalg.norm(_build_density_from_occ(c, result.mo_occ) - d),
                min_overlap_eigenvalue=jnp.linalg.eigvalsh(s)[0], reconstruction_error=jnp.abs(energy - result.total_energy))
        self.solve = jax.jit(solve, static_argnums=(4,))
        for row in rows:
            with jax.default_device(self.cpu):
                atom = list(zip(row['symbols'], row['coords_angstrom']))
                layout = prepare_direct_basis(atom, unit='Angstrom', cart=False, basis_family='szp663_direct', core_primitives=6)
                from ..integrals import prepare_native_context
                tick = time.perf_counter()
                context = prepare_native_context(layout, atom, args.auxbasis)
                enuc = scf.nuclear_repulsion_energy(layout.parameters.nuclear_coords, jnp.asarray(layout.topology.nuclear_charges))
                context['enuc'] = jax.device_put(enuc, self.device)
                self.timings.append(dict(phase='auxiliary-metric', structure_id=row['structure_id'], seconds=time.perf_counter() - tick))
            with jax.default_device(self.device):
                context['graph'] = build_graph([atomic_number(s) for s in row['symbols']],
                                               row['coords_angstrom'], element_order=model.elements)
            self.contexts[row['structure_id']] = context

    def predictions(self, theta):
        if self.prediction_theta is None or not np.array_equal(theta, self.prediction_theta):
            tick = time.perf_counter(); self.prediction_cache = {}
            for identifier, context in self.contexts.items():
                value = self.predict(self.jax.device_put(theta, self.device), context['graph']); value.block_until_ready()
                self.prediction_cache[identifier] = value
            self.prediction_theta = np.array(theta, copy=True)
            self.timings.append(dict(phase='nn-forward-batch', seconds=time.perf_counter() - tick))
        return self.prediction_cache

    def prepare(self, beta, affected, previous):
        experiments = dict(previous['experiments']) if previous else {}
        manifests = dict(previous['manifests']) if previous else {}
        for row in self.rows:
            identifier = row['structure_id']
            if affected is not None and identifier not in affected: continue
            context = self.contexts[identifier]; tick = time.perf_counter()
            with self.jax.default_device(self.cpu):
                ex = self.experiment(geometry=row, basis_family='szp663_direct', core_primitives=6,
                    jk_backend='df', auxbasis=self.args.auxbasis, implicit_tolerance=self.args.implicit_tolerance,
                    scf_rescue_level_shift=self.args.scf_rescue_level_shift,
                    log_exponent_scales=filtered_scales(beta, context['layout'].exponent_scale_keys),
                    df_metric_factor=context['metric'])
            if np.all(beta == 0.) and ex.integral_signature != self.parent_identity['caches'][identifier]['integral_signature']:
                raise ValueError(f'{identifier}: original geometry/primitive signature differs from parent training.')
            self.generation += 1
            path = Path(self.args.cache_dir) / hashlib.sha256(identifier.encode()).hexdigest()[:16] / f'{ex.integral_signature}-{self.generation}.npz'
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists(): raise FileExistsError('Joint cache paths are immutable; use a fresh cache directory.')
            ex.write_integral_cache(path)
            manifests[identifier] = dict(path=str(path.resolve()), sha256=pilot.file_hash(path),
                integral_signature=ex.integral_signature,
                log_exponent_scales={f'{key[0]}:{key[1]}': value for key, value in
                    filtered_scales(beta, context['layout'].exponent_scale_keys).items()}, df_shape=list(ex.rep.shape))
            self.timings.append(dict(phase='primitive-integrals-write', structure_id=identifier, seconds=time.perf_counter() - tick))
            tick = time.perf_counter()
            ex.ps, ex.ph, ex.rep, ex.enuc = self.jax.device_put((ex.ps, ex.ph, ex.rep, ex.enuc), self.device)
            self.jax.block_until_ready((ex.ps, ex.ph, ex.rep, ex.enuc)); experiments[identifier] = ex
            self.timings.append(dict(phase='primitive-transfer', structure_id=identifier, seconds=time.perf_counter() - tick))
        return dict(experiments=experiments, manifests=manifests)

    def forward_at(self, theta, identifier, beta):
        from gradscf.integrals.molecular.density_fitting import unpack_factors
        context = self.contexts[identifier]; outputs = self.predictions(theta)[identifier]
        tick = time.perf_counter()
        with self.jax.default_device(self.cpu):
            layout = context['layout'].with_log_exponent_scales(filtered_scales(beta, context['layout'].exponent_scale_keys))
            bound = layout.bind(self.jax.device_put(outputs, self.cpu))
            s = context['plan'].evaluate('overlap', bound)
            h = context['plan'].evaluate('kinetic', bound) + context['plan'].evaluate('nuclear', bound)
            packed = context['aux'].factors(bound, context['ap'], metric_factor=context['metric'])
            factors = unpack_factors(packed, layout.topology.nao)
            self.jax.block_until_ready((s, h, factors))
        self.timings.append(dict(phase='contracted-integrals', structure_id=identifier, seconds=time.perf_counter() - tick))
        tick = time.perf_counter(); s, h, factors = self.jax.device_put((s, h, factors), self.device)
        self.jax.block_until_ready((s, h, factors))
        self.timings.append(dict(phase='contracted-transfer', structure_id=identifier, seconds=time.perf_counter() - tick))
        tick = time.perf_counter(); energy, info = self.solve(s, h, factors, context['enuc'], layout.nelectron)
        self.jax.block_until_ready((energy, info)); details = self.experiment._checked_info(energy, info)
        self.timings.append(dict(phase='contracted-scf-validation', structure_id=identifier, seconds=time.perf_counter() - tick))
        return float(energy), details

    def forward(self, identifier, beta):
        return self.forward_at(self.theta, identifier, beta)

    def evaluate(self, theta, beta, cache):
        outputs = self.predictions(theta); evaluators = []
        for row in self.rows:
            identifier = row['structure_id']; ex = cache['experiments'][identifier]
            context = self.contexts[identifier]
            def evaluate(vector, identifier=identifier, ex=ex, context=context):
                tick = time.perf_counter(); value, cotangent, details = ex.evaluate(outputs[identifier])
                self.jax.block_until_ready(cotangent)
                self.timings.append(dict(phase='primitive-scf-implicit', structure_id=identifier, seconds=time.perf_counter() - tick))
                tick = time.perf_counter()
                gradient = self.backward(self.jax.device_put(vector, self.device), context['graph'], cotangent)
                gradient.block_until_ready()
                self.timings.append(dict(phase='nn-vjp', structure_id=identifier, seconds=time.perf_counter() - tick))
                return value, np.asarray(gradient), details
            evaluators.append((identifier, evaluate))
        loss, gradient, records = trainer.evaluate_batch(theta, evaluators)
        native = {row['structure_id']: self.forward_at(theta, row['structure_id'], beta)[0] for row in self.rows}
        check_energy_parity({r['structure_id']: r['energy_hartree'] for r in records}, native)
        for record in records: record['native_parity_error_hartree'] = abs(record['energy_hartree'] - native[record['structure_id']])
        self.theta = np.array(theta, copy=True)
        return loss, gradient, records
