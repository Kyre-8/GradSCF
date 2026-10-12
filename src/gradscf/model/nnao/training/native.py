"""Shared MACE/native basis gradients through implicit RHF."""
from __future__ import annotations
import json
import time
import numpy as np

from . import alternating as base
from ..config import EXPONENT_KEYS
from ..integrals.variable_exponents import native_projection
MODEL_CONFIG = json.loads(json.dumps(base.MODEL_CONFIG))

def compilation_cache_path(args, native_hash):
    """Keep custom cache roots isolated across native FFI binary versions too."""
    root=args.compilation_cache_dir or args.output_dir/'compilation-cache'
    return root/native_hash[:16]

def read_initial_parameters(path, *, log_bound):
    """Load compatible MACE weights, retaining joint log-exponent scales."""
    with np.load(path, allow_pickle=False) as saved:
        vector=np.array(saved['parameters'])
        metadata=json.loads(str(saved['metadata'].item()))
    identity=metadata.get('identity',{})
    if identity.get('model_config') != MODEL_CONFIG:
        raise ValueError('Checkpoint MACE model configuration differs.')
    if vector.dtype != np.float64 or vector.shape not in {(7171,), (7185,)} or not np.isfinite(vector).all():
        raise ValueError('Expected finite float64 MACE or joint parameters.')
    if vector.size == 7185:
        if identity.get('exponent_keys') != [list(k) for k in base.EXPONENT_KEYS]:
            raise ValueError('Checkpoint exponent channel order differs.')
        theta,beta=vector[:-14],vector[-14:]
    else:
        theta,beta=vector,np.zeros(14)
    if np.any(np.abs(beta)>log_bound+1e-12):
        raise ValueError('Checkpoint exponent scales exceed the requested log bound.')
    return theta,beta,metadata

class JointAdam(base.trainer.Adam):
    """Separate learning rates, shared clipping, projected log-exponent bounds."""
    def __init__(self, ntheta, *, learning_rate, exponent_learning_rate, log_bound, trainable_mask=None):
        super().__init__(1.)
        if min(learning_rate, exponent_learning_rate, log_bound) <= 0:
            raise ValueError('Learning rates and exponent bound must be positive.')
        self.ntheta = ntheta
        self.coefficient_rate = learning_rate
        self.exponent_rate = exponent_learning_rate
        self.log_bound = log_bound
        self.trainable_mask = (np.ones(ntheta, dtype=bool) if trainable_mask is None
                               else np.asarray(trainable_mask, dtype=bool))
        if self.trainable_mask.shape != (ntheta,):
            raise ValueError('Trainable mask must match the flat MACE vector.')

    def _step(self, vector, gradient, state):
        import jax.numpy as jnp
        import optax
        gradient = gradient.at[:self.ntheta].set(gradient[:self.ntheta]*self.trainable_mask)
        updates, state = self.transform.update(gradient, state, vector)
        rates = jnp.where(jnp.arange(vector.size) < self.ntheta,
                          self.coefficient_rate, self.exponent_rate)
        candidate = optax.apply_updates(vector, updates*rates)
        candidate = candidate.at[:self.ntheta].set(jnp.where(
            self.trainable_mask, candidate[:self.ntheta], vector[:self.ntheta]))
        candidate = candidate.at[self.ntheta:].set(
            jnp.clip(candidate[self.ntheta:], -self.log_bound, self.log_bound))
        return candidate, state

    def learning_rate_at(self, count):
        return self.coefficient_rate

class NativeJointEngine(base.JointEngine):
    """Reuse model/graph/metric setup; bypass the legacy primitive-cache path."""
    def __init__(self, rows, args, parent_identity, parent_energies):
        super().__init__(rows, args, parent_identity)
        from gradscf.tools.jax_runtime import BoundedJitCache
        self.cpu_jit_cache=BoundedJitCache(max_entries=getattr(args,'cpu_jit_cache_size',8))
        self.parent_energies = parent_energies
        self.batch = 0
        for context in self.contexts.values():
            context['physical'], context['pullback'] = native_projection(context, EXPONENT_KEYS)
            context['value'], context['gradient'] = self.implicit_solver(
                context['layout'].nelectron, args.implicit_tolerance, args.scf_rescue_level_shift)

    def molecule(self, vector, identifier, *, gradient=True):
        jax = self.jax
        theta, beta = vector[:-len(base.EXPONENT_KEYS)], vector[-len(base.EXPONENT_KEYS):]
        context = self.contexts[identifier]
        self.cpu_jit_cache.activate(identifier,(context['physical'],context['pullback']))
        tick = time.perf_counter()
        theta = jax.device_put(theta, self.device)
        outputs = self.predict(theta, context['graph'])
        with jax.default_device(self.cpu):
            outputs_cpu, beta_cpu = jax.device_put((outputs, beta), self.cpu)
            physical = context['physical'](outputs_cpu, beta_cpu)
            jax.block_until_ready(physical)
        integral_seconds = time.perf_counter()-tick
        tick = time.perf_counter()
        inputs = jax.device_put(physical, self.device)
        if gradient:
            (value, info), physical_cotangent = context['gradient'](inputs, context['enuc'])
        else:
            value, info = context['value'](inputs, context['enuc'])
        details = self.experiment._checked_info(value, info)
        scf_seconds = time.perf_counter()-tick
        if not gradient:
            return float(value), details
        tick = time.perf_counter()
        with jax.default_device(self.cpu):
            output_cotangent, beta_cotangent = context['pullback'](
                outputs_cpu, beta_cpu, jax.device_put(physical_cotangent, self.cpu))
            jax.block_until_ready((output_cotangent, beta_cotangent))
        native_backward_seconds = time.perf_counter()-tick
        theta_cotangent = self.backward(theta, context['graph'], jax.device_put(output_cotangent, self.device))
        result = np.concatenate((np.asarray(theta_cotangent), np.asarray(beta_cotangent)))
        details.update(integral_seconds=integral_seconds, scf_seconds=scf_seconds,
            native_backward_seconds=native_backward_seconds,
            exponent_gradient_norm=float(np.linalg.norm(result[-len(base.EXPONENT_KEYS):])),
            cpu_jit_cache_entries=len(self.cpu_jit_cache),
            cpu_jit_cache_evictions=self.cpu_jit_cache.evictions,
            jit_cache_global_clears=self.cpu_jit_cache.global_clears,
            mappings_before_dispatch=self.cpu_jit_cache.mapping_count,
            mapping_limit=self.cpu_jit_cache.mapping_limit)
        return float(value), result, details

    def evaluate(self, vector):
        beta = np.asarray(vector[-len(base.EXPONENT_KEYS):])
        if not np.isfinite(beta).all() or np.any(np.abs(beta) > self.args.log_bound+1e-12):
            raise ValueError('Invalid bounded exponent scales.')
        self.batch += 1
        evaluators = [(r['structure_id'], lambda x, key=r['structure_id']: self.molecule(x, key)) for r in self.rows]
        def progress(records):
            base.pilot.atomic_json(self.args.output_dir/'batch-progress.json', dict(batch=self.batch,
                evaluated=len(records), total=len(self.rows), last=records[-1], beta=beta.tolist()))
        loss, g, records = base.trainer.evaluate_batch(vector, evaluators, progress=progress)
        if self.parent_energies is not None:
            if np.any(beta != 0.):
                raise ValueError('Parent parity gate requires zero initial exponent scales.')
            # Preserve the original integral/SCF parity check before changing
            # root; stable energies are a separate variational baseline.
            actual = {r['structure_id']: r['initial_scf_energy'] for r in records}
            base.check_energy_parity(self.parent_energies, actual, tolerance=1e-7)
            base.pilot.atomic_json(self.args.output_dir/'parent-parity.json', dict(
                max_absolute_error_hartree=max(abs(actual[k]-self.parent_energies[k]) for k in actual),
                energies=actual, tolerance_hartree=1e-7, all_gradients_finite=True,
                stable_energies={r['structure_id']:r['energy_hartree'] for r in records}))
            self.parent_energies = None
        return loss, g, records

    def check_gradient(self, vector, identifier):
        direction = np.sin(np.arange(vector.size)+1.)
        direction[:-14] *= self.trainable_mask
        direction[:-14] /= np.linalg.norm(direction[:-14])
        direction[-14:] *= .1
        value, g, details = self.molecule(vector, identifier)
        analytic = float(g@direction)
        h = 1e-4
        energies = {k: self.molecule(vector+k*h*direction, identifier, gradient=False)[0]
                    for k in (-2, -1, 1, 2)}
        fd = (8*(energies[1]-energies[-1])-energies[2]+energies[-2])/(12*h)
        if not np.isfinite(analytic) or abs(analytic-fd) > 2e-6+2e-5*abs(fd):
            raise RuntimeError(f'{identifier}: joint GPU/native gradient mismatch: {analytic} vs {fd}')
        return dict(structure_id=identifier, energy_hartree=value, analytic=analytic,
            finite_difference=fd, absolute_error=abs(analytic-fd), step=h, diagnostics=details)
