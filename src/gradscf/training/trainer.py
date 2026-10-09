from __future__ import annotations

from typing import Any, Callable, Sequence
from math import isfinite

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from jaxtyping import Array, PRNGKeyArray

from .config import MolecularTrainingDatum, MolecularTrainingConfig
from .targets import density_on_grid, molecular_loss


def _tree_l2_norm(tree: Any, *, sanitize: bool = False) -> Array:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float32)

    total = jnp.asarray(0.0, dtype=jnp.float32)
    for leaf in leaves:
        arr = jnp.asarray(leaf)
        if sanitize:
            arr = jnp.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        total = total + jnp.sum(jnp.square(arr.astype(jnp.float32)))
    return jnp.sqrt(total)


def _tree_abs_max(tree: Any, *, sanitize: bool = False) -> Array:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float32)

    current_max = jnp.asarray(0.0, dtype=jnp.float32)
    for leaf in leaves:
        arr = jnp.asarray(leaf)
        if sanitize:
            arr = jnp.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        current_max = jnp.maximum(current_max, jnp.max(jnp.abs(arr.astype(jnp.float32))))
    return current_max


def _sanitize_gradients(tree: Any) -> tuple[Any, Array]:
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    if not leaves:
        return tree, jnp.asarray(0.0, dtype=jnp.float32)

    cleaned_leaves = []
    nonfinite_total = jnp.asarray(0.0, dtype=jnp.float32)
    element_total = jnp.asarray(0.0, dtype=jnp.float32)
    for leaf in leaves:
        arr = jnp.asarray(leaf)
        cleaned_leaves.append(jnp.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0))
        nonfinite_total = nonfinite_total + jnp.sum((~jnp.isfinite(arr)).astype(jnp.float32))
        element_total = element_total + jnp.asarray(arr.size, dtype=jnp.float32)

    cleaned_tree = jax.tree_util.tree_unflatten(treedef, cleaned_leaves)
    fraction = nonfinite_total / jnp.maximum(element_total, 1.0)
    return cleaned_tree, fraction


def _functional_apply(functional):
    """TrainState bookkeeping for external callbacks or existing Flax models."""
    apply = getattr(functional, 'apply', None)
    return apply if callable(apply) else functional.model.apply


def create_train_state(
    functional: Any,
    rng: PRNGKeyArray,
    sample_density: Array,
    tx: optax.GradientTransformation,
) -> TrainState:
    """Initialize a Flax/Optax train state for a neural XC functional."""

    params = functional.init(rng, sample_density)
    return TrainState.create(apply_fn=_functional_apply(functional), params=params, tx=tx)


def create_train_state_from_molecule(
    functional: Any,
    rng: PRNGKeyArray,
    molecule: Any,
    tx: optax.GradientTransformation,
) -> TrainState:
    """Initialize a train state from a molecule-like object's density."""

    if hasattr(functional, "init_from_molecule"):
        params = functional.init_from_molecule(rng, molecule)
        return TrainState.create(apply_fn=_functional_apply(functional), params=params, tx=tx)
    sample_density = density_on_grid(molecule)
    return create_train_state(functional, rng, sample_density, tx)


def make_molecular_loss_and_grad(
    functional: Any,
    training_config: MolecularTrainingConfig | None = None,
    predictor: Callable[[Any, Any], tuple[Array, Any]] | None = None,
):
    """Create a params-only molecular objective and gradient kernel."""

    config = MolecularTrainingConfig() if training_config is None else training_config

    def compute_loss(local_params, local_data):
        kwargs = {"training_config": config}
        if predictor is not None:
            kwargs["predictor"] = predictor
        return molecular_loss(
            local_params,
            functional,
            local_data,
            **kwargs,
        )

    loss_value_and_grad = jax.value_and_grad(
        compute_loss,
        has_aux=True,
        argnums=0,
    )

    def loss_and_grad(
        params: Any,
        data: MolecularTrainingDatum | Sequence[MolecularTrainingDatum],
    ):
        (loss, metrics), grads = loss_value_and_grad(params, data)
        cleaned_grads, nonfinite_grad_fraction = _sanitize_gradients(grads)
        metrics = dict(metrics)
        metrics["total_loss"] = loss
        metrics["grad_norm"] = jnp.asarray([_tree_l2_norm(cleaned_grads, sanitize=True)], dtype=loss.dtype)
        metrics["nonfinite_grad_fraction"] = jnp.asarray([nonfinite_grad_fraction], dtype=loss.dtype)
        return loss, metrics, cleaned_grads

    return loss_and_grad


def make_molecular_eval(
    functional: Any,
    training_config: MolecularTrainingConfig | None = None,
    predictor: Callable[[Any, Any], tuple[Array, Any]] | None = None,
):
    """Create a params-only evaluation kernel aligned with the train-step policy."""

    config = MolecularTrainingConfig() if training_config is None else training_config

    def evaluate(
        params: Any,
        data: MolecularTrainingDatum | Sequence[MolecularTrainingDatum],
    ):
        kwargs = {"training_config": config}
        if predictor is not None:
            kwargs["predictor"] = predictor
        return molecular_loss(
            params,
            functional,
            data,
            **kwargs,
        )

    return evaluate


def make_molecular_train_step(
    functional: Any,
    training_config: MolecularTrainingConfig | None = None,
    predictor: Callable[[Any, Any], tuple[Array, Any]] | None = None,
):
    """Create one molecular training step."""

    config = MolecularTrainingConfig() if training_config is None else training_config
    loss_and_grad = make_molecular_loss_and_grad(
        functional,
        training_config=training_config,
        predictor=predictor,
    )

    def train_step(
        state: TrainState,
        data: MolecularTrainingDatum | Sequence[MolecularTrainingDatum],
    ):
        loss, metrics, cleaned_grads = loss_and_grad(state.params, data)
        valid = jnp.isfinite(loss) & jnp.all(metrics["nonfinite_grad_fraction"] == 0)
        if config.requires_scf_convergence(functional):
            valid = valid & jnp.all(metrics["scf_converged"] == 1)
        new_state = jax.lax.cond(valid,
            lambda s: s.apply_gradients(grads=cleaned_grads), lambda s: s, state)
        param_delta = jax.tree_util.tree_map(lambda new, old: new - old, new_state.params, state.params)
        metrics = dict(metrics)
        metrics["update_accepted"] = valid
        metrics["param_update_norm"] = jnp.asarray(
            [_tree_l2_norm(param_delta, sanitize=True)],
            dtype=loss.dtype,
        )
        metrics["param_norm"] = jnp.asarray([_tree_l2_norm(state.params, sanitize=True)], dtype=loss.dtype)
        return new_state, metrics

    return train_step


_LOSS_FIELDS = {
    'energy': {'mse': 'e0_total_mse_weight', 'mae': 'e0_total_mae_weight'},
    'density': {'mse': 'grid_density_mse_weight'},
    'orbital_energy': {'mse': 'orbital_energy_mse_weight', 'mae': 'orbital_energy_mae_weight'},
    's1_energy': {'mse': 's1_total_mse_weight', 'mae': 's1_total_mae_weight'},
    'excitation': {'mse': 'excitation_gap_mse_weight', 'mae': 'excitation_gap_mae_weight'},
    'oscillator_strength': {'mse': 'oscillator_strength_mse_weight', 'mae': 'oscillator_strength_mae_weight'},
    'spectrum': {'mse': 'spectrum_mse_weight'},
    'xc_potential': {'mse': 'xc_potential_mse_weight'},
    'xc_kernel': {'mse': 'xc_kernel_mse_weight'},
}


class Trainer:
    """Object-style training through the shared loss, SCF and optimizer step.

    run(data, steps=N) returns self; kernel returns the resulting parameters.
    History contains step zero and the metrics AFTER each attempted update.
    Repeated run calls continue optimizer state and history. Replacing params
    or changing learning_rate restarts the optimizer/history explicitly.
    """
    def __init__(self, functional, *, params=None):
        self.functional = functional
        self.params = params
        self.mode = 'fixed_density'
        self.loss = {'energy': {'mse': 1.}}
        self.learning_rate = 1e-4
        self.seed = 0
        self.scf = {}
        self.adjoint = {}
        self.history = {}
        self.metrics = {}
        self.steps = 0
        self._state = None
        self._state_lr = None
        self._compiled = {}
        self._last_accepted = None

    def _configuration(self, mode=None):
        mode = self.mode if mode is None else mode
        if mode not in ('fixed_density', 'explicit', 'implicit'):
            raise ValueError('mode must be fixed_density, explicit or implicit.')
        settings = {}
        for target, terms in self.loss.items():
            if target not in _LOSS_FIELDS:
                raise ValueError(f'Unknown loss target: {target}')
            for metric, weight in terms.items():
                if metric not in _LOSS_FIELDS[target]:
                    raise ValueError(f'Unsupported {target} loss: {metric}')
                settings[_LOSS_FIELDS[target][metric]] = weight
        if mode == 'fixed_density' and any(settings.get(k, 0.) > 0 for k in
                ('grid_density_mse_weight', 'orbital_energy_mse_weight', 'orbital_energy_mae_weight')):
            raise ValueError('Density/orbital-energy supervision requires a self-consistent mode.')
        for name, value in self.scf.items():
            field = 'scf_' + name
            if field not in MolecularTrainingConfig.__dataclass_fields__ or name in ('gradient_mode',):
                raise ValueError(f'Unknown SCF option: {name}')
            settings[field] = value
        adjoint_fields = {'tolerance': 'scf_implicit_diff_tolerance',
                          'max_iter': 'scf_implicit_diff_max_iter',
                          'regularization': 'scf_implicit_diff_regularization'}
        for name, value in self.adjoint.items():
            if name not in adjoint_fields:
                raise ValueError(f'Unknown adjoint option: {name}')
            settings[adjoint_fields[name]] = value
        return MolecularTrainingConfig(
            mode='fixed_density' if mode == 'fixed_density' else 'self_consistent',
            scf_gradient_mode='implicit' if mode == 'fixed_density' else mode,
            **settings)

    @staticmethod
    def _samples(data):
        data = (data,) if isinstance(data, MolecularTrainingDatum) else tuple(data)
        if not data:
            raise ValueError('Training requires at least one Sample.')
        if not all(isinstance(x, MolecularTrainingDatum) for x in data):
            raise TypeError('data must contain training.Sample records.')
        return data

    def _initialize(self, data):
        if (jnp.shape(self.learning_rate) != ()
                or not isfinite(float(self.learning_rate)) or float(self.learning_rate) <= 0):
            raise ValueError('learning_rate must be a finite positive scalar.')
        if (self._state is not None and self.params is self._state.params
                and self.learning_rate == self._state_lr):
            return
        tx = optax.adam(self.learning_rate)
        if self.params is None:
            self._state = create_train_state_from_molecule(
                self.functional, jax.random.PRNGKey(self.seed), data[0].molecule, tx)
        else:
            self._state = TrainState.create(
                apply_fn=_functional_apply(self.functional), params=self.params, tx=tx)
        self.params = self._state.params
        self._state_lr = self.learning_rate
        self.history, self.metrics = {}, {}
        self.steps, self._last_accepted = 0, None

    def _functions(self, config, data):
        from ..scf.differentiable import _is_traceable_pytree
        use_jit = _is_traceable_pytree(data)
        key = (id(self.functional), config, use_jit)
        if key not in self._compiled:
            functions = (make_molecular_train_step(self.functional, config),
                         make_molecular_eval(self.functional, config))
            self._compiled[key] = tuple(jax.jit(fn) for fn in functions) if use_jit else functions
        return self._compiled[key]

    @staticmethod
    def _metrics(raw, data):
        weights = jnp.asarray([x.weight for x in data])
        def mean(key):
            values = jnp.asarray(raw[key])
            return jnp.sum(weights*values)/jnp.sum(weights)
        has_scf = raw['scf_converged'].size > 0
        return dict(loss=jnp.asarray(raw['total_loss']).reshape(()),
            energy=raw['predicted_e0_total_h'],
            energy_mse=mean('e0_total_mse'), energy_mae=mean('e0_total_mae'),
            density_mse=mean('grid_density_mse'),
            scf_converged=jnp.all(raw['scf_converged'] == 1) if has_scf else None,
            scf_cycles=jnp.max(raw['scf_cycles']) if has_scf else jnp.asarray(0))

    def evaluate(self, data, *, mode=None, params=None):
        """Evaluate loss and diagnostics in the requested mode; no optimizer step."""
        config, data = self._configuration(mode), self._samples(data)
        if params is None:
            self._initialize(data)
            params = self.params
        _, evaluate = self._functions(config, data)
        _, raw = evaluate(params, data)
        return self._metrics(raw, data)

    def predict(self, molecule, *, mode=None, params=None):
        """Return (energy, electronic_state), reusing the existing predictor."""
        from .predictors import make_ground_state_predictor
        from .config import Sample
        sample = Sample(molecule)
        if params is None:
            self._initialize((sample,))
            params = self.params
        molecule = sample.molecule
        predictor = make_ground_state_predictor(self.functional, training_config=self._configuration(mode))
        return predictor(params, molecule)

    def _record(self, metrics):
        record = dict(step=self.steps, optimizer_step=int(self._state.step),
                      update_accepted=self._last_accepted)
        record.update({k: None if v is None else v.tolist()
                       for k, v in jax.device_get(metrics).items()})
        replace_last = bool(self.history) and self.history['step'][-1] == self.steps
        for key, value in record.items():
            column = self.history.setdefault(key, [])
            if replace_last:
                column[-1] = value
            else:
                column.append(value)
        self.metrics = metrics

    def run(self, data, *, steps):
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
            raise ValueError('steps must be a nonnegative integer.')
        config, data = self._configuration(), self._samples(data)
        self._initialize(data)
        train_step, evaluate = self._functions(config, data)
        for _ in range(steps):
            state, raw = train_step(self._state, data)
            self._record(self._metrics(raw, data))
            self._state, self.params = state, state.params
            self._last_accepted = bool(raw['update_accepted'])
            self.steps += 1
        _, raw = evaluate(self.params, data)
        self._record(self._metrics(raw, data))
        return self

    def kernel(self, data, *, steps):
        return self.run(data, steps=steps).params
