"""Pure JAX force supervision for differentiable scalar energy callbacks.

Coordinates are in Bohr. An energy callback returning Hartree therefore gives
forces in Hartree/Bohr. The callback owns the electronic model, SCF derivative
mode, and convergence checks; these helpers neither change nor approximate its
derivatives. Force training requires its mixed parameter/coordinate derivatives.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp


EnergyFn = Callable[[Any, jax.Array], jax.Array]


class EnergyAndForces(NamedTuple):
    """JAX PyTree containing scalar energy and ``(natom, 3)`` forces."""

    energy: jax.Array
    forces: jax.Array


def _coordinates_array(coordinates: jax.Array) -> jax.Array:
    coordinates = jnp.asarray(coordinates)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3 or coordinates.shape[0] == 0:
        raise ValueError("coordinates must have shape (natom, 3) with natom > 0")
    if not jnp.issubdtype(coordinates.dtype, jnp.floating):
        raise TypeError("coordinates must have a real floating-point dtype")
    return coordinates


def energy_and_forces(
    energy_fn: EnergyFn,
    params: Any,
    coordinates: jax.Array,
) -> EnergyAndForces:
    """Evaluate ``energy_fn(params, coordinates)`` and ``-dE/dcoordinates``.

    ``energy_fn`` must return a real scalar. Parameters may be any JAX PyTree;
    coordinates must be a real floating-point array of shape ``(natom, 3)``.
    All differentiation paths are retained for nested ``grad`` and ``jit``.
    Close over ``energy_fn`` (or mark it static) when applying ``jax.jit``.
    """
    coordinates = _coordinates_array(coordinates)
    energy, gradient = jax.value_and_grad(energy_fn, argnums=1)(params, coordinates)
    return EnergyAndForces(energy=energy, forces=-gradient)


def force_matching_loss(
    energy_fn: EnergyFn,
    params: Any,
    coordinates: jax.Array,
    target_forces: jax.Array,
) -> jax.Array:
    """Return ``sum((forces - target_forces)**2) / (2 * 3 * natom)``.

    The mean is over every atom and Cartesian component, with no batch axis,
    mask, or additional energy target. Targets must match the coordinate shape
    exactly, have a real floating-point dtype, and use the predicted force units.
    The scalar loss has squared force units. Inputs retain their floating dtype
    subject to JAX's configured precision and normal type promotion rules.
    """
    coordinates = _coordinates_array(coordinates)
    target_forces = jnp.asarray(target_forces)
    if target_forces.shape != coordinates.shape:
        raise ValueError("target_forces must have the same (natom, 3) shape as coordinates")
    if not jnp.issubdtype(target_forces.dtype, jnp.floating):
        raise TypeError("target_forces must have a real floating-point dtype")
    prediction = energy_and_forces(energy_fn, params, coordinates)
    return 0.5 * jnp.mean(jnp.square(prediction.forces - target_forces))


def make_force_loss_and_grad(
    energy_fn: EnergyFn,
) -> Callable[[Any, jax.Array, jax.Array], tuple[jax.Array, Any]]:
    """Return ``(params, coordinates, target_forces) -> (loss, params_grad)``.

    The returned pure function differentiates only with respect to ``params``
    and can be wrapped in ``jax.jit`` or used with an external Optax optimizer.
    No optimization state, compilation, or gradient truncation is introduced.
    """
    def loss(params: Any, coordinates: jax.Array, target_forces: jax.Array) -> jax.Array:
        return force_matching_loss(energy_fn, params, coordinates, target_forces)

    return jax.value_and_grad(loss)
