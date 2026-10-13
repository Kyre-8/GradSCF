from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree


XCEnergyFn = Callable[[PyTree, Any, Array], Array]


@dataclass(frozen=True)
class XCEnergyPotentialResult:
    """XC energy and density-matrix derivative for SCF Fock construction."""

    xc_energy: Array
    vxc_matrix: Array
    exact_exchange_fraction: Array
    extra_fock_matrix: Array
    aux: Any | None = None


def xc_energy_and_potential_from_density(
    params: PyTree,
    *,
    molecule: Any,
    density: Array,
    xc_energy_fn: XCEnergyFn,
    exact_exchange_fraction: Array | float = 0.0,
    extra_fock_matrix: Array | None = None,
    symmetrize: bool = True,
    has_aux: bool = False,
    density_weights: Array | None = None,
) -> XCEnergyPotentialResult:
    """Differentiate a real scalar XC energy into the physical AO potential.

    For complex Hermitian densities, ``dE = Re sum_b w_b Tr(V_b dD_b)``.
    Conjugate JAX's gradient and divide by the supplied block weights (e.g.
    normalized k-point weights). Molecular densities use unit weights.
    """

    density_arr = jnp.asarray(density)

    aux = None
    if has_aux:

        def _energy_for_density(density_var: Array) -> tuple[Array, Any]:
            energy, local_aux = xc_energy_fn(params, molecule, density_var)
            return jnp.asarray(energy), local_aux

        (xc_energy, aux), vxc_matrix = jax.value_and_grad(
            _energy_for_density,
            has_aux=True,
        )(density_arr)
    else:

        def _energy_for_density(density_var: Array) -> Array:
            return jnp.asarray(xc_energy_fn(params, molecule, density_var))

        xc_energy, vxc_matrix = jax.value_and_grad(_energy_for_density)(density_arr)
    # For a real energy, JAX's complex gradient is conjugate to the physical
    # Hermitian potential in dE = Re sum_b w_b Tr(V_b dD_b).
    vxc_matrix = vxc_matrix.conj()
    if density_weights is not None:
        weight = jnp.broadcast_to(jnp.asarray(density_weights),density_arr.shape[:-2])
        valid = jnp.all(jnp.isfinite(weight)&(weight>0))
        vxc_matrix = vxc_matrix/jnp.where(weight>0,weight,1.)[...,None,None]*jnp.where(valid,1.,jnp.nan)
    if symmetrize:
        vxc_matrix = 0.5 * (vxc_matrix + jnp.swapaxes(vxc_matrix.conj(), -1, -2))

    if extra_fock_matrix is None:
        extra_fock = jnp.zeros_like(vxc_matrix)
    else:
        extra_fock = jnp.asarray(extra_fock_matrix, dtype=vxc_matrix.dtype)
        extra_fock = jnp.nan_to_num(extra_fock, nan=0.0, posinf=0.0, neginf=0.0)
        if symmetrize:
            extra_fock = 0.5 * (extra_fock + jnp.swapaxes(extra_fock.conj(), -1, -2))

    alpha = jnp.asarray(exact_exchange_fraction, dtype=density_arr.real.dtype)
    alpha = jnp.clip(jnp.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

    return XCEnergyPotentialResult(
        xc_energy=jnp.asarray(xc_energy, dtype=density_arr.real.dtype),
        vxc_matrix=vxc_matrix,
        exact_exchange_fraction=alpha,
        extra_fock_matrix=extra_fock,
        aux=aux,
    )


def xc_kernel_action(params, *, molecule, density, tangent, xc_energy_fn, density_weights=None):
    """Full AO density-matrix XC Hessian action, preserving parameter AD.

    The derivative includes feature construction. No locality assumption or
    (nao**4) Hessian allocation is made. Spin-stacked densities are supported.
    """
    density = jnp.asarray(density)
    tangent = jnp.asarray(tangent)
    if tangent.shape != density.shape:
        raise ValueError('XC kernel tangent must have the density shape.')
    def potential(dm):
        return xc_energy_and_potential_from_density(
            params, molecule=molecule, density=dm, xc_energy_fn=xc_energy_fn,density_weights=density_weights,
        ).vxc_matrix
    return jax.jvp(potential, (density,), (tangent,))[1]
