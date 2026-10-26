"""Libcint ATM/BAS topology tables and differentiable ENV packing.

Nuclear atoms and floating basis-center atoms have independent storage.
"""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from ...basis.types import check_parameter_shapes
from ...basis.normalization import normalized_shell_coefficients




def pack(plan, parameters, *, origin=None):
    """Pack libcint ATM/BAS/ENV without freezing dynamic numerical values.

    Nuclear atoms and zero-charge basis-center atoms are separate. This
    permits floating centers without spuriously moving nuclear potentials.
    """
    check_parameter_shapes(plan.topology, parameters)
    coefficients = tuple(normalized_shell_coefficients(l, a, c)
                         for l, a, c in zip(plan.topology.angular_momenta,
                                           parameters.exponents, parameters.coefficients))
    return pack_environment(plan, parameters, coefficients, origin=origin)

def pack_environment(plan, parameters, coefficients, *, origin=None):
    check_parameter_shapes(plan.topology, parameters)
    if not jax.config.x64_enabled:
        raise ValueError("The native integral backend requires JAX float64 enabled.")
    env = jnp.zeros(plan.env_size, dtype=jnp.float64)
    if origin is not None:
        origin = jnp.asarray(origin)
        if origin.shape != (3,):
            raise ValueError("Integral origin must have shape (3,).")
        env = env.at[1:4].set(origin)
    coords = jnp.concatenate([parameters.nuclear_coords, parameters.centers], axis=0)
    for i in range(len(plan.atm)):
        offset = int(plan.atm[i, 1])
        env = env.at[offset:offset+3].set(coords[i])
    for i, (a, c) in enumerate(zip(parameters.exponents, coefficients)):
        ae, ce = int(plan.bas[i, 5]), int(plan.bas[i, 6])
        env = env.at[ae:ae+a.size].set(a)
        env = env.at[ce:ce+c.size].set(c.T.ravel())
    return plan.atm, plan.bas, env

def coefficient_data(plan, parameters, *, active_shells=None, separate_exponents=False):
    """Separate active normalized coefficients and optionally exponents.

    Normalization follows the actual exponents in JAX. With separate
    exponents, their ENV placeholders are constant so native exponent
    products and the normalization chain both contribute to the gradient.
    ``active_shells`` selects leading orbital shells for a combined RI plan.
    """
    check_parameter_shapes(plan.topology, parameters)
    count = len(plan.bas) if active_shells is None else int(active_shells)
    if not 0 < count <= len(plan.bas):
        raise ValueError("Invalid active coefficient shell count.")
    normalized = tuple(normalized_shell_coefficients(l, a, c)
                       for l, a, c in zip(plan.topology.angular_momenta,
                                         parameters.exponents, parameters.coefficients))
    fixed = tuple(jnp.zeros_like(c) if i < count else c
                  for i, c in enumerate(normalized))
    packed_parameters = parameters
    if separate_exponents:
        packed_parameters = replace(parameters, exponents=tuple(
            jnp.ones_like(a) if i < count else a
            for i,a in enumerate(parameters.exponents)))
    atm, bas, env = pack_environment(plan, packed_parameters, fixed)
    coefficients = jnp.concatenate(tuple(c.T.ravel() for c in normalized[:count]))
    if separate_exponents:
        return atm, bas, env, coefficients, jnp.concatenate(parameters.exponents[:count])
    return atm, bas, env, coefficients


def build_tables(topology):
    nnuc = len(topology.nuclear_charges)
    nsh = len(topology.angular_momenta)
    atm = np.zeros((nnuc+nsh, 6), dtype=np.int32)
    atm[:nnuc, 0] = topology.nuclear_charges
    atm[:, 2] = 1  # point nuclear model; basis-center atoms have zero charge
    atm[:, 1] = 20 + np.arange(nnuc+nsh)*3
    offset = 20 + 3*(nnuc+nsh)
    bas = np.zeros((nsh, 8), dtype=np.int32)
    for i, (l, np_, nc) in enumerate(zip(topology.angular_momenta,
                                        topology.primitive_counts, topology.contraction_counts)):
        bas[i, :4] = (nnuc+i, l, np_, nc)
        bas[i, 5] = offset
        bas[i, 6] = offset + np_
        offset += np_ + np_*nc
    atm.setflags(write=False)
    bas.setflags(write=False)
    return atm, bas, offset
