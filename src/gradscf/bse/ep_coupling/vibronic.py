"""Harmonic vibrational-space assembly; numerical solves belong to solvers."""

import jax.numpy as jnp
import numpy as np

from ...gw.ep_coupling.types import _require
from .types import validate_model


def _partitions(total, modes):
    if modes == 1:
        yield (total,)
    else:
        for first in range(total + 1):
            for rest in _partitions(total - first, modes - 1):
                yield (first,) + rest


def _steps(state, mode):
    for change in (-1, 1):
        occupation = state[mode] + change
        if occupation >= 0:
            target = list(state)
            target[mode] = occupation
            yield tuple(target), np.sqrt(state[mode] + (change == 1))


def vibronic_hamiltonian(hamiltonian, model, dipoles, *, max_quanta=6):
    """Return a finite vibronic Hamiltonian (Ha) and Condon moments (bohr).

    H = H_X + sum_l omega_l b_l† b_l + G_l (b_l+b_l†)
        + 1/2 sum_lm Lambda_lm (b_l+b_l†)(b_m+b_m†).
    The basis contains all occupations with sum(n_l) <= max_quanta, ordered
    by total occupation, then lexicographically, with electronic state last.
    Ground-surface zero-point energy is subtracted. The dipole creates an
    excitation from the ground vibrational vacuum: this is a zero-temperature
    Condon model, without thermally occupied initial levels or dipole response.

    Quadratic vertices are relative to the ground surface. Products are formed
    BEFORE projecting into the finite basis, retaining virtual intermediates
    outside the cutoff. Do not add a separate DW shift to this Hamiltonian.
    Increasing max_quanta is necessary to establish convergence. A finite
    matrix does not establish stability of the infinite quadratic model.

    This dense assembly is JIT/AD compatible for static max_quanta; use shared
    eigensolvers or resolvents for response. Individual degenerate eigenvector
    derivatives remain subject to the shared solver's observable constraints.
    """
    if type(max_quanta) is not int or max_quanta < 0:
        raise ValueError("max_quanta must be a nonnegative static integer")
    h, d = jnp.asarray(hamiltonian), jnp.asarray(dipoles)
    if (
        h.ndim not in (1, 2)
        or not h.shape[0]
        or (h.ndim == 2 and h.shape[0] != h.shape[1])
    ):
        raise ValueError(
            "hamiltonian must be a nonempty energy vector or square matrix"
        )
    if h.ndim == 1 and jnp.iscomplexobj(h):
        raise ValueError("Excitation energies must be real")
    h = jnp.diag(h) if h.ndim == 1 else h
    validate_model(model, nstates=h.shape[0])
    if d.shape != (h.shape[0], 3):
        raise ValueError("dipoles must have shape (nstate,3)")
    _require(
        jnp.all(jnp.isfinite(h))
        & jnp.all(jnp.isfinite(d))
        & jnp.allclose(h, h.conj().T, rtol=1e-10, atol=1e-12),
        "Hamiltonian must be finite Hermitian and dipoles finite",
    )
    energies, g = jnp.asarray(model.energies), jnp.asarray(model.couplings)
    modes, states = g.shape[:2]
    basis = tuple(
        s for total in range(max_quanta + 1) for s in _partitions(total, modes)
    )
    index = {state: i for i, state in enumerate(basis)}
    size = len(basis)
    dtype = jnp.result_type(
        h, d, energies, g, 0.0 if model.quadratic is None else model.quadratic
    )
    identity = jnp.eye(size, dtype=dtype)
    matrix = jnp.kron(identity, h.astype(dtype)) + jnp.kron(
        jnp.diag(jnp.asarray(basis) @ energies), jnp.eye(states, dtype=dtype)
    )

    def coordinate(first, second=None):
        operator = np.zeros((size, size))
        for col, state in enumerate(basis):
            for intermediate, a in _steps(state, first):
                targets = (
                    ((intermediate, 1.0),)
                    if second is None
                    else _steps(intermediate, second)
                )
                for target, b in targets:
                    if target in index:
                        operator[index[target], col] += a * b
        return jnp.asarray(operator, dtype=dtype)

    for mode in range(modes):
        matrix += jnp.kron(coordinate(mode), g[mode].astype(dtype))
    if model.quadratic is not None:
        quadratic = jnp.asarray(model.quadratic, dtype=dtype)
        for first in range(modes):
            seconds = (first,) if quadratic.ndim == 3 else range(modes)
            for second in seconds:
                vertex = (
                    quadratic[first]
                    if quadratic.ndim == 3
                    else quadratic[first, second]
                )
                matrix += 0.5 * jnp.kron(coordinate(first, second), vertex)
    moments = jnp.zeros((size * states, 3), dtype=dtype).at[:states].set(d)
    return matrix, moments


__all__ = ["vibronic_hamiltonian"]
