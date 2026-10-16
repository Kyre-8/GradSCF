"""Harmonic vacuum overlaps in one common mass-weighted coordinate space.

The final normal coordinates obey Q_f = J Q_i + K, with orthogonal J.
Frequencies are positive oscillator energies in Hartree (hbar=1), and Q/K
use bohr*sqrt(electron mass). This module does not align molecules or replace
a nonorthogonal overlap of two projected molecular mode spaces by a rotation.

The Gaussian generating function is differentiated by JAX for the 0-0
probability. Higher final occupations use a host recurrence with an explicit
total-quanta cutoff; that enumeration is not a differentiable spectrum API.
"""
from itertools import combinations_with_replacement
from math import comb

import jax
import jax.numpy as jnp
import numpy as np


def _inputs(initial, final, rotation, shift):
    if not jax.config.x64_enabled:
        raise ValueError("Harmonic overlaps require JAX float64 enabled.")
    arrays = tuple(jnp.asarray(x) for x in (initial, final, rotation, shift))
    if any(jnp.iscomplexobj(x) for x in arrays):
        raise ValueError("Frequencies, rotation and shift must be real.")
    wi, wf, j, k = (x.astype(jnp.float64) for x in arrays)
    if wi.ndim != 1 or not wi.size or wf.shape != wi.shape:
        raise ValueError("Both frequency arrays must have the same nonempty shape.")
    if j.shape != (wi.size, wi.size) or k.shape != wi.shape:
        raise ValueError("Rotation and shift must match the number of modes.")
    finite = jnp.all(jnp.isfinite(wi)) & jnp.all(jnp.isfinite(wf))
    finite &= jnp.all(jnp.isfinite(j)) & jnp.all(jnp.isfinite(k))
    valid = finite & jnp.all(wi > 0) & jnp.all(wf > 0)
    orthogonal = jnp.max(jnp.abs(j.T@j-jnp.eye(wi.size))) < 1e-10
    if not isinstance(valid, jax.core.Tracer):
        if not bool(valid):
            raise ValueError("Harmonic inputs must be finite, with positive frequencies.")
        if not bool(orthogonal):
            raise ValueError("The normal-coordinate rotation must be orthogonal; projected raw J is not silently corrected.")
    return wi, wf, j, k, valid & orthogonal


def _gaussian_generator(initial, final, rotation, shift):
    """Vacuum generator C,r,R and initial-coordinate insertion mean/source.

    F(z)=sum_n <n_f|0_i> z**n/sqrt(n!)=C exp(r.z+z.R.z/2).
    Inserting Q_i under that Gaussian integral gives (mean+source@z)*F(z).
    """
    wi, wf, j, k, valid = _inputs(initial, final, rotation, shift)
    matrix = jnp.diag(wi)+j.T@(wf[:, None]*j)
    linear = -j.T@(wf*k)
    center = jnp.linalg.solve(matrix, linear)
    _, log_determinant = jnp.linalg.slogdet(matrix)
    log_vacuum = (.25*jnp.sum(jnp.log(wi))+.25*jnp.sum(jnp.log(wf))
        +.5*wi.size*jnp.log(2.)-.5*log_determinant
        -.5*jnp.vdot(k, wf*k)+.5*jnp.vdot(linear, center))
    root_final = jnp.sqrt(wf)
    vector = jnp.sqrt(2.)*root_final*(k+j@center)
    mixed = jnp.linalg.solve(matrix, j.T*root_final[None, :])
    quadratic = 2*(root_final[:, None]*j)@mixed-jnp.eye(wi.size)
    quadratic = .5*(quadratic+quadratic.T)
    return tuple(jnp.where(valid, value, jnp.nan)
                 for value in (jnp.exp(log_vacuum), vector, quadratic, center, jnp.sqrt(2.)*mixed))


def zero_zero_probability(initial, final, rotation, shift):
    """Differentiable |<0_f|0_i>|² for positive frequencies and fixed J/K convention.

    Frequency/shift sensitivities are ordinary JAX derivatives. A varying J
    must remain on the orthogonal manifold. Invalid traced inputs return NaN;
    invalid concrete inputs are rejected. No wavefunction Jacobian is omitted:
    the normalized Gaussian prefactors include frequency scaling, and |det J|=1.
    """
    amplitude, *_ = _gaussian_generator(initial, final, rotation, shift)
    return amplitude**2


def ground_state_overlaps(initial, final, rotation, shift, *, max_quanta=12):
    """Return (occupations, signed amplitudes) with sum(n_i) <= max_quanta.

    Host-side final-state enumeration is ordered by total occupation then
    lexicographically. Probabilities are amplitude**2 and are never renormalized
    after truncation. The explicit two-million-state bound limits this small
    workflow helper; it is not a large-molecule FC screening implementation.
    """
    if type(max_quanta) is not int or max_quanta < 0:
        raise ValueError("max_quanta must be a nonnegative static integer.")
    if any(isinstance(x, jax.core.Tracer) for x in (initial, final, rotation, shift)):
        raise TypeError("Occupation enumeration is host-side; use zero_zero_probability for JAX response.")
    vacuum, vector, quadratic, *_ = map(np.asarray, _gaussian_generator(initial, final, rotation, shift))
    dimension = vector.size
    if comb(max_quanta+dimension, dimension) > 2_000_000:
        raise ValueError("Total-quanta enumeration exceeds the two-million-state bound.")
    states = []
    for degree in range(max_quanta+1):
        level = []
        for occupied in combinations_with_replacement(range(dimension), degree):
            counts = np.bincount(np.asarray(occupied, dtype=int), minlength=dimension)
            level.append(tuple(counts.tolist()))
        states.extend(sorted(level))
    lookup = {state: index for index, state in enumerate(states)}
    amplitudes = np.empty(len(states), dtype=float)
    amplitudes[0] = float(vacuum)
    square_roots = np.sqrt(np.arange(max_quanta+1))
    for index, state in enumerate(states[1:], 1):
        first = next(mode for mode, count in enumerate(state) if count)
        predecessor = list(state)
        predecessor[first] -= 1
        value = vector[first]*amplitudes[lookup[tuple(predecessor)]]
        for mode, count in enumerate(predecessor):
            if count:
                before_pair = predecessor.copy()
                before_pair[mode] -= 1
                value += quadratic[first, mode]*square_roots[count]*amplitudes[lookup[tuple(before_pair)]]
        amplitudes[index] = value/square_roots[state[first]]
    if not np.isfinite(amplitudes).all():
        raise FloatingPointError("Harmonic overlap recurrence produced nonfinite amplitudes.")
    return np.asarray(states, dtype=np.int32), amplitudes


def ground_state_coordinate_overlaps(initial, final, rotation, shift, *, max_quanta=12):
    """Return occupations, <n_f|0_i>, and <n_f|Q_i,k|0_i> for a linear dipole.

    Coordinate matrix elements have shape (nstate,nmode), in the same initial
    mass-weighted coordinates used by the supplied dipole derivatives. This
    reuses the vacuum recurrence; no separate FC solver or overlap convention
    is introduced. The host-side cutoff and raw-tail contract are unchanged.
    """
    states, amplitudes = ground_state_overlaps(initial, final, rotation, shift, max_quanta=max_quanta)
    *_, mean, source = map(np.asarray, _gaussian_generator(initial, final, rotation, shift))
    coordinates = amplitudes[:, None]*mean[None, :]
    lookup = {tuple(state.tolist()): index for index, state in enumerate(states)}
    for mode in range(states.shape[1]):
        selected = np.flatnonzero(states[:, mode] > 0)
        predecessors = states[selected].copy()
        predecessors[:, mode] -= 1
        indices = np.fromiter((lookup[tuple(state.tolist())] for state in predecessors),
                              dtype=np.int64, count=len(predecessors))
        factor = np.sqrt(states[selected, mode])*amplitudes[indices]
        coordinates[selected] += factor[:, None]*source[:, mode][None, :]
    if not np.isfinite(coordinates).all():
        raise FloatingPointError("Harmonic coordinate overlaps are nonfinite.")
    return states, amplitudes, coordinates


def ground_state_quadratic_coordinate_overlaps(initial, final, rotation, shift, *, max_quanta=12):
    """Also return <n_f|Q_i,a Q_i,b|0_i>, with shape (nstate,nmode,nmode).

    The product Gaussian has covariance Sigma=(diag(wi)+J.T diag(wf) J)^-1.
    Its second insertion is [Sigma+(mean+Bz)(mean+Bz).T]*F(z). Reusing the
    first-coordinate overlaps evaluates this expression with one predecessor
    per occupied mode. Sigma must be retained even for identical oscillators.
    This host recurrence has the same raw cutoff contract as the linear API.
    """
    states, amplitudes, coordinates = ground_state_coordinate_overlaps(
        initial, final, rotation, shift, max_quanta=max_quanta)
    wi, wf, j, _, _ = _inputs(initial, final, rotation, shift)
    dimension = wi.size
    if len(states)*dimension**2 > 100_000_000:
        raise ValueError("Quadratic coordinate overlaps exceed the 100-million-element bound.")
    *_, mean, source = map(np.asarray, _gaussian_generator(initial, final, rotation, shift))
    covariance = np.asarray(jnp.linalg.solve(jnp.diag(wi)+j.T@(wf[:, None]*j), jnp.eye(dimension)))
    quadratic = amplitudes[:, None, None]*covariance[None, :, :]
    quadratic += mean[None, :, None]*coordinates[:, None, :]
    lookup = {tuple(state.tolist()): index for index, state in enumerate(states)}
    for mode in range(dimension):
        selected = np.flatnonzero(states[:, mode] > 0)
        predecessors = states[selected].copy()
        predecessors[:, mode] -= 1
        indices = np.fromiter((lookup[tuple(state.tolist())] for state in predecessors),
                              dtype=np.int64, count=len(predecessors))
        factor = np.sqrt(states[selected, mode])[:, None]*coordinates[indices]
        quadratic[selected] += source[None, :, mode, None]*factor[:, None, :]
    if not np.isfinite(quadratic).all():
        raise FloatingPointError("Harmonic quadratic coordinate overlaps are nonfinite.")
    return states, amplitudes, coordinates, quadratic


__all__ = ["ground_state_overlaps", "ground_state_coordinate_overlaps",
           "ground_state_quadratic_coordinate_overlaps", "zero_zero_probability"]
