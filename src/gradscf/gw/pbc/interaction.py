"""Finite-model Coulomb conventions for periodic correlation.

Three-dimensional kernels retain their original value/AD path. Reduced-D
correlation is eager and uses the existing finite-part electrostatic gauge,
without q-cell averaging or a transplanted three-dimensional head correction.
Only a negative exact-zero head is omitted from square-root Wc factors.
The raw finite head remains part of bare interactions and exchange.

Before using this split, callers must verify the unweighted OV and VO Fourier
heads vanish in the same FFT overlap metric. Removing a nonzero transition
head would change screening; AO-metric orthogonality alone is insufficient if
that metric differs from the actual sampled one.
"""
import jax
import jax.numpy as jnp
import numpy as np

from ...integrals.periodic.coulomb import coulomb_kernel


def _eager(*values):
    if any(isinstance(value, jax.core.Tracer) for value in jax.tree.leaves(values)):
        raise NotImplementedError('Reduced-dimensional correlation kernels are eager; AD is not implemented.')


def _check_periodic_directions(inputs, vectors):
    """Check raw k/q momenta before reciprocal wrapping; G vectors are exempt."""
    boundary = getattr(inputs, 'boundary', None)
    if boundary is None:
        return
    if getattr(boundary, 'dimension', None) not in (1, 2):
        raise NotImplementedError('Use three-dimensional inputs or matching slab/wire boundaries.')
    _eager(vectors, boundary)
    points = np.asarray(vectors)
    if points.ndim < 1 or points.shape[-1] != 3 or np.iscomplexobj(points) or not np.isfinite(points).all():
        raise ValueError('Use finite real Cartesian k/q vectors.')
    fractional = points@np.asarray(boundary.lattice).T/(2*np.pi)
    if np.any(abs(fractional[..., boundary.dimension:]) > 1e-12):
        raise ValueError('k/q momenta must vanish in nonperiodic directions before wrapping.')


def _raw_low_dimensional(inputs, vectors):
    boundary = inputs.boundary
    _eager(vectors, boundary)
    vectors = jnp.asarray(vectors)
    if vectors.ndim != 2 or vectors.shape[1] != 3 or not len(vectors) or jnp.iscomplexobj(vectors):
        raise ValueError('Use real reciprocal vectors with shape (nG,3).')
    if not np.isfinite(vectors).all():
        raise ValueError('Reciprocal vectors must be finite.')
    head = np.all(np.asarray(vectors) == 0., axis=1)
    if np.count_nonzero(head) > 1:
        raise ValueError('The reciprocal grid must have at most one exact-zero head.')
    values = coulomb_kernel(vectors, boundary=boundary)
    if not np.isfinite(values).all():
        raise ValueError('Reduced-dimensional Coulomb values must be finite.')
    return values, head


def correlation_kernel(inputs, vectors):
    """Coulomb values usable in real square-root correlation factors.

    No non-head value is clipped, including tiny negative numbers. The head
    mask uses vectors==0 exactly, not a tolerance or the first array index.
    A positive finite head is retained; a bare caller must not add it twice.
    """
    if getattr(inputs, 'boundary', None) is None:
        return coulomb_kernel(vectors)
    values, head = _raw_low_dimensional(inputs, vectors)
    if np.any(np.asarray(values)[~head] < 0.):
        raise ValueError('Reduced-dimensional correlation has a negative non-head Coulomb value.')
    return jnp.where(jnp.asarray(head) & (values < 0.), 0., values)


def finite_head(inputs):
    """Return raw v(G=0)/V in Hartree, including its finite-part sign.

    V uses the same uniform FFT weights[0]*Ngrid normalization as the product
    factors and must match the boundary lattice volume. Bare BSE/Σx callers
    restore this term only when its negative value was actually omitted by
    correlation_kernel. This scalar is not a q-cell/head-wing correction.
    """
    if getattr(inputs, 'boundary', None) is None:
        return jnp.zeros((), dtype=jnp.result_type(jnp.asarray(inputs.gvectors), 1.))
    _eager(inputs.weights)
    values, head = _raw_low_dimensional(inputs, inputs.gvectors)
    if np.count_nonzero(head) != 1:
        raise ValueError('The unshifted FFT grid must contain exactly one zero head.')
    weights = np.asarray(inputs.weights)
    if (weights.shape != (len(values),) or np.iscomplexobj(weights)
            or not np.isfinite(weights).all() or np.any(weights <= 0.)
            or not np.allclose(weights, weights[0], atol=0., rtol=1e-12)):
        raise ValueError('Finite-head normalization requires positive uniform FFT weights.')
    volume = float(weights[0])*len(weights)
    lattice_volume = float(jnp.linalg.det(inputs.boundary.lattice))
    if not np.isfinite(lattice_volume) or lattice_volume <= 0. or not np.isclose(volume, lattice_volume, atol=0., rtol=1e-12):
        raise ValueError('FFT weights and boundary lattice volume do not match.')
    return values[int(np.flatnonzero(head)[0])]/volume


__all__ = ['correlation_kernel', 'finite_head']
