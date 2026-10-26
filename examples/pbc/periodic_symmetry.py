"""Space-group and Cartesian AO maps for a hexagonal two-He cell.

Requires the optional symmetry extra (spglib). The matrix data below are
synthetic: this demonstrates basis/gauge transformations, not an SCF calculation.
An operation of the crystal need not preserve a finite FFT discretization.
"""
import jax
import jax.numpy as jnp
import numpy as np
from gradscf.pbc import gto, symmetry

jax.config.update('jax_enable_x64', True)
cell = gto.M(
    atom='He 0 0 0; He 0 0 3.5',
    a=[[5., 0., 0.], [2.5, 2.5*np.sqrt(3), 0.], [0., 0., 7.]],
    unit='Bohr', basis={'He': [[0, [.8, 1.]], [1, [.6, 1.]], [2, [.5, 1.]]]},
    pseudo='gth-pade', mesh=(9, 9, 9),
)
operations = symmetry.space_group(cell)
print('Spatial operations:', len(operations))

# 60-degree rotation followed by a half-cell translation along z.
operation = symmetry.Operation.from_cell(
    cell, [[0, -1, 0], [1, 1, 0], [0, 0, 1]], [0., 0., .5]
)
operation.validate(cell)
k = jnp.array([.19, -.11, .07])  # fractional reciprocal coordinates
print('Rotated fractional k:', np.asarray(operation.rotate_k(k)))

rng = np.random.default_rng(0)
n = cell.topology.nao
coefficients = jnp.asarray(rng.normal(size=(n, 4)) + 1j*rng.normal(size=(n, 4)))
raw = jnp.asarray(rng.normal(size=(n, n)) + 1j*rng.normal(size=(n, n)))
density = coefficients @ coefficients.conj().T
operator = raw + raw.conj().T
rotated_c = operation.coefficients(k, coefficients)
rotated_d = jax.jit(operation.density)(k, density)
rotated_f = operation.operator(k, operator)
np.testing.assert_allclose(rotated_d, rotated_c @ rotated_c.conj().T, atol=1e-12)
trace_error = abs(jnp.trace(rotated_d @ rotated_f) - jnp.trace(density @ operator))
assert trace_error < 1e-10
print('Trace invariance error:', float(trace_error))

# This is a derivative of a representation's Bloch phase, not a material response.
entry = lambda point: operation.matrix(point)[0, n//2].real
gradient = jax.jit(jax.grad(entry))(k)
print('Bloch-phase gradient:', np.asarray(gradient))

# jy_gpu_1, CPU float64, JAX 0.10.0, spglib 2.7.0, seed 0:
# Spatial operations: 48
# Rotated fractional k: [0.3  0.19 0.07]
# Trace invariance error: 1.2079226507921703e-13
# Bloch-phase gradient: [ 0.          0.         -2.67525019]
