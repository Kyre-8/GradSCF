"""Static finite-q electronic screening in a periodic H2 cell.

GTH-SZV/GTH-Pade, 6-Bohr cubic cell, 3x1x1 k mesh, 15x15x15 FFT grid,
kBT=0.15 Eh; CPU float64. This is an API example, not a phonon calculation.
The real potential 0.06*cos(q.r) Eh has positive-q envelope 0.03 Eh.
Its physical density response combines the computed +q part and its adjoint.
"""
import jax
import jax.numpy as jnp
import numpy as np
from gradscf import dfpt
from gradscf.pbc import gto, scf
from gradscf.pbc.momentum import momentum_transfer
from gradscf.pbc.response import project_q_potential

jax.config.update('jax_enable_x64', True)

cell = gto.M(atom='H .4 .5 .6; H 1.8 .8 .9', a=np.eye(3)*6., unit='Bohr',
             basis='gth-szv', pseudo='gth-pade', mesh=(15,)*3)
kpts = cell.make_kpts((3, 1, 1))
mf = scf.KRHF(cell, kpts=kpts, temperature=.15).run(
    conv_tol=1e-12, conv_tol_density=1e-11, conv_tol_grad=1e-10, damping=.1)
assert mf.converged

q = np.array([1/3, 0., 0.])  # Fractional reciprocal coordinates.
fractional_kpts = np.asarray(kpts) @ np.asarray(cell.lattice).T / (2*np.pi)
transfer = momentum_transfer(fractional_kpts, q)
potential = project_q_potential(mf.inputs, transfer, .03*jnp.ones(len(mf.inputs.weights)))
problem = dfpt.DFPT(mf, q=q)
result = problem.response(potential=potential)
assert result.converged

print('k+q partner indices:', transfer.partners)
print('Reciprocal shifts:\n', transfer.shifts)
print('Linear residual:', float(result.residual_norm))
print('Density response norm:', float(jnp.linalg.norm(result.density)))
rho_q = result.values['rho_spin'].sum(axis=0)
print('Max density envelope (electrons/Bohr^3):', float(jnp.max(jnp.abs(rho_q))))
print('Modulation-averaged electron-number response:', float(result.values['electron_number']))

# result.density[s,k] connects row k+q and column k, so an individual AO
# block is not Hermitian. For q=-q modulo G, combine the physical potential
# blocks consistently before solving; do not add the same adjoint twice.

# Measured CPU float64 output, JAX 0.8.1:
# k+q partner indices: [1 2 0]
# Reciprocal shifts:
# [[ 0  0  0]
#  [-1  0  0]
#  [ 0  0  0]]
# Linear residual: 3.7034876822405543e-14
# Density response norm: 0.049728710427918786
# Max density envelope (electrons/Bohr^3): 0.004787187843524231
# Modulation-averaged electron-number response: 0.0
