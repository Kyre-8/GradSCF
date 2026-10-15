"""Full/irreducible HF quadrature on a coarse periodic H2 cell.

The source-k sum and AO inputs remain complete. Only representative target
J/K, XC projection and SCF diagonalization are evaluated during iterations.
The final state and implicit response are certified on the full k mesh.
Automatic spatial symmetry discovery requires the optional symmetry extra.
"""
import jax
import numpy as np
from gradscf.pbc import gto, scf

jax.config.update('jax_enable_x64', True)
cell = gto.M(
    atom='H -.7 0 0; H .7 0 0', a=np.eye(3)*6., unit='Bohr', mesh=(9,)*3,
)
kpoints = cell.make_kpts((3, 1, 1))
controls = dict(max_cycle=100, damping=.1, conv_tol=1e-12,
                conv_tol_density=1e-11, conv_tol_grad=1e-10)
full = scf.KRHF(cell, kpts=kpoints, temperature=.2, exxdiv=None).run(**controls)
reduced = scf.KRHF(
    cell, kpts=kpoints, temperature=.2, exxdiv=None,
    symmetry=True, time_reversal=True,
).run(**controls)
assert full.converged and reduced.converged
np.testing.assert_allclose(reduced.e_free, full.e_free, atol=1e-10)
np.testing.assert_allclose(reduced.make_rdm1(), full.make_rdm1(), atol=1e-9)
plan = reduced.reduction
print('Full/irreducible k points:', len(kpoints), len(plan.representatives))
print('Irreducible weights:', np.asarray(plan.reduce_weights(reduced.inputs.kweights)))
print('Free energy full/irreducible (Eh/cell):', float(full.e_free), float(reduced.e_free))
print('Maximum density difference:',
      float(np.max(np.abs(reduced.make_rdm1()-full.make_rdm1()))))
print('Exchange target/source pairs:',
      len(plan.representatives)*len(kpoints), 'versus', len(kpoints)**2)

# Measured 2026-10-15 on jy_gpu_1: CPU JAX 0.10.0, float64, OMP/BLAS threads=2.
# Full/irreducible k points: 3 2
# Irreducible weights: [0.33333333 0.66666667]
# Free energy full/irreducible (Eh/cell): -1.1169946341008274 -1.1169946341008274
# Maximum density difference: 1.816997577726786e-13
# Exchange target/source pairs: 6 versus 9
