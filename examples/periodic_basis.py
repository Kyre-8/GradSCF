"""Offline periodic GTH basis catalog and a small MOLOPT-SR HF calculation."""
import jax
import numpy as np
from gradscf.pbc import gto, scf

jax.config.update('jax_enable_x64', True)

print('Named basis families:', len(gto.basis_names()))
print('MOLOPT-SR element coverage:', len(gto.basis_elements('DZVP-MOLOPT-SR-GTH')))
print('Silicon families:', gto.basis_names('Si'))

# PySCF-style names and CP2K header aliases are both accepted. The grid must
# resolve the chosen Gaussian exponents; changing basis may require a finer grid.
cell = gto.M(atom='H 0 0 0; H 0 0 1.4', a=np.eye(3)*6., unit='Bohr',
             basis='SZV-MOLOPT-SR-GTH', pseudo='GTH-PADE-q1',
             mesh=(41,41,41), precision=1e-10)
mf = scf.RHF(cell).run(conv_tol=1e-11, conv_tol_density=1e-10)
assert mf.converged
print('HF energy / Eh per cell:', float(mf.e_tot))

# Raw shell data remain accepted for custom and differentiable basis workflows.
shells = gto.load_basis('gth-dzvp-molopt-sr', 'Si')
print('Silicon DZVP-MOLOPT-SR shell blocks:', len(shells))

# CPU float64, JAX 0.8.1:
# Named basis families: 25
# MOLOPT-SR element coverage: 71
# HF energy / Eh per cell: -1.25167039762391
# Silicon DZVP-MOLOPT-SR shell blocks: 3
