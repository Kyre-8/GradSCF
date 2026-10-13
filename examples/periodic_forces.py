"""Periodic HF and OFDFT forces and stress from the shared geometry driver.

H2 in a cubic 6-Bohr cell; GTH-SZV/GTH-Pade, CPU float64.
HF uses a shifted 2x1x1 k mesh and kBT=0.1 Eh. OFDFT uses TF+vW with no XC
and the local H pseudopotential. HF uses a 41x41x41 FFT grid; OFDFT uses a
coarser 21x21x21 grid. These illustrate the API, not converged material
properties. Geometry derivatives reconverge the electrons
with the existing explicit/implicit policy and the full k mesh.
"""
import jax
import jax.numpy as jnp
from gradscf import ofdft
from gradscf.pbc import gto, scf
from gradscf.scf import SCFDifferentiationConfig

jax.config.update('jax_enable_x64', True)

cell = gto.M(atom='H .4 .5 .6; H 1.8 .8 .9', a=jnp.eye(3)*6., unit='Bohr',
             basis='gth-szv', pseudo='gth-pade', mesh=(41,)*3)
backward = SCFDifferentiationConfig(mode='implicit', tolerance=1e-10, max_iter=80)
mf = scf.KRHF(cell, kpts=cell.make_kpts((2, 1, 1), shift=(.5, 0, 0)),
              temperature=.1, differentiation=backward).run(
                  conv_tol=1e-12, conv_tol_density=1e-10, conv_tol_grad=1e-9)
assert mf.converged
gradient = mf.nuc_grad_method()
de = gradient.kernel()  # dF/dR, Eh/Bohr; forces have the opposite sign.
print('HF free energy (Eh):', float(gradient.result.energy))
print('HF forces (Eh/Bohr):\n', -de)
print('HF stress (Eh/Bohr^3):\n', gradient.stress())
print('HF pressure (Eh/Bohr^3):', float(-jnp.trace(gradient.stress())/3))

ofcell = gto.M(atom=cell.atom, a=cell.a, unit=cell.unit,
               basis=cell.basis, pseudo=cell.pseudo, mesh=(21,)*3)
calc = ofdft.OFDFT(ofcell, xc=None, tolerance=1e-8, maxiter=500, differentiation=backward)
gradient = calc.nuc_grad_method()
gradient.kernel()
print('OFDFT energy (Eh):', float(gradient.result.energy))
print('OFDFT forces (Eh/Bohr):\n', gradient.result.forces)
print('OFDFT stress (Eh/Bohr^3):\n', gradient.stress())

# Measured CPU float64 output, JAX 0.8.1:
# HF free energy (Eh): -1.1787863634533662
# HF forces (Eh/Bohr):
# [[-0.06582420 -0.01850994 -0.01850994]
#  [ 0.06582420  0.01850992  0.01850991]]
# HF stress (Eh/Bohr^3):
# [[-2.20301106e-04 -1.19049568e-04 -1.19049568e-04]
#  [-1.19049568e-04  6.18097985e-04 -2.71723937e-05]
#  [-1.19049568e-04 -2.71723937e-05  6.18097751e-04]]
# HF pressure (Eh/Bohr^3): -0.00033863154348003034
# OFDFT energy (Eh): -0.23947291459672504
# OFDFT forces (Eh/Bohr):
# [[-0.29926762 -0.06430543 -0.06496477]
#  [ 0.29874355  0.06511428  0.06439417]]
# OFDFT stress (Eh/Bohr^3):
# [[-1.76614622e-03 -4.13715730e-04 -4.13680938e-04]
#  [-4.13715730e-04  1.52582497e-04 -8.94297090e-05]
#  [-4.13680938e-04 -8.94297090e-05  1.51911381e-04]]
