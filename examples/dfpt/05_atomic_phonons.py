"""AD atomic response and commensurate reference phonons for a periodic H2 cell.

GTH-SZV/GTH-Pade, 6-Bohr cubic cell, 3x1x1 k mesh, 31-cubed FFT mesh,
CPU float64. The finite-q reference uses a three-cell supercell. This input
geometry is not optimized: signed frequencies describe its local curvatures,
and negative values are retained. It is not a converged material benchmark.
"""
import jax
import jax.numpy as jnp
import numpy as np
from scipy.constants import physical_constants, h, c
from gradscf import dfpt
from gradscf.pbc import gto, scf
from gradscf.pbc.phonon import Phonons

jax.config.update('jax_enable_x64', True)

cell = gto.M(atom='H .4 .5 .6; H 1.8 .8 .9', a=np.eye(3)*6., unit='Bohr',
             basis='gth-szv', pseudo='gth-pade', mesh=(31,)*3)
mf = scf.KRHF(cell, kpts=cell.make_kpts((3, 1, 1))).run(
    conv_tol=1e-12, conv_tol_density=1e-11, conv_tol_grad=1e-10, damping=.1)
assert mf.converged

displacement = jnp.zeros_like(cell.coords).at[1, 0].set(1.)
atomic = dfpt.atomic_problem(mf)
response = atomic.response(tangent=displacement)
assert response.converged
print('Gamma force-constant column (Eh/Bohr^2):\n', -response.values['forces'])

phonons = Phonons(mf, masses=[1.00782503223, 1.00782503223],  # amu
                  max_grid=100_000, max_pair_elements=4_000_000)
to_cm1 = physical_constants['Hartree energy'][0] / (h*c*100)
gamma = phonons.kernel(asr=True)
print('Raw Gamma ASR residual (Eh/Bohr^2):', float(gamma.asr_residual))
print('Gamma frequencies with ASR (cm^-1):', np.asarray(gamma.frequencies)*to_cm1)

q = np.array([1/3, 0., 0.])
source = phonons.displacement(q, displacement)
screened = dfpt.DFPT(mf, q=q).response(tangent=source.perturbation)
assert screened.converged
print('Finite-q overlap-source norm (1/Bohr):', float(jnp.linalg.norm(source.perturbation.overlap)))
print('Finite-q coefficient-density response norm:', float(jnp.linalg.norm(screened.density)))

finite = phonons.kernel(q)
print('q=(1/3,0,0) frequencies (cm^-1):', np.asarray(finite.frequencies)*to_cm1)
print('Force-constant Hermiticity error:', float(finite.hermiticity_error))
print('Maximum linear residual:', float(finite.residual_norm))

# Measured CPU float64 output, JAX 0.8.1:
# Gamma force-constant column (Eh/Bohr^2):
# [[-0.44937907 -0.10217584 -0.10217450]
#  [ 0.44928785  0.10217638  0.10217402]]
# Raw Gamma ASR residual (Eh/Bohr^2): 0.0005457780540763559
# Gamma frequencies with ASR (cm^-1):
# [-1736.00227 -1710.11028 ~0 ~0 ~0 5072.93120]
# (The three projected translation frequencies are below 2e-5 cm^-1.)
# Finite-q overlap-source norm (1/Bohr): 0.7864469202751011
# Finite-q coefficient-density response norm: 0.22972737639183513
# q=(1/3,0,0) frequencies (cm^-1):
# [-1732.45234 -1708.19684 -61.59887 -46.30911 243.69684 5070.43139]
# Force-constant Hermiticity error: 8.483791559210257e-15
# Maximum linear residual: 2.093895830192015e-16
