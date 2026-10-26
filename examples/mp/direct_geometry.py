"""Direct MP2 nuclear gradient and stretch HVP at N2's internal pi degeneracy.

SCF and correlation use GradSCF. No full AO/MO ERI tensor is requested by this
path; the noncanonical MP2 resolvent still keeps its necessary doubles state.
Coordinates are Bohr, energies Hartree. max_memory is an array-planning budget
in decimal MB, not an operating-system RSS ceiling.
"""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)

from gradscf import gto, scf
from gradscf_tools.correlated_response import CorrelatedEnergy

mol = gto.M(atom='N 0 0 -1.04; N 0 0 1.04', basis='sto-3g', unit='Bohr', cart=True)
# The ten STO-3G AOs give Tr[D S]=14 for this symmetric density seed,
# selecting the usual closed-shell branch rather than a broken-symmetry root.
mf = scf.RHF(mol, integral_backend='native', jk_backend='direct', direct_scf_tol=0.,
    init_guess=1.4*np.eye(10), max_memory=128., max_cycle=250,
    conv_tol=1e-13, conv_tol_density=1e-11, conv_tol_grad=1e-10).run()
assert mf.converged
assert np.min(np.abs(np.diff(np.asarray(mf.mo_energy)))) < 1e-9

energy = CorrelatedEnergy(mf, 'mp2', jk_backend='direct')
coordinates = energy.coordinates
value, gradient = jax.jit(jax.value_and_grad(energy))(coordinates)
stretch = jnp.array([[0., 0., -.5], [0., 0., .5]])
hvp = jax.jit(lambda r: jax.jvp(jax.grad(energy), (r,), (stretch,))[1])(coordinates)
assert np.isfinite(value) and np.isfinite(np.asarray(hvp)).all()
np.testing.assert_allclose(gradient.sum(axis=0), 0., atol=2e-8, rtol=0)

np.set_printoptions(precision=9, suppress=True)
print('MP2 total energy / Hartree: %.12f' % value)
print('Nuclear gradient / Hartree/Bohr:\n', np.asarray(gradient))
print('Stretch HVP / Hartree/Bohr^2:\n', np.asarray(hvp))

# Output: CPU, JAX 0.10.0, float64, native integrals.
# MP2 total energy / Hartree: -107.651843995332
# Nuclear gradient / Hartree/Bohr:
#  [[-0.           0.           0.323413809]
#   [ 0.          -0.          -0.323413809]]
# Stretch HVP / Hartree/Bohr^2:
#  [[ 0.          -0.          -2.096236923]
#   [-0.           0.           2.096236923]]
