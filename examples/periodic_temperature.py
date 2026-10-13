"""Finite-temperature periodic HF, thermodynamic AD and static response.

FCC Al, GTH-SZV/GTH-PADE, two shifted k points and a coarse 15^3 FFT mesh.
This demonstrates the numerical interfaces; it is not a converged bulk-Al
benchmark. temperature means kBT in Hartree, entropy means S/kB per cell.
"""
import jax
import jax.numpy as jnp
import numpy as np
from gradscf.pbc import gto,scf
from gradscf.pbc.kpoints import symmetry_reduce
from gradscf.pbc.scf import run_kpoint_scf
from gradscf.scf import SCFDifferentiationConfig
from gradscf import dfpt

jax.config.update('jax_enable_x64',True)

lattice=7.65/2*np.array([[0.,1.,1.],[1.,0.,1.],[1.,1.,0.]])
cell=gto.M(atom='Al 0 0 0',a=lattice,unit='Bohr',mesh=(15,)*3)
points=cell.make_kpts((2,1,1),shift=(.5,0.,0.))
controls=dict(max_cycle=180,damping=.3,conv_tol=1e-11,
              conv_tol_density=1e-10,conv_tol_grad=1e-9)
mf=scf.KRHF(cell,kpts=points,temperature=.08,exxdiv=None).run(**controls)
assert mf.converged
print('Spin populations per cell:',cell.nelec)
print('E / Eh:',float(mf.e_tot))
print('F / Eh:',float(mf.e_free))
print('S/kB:',float(mf.entropy))
print('Chemical potentials / Eh:',np.asarray(mf.chemical_potential))

def free_energy(temperature):
    result=run_kpoint_scf(mf.inputs,mesh=cell.mesh,nelec=cell.nelec,
        temperature=temperature,exxdiv=None,
        differentiation=SCFDifferentiationConfig(mode='implicit',tolerance=1e-10,max_iter=80),**controls)
    return result.free_energy

value,slope=jax.jit(jax.value_and_grad(free_energy))(.08)
print('dF/d(kBT):',float(slope))
np.testing.assert_allclose(slope,-mf.entropy,atol=1e-7)
potential=jnp.broadcast_to(jnp.diag(jnp.array([.1,-.2,.05,.05])),mf.inputs.hcore.shape)
response=dfpt.DFPT(mf).response(potential=potential)
assert response.converged
print('Electron-number response:',float(response.values['electron_number']))

# Reduction is metadata for scalar sums. SCF above still uses the FULL mesh.
fractional=np.asarray(points)@np.asarray(cell.lattice).T/(2*np.pi)
reduction=symmetry_reduce(fractional,time_reversal=True)
print('Time-reversal orbit weights:',reduction.weights)

# CPU float64, JAX 0.8.1:
# Spin populations per cell: (1.5, 1.5)
# E / Eh: -1.4633358455321515
# F / Eh: -1.6102935378254635
# S/kB: 1.8369711536663993
# Chemical potentials / Eh: [0.72089055 0.72089055]
# dF/d(kBT): -1.8369711536330922
# Electron-number response: -4.44e-15
# Time-reversal orbit weights: [1.]
