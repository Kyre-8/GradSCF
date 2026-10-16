"""An external nonlinear XC energy and its AD-derived finite-q response.

Run with JAX_ENABLE_X64=1 PYTHONPATH=src python examples/dfpt/03_external_finite_q.py.
The three translated grids specify this model's physical q/-q extension.
Only scalar energies are supplied; GradSCF differentiates the kernel and solves
the screened response with its shared DFPT solver. Atomic units throughout.
"""
import jax
import jax.numpy as jnp
import numpy as np
from gradscf import dft, dfpt
from gradscf.pbc import gto, dft as pdft
from gradscf.pbc.functional import density_inputs
from gradscf.pbc.kpoints import reciprocal_vectors
from gradscf.pbc.momentum import momentum_transfer
from gradscf.pbc.response import q_density, project_q_potential

jax.config.update('jax_enable_x64', True)


def energy(params, features):
    rho, weights = features
    return params['scale']*jnp.sum(weights*rho**3)/3


def q_energy(params, inputs, reference, transfer, delta_q):
    rho = density_inputs(inputs, reference).rho
    envelope = q_density(inputs, transfer, delta_q).sum(axis=0)
    translations = jnp.arange(3)[:, None]*inputs.lattice[0]
    wavevector = jnp.asarray(transfer.q) @ reciprocal_vectors(inputs.lattice)
    phase = jnp.exp(1j*(inputs.coords[None]+translations[:, None]) @ wavevector)
    # D + Z_q + Z_q†, including the physical Bloch phase. The callback must
    # return energy PER PRIMITIVE CELL and agree with energy() at delta_q=0.
    physical_rho = rho+2*(phase*envelope).real
    return params['scale']*jnp.mean(jnp.sum(inputs.weights*physical_rho**3, axis=-1))/3


model = dft.Functional(lambda x: (x.rho, x.weights), energy, q_energy_fn=q_energy)
cell = gto.M(
    atom='H .4 .5 .6; H 1.8 .8 .9', a=np.eye(3)*6., unit='Bohr', mesh=(9,)*3,
)
k = cell.make_kpts((3, 1, 1))
q = np.array([1/3, 0., 0.])
mf = pdft.KRKS(
    cell, kpts=k, functional=model, params={'scale': jnp.array(.15)}, temperature=.12,
).run(conv_tol=1e-12, conv_tol_density=1e-11, conv_tol_grad=1e-10, damping=.1)
transfer = momentum_transfer(np.asarray(k) @ np.asarray(cell.lattice).T/(2*np.pi), q)
potential = project_q_potential(
    mf.inputs, transfer, (.03+.01j)*jnp.ones(len(mf.inputs.weights)),
)
response = dfpt.DFPT(mf, q=q).response(potential=potential)
assert mf.converged and response.converged
print('SCF free energy (Eh/cell):', float(mf.result.free_energy))
print('Finite-q density response norm:', float(jnp.linalg.norm(response.density)))
print('For parameter training, construct finite_q_problem inside the differentiated function,')
print('passing functional=model, params=params and the relaxed run_kpoint_scf result.')

# Measured 2026-10-15 on jy_gpu_1: CPU JAX 0.10.0, float64, OMP/BLAS threads=2.
# Both convergence assertions passed; external-example.log recorded:
# SCF free energy (Eh/cell): -0.6104181508891107
# Finite-q density response norm: 0.05878070762111205
