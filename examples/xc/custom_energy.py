"""A handwritten exchange energy using the same port as an external network.

H2/STO-3G, grid level 1, float64. This example supplies spin-scaled Dirac
exchange only: it deliberately contains no correlation energy. energy_fn
returns the TOTAL exchange energy in Hartree, including grid quadrature.
GradSCF derives the AO potential and its Hessian action from this energy.
"""
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

from gradscf import dft, gto


def make_inputs(state):
    # Evaluated again for each CURRENT SCF trial density. The input schema
    # belongs to the external functional, just as a neural model's would.
    return {'rho_spin': state.rho_spin, 'weights': state.weights}


def exchange_energy(params, inputs):
    rho_spin = jnp.maximum(inputs['rho_spin'], 1e-18)
    exchange_density = -.75*(6/jnp.pi)**(1/3)*jnp.sum(rho_spin**(4/3), axis=0)
    return params['scale']*jnp.dot(inputs['weights'], exchange_density)


functional = dft.custom_xc(input_fn=make_inputs, energy_fn=exchange_energy,
                          name='scaled_dirac_exchange')
params = {'scale': jnp.asarray(1.)}
mol = gto.M(atom='H 0 0 0; H 0 0 .74', basis='sto-3g')
mf = dft.RKS(mol, xc=functional, params=params, grids_level=1,
             integral_backend='native', conv_tol=1e-11, max_cycle=80).run()
if not mf.converged:
    raise RuntimeError('Custom-exchange SCF did not converge.')

reference, density = mf.to_reference(), mf.make_rdm1()
potential = functional.potential(params, reference, density)
tangent = jnp.eye(density.shape[0])
kernel_action = functional.kernel_action(params, reference, density, tangent)
print('Self-consistent total energy / Eh:', float(mf.e_tot))
print('XC potential shape:', potential.shape)
print('XC kernel action norm:', float(jnp.linalg.norm(kernel_action)))

# This derivative holds the displayed density fixed. Self-consistent training
# uses the same Functional with Trainer.mode = 'explicit' or 'implicit'.
derivative = jax.grad(lambda scale: functional.energy_for_density(
    {'scale': scale}, reference, density))(params['scale'])
print('Fixed-density dExc/dscale / Eh:', float(derivative))

# A neural energy uses the identical constructor and explicit params:
# see ../training/density_matrix_mlp.py for an external four-layer network,
# and ../training/external_functional.py for all three training modes.

# Measured on jy_gpu_1, CPU float64, JAX 0.10.0, 2026-10-20:
# Self-consistent total energy / Eh: -1.0250090882481175
# XC potential shape: (2, 2)
# XC kernel action norm: 0.22127771006605085
# Fixed-density dExc/dscale / Eh: -0.5830057076661403
