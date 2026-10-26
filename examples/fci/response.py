"""AD of a two-site Hubbard ground energy with respect to hopping t.

U=4 Hartree is fixed; h12=h21=-t. t is a model parameter, not a nuclear
coordinate. Static FCI topology is constructed outside jit.
"""
import jax
import jax.numpy as jnp
from gradscf import fci
from gradscf.solvers import EigenSolverConfig

jax.config.update('jax_enable_x64',True)
space=fci.make_fci_space(2,(1,1))
config=EigenSolverConfig(method='dense')
eri=jnp.zeros((2,)*4).at[0,0,0,0].set(4.).at[1,1,1,1].set(4.)

def energy(t):
    h=jnp.array([[0.,-t],[-t,0.]])
    return fci.solve_fci(h,eri,space,config=config).total_energies[0]

value,derivative=jax.jit(jax.value_and_grad(energy))(1.)
print('Ground energy / Hartree:',value)
print('dE/dt at t=1:',derivative)
print('Analytic dE/dt:',-8/jnp.sqrt(4.**2+16))
