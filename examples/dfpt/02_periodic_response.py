"""Periodic q=0 AO-potential response; no macroscopic electric-field claim."""
import jax
import jax.numpy as jnp
from gradscf import dfpt
from gradscf.pbc import gto, scf

jax.config.update('jax_enable_x64', True)

cell = gto.M(atom='H .2 .3 .4; H 1.6 .3 .4', a=jnp.eye(3)*6,
             unit='Bohr', basis='gth-szv', pseudo='gth-pade', mesh=(15,15,15))
mf = scf.RHF(cell).run(conv_tol=1e-12, conv_tol_density=1e-11, conv_tol_grad=1e-10)
problem = dfpt.DFPT(mf)
potential = jnp.array([[[.2, .1], [.1, -.1]]])
first = problem.response(potential=potential)
series = problem.series(potential=potential, order=2)
assert first.converged and series.converged

print('density response shape (spin,k,ao,ao):', first.density.shape)
print('electron-number response:', float(first.values['electron_number']))
print('dE/dt, d2E/dt2 / Eh:', float(series.energy[0]), float(2*series.energy[1]))

# CPU, float64:
# density response shape (spin,k,ao,ao): (2, 1, 2, 2)
# electron-number response: 0.0
# dE/dt, d2E/dt2 / Eh: 0.14724330889701667 -0.19702722399995087
