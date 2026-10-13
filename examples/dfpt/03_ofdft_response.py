"""Grid OFDFT: screened response and normalized Taylor energy coefficients.

A neutralizing background is implicit in the periodic Coulomb convention.
TF+vW, no XC; this is a response demonstration, not a material benchmark.
"""
from dataclasses import replace
import jax
import jax.numpy as jnp
from gradscf import ofdft, dfpt

jax.config.update('jax_enable_x64', True)

data = ofdft.periodic_inputs(jnp.eye(3)*5., (3,3,3), nelectron=2.)
probe = jnp.cos(2*jnp.pi*data.coordinates[:,0]/5.)
data = replace(data, external_potential=.1*probe)
config = ofdft.OFDFTConfig(xc=None, tolerance=1e-10, maxiter=400)
ground = ofdft.run_ofdft(data, config=config)
assert ground.converged
problem = dfpt.ofdft_problem(data, ground, config=config)
first = problem.response(potential=probe)
series = problem.series(potential=probe, order=2)
assert first.converged and series.converged

print('E(0) / Eh:', float(ground.total_energy))
print('electron-number response:', float(first.values['electron_number']))
print('dE/dt, d2E/dt2 / Eh:', float(series.energy[0]), float(2*series.energy[1]))
print('integral v^(1)*rho^(1) / Eh:', float(jnp.sum(data.weights*probe*first.density)))

# CPU, float64:
# E(0) / Eh: 0.3578772782697514
# electron-number response: -1.21e-15
# dE/dt, d2E/dt2 / Eh: -0.13332362138916756 -1.2833547732130917
# integral v^(1)*rho^(1) / Eh: -1.2833547732130892
