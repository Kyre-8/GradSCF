"""Gaussian H2 OFDFT and a differentiable kinetic-functional parameter.

Atomic units; 6-31G*, Cartesian AOs, grid level 1, native CPU integrals.
This intentionally omits XC to isolate the KEDF/Hartree example. With jax-xc
installed, set xc='pbe' or 'svwn' in both the facade and the response config.
TF+vW is a numerical baseline, not an accurate molecular KEDF.
"""
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

from gradscf import gto, ofdft

mol = gto.M(atom='H 0 0 0; H 0 0 .74', basis='6-31g*', unit='Angstrom')
calculation = ofdft.OFDFT(mol, xc=None, tolerance=1e-8).run()
if not calculation.converged:
    raise RuntimeError('OFDFT did not reach a stationary density')
print('Energy / Hartree:', float(calculation.e_tot))
print('Energy components:', calculation.result.components)
print('Electron number:', float(calculation.result.electron_number))
print('Grid electron number:', float(calculation.result.grid_electron_number))
print('Stationarity residual:', float(calculation.result.residual_norm))

# Full density response to the vW weight, at fixed nuclear coordinates/basis.
data = calculation.inputs
config = ofdft.OFDFTConfig(xc=None, tolerance=1e-9)
probe = jnp.exp(-jnp.sum(data.coordinates**2, axis=1))

def density_probe(vw_weight):
    result = ofdft.run_ofdft(data, kinetic_params={'vw': vw_weight}, config=config)
    return jnp.sum(data.weights*probe*result.density)

value, derivative = jax.jit(jax.value_and_grad(density_probe))(1.)
step = 1e-4
finite_difference = (density_probe(1.+step)-density_probe(1.-step))/(2*step)
print('Density probe:', float(value))
print('d(probe)/d(vW weight):', float(derivative))
print('Finite difference:', float(finite_difference))
assert jnp.isclose(derivative,finite_difference,rtol=1e-4,atol=1e-6)

# CPU float64 example output (XC omitted):
# Energy / Hartree: 0.05889014102429713
# Electron number: 2.0000000000000004
# Grid electron number: 2.0000032630335274
# Stationarity residual: 5.924241233606791e-10
# Density probe: 0.2537764625011235
# d(probe)/d(vW weight): -0.07886838837426632
# Finite difference: -0.07886838261167428
