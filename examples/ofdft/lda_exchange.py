"""Pure-JAX Dirac LDA exchange with OFDFT, independent of jax-xc.

This is exchange only: E_x = -3/4 (3/pi)^(1/3) integral(n^(4/3)).
No correlation, hybrid exchange, empirical fit or substitute library is used.
The system is an eight-electron periodic model with a neutralizing background
and smooth local external potential; no atom/ionic energy is present.
"""
from dataclasses import replace
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)
from gradscf import ofdft


def exchange_energy(params, features):
    del params
    # Equivalent amplitude form has a finite Hessian even at density nodes.
    return -.75*(3/jnp.pi)**(1/3)*jnp.sum(features.weights*jnp.abs(features.phi)**(8/3))


def model_inputs(mesh=(7,7,7)):
    data = ofdft.periodic_inputs(jnp.eye(3)*6.,mesh,nelectron=8.)
    phase = 2*jnp.pi*data.coordinates/6.
    potential = .15*jnp.cos(phase[:,0])+.1*jnp.sin(phase[:,1])+.05*jnp.cos(phase[:,2])
    return replace(data,external_potential=potential)


def main():
    data = model_inputs()
    for name in ('tfvw','wt'):
        result = ofdft.run_ofdft(data,kinetic=ofdft.KineticFunctional(name),
            xc_energy_fn=exchange_energy,config=ofdft.OFDFTConfig(xc=None,tolerance=1e-9,maxiter=700))
        print(name,'energy / Hartree:',float(result.total_energy),
              'N:',float(result.electron_number),'residual:',float(result.residual_norm))
        if not result.converged:
            raise RuntimeError('OFDFT density did not converge')


if __name__=='__main__':
    main()
