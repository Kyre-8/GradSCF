"""WGC99 second-order OFDFT for Al and Mg, using GradSCF only.

The physical input helpers provide OEPP and an explicit pure-JAX PZ-LDA
energy. No KS orbitals or DFTpy results enter the OF minimization.
"""
import jax
jax.config.update('jax_enable_x64', True)

from gradscf import ofdft
from atlas_inputs import crystal, gaussian_free_inputs, pz_lda_energy


for symbol in ('Al', 'Mg'):
    atoms, mesh = crystal(symbol, spacing=.18)
    inputs = gaussian_free_inputs(atoms, mesh)
    calculation = ofdft.OFDFT(inputs, kinetic='wgc', xc=None,
                            xc_energy_fn=pz_lda_energy,
                            maxiter=1000, tolerance=1e-8).run()
    print(symbol, 'mesh', mesh)
    print('  Energy / Hartree per atom:', float(calculation.e_tot) / len(atoms))
    print('  Residual:', float(calculation.result.residual_norm))
    if not calculation.converged:
        raise RuntimeError('WGC density is not stationary')

# CPU float64, target spacing 0.18 Angstrom:
# Al   -2.087391936818388 Ha/atom
# Mg   -0.905887662700436 Ha/atom
# alpha=(5+sqrt(5))/6, beta=(5-sqrt(5))/6, gamma=2.7.
# See WGC_REPRODUCTION.md for the differences from the original paper inputs.
