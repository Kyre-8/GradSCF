"""Periodic FFT and Gaussian OFDFT with a local GTH H pseudopotential.

Atomic units. This small 9^3 grid is a solver demonstration, not a converged
materials benchmark. XC is explicitly disabled; set xc='pbe' with jax-xc.
Use a local OF-compatible potential for other elements: nonlocal GTH projector
channels are rejected. Grid and Gaussian minima need not agree at finite basis.
"""
import jax
import numpy as np

jax.config.update('jax_enable_x64', True)

from gradscf.pbc import gto
from gradscf import ofdft

cell = gto.M(atom='H 1 1 1; H 2.4 1 1', a=np.eye(3)*6., unit='Bohr',
             basis='gth-szv', pseudo='gth-pade', mesh=(9,9,9))
for representation in ('periodic', 'periodic_gaussian'):
    for kinetic in ('tfvw', 'wt'):
        calculation = ofdft.OFDFT(cell, representation=representation, kinetic=kinetic,
                                 xc=None, tolerance=1e-7, maxiter=600).run()
        print(representation, kinetic, 'energy / Hartree:', float(calculation.e_tot))
        print('  converged:', calculation.converged,
              'residual:', float(calculation.result.residual_norm),
              'electrons:', float(calculation.result.electron_number))
        if not calculation.converged:
            raise RuntimeError('OFDFT did not reach stationarity')

# CPU float64 example output (9^3 mesh; XC omitted):
# representation       KEDF    energy / Hartree       residual
# periodic             tfvw   -0.20886008759121416    9.24e-08
# periodic             wt     -0.6083608050358628     6.43e-08
# periodic_gaussian    tfvw    0.07426967072754997    1.95e-10
# periodic_gaussian    wt     -0.5688988000548016     5.80e-09
# All four runs converged; electron numbers agree with 2 within 2e-15.
