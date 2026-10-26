"""OFDFT with the project's actual jax-xc LDA/GGA backend (required).

H2/STO-3G, Angstrom, TF+vW KEDF. No substitute XC or silent fallback is used.
"""
import jax
jax.config.update('jax_enable_x64', True)
from gradscf import gto, ofdft

mol = gto.M(atom='H 0 0 0; H 0 0 .74',basis='sto-3g')
for xc in ('svwn','pbe'):
    calculation = ofdft.OFDFT(mol,xc=xc,tolerance=1e-8).run()
    print(xc,'energy / Hartree:',float(calculation.e_tot),
          'converged:',calculation.converged)
    if not calculation.converged:
        raise RuntimeError('OFDFT did not converge')
