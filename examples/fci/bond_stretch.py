"""H2 bond stretching: static-orbital full CI and natural occupations.

Near dissociation two spatial natural orbitals acquire occupations near one.
Each geometry has a fresh native RHF reference; no nuclear gradients are taken.
"""
import jax
import numpy as np
from gradscf import gto, scf, fci

jax.config.update('jax_enable_x64',True)
for distance in (.74,1.5,3.):
    mol=gto.M(atom=[('H',(0,0,0)),('H',(0,0,distance))],basis='sto-3g')
    mf=scf.RHF(mol,conv_tol=1e-12).run()
    result=fci.FCI(mf,solver='dense').run()
    occupations=np.linalg.eigvalsh(result.make_rdm1())[::-1]
    print('R / Angstrom:',distance,'E_FCI / Hartree:',float(result.e_tot),
          'natural occupations:',occupations)
