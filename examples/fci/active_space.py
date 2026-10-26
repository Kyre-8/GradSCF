"""Fixed-orbital active-space FCI with a doubly occupied Li 1s core.

The FCI engine receives a (2 electrons, 4 orbitals) problem. This demonstrates
core folding and the future CAS solver interface; no orbital optimization.
"""
import jax
from gradscf import gto, scf, fci

jax.config.update('jax_enable_x64',True)
mf=scf.RHF(gto.M(atom='Li 0 0 0; H 0 0 1.6',basis='sto-3g')).run()
solver=fci.FCI(mf,core=1,active=(1,2,3,4)).run()
print('Embedded active-space FCI energy / Hartree:',solver.e_tot)
print('Active electrons:',solver.space.nelec)
print('Active 1-RDM:',solver.make_rdm1())
