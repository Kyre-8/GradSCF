"""Native HF -> FCI energy, density and spin diagnostics for H2."""
import jax
from gradscf import gto, scf, fci

jax.config.update('jax_enable_x64',True)
mol=gto.M(atom='H 0 0 0; H 0 0 .74',basis='sto-3g')
mf=scf.RHF(mol,conv_tol=1e-12).run()
solver=fci.FCI(mf,nroots=2,solver='dense').run()
print('FCI total energies / Hartree:',solver.e_tot)
print('CI tensor shape:',solver.ci.shape)
print('Ground-state 1-RDM:',solver.make_rdm1())
print('Ground-state S^2 / effective multiplicity:',solver.spin_square())
