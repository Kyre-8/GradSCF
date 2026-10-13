"""Static H2 polarizability from a converged GradSCF reference (atomic units).

RHF needs no optional XC runtime. With jax-xc installed, replace scf.RHF with
dft.RKS(mol, xc='pbe') to include the self-consistent GGA response kernel.
"""
import jax
import jax.numpy as jnp
from gradscf import gto, scf, dfpt

jax.config.update('jax_enable_x64', True)

mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', unit='Bohr')
mf = scf.RHF(mol, conv_tol=1e-12).run()
reference = mf.to_reference()
response = dfpt.DFPT(mf)

# Electronic perturbation +F*z; mu_e = -Tr(D*z), alpha = -Tr(D^(1)*z).
z = reference.dipole_integrals[2]
first = response.response(potential=z)
assert first.converged
print('E(0) / Eh:', float(mf.e_tot))
print('alpha_zz / bohr^3:', float(-jnp.sum(first.density*z.T)))
print('electron-number response:', float(first.values['electron_number']))
print('linear residual:', float(first.residual_norm))

# CPU, float64:
# E(0) / Eh: -1.1167143250625502
# alpha_zz / bohr^3: 3.071895559766628
# electron-number response: 4.01e-16
# linear residual: 8.71e-17
