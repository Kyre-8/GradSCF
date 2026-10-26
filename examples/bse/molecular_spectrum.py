"""H2/STO-3G: evGW0 -> static full BSE -> polarized absorption.

CPU float64, energies and linewidth in Hartree. This small-basis example
illustrates the interface; it is not an experimental spectrum prediction.
MolGW comparisons live separately in tests/comparisons/compare_molgw_gw_bse.py.
"""
import jax
import numpy as np

jax.config.update('jax_enable_x64', True)

from gradscf import gto, scf, gw, bse
from gradscf.solvers import LinearSolverConfig

mol = gto.M(atom='H 0 0 0; H 0 0 .74', basis='sto-3g')
mf = scf.RHF(mol, conv_tol=1e-12).run()
nocc = np.count_nonzero(mf.mo_occ)
nmo = len(mf.mo_energy)

# Target QP states, intermediate G states and screening transitions are
# independent original-MO windows. Here all are retained explicitly.
mygw = gw.GW(
    mf, method='evgw0', nw=100, eta=1e-5,
    max_cycle=80, conv_tol=1e-10,
    g_orbitals=range(nmo),
    screening_occupied=range(nocc),
    screening_virtual=range(nocc, nmo),
).run(orbs=range(nmo))
assert mygw.converged
print('QP energies / Ha:', mygw.mo_energy)
print('QP weights:', mygw.result.qp_weight)
print('W0 spectrum / Ha:', mygw.result.screening_energy)

# BSE inherits the recorded W0 spectrum and screening windows. Choosing the
# optical occupied/virtual window does not change screening.
response = bse.BSE(
    mygw, nroots=1, tda=False,
    screening_config=LinearSolverConfig(method='gmres', rtol=1e-11, atol=1e-13),
).run()
assert response.converged.all()
print('BSE energies / Ha:', response.e)
print('Oscillator strengths:', response.oscillator_strength())
print('Static polarizability / a0^3:\n', response.polarizability().real)

omega = np.linspace(0., 1.5, 601)
alpha = response.polarizability(omega, eta=.01)  # complex (nfrequency,3,3)
sigma = response.absorption_cross_section(omega, eta=.01, unit='Mb')
sigma_z = response.absorption_cross_section(
    omega, eta=.01, polarization=(0., 0., 1.), unit='Mb',
)
print('Sampled isotropic peak / Mb:', float(sigma.max()))
print('Sampled z-polarized peak / Mb:', float(sigma_z.max()))
# The tensor/absorption include only the computed roots. Increase nroots and
# the orbital/basis windows before drawing conclusions about spectral sums.

# Measured on CPU float64, JAX 0.8.1:
# QP energies / Ha: [-0.59684064  0.68943027]
# QP weights: [0.99364029 0.99364029]
# W0 spectrum / Ha: [-0.57855386  0.67114349]
# BSE energies / Ha: [0.95331811]
# Oscillator strengths: [0.85121264]
# Static alpha_zz / a0^3: 2.80985323 (other components zero)
# Sampled isotropic peak / Mb: 108.46849186864853
# Sampled z-polarized peak / Mb: 325.4054756059456
