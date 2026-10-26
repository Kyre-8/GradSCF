"""OH doublet: native UHF -> evGW0 -> spin-conserving TDA-BSE.

STO-3G, fixed geometry in Angstrom, CPU float64. The optical window includes
only computed QP levels; all occupied/virtual transitions screen W0. States
conserve M_S and are not assigned spin-pure multiplicities. No PySCF or CLI.
"""

import jax
from gradscf import bse, gto, scf, gw
from gradscf.tools.spectra import HARTREE_TO_EV

jax.config.update("jax_enable_x64", True)

mol = gto.M(atom="O 0 0 0; H 0 0 .9697", basis="sto-3g", spin=1)
mf = scf.UHF(mol, conv_tol=1e-10).run()
mygw = gw.UGW(mf, method="evgw0", nw=48,
              max_cycle=80, conv_tol=1e-8, damp=.3).run(orbs=(3, 4, 5))
response = bse.BSE(mygw, occupied=((4,), (3,)), virtual=((5,), (4, 5)),
                   nroots=2, tda=True, solver="dense", conv_tol=1e-10).run()

print("GW converged:", bool(mygw.converged))
print("Maximum QP residual / Ha:", float(abs(mygw.result.qp_residual).max()))
for root, (energy, strength) in enumerate(zip(response.e, response.oscillator_strength())):
    print(f"Root {root}: {float(energy * HARTREE_TO_EV):.8f} eV; f = {float(strength):.8f}")
print("BSE residuals / Ha:", response.result.residual_norms)
print("Alpha/beta amplitude shapes:", [x.shape for x in response.result.x_amplitudes])

# CPU float64 output (2026-10-04; JAX_PLATFORMS=cpu):
# GW converged: True
# Maximum QP residual / Ha: 6.407554750675892e-09
# Root 0: 0.28222595 eV; f = 0.00000000
# Root 1: 11.44647563 eV; f = 0.00022168
# BSE residuals / Ha: [1.64826641e-17 1.17365659e-16]
# Alpha/beta amplitude shapes: [(2, 1, 1), (2, 1, 2)]
