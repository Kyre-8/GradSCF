"""Native GradSCF RHF -> G0W0/W0 -> singlet/triplet TDA-BSE for water.

STO-3G, fixed coordinates in Angstrom, real float64 CPU reference calculation.
All QP levels and the full screening space are retained. No PySCF or CLI.
"""

import jax
from gradscf import bse, gto, scf, gw
from gradscf.tools.spectra import HARTREE_TO_EV

jax.config.update("jax_enable_x64", True)
mol = gto.M(atom="O 0 0 0; H 0 -.757 .587; H 0 .757 .587", basis="sto-3g")
mf = scf.RHF(mol, conv_tol=1e-12).run()
mygw = gw.GW(mf, nw=100).run()
print("G0W0 converged:", bool(mygw.converged))

for singlet in (True, False):
    response = bse.BSE(mygw, singlet=singlet, nroots=3, conv_tol=1e-10).run()
    strengths = response.oscillator_strength()
    print("Singlets" if singlet else "Triplets")
    for root, (energy, strength) in enumerate(zip(response.e, strengths)):
        print(
            root,
            "energy / eV:",
            float(energy * HARTREE_TO_EV),
            "oscillator strength:",
            float(strength),
        )
    print("Residuals / Hartree:", response.result.residual_norms)
