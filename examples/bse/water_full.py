"""Native RHF/G0W0 water: compare TDA and stable full BSE optical excitations.

STO-3G, coordinates in Angstrom, float64. Full BSE compares the bounded dense
reference solver with matrix-free Davidson and optional GMRES screening. No CLI.
"""

import jax
from gradscf import bse, gto, scf, gw
from gradscf.solvers import LinearSolverConfig
from gradscf.tools.spectra import HARTREE_TO_EV

jax.config.update("jax_enable_x64", True)
mol = gto.M(atom="O 0 0 0; H 0 -.757 .587; H 0 .757 .587", basis="sto-3g")
mf = scf.RHF(mol, conv_tol=1e-12).run()
mygw = gw.GW(mf, nw=100).run()
print("G0W0 converged:", bool(mygw.converged))

for singlet in (True, False):
    print("Singlet" if singlet else "Triplet")
    for tda, method, screening in (
        (True, "davidson", None),
        (False, "dense", None),
        (False, "davidson", None),
        (False, "davidson", LinearSolverConfig(method="gmres", rtol=1e-11, atol=1e-13)),
    ):
        response = bse.BSE(
            mygw,
            tda=tda,
            solver=method,
            singlet=singlet,
            nroots=3,
            conv_tol=1e-10,
            screening_config=screening,
        ).run()
        print("TDA" if tda else "Full BSE", method,
              "screening:", "direct" if screening is None else screening.method)
        print("Energies / eV:", response.e * HARTREE_TO_EV)
        print("Oscillator strengths:", response.oscillator_strength())
        print("Residuals / Hartree:", response.result.residual_norms)
        if not tda:
            print(
                "Stability margins / Hartree:",
                response.result.stability_margins,
            )

            print("Global stability certified:", response.result.stability_certified)
