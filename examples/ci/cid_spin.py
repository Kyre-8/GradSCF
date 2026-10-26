"""CID/CISD energies and total-spin diagnostics, using GradSCF throughout.

Real orbitals, STO-3G, fixed geometries in Angstrom, energies in Hartree.
Effective multiplicity is diagnostic: mixed-spin states need not give integers.
No PySCF or CLI.
"""
import jax
from gradscf import ci, gto, scf

jax.config.update("jax_enable_x64", True)
mol = gto.M(atom="H 0 0 0; H 0 0 .8; H 0 0 1.9; H 0 0 3.1", basis="sto-3g")
mf = scf.RHF(mol, conv_tol=1e-12).run()
for method in (ci.CID, ci.CISD):
    calculation = method(mf, nroots=2, conv_tol=1e-11).run()
    for root in range(2):
        ss, multiplicity = calculation.spin_square(root=root)
        print(method.__name__, "root", root, "energy:", float(calculation.e_tot[root]),
              "<S^2>:", float(ss), "effective multiplicity:", float(multiplicity))

radical = gto.M(atom="H 0 0 0; H 0 0 .85; H 0 0 1.9", basis="sto-3g", spin=1)
for reference in (scf.UHF, scf.ROHF):
    mf = reference(radical, conv_tol=1e-12, max_cycle=150).run()
    calculation = mf.CID(conv_tol=1e-11).run()
    ss, multiplicity = calculation.spin_square()
    print(reference.__name__, "CID energy:", float(calculation.e_tot),
          "<S^2>:", float(ss), "effective multiplicity:", float(multiplicity))
