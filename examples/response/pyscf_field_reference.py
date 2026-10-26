"""Independent finite-field RHF/PBE reference used only by comparisons.

No GradSCF solver, integral or derivative array is imported. Electron and
nuclear field energies use the same fixed origin and atomic units.
"""
import numpy as np
from pyscf import dft, gto, scf
from pyscf.dft import gen_grid, radi


def energy_and_dipole(symbols, coordinates_bohr, basis, field_au, *, method="HF", grid_level=0):
    method = method.upper()
    if method not in ("HF", "RHF", "PBE"):
        raise ValueError("The independent reference supports neutral closed-shell HF/PBE.")
    coordinates, field = np.asarray(coordinates_bohr), np.asarray(field_au)
    mol = gto.M(atom=list(zip(symbols, coordinates)), basis=basis, unit="Bohr",
                cart=True, charge=0, spin=0, verbose=0)
    position = mol.intor_symmetric("int1e_r", comp=3)
    hcore = scf.RHF(mol).get_hcore()
    nuclear_dipole = mol.atom_charges() @ coordinates
    mf = dft.RKS(mol) if method == "PBE" else scf.RHF(mol)
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-13, 1e-11, 150
    if method == "PBE":
        mf.xc, mf.small_rho_cutoff = "pbe", 0
        mf.grids.level, mf.grids.alignment = grid_level, 1
        mf.grids.radi_method, mf.grids.prune = radi.treutler_ahlrichs, gen_grid.nwchem_prune
        mf.grids.radii_adjust, mf.grids.atomic_radii = radi.treutler_atomic_radii_adjust, radi.BRAGG_RADII
        mf.grids.becke_scheme = gen_grid.original_becke
    mf.get_hcore = lambda *args: hcore+np.einsum("x,xpq->pq", field, position)
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("An independent finite-field SCF did not converge.")
    # PySCF's default nuclear energy has no external-field contribution.
    total_energy = float(mf.e_tot)-field@nuclear_dipole
    dipole = nuclear_dipole-np.einsum("pq,xqp->x", mf.make_rdm1(), position)
    return np.r_[total_energy, dipole]
