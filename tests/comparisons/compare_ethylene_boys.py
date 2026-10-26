"""Compare AD-Boys with PySCF Boys, using the same GradSCF occupied orbitals.

No PySCF SCF calculation is performed. Different starting rotations may reach
different local minima; compare the score before interpreting matched orbitals.
"""

from pathlib import Path
import runpy

import numpy as np
from pyscf import gto, lo
from scipy.linalg import expm
from scipy.optimize import linear_sum_assignment


def compare(example):
    source = example["mol"]
    mol = gto.M(atom=source.atom, basis=source.basis, unit=source.unit,
                cart=source.cart, verbose=0)
    c_occ = np.asarray(example["c_occ"])
    c_boys = np.asarray(example["c_boys"])
    overlap = np.asarray(example["overlap"])
    dipole = np.asarray(example["dipole_ao"])

    def score(c):
        centers = np.einsum("mi,xmn,ni->xi", c, dipole, c)
        return np.sum(centers**2)

    # PySCF runs independently from canonical/atomic and random guesses.
    # In particular, do not initialize it with the AD-localized orbitals.
    default = lo.Boys(mol, c_occ)
    default.conv_tol = 1e-12
    default.conv_tol_grad = 1e-9
    default.max_cycle = 200
    candidates = [default.kernel()]
    p, q = np.triu_indices(c_occ.shape[1], 1)
    rng = np.random.default_rng(19)
    for _ in range(7):
        kappa = np.zeros((c_occ.shape[1], c_occ.shape[1]))
        kappa[p, q] = 1e-2 * rng.normal(size=len(p))
        initial = c_occ @ expm(kappa - kappa.T)
        reference = lo.Boys(mol, initial)
        reference.conv_tol = 1e-12
        reference.conv_tol_grad = 1e-9
        reference.max_cycle = 200
        candidates.append(reference.kernel(mo_coeff=initial))
    c_reference = max(candidates, key=score)

    # Match permutations and signs. Equal scores need not imply identical
    # columns: symmetry-related local minima can still differ.
    matrix = c_boys.T @ overlap @ c_reference
    rows, cols = linear_sum_assignment(-np.abs(matrix))
    signs = np.where(matrix[rows, cols] >= 0, 1., -1.)
    c_matched = c_reference[:, cols] * signs
    matched_overlaps = np.abs(matrix[rows, cols])
    difference = score(c_boys) - score(c_reference)
    print(f"AD Boys score              = {score(c_boys):.12f}")
    print(f"PySCF default score        = {score(candidates[0]):.12f}")
    print(f"PySCF best multistart score = {score(c_reference):.12f}")
    print(f"Score difference           = {difference:.3e}")
    print("Matched absolute overlaps  =", matched_overlaps)
    return c_matched, matched_overlaps, difference


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    example = runpy.run_path(str(root / "examples" / "localization" / "ethylene_boys.py"))
    compare(example)
