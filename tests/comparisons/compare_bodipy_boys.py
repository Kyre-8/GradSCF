"""Independent PySCF Boys comparison using saved GradSCF BODIPY orbitals.

Run examples/localization/bodipy_boys.py first. Neither SCF nor the AD localization is rerun.
"""

from pathlib import Path
import runpy

import numpy as np
from gradscf import gto

root = Path(__file__).resolve().parents[2]
with np.load(root / "artifacts" / "bodipy_boys" / "trajectory.npz") as archive:
    data = dict(archive)

mol = gto.M(atom=str(data["atom"]), basis=str(data["basis"]), cart=bool(data["cart"]))
example = dict(mol=mol, c_occ=data["c_occ"], c_boys=data["c_boys"],
               overlap=data["overlap"], dipole_ao=data["dipole_ao"])
compare = runpy.run_path(str(Path(__file__).with_name("compare_ethylene_boys.py")))["compare"]
c_reference, matched_overlaps, score_difference = compare(example)
