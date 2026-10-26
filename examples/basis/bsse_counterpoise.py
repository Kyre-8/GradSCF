#!/usr/bin/env python
"""Boys--Bernardi counterpoise energies through the public GradSCF RHF API.

Fixed S22 geometries; Cartesian STO-3G, 3-21G and def2-SVP; CPU float64.
Run from the GradSCF root:
    PYTHONPATH=src python examples/basis/bsse_counterpoise.py
"""
import json
import os
from dataclasses import replace
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

import jax
import numpy as np
from gradscf import gto, scf
from gradscf.data.molecule import parse_molecule_spec


def make_mol(atom, basis, ghost_atoms=()):
    """Ghosts keep the element basis, with zero nuclear charge and electrons."""
    spec = parse_molecule_spec(atom, unit="Angstrom")
    charges = np.asarray(spec.charges).copy()
    charges[list(ghost_atoms)] = 0.0
    return gto.M(atom=replace(spec, charges=charges), basis=basis, cart=True)


geometries = json.loads(
    Path(__file__).with_name("s22_hbond_geometries.json").read_text()
)["systems"]
systems = ("Water_dimer", "Ammonia_dimer", "Formic_acid_dimer")
bases = ("sto-3g", "3-21g", "def2-svp")
hartree_to_kcal_mol = 627.5094740631

for name in systems:
    system = geometries[name]
    atom = list(zip(system["symbols"], system["positions_angstrom"]))
    atoms_a, atoms_b = system["fragments"]
    fragment_a = [atom[i] for i in atoms_a]
    fragment_b = [atom[i] for i in atoms_b]

    print("\n" + name)
    print("Basis        Raw interaction     CP interaction       BSSE   (kcal/mol)")
    for basis in bases:
        # AB, A, B, A with ghost B, and B with ghost A, at the dimer geometry.
        molecules = (
            make_mol(atom, basis),
            make_mol(fragment_a, basis),
            make_mol(fragment_b, basis),
            make_mol(atom, basis, ghost_atoms=atoms_b),
            make_mol(atom, basis, ghost_atoms=atoms_a),
        )
        energies = []
        for mol in molecules:
            mf = scf.RHF(mol, conv_tol=1e-12, conv_tol_density=1e-10,
                         conv_tol_grad=1e-9, max_cycle=200)
            energies.append(float(mf.kernel()))
            if not mf.converged:
                raise RuntimeError("RHF did not converge.")

        e_ab, e_a, e_b, e_a_ab, e_b_ab = energies
        interaction_raw = e_ab - e_a - e_b
        interaction_cp = e_ab - e_a_ab - e_b_ab
        bsse = (e_a - e_a_ab) + (e_b - e_b_ab)
        print(f"{basis:10s} {interaction_raw * hartree_to_kcal_mol:18.9f}"
              f" {interaction_cp * hartree_to_kcal_mol:18.9f}"
              f" {bsse * hartree_to_kcal_mol:12.9f}")

    jax.clear_caches()


# Reference results (2026-10-08).
# Fixed S22 geometries; RHF; Cartesian AOs; native CPU / float64.
# Energies in kcal/mol; CP interaction = raw interaction + BSSE.
#
# System              Basis        Raw interaction     CP interaction       BSSE
# Water_dimer         sto-3g           -5.530692914       -1.401174239  4.129518675
# Water_dimer         3-21g           -10.545679548       -5.942393542  4.603286006
# Water_dimer         def2-svp         -5.957259564       -4.022647169  1.934612395
# Ammonia_dimer       sto-3g           -1.577567897       -0.567480522  1.010087376
# Ammonia_dimer       3-21g            -5.440055464       -2.243155448  3.196900017
# Ammonia_dimer       def2-svp         -3.257109594       -1.382486015  1.874623579
# Formic_acid_dimer   sto-3g          -18.716684509       -4.421627448 14.295057061
# Formic_acid_dimer   3-21g           -27.926726980      -18.422954442  9.503772538
# Formic_acid_dimer   def2-svp        -18.990793006      -15.615700113  3.375092893
