"""Independent PySCF FCI check of the native H2/STO-3G EOM Dyson example.

This script imports no GradSCF runtime. Each displaced geometry reconverges HF
and rediagonalizes all three particle-number sectors before contracting the
physical creation/annihilation operators. Compare residues/weights, not the
separately gauge-dependent left and right CC Dyson amplitudes.
"""
import json
from pathlib import Path

import numpy as np
from pyscf import ao2mo, fci, gto, scf


output = Path("reproducibility/ad_showcase/X14_h2_eom_dyson")
native = json.loads((output/"gradscf.json").read_text())


def calculate(bond):
    mol = gto.M(atom=[("H", (0, 0, -bond/2)), ("H", (0, 0, bond/2))],
                basis="sto-3g", unit="Bohr", cart=True, verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-13)
    if not mf.converged:
        raise RuntimeError("Independent RHF did not converge.")
    c = mf.mo_coeff
    h = c.T @ mf.get_hcore() @ c
    eri = ao2mo.restore(1, ao2mo.kernel(mol, c), 2)
    e0, c0 = fci.direct_spin1.kernel(h, eri, 2, (1, 1), tol=1e-13)
    energies, weights = [], []
    for nelec, apply in (((0, 1), fci.addons.des_a), ((2, 1), fci.addons.cre_a)):
        eigenvalues, states = fci.direct_spin1.kernel(h, eri, 2, nelec, nroots=2, tol=1e-13)
        amplitude = np.asarray([[np.vdot(state, apply(c0, 2, (1, 1), p))
                                 for state in states] for p in range(2)])
        energies.append(np.asarray(eigenvalues)-e0)
        weights.append(np.sum(amplitude**2, axis=0))
    return np.concatenate(energies+weights)


values = calculate(native["bond_bohr"])
checks = []
for step in (1e-3, 3e-4, 1e-4, 3e-5, 1e-5):
    derivative = (calculate(native["bond_bohr"]+step)-calculate(native["bond_bohr"]-step))/(2*step)
    error = float(np.max(np.abs(derivative-np.asarray(native["derivative_per_bohr"]))))
    checks.append(dict(step_bohr=step, derivative=derivative.tolist(), max_error=error))
    print(f"Independent FCI h={step:.1e}: derivative error {error:.3e}", flush=True)
record = dict(method="PySCF exact FCI", basis="sto-3g", cart=True,
    values=values.tolist(), max_value_error=float(np.max(np.abs(values-native["values"]))),
    finite_differences=checks)
(output/"pyscf_fci.json").write_text(json.dumps(record, indent=2)+"\n")
np.testing.assert_allclose(values, native["values"], atol=2e-9, rtol=0)
np.testing.assert_allclose(checks[-1]["derivative"], native["derivative_per_bohr"],
                           atol=2e-7, rtol=2e-6)
print(f"Maximum value error: {record['max_value_error']:.3e}", flush=True)

# Measured independent FCI errors, CPU float64, 2026-10-16:
# Values: 9.992e-16. Bond derivatives at decreasing displacements:
# [2.703e-7, 2.432e-8, 2.683e-9, 2.163e-10, 1.608e-10].
