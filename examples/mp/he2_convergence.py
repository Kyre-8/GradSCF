"""He2(2+): native GradSCF RMP/UMP curves in Gill and Radom's minimal basis.

Reference: Chem. Phys. Lett. 132, 16 (1986), Tables 2--3 and Figs. 1--2.
https://doi.org/10.1016/0009-2614(86)80686-8
The (3s)->[1s] basis is contracted from the occupied 3-21G RHF orbital at
R=0.7 Angstrom, then held fixed at every distance. It is NOT STO-3G.
Run from the repository root; no PySCF calculation enters the native path.
"""
from functools import cache
import json
from pathlib import Path
from time import perf_counter

import jax
import numpy as np
from scipy.optimize import minimize_scalar

jax.config.update("jax_enable_x64", True)

from gradscf import gto, scf, mp, fci
from gradscf.integrals.basis import prepare_basis
from gradscf.integrals.basis.normalization import normalized_shell_coefficients, radial_primitive_norm
from gradscf.scf.reference import reference_from_source, unrestricted_reference_from_source
from gradscf.solvers import EigenSolverConfig

order = 30
start = perf_counter()
seed_mol = gto.M(atom="He 0 0 0; He 0 0 .7", basis="3-21g", charge=2)
seed_mf = scf.RHF(seed_mol, conv_tol=1e-13, conv_tol_grad=1e-11).run()
assert seed_mf.converged
topology, parameters = prepare_basis(seed_mol.atom, seed_mol.basis, charge=2)
rows = []
for shell in range(2):
    a, c = parameters.exponents[shell], parameters.coefficients[shell]
    normalized = normalized_shell_coefficients(0, a, c)[:, 0] / radial_primitive_norm(0, a)
    rows.extend([[float(x), float(y)*float(seed_mf.mo_coeff[shell, 0])]
                 for x, y in zip(a, normalized)])
basis = {"He": [[0, *rows]]}
print("Fixed minimal basis:", basis, flush=True)

config = mp.MPConfig(order=order, algorithm="tensor", with_t2=False)

@jax.jit
def restricted_mp(h, g):
    result = mp.run_mp(h, g, nocc=1, config=config)
    return result.corrections, result.valid

@jax.jit
def unrestricted_mp(h, g):
    result = mp.run_mp(h, g, nocc=(1, 1), config=config)
    return result.corrections, result.valid

space = fci.make_fci_space(2, (1, 1))

@jax.jit
def exact_energy(h, g, enuc):
    result = fci.solve_fci(h, g, space, ecore=enuc,
                         config=EigenSolverConfig(method="dense", nroots=1))
    return result.total_energies[0]

@cache
def point(distance):
    mol = gto.M(atom=[("He", (0, 0, 0)), ("He", (0, 0, distance))],
                basis=basis, charge=2, unit="Angstrom")
    rhf = scf.RHF(mol, conv_tol=1e-13, conv_tol_grad=1e-10, max_cycle=400).run()
    # One electron on each atom explicitly selects the broken-symmetry branch.
    guess = np.array([np.diag([1., 0.]), np.diag([0., 1.])])
    uhf = scf.UHF(mol, init_guess=guess, conv_tol=1e-13,
                  conv_tol_grad=1e-10, max_cycle=400).run()
    if not rhf.converged or not uhf.converged:
        raise RuntimeError("HF did not converge at R=%.9f Angstrom" % distance)
    rref = reference_from_source(rhf)
    uref = unrestricted_reference_from_source(uhf)
    er, vr = restricted_mp(rref.h1, rref.eri)
    eu, vu = unrestricted_mp(uref.h1, uref.eri)
    if not vr or not vu:
        raise RuntimeError("MP reference/response invalid at R=%.9f Angstrom" % distance)
    overlap = np.asarray(rhf.scf_result.overlap_matrix)
    ca, cb = np.asarray(uhf.mo_coeff)[:, :, 0]
    s2 = 1.-float(ca @ overlap @ cb)**2
    exact = float(exact_energy(rref.h1, rref.eri, rref.nuclear_repulsion))
    return dict(distance=distance, rhf=float(rhf.e_tot), uhf=float(uhf.e_tot),
                rmp=np.asarray(er).tolist(), ump=np.asarray(eu).tolist(),
                fci=exact, spin_square=s2)

distances = np.unique(np.r_[np.linspace(.55, 1.6, 43), .7, 1.074, 1.115, 1.8])
curves = []
for distance in distances:
    data = point(float(distance))
    curves.append(data)
    print("R=%.4f  RHF=% .10f UHF=% .10f FCI=% .10f <S2>=%.6f" %
          (distance, data["rhf"], data["uhf"], data["fci"], data["spin_square"]), flush=True)

def partial(distance, method, degree):
    data = point(float(distance))
    if method == "fci":
        return data["fci"]
    return data["rhf" if method == "rmp" else "uhf"] + sum(data[method][:degree-1])

extrema = []
for method, degree in [(m, n) for m in ("rmp", "ump") for n in (2, 3, 4, 10, 20)] + [("fci", 0)]:
    minimum = minimize_scalar(lambda r: partial(r, method, degree),
        bounds=(.6, .8), method="bounded", options={"xatol":1e-7})
    maximum = minimize_scalar(lambda r: -partial(r, method, degree),
        bounds=(.9, 1.4), method="bounded", options={"xatol":1e-7})
    if not minimum.success or not maximum.success:
        raise RuntimeError("Extremum search failed")
    barrier = (-maximum.fun-minimum.fun)*2625.4996394799
    extrema.append(dict(method=method, order=degree, r_min=float(minimum.x),
                        r_max=float(maximum.x), barrier_kj_mol=float(barrier)))
    print("%s %2d  r_min=%.6f r_max=%.6f  barrier=%.3f kJ/mol" %
          (method, degree, minimum.x, maximum.x, barrier), flush=True)

output = Path("artifacts/mp_convergence/he2_native.json")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(dict(basis=basis, order=order, curves=curves,
    extrema=extrema, elapsed_seconds=perf_counter()-start, jax=jax.__version__,
    backend=str(jax.devices()), dtype="float64"), indent=2)+"\n")
print("Saved:", output, flush=True)

# Measured output, Apple M4 Pro CPU, float64, 2026-10-10:
# order  RMP barrier / kJ mol^-1  UMP barrier / kJ mol^-1
#   2       312.542                  228.278
#   3       145.121                  250.386
#   4        96.332                  251.747
#  10       104.948                  209.336
#  20       102.264                  167.490
# FCI barrier = 102.249 kJ/mol, maximum at 1.074029 Angstrom.
# UMP20 maximum = 1.10802246 Angstrom (independent PySCF: 1.10802244).
# The original Table 3 prints 1.1146 Angstrom; see CONVERGENCE.md.
# At 1.8 Angstrom: E_RMP^(30) = +17211.944087 Ha, E_UMP^(30) = -4.80746e-7 Ha.
# Independent coefficient checks: atol=2e-10 Ha, rtol=2e-9 through MP30.
