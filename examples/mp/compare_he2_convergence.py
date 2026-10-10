"""Independent PySCF/RS check of he2_convergence.py, including frozen basis.

Run the native example first. PySCF independently reconstructs the contracted
basis and converges HF. It supplies FCI actions to a NumPy RS recurrence,
not a native PySCF high-order MP API. Distances are Angstrom, energies Hartree.
"""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from pyscf import gto, scf, fci
from _pyscf_reference import mp_coefficients

folder = Path("artifacts/mp_convergence")
native = json.loads((folder/"he2_native.json").read_text())
seed_mol = gto.M(atom="He 0 0 0; He 0 0 .7",basis="3-21g",charge=2,verbose=0)
seed = scf.RHF(seed_mol).run(conv_tol=1e-13,conv_tol_grad=1e-11)
rows = []
for shell in range(2):
    rows.extend([[float(a),float(c)*seed.mo_coeff[shell,0]]
        for a,c in zip(seed_mol.bas_exp(shell),seed_mol.bas_ctr_coeff(shell)[:,0])])
# A global sign of the contracted AO has no physical effect.
if rows[0][1]*native["basis"]["He"][0][1][1] < 0:
    rows = [[a,-c] for a,c in rows]
np.testing.assert_allclose(rows,native["basis"]["He"][0][1:],atol=2e-9,rtol=0.)
basis = {"He":[[0,*rows]]}
records = []
for data in native["curves"]:
    distance = data["distance"]
    mol = gto.M(atom=[("He",(0,0,0)),("He",(0,0,distance))],
                basis=basis,charge=2,cart=True,verbose=0)
    rhf = scf.RHF(mol).run(conv_tol=1e-13,conv_tol_grad=1e-10,max_cycle=400)
    dm = np.array([np.diag([1.,0.]),np.diag([0.,1.])])
    uhf = scf.UHF(mol).run(dm,conv_tol=1e-13,conv_tol_grad=1e-10,max_cycle=400)
    if not rhf.converged or not uhf.converged:
        raise RuntimeError("PySCF reference did not converge")
    er,eu = [mp_coefficients(mf,native["order"]) for mf in (rhf,uhf)]
    exact = fci.FCI(rhf).kernel()[0]
    record = dict(distance=distance,rhf=float(rhf.e_tot),uhf=float(uhf.e_tot),
        rmp=er.tolist(),ump=eu.tolist(),fci=float(exact),spin_square=uhf.spin_square()[0])
    records.append(record)
    for method in ("rmp","ump"):
        error = np.max(np.abs(np.asarray(data[method])-record[method]))
        print("R=%.4f %s max E2--E%d difference %.3e Ha" %
              (distance,method,native["order"],error),flush=True)
    np.testing.assert_allclose([data[k] for k in ("rhf","uhf","fci")],
                              [record[k] for k in ("rhf","uhf","fci")],atol=3e-10,rtol=0.)
(folder/"he2_reference.json").write_text(json.dumps(dict(basis=basis,curves=records),indent=2)+"\n")
for actual,expected in zip(native["curves"],records):
    for method in ("rmp","ump"):
        np.testing.assert_allclose(actual[method],expected[method],atol=2e-10,rtol=2e-9)
print("All HF, FCI and MP coefficient checks passed.")

# Check the optimized UMP20 barrier separately: matching a rounded barrier
# height does not by itself establish agreement in its optimized distance.
def ump20(distance):
    mol = gto.M(atom=[("He",(0,0,0)),("He",(0,0,distance))],
                basis=basis,charge=2,cart=True,verbose=0)
    dm = np.array([np.diag([1.,0.]),np.diag([0.,1.])])
    mf = scf.UHF(mol).run(dm,conv_tol=1e-13,conv_tol_grad=1e-11,max_cycle=400)
    assert mf.converged
    return mf.e_tot+sum(mp_coefficients(mf,20))
maximum = minimize_scalar(lambda r:-ump20(r),bounds=(.95,1.3),method="bounded",
                          options={"xatol":1e-7})
native_max = next(x for x in native["extrema"] if x["method"] == "ump" and x["order"] == 20)
print("UMP20 maximum: native %.8f, independent %.8f Angstrom" %
      (native_max["r_max"],maximum.x))
np.testing.assert_allclose(native_max["r_max"],maximum.x,atol=5e-5,rtol=0.)
