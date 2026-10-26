"""Independent GradSCF/PySCF RHF -> CCSD -> EE/IP/EA water comparison.

Keep reference software imports here, separate from the native EOM examples.
This small STO-3G calculation validates energies, not basis-set convergence.
"""

import json
import platform
from time import perf_counter

import jax
import numpy as np
import pyscf
from pyscf import gto as py_gto, scf as py_scf

jax.config.update("jax_enable_x64", True)

from gradscf import gto, scf, cc

start = perf_counter()
atom = "O 0 0 0; H 0 -.757 .587; H 0 .757 .587"
mf = scf.RHF(
    gto.M(atom=atom, basis="sto-3g", unit="Angstrom"),
    conv_tol=1e-12,
).run()
mycc = cc.CCSD(mf, conv_tol=1e-12, residual_tol=1e-11).run()
py_mol = py_gto.M(atom=atom, basis="sto-3g", unit="Angstrom", verbose=0)
py_mf = py_scf.RHF(py_mol).run(conv_tol=1e-13)
py_cc = py_mf.CCSD().run(conv_tol=1e-13, conv_tol_normt=1e-12)
py_cc.conv_tol = 1e-11  # PySCF EOM iterative energy tolerance

report = {
    "atom": atom,
    "basis": "sto-3g",
    "geometry_unit": "Angstrom",
    "energy_unit": "Hartree",
    "machine": platform.machine(),
    "jax": jax.__version__,
    "pyscf": pyscf.__version__,
    "dtype": "float64",
    "devices": [str(d) for d in jax.devices()],
    "hf_absolute_error": float(abs(mf.e_tot - py_mf.e_tot)),
    "ccsd_absolute_error": float(abs(mycc.e_tot - py_cc.e_tot)),
    "sectors": {},
}
for cls, oracle in (
    (cc.EOMEE, py_cc.eomee_ccsd_singlet),
    (cc.EOMIP, py_cc.ipccsd),
    (cc.EOMEA, py_cc.eaccsd),
):
    result = cls(mycc, nroots=3).run()
    expected = np.asarray(oracle(nroots=3)[0])
    actual = np.asarray(result.e)
    if not np.all(result.converged & result.result.response_valid):
        raise RuntimeError(f"Invalid {result.sector} solution")
    np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=0)
    report["sectors"][result.sector] = {
        "gradscf": actual.tolist(),
        "pyscf": expected.tolist(),
        "max_absolute_error": float(np.max(abs(actual - expected))),
        "max_right_residual": float(np.max(result.result.residual_norms)),
        "max_left_residual": float(np.max(result.result.left_residual_norms)),
        "biorthogonality_error": float(result.result.biorthogonality_error),
    }
report["seconds_including_compile"] = perf_counter() - start
print(json.dumps(report, indent=2))
