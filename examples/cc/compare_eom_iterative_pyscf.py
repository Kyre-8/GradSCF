"""Independent GradSCF/PySCF RHF -> CCSD -> EE/IP/EA water comparison.

Keep reference software imports here, separate from the native EOM examples.
Water/6-31G exercises iterative spaces above the old dense capacity.
"""

import json
import platform
from time import perf_counter

import jax
import numpy as np
import pyscf
from pyscf import gto as py_gto, scf as py_scf
from pyscf.cc import eom_rccsd

jax.config.update("jax_enable_x64", True)

from gradscf import gto, scf, cc

start = perf_counter()
atom = "O 0 0 0; H 0 -.757 .587; H 0 .757 .587"
mf = scf.RHF(
    gto.M(atom=atom, basis="6-31g", unit="Angstrom"),
    conv_tol=1e-12,
).run()
mycc = cc.CCSD(mf, conv_tol=1e-12, residual_tol=1e-11).run()
py_mol = py_gto.M(atom=atom, basis="6-31g", unit="Angstrom", verbose=0)
py_mf = py_scf.RHF(py_mol).run(conv_tol=1e-13)
py_cc = py_mf.CCSD().run(conv_tol=1e-13, conv_tol_normt=1e-12)
py_cc.conv_tol = 1e-11  # PySCF EOM iterative energy tolerance

report = {
    "atom": atom,
    "basis": "6-31g",
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
for cls, oracle_class in (
    (cc.EOMEE, eom_rccsd.EOMEESinglet),
    (cc.EOMIP, eom_rccsd.EOMIP),
    (cc.EOMEA, eom_rccsd.EOMEA),
):
    result = cls(mycc, nroots=2, solver="davidson", max_space=40, max_cycle=150).run()
    oracle = oracle_class(py_cc)
    imds = oracle.make_imds()
    default_roots = np.asarray(oracle.kernel(nroots=2, imds=imds)[0])
    # An independent full reference avoids treating the iterative driver's
    # root selection as a proof of lowest-root ordering. Reference code only.
    matrix = np.column_stack(
        [oracle.matvec(v, imds) for v in np.eye(oracle.vector_size())]
    )
    spectrum = np.linalg.eigvals(matrix)
    order = np.lexsort((spectrum.imag, spectrum.real))
    selected = spectrum[order[:2]]
    assert np.max(abs(selected.imag)) < 1e-10
    expected = selected.real
    actual = np.asarray(result.e)
    if not np.all(result.converged & result.result.response_valid):
        raise RuntimeError(f"Invalid {result.sector} solution")
    np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=0)
    report["sectors"][result.sector] = {
        "dimension": result.space.size,
        "subspace_dimension": int(result.result.subspace_dimension),
        "iterations": int(result.result.iterations),
        "restarts": int(result.result.restarts),
        "spectrum_complete": bool(result.result.spectrum_complete),
        "gradscf": actual.tolist(),
        "pyscf_full_action_spectrum": expected.tolist(),
        "pyscf_default_two_roots": default_roots.tolist(),
        "max_absolute_error": float(np.max(abs(actual - expected))),
        "max_right_residual": float(np.max(result.result.residual_norms)),
        "max_left_residual": float(np.max(result.result.left_residual_norms)),
        "biorthogonality_error": float(result.result.biorthogonality_error),
    }
report["seconds_including_compile"] = perf_counter() - start
print(json.dumps(report, indent=2))
