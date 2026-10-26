"""Native water/6-31G EE/IP/EA with the shared non-Hermitian Davidson solver.

The EE amplitude space exceeds the default dense capacity. No CLI or PySCF
solver is used. Ritz gaps are estimates; see spectrum_complete diagnostics.
"""

import json
import platform
from time import perf_counter
import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from gradscf import gto, scf, cc

start = perf_counter()
mf = scf.RHF(
    gto.M(
        atom="O 0 0 0; H 0 -.757 .587; H 0 .757 .587", basis="6-31g", unit="Angstrom"
    ),
    conv_tol=1e-12,
).run()
mycc = cc.CCSD(mf, conv_tol=1e-12, residual_tol=1e-11).run()
report = {
    "machine": platform.machine(),
    "jax": jax.__version__,
    "devices": [str(d) for d in jax.devices()],
    "basis": "6-31g",
    "dtype": "float64",
    "energy_unit": "Hartree",
    "sectors": {},
}
for method in (cc.EOMEE, cc.EOMIP, cc.EOMEA):
    before = perf_counter()
    eom = method(
        mycc, nroots=2, solver="davidson", max_space=40, max_cycle=150, conv_tol=1e-9
    ).run()
    if not np.all(eom.converged & eom.result.response_valid):
        raise RuntimeError(
            f"{eom.sector}: failed Ritz or response checks: {eom.result}"
        )
    report["sectors"][eom.sector] = {
        "energies": np.asarray(eom.e).tolist(),
        "dimension": eom.space.size,
        "subspace_dimension": int(eom.result.subspace_dimension),
        "iterations": int(eom.result.iterations),
        "restarts": int(eom.result.restarts),
        "max_right_residual": float(np.max(eom.result.residual_norms)),
        "max_left_residual": float(np.max(eom.result.left_residual_norms)),
        "guard_residuals": np.asarray(eom.result.guard_residual_norms).tolist(),
        "spectral_gaps": np.asarray(eom.result.spectral_gaps).tolist(),
        "spectrum_complete": bool(eom.result.spectrum_complete),
        "seconds_including_compile": perf_counter() - before,
    }
report["seconds_including_compile"] = perf_counter() - start
print(json.dumps(report, indent=2))
