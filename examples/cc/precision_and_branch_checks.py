"""Native restricted branch search and SCF->CC->EOM residual diagnostics."""

from dataclasses import asdict, replace
import json
import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
from gradscf import gto, scf, cc

mf = scf.RHF(
    gto.M(atom="F 0 0 0; F 0 0 2.2", basis="sto-3g"),
    conv_tol=1e-12,
    conv_tol_density=1e-10,
    conv_tol_grad=1e-9,
    max_cycle=250,
)
search = mf.multistart(amplitudes=(0.5, 1.0), seeds=(0, 1), require_stable=True)
report = {
    "branch_search": {
        "attempts": [asdict(a) for a in search.attempts],
        "selected_index": search.selected_index,
    }
}
if not search.converged:
    raise RuntimeError("No converged candidate passed the internal stability check")
report["branch_search"]["selected_energy"] = float(search.selected.e_tot)
report["branch_search"]["stability"] = float(
    search.selected.stability().minimum_curvature
)

helium = scf.RHF(
    gto.M(atom="He 0 0 0; He 0 0 3", basis="6-31g"), conv_tol=1e-12
)
report["precision"] = {}
for label, source in (
    ("default", helium),
    (
        "tight",
        replace(
            helium,
            conv_tol=1e-14,
            conv_tol_density=1e-12,
            conv_tol_grad=1e-11,
            max_cycle=150,
        ),
    ),
):
    source.run()
    ground = cc.CCSD(source, conv_tol=1e-12, residual_tol=1e-11).run()
    eom = ground.EOMIP(nroots=3).run()
    check = eom.diagnostics(scf_gradient_tol=1e-11)
    report["precision"][label] = {
        "scf": asdict(check.scf),
        "cc_residual": float(check.cc.residual_norm),
        "eom_right_residuals": np.asarray(check.eom.residual_norms).tolist(),
        "eom_left_residuals": np.asarray(check.eom.left_residual_norms).tolist(),
        "energies": np.asarray(eom.e).tolist(),
        "cc_ok": check.cc_ok,
        "eom_ok": check.eom_ok,
        "all_passed": check.all_passed,
    }
print(json.dumps(report, indent=2))
# Targets identify a stage to revisit. They are not a bound on spectral error,
# and internal real restricted stability is not spin/complex/global stability.
