"""Native internal and spin-channel checks, followed by explicit mode steps.

RHF/RKS->UHF/UKS changes the reference class only when channel='spin' is
explicitly requested. The original restricted source is preserved.
"""

import json
from dataclasses import asdict
from time import perf_counter
import jax

jax.config.update("jax_enable_x64", True)
from gradscf import gto, scf

start = perf_counter()
report = {"dtype": "float64", "devices": [str(d) for d in jax.devices()], "cases": {}}
for name, atom, channel in (
    ("H2_stretched", "H 0 0 0; H 0 0 2.5", "spin"),
    ("N2", "N 0 0 0; N 0 0 1.1", "internal"),
    ("F2_stretched", "F 0 0 0; F 0 0 2.2", "internal"),
):
    mf = scf.RHF(
        gto.M(atom=atom, basis="sto-3g", unit="Angstrom"),
        conv_tol=1e-12,
        conv_tol_density=1e-10,
        conv_tol_grad=1e-9,
        max_cycle=250,
    ).run()
    internal = mf.stability(solver="dense")
    spin = mf.stability(channel="spin", solver="dense")
    result = mf.stabilize(channel=channel, max_restarts=5, step_sizes=(0.5, 1.0))
    row = {
        "atom_angstrom": atom,
        "basis": "sto-3g",
        "channel": channel,
        "initial_energy": float(mf.e_tot),
        "internal_curvature": internal.minimum_curvature,
        "spin_curvature": spin.minimum_curvature,
        "final_energy": float(result.result.e_tot),
        "final_model": type(result.result).__name__,
        "status": result.status,
        "stable": result.stable,
        "restarts": result.restarts,
        "energy_history": result.energy_history,
        "attempts": [asdict(t) for t in result.attempts],
        "final_minimum_curvature": result.minimum_curvature,
        "final_spectrum_certified": result.stability.spectrum_certified,
    }
    report["cases"][name] = row
    print(
        json.dumps(
            {
                "progress": name,
                "status": result.status,
                "final_energy": row["final_energy"],
            }
        ),
        flush=True,
    )
report["seconds_including_compile"] = perf_counter() - start
print(json.dumps(report, indent=2))
