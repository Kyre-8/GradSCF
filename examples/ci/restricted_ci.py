"""Closed-shell H4 CI hierarchy and singlet CIS(D), with JSON diagnostics.

Run with PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1.
Build the repository's native integral extension first if it is not installed.
"""
import json
import platform
from time import perf_counter

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from gradscf import ci, gto, scf


def main():
    atom = "H 0 0 0; H 0 0 0.8; H 0 0 1.9; H 0 0 3.1"
    start = perf_counter()
    mol = gto.M(atom=atom, basis="sto-3g", unit="Angstrom")
    mf = scf.RHF(mol, conv_tol=1e-12).run()
    if not mf.converged:
        raise RuntimeError("HF did not converge")
    report = {
        "molecule": atom, "basis": "sto-3g", "geometry_unit": "Angstrom",
        "energy_unit": "Hartree", "machine": platform.machine(),
        "jax_version": jax.__version__, "devices": [str(d) for d in jax.devices()],
        "float64": bool(jax.config.jax_enable_x64), "integral_backend": mf.integral_backend,
        "hf_energy": float(mf.e_tot), "hf_seconds": perf_counter() - start,
        "methods": {},
    }
    for name in ("CISD", "CISDT", "CISDTQ"):
        start = perf_counter()
        solver = getattr(ci, name)(mf, conv_tol=1e-10).run()
        if not solver.converged:
            raise RuntimeError(f"{name} did not converge")
        report["methods"][name] = {
            "total_energy": float(solver.e_tot), "correlation_energy": float(solver.e_corr),
            "determinants": solver.space.size,
            "residual_norm": float(solver.result.residual_norms[0]),
            "seconds_including_compile": perf_counter() - start,
        }
    start = perf_counter()
    singles = ci.CIS_D(mf, nroots=3, conv_tol=1e-10).run()
    if not np.all(singles.result.valid):
        raise RuntimeError("CIS(D) roots did not pass convergence/denominator checks")
    report["methods"]["CIS(D)"] = {
        "cis_excitation_energies": np.asarray(singles.e_cis).tolist(),
        "corrections": np.asarray(singles.correction).tolist(),
        "excitation_energies": np.asarray(singles.e).tolist(),
        "min_abs_denominators": np.asarray(singles.result.min_abs_denominators).tolist(),
        "seconds_including_compile": perf_counter() - start,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
