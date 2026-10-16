"""A06: reuse the ethylene Boys example and archive its independent checks.

The occupied subspace is fixed after native RHF. PySCF localizes those same
orbitals independently; it is not a second independent SCF calculation.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import runpy
from time import perf_counter

import jax
import numpy as np


def main():
    start = perf_counter()
    root = Path(__file__).resolve().parents[2]
    source = root/"examples/localization/ethylene_boys.py"
    example = runpy.run_path(str(source))
    path = root/"tests/comparisons/compare_ethylene_boys.py"
    spec = importlib.util.spec_from_file_location("independent_boys", path)
    comparison = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparison)
    _, overlaps, difference = comparison.compare(example)
    assert abs(difference) < 2e-7
    history = []
    dipoles = np.asarray(example["dipole_mo"])
    for u in example["rotation_history"]:
        rotated = np.einsum("pi,xpq,qj->xij", u, dipoles, u)
        history.append(float(np.sum(np.diagonal(rotated, axis1=1, axis2=2)**2)))
    gradient_error = abs(float(example["grad"]@example["vector"]-example["fd"]))
    hvp_error = float(np.max(np.abs(np.asarray(example["hvp"]-example["fd_hvp"]))))
    analytic_error = float(np.max(np.abs(np.asarray(example["local_ad"]-example["analytic"]))))
    digest = hashlib.sha256()
    for file in sorted((root/"src").rglob("*.py")):
        digest.update(file.relative_to(root).as_posix().encode()); digest.update(file.read_bytes())
    report = dict(complete=True, source_python_sha256=digest.hexdigest(),
        executed_scripts={str(file.relative_to(root)):hashlib.sha256(file.read_bytes()).hexdigest()
            for file in (source, path, Path(__file__))},
        hostname=platform.node(), python=platform.python_version(), jax=jax.__version__,
        devices=[str(d) for d in jax.devices()], dtype="float64", elapsed_seconds=perf_counter()-start,
        molecule="ethylene", basis="cc-pvdz", cart=True, noccupied=int(example["nocc"]),
        nrotation_parameters=int(example["nparam"]), fixed_subspace=True,
        rhf_energy_hartree=float(example["mf"].e_tot),
        score_units="bohr^2", initial_score=float(example["score_initial"]), final_score=float(example["score_final"]),
        local_gradient_norm=float(np.linalg.norm(example["local_gradient"])),
        orthogonality_error=float(example["orth_error"]), density_error=float(example["density_error"]),
        fd_step=float(example["step"]), gradient_fd_error=gradient_error, hvp_fd_error=hvp_error,
        analytic_local_gradient_error=analytic_error, independent_pyscf_score_difference=float(difference),
        matched_absolute_overlaps=np.asarray(overlaps).tolist(),
        multistart_scores=[float(-r.fun) for r in example["results"]],
        trajectory_scores=history, trajectory_stages=example["iteration_stages"],
        scope="AD acts on orthogonal rotations in a fixed occupied subspace. No SCF/nuclear derivative, global minimum or unique-orbital claim.")
    output = root/"reproducibility/ad_showcase/A06_boys_localization"
    output.mkdir(parents=True, exist_ok=True)
    (output/"results.json").write_text(json.dumps(report, indent=2)+"\n")
    print("A06 native FD/HVP/invariance and independent multistart Boys gates PASS", flush=True)


if __name__ == "__main__":
    main()

# jy_gpu_1 CPU/x64: native and independent Boys checks PASS in 21.38 s.
# Final score25.574296815574 bohr^2; independent difference6.40e-14 bohr^2.
# Fixed-subspace gradient/HVP FD errors4.56e-10/8.95e-10 bohr^2.
