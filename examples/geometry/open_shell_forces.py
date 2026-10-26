"""G03: UHF, UKS-PBE, UMP2 and UCCSD gradients for real OH/NO doublets.

An internal stability check and spin diagnostics accompany each stationary
reference branch; post-HF spin/stability are UHF determinant diagnostics.
The independent reference receives its density only as an initial
guess to select the same real pi orientation; it recomputes all observables.
"""
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from gradscf import dft, gto, scf
from gradscf_tools.molecular_response import MolecularEnergy
from gradscf_tools.correlated_response import CorrelatedEnergy


SYSTEMS = {"OH": "O 0 0 0; H 0 0 1.85", "NO": "N 0 0 0; O 0 0 2.18"}
METHODS = ("HF", "PBE", "MP2", "CCSD")
OUTPUT = Path("reproducibility/ad_showcase/G03_open_shell_forces")


def run(name, method):
    if method not in {"HF", "PBE", "MP2", "CCSD"}:
        raise ValueError("Choose HF, PBE, MP2 or CCSD.")
    mol = gto.M(atom=SYSTEMS[name], spin=1, basis="6-31g*", unit="Bohr", cart=True)
    controls = dict(conv_tol=1e-12, conv_tol_density=1e-11, conv_tol_grad=1e-10,
                    max_cycle=250, integral_backend="native")
    mf = (scf.UHF(mol, **controls) if method != "PBE"
          else dft.UKS(mol, xc="pbe", grids_level=3, **controls)).run()
    if not mf.converged:
        raise RuntimeError(f"{name}/{method} initial SCF failed.")
    initial_energy = float(mf.e_tot)
    following = dict(status="initial branch stable", restarts=0, energy_history=[initial_energy])
    # For these small systems inspect the complete real orbital Hessian.
    # Partial Davidson roots can miss the nearly flat pi-rotation direction.
    stability_options = dict(solver="dense", stability_tol=1e-5, eigensolver_tol=1e-9)
    check = mf.stability(**stability_options)
    print(name, method, "initial internal curvature", check.minimum_curvature,
          "resolved", check.eigensolver_converged, flush=True)
    if not check.stationary or not check.eigensolver_converged:
        raise RuntimeError(f"{name}/{method} internal stability remains unresolved.")
    if not check.stable:
        followed = mf.stabilize(max_restarts=4, **stability_options)
        if not followed.stable:
            raise RuntimeError(f"{name}/{method} bounded stability following failed.")
        following = dict(status=followed.status, restarts=followed.restarts,
                         energy_history=list(followed.energy_history))
        mf, check = followed.result, followed.stability
    correlated = method in {"MP2", "CCSD"}
    model = (CorrelatedEnergy(mf, method) if correlated
             else MolecularEnergy(mf, gradient_tolerance=1e-10))
    coordinates = model.coordinates
    value_gradient = jax.jit(jax.value_and_grad(model))
    start = perf_counter()
    value, gradient = value_gradient(coordinates)
    gradient = np.asarray(gradient)
    cold = perf_counter()-start
    if not np.isfinite(value) or not np.isfinite(gradient).all():
        raise RuntimeError(f"{name}/{method} energy/force response failed.")
    if not correlated:
        np.testing.assert_allclose(value, mf.e_tot, atol=2e-9, rtol=0)
    np.testing.assert_allclose(gradient.sum(axis=0), 0., atol=2e-7, rtol=0)
    np.testing.assert_allclose(np.cross(np.asarray(coordinates), gradient).sum(axis=0), 0., atol=2e-6, rtol=0)
    warm_gradient = []
    for _ in range(3):
        start = perf_counter()
        np.asarray(value_gradient(coordinates)[1])
        warm_gradient.append(perf_counter()-start)
    direction = jnp.array([[0., 0., -.5], [0., 0., .5]])
    forward = jax.jit(model)
    start = perf_counter()
    forward(coordinates).block_until_ready()
    cold_forward = perf_counter()-start
    warm_forward = []
    for _ in range(3):
        start = perf_counter()
        forward(coordinates).block_until_ready()
        warm_forward.append(perf_counter()-start)
    checks = []
    for step in (1e-3, 3e-4, 1e-4):
        finite = float((forward(coordinates+step*direction)-forward(coordinates-step*direction))/(2*step))
        if not np.isfinite(finite):
            raise RuntimeError("A displaced open-shell state failed.")
        checks.append(dict(step_bohr=step, derivative=finite,
                           error=abs(finite-float(np.sum(gradient*np.asarray(direction))))))
    assert checks[-1]["error"] < 3e-6
    spin_density = np.asarray(mf.make_rdm1())
    reference_model = model.reference if correlated else model
    metric = np.asarray(reference_model.inputs(coordinates)["overlap"])
    electrons = np.einsum("spq,qp->s", spin_density, metric)
    spin_z = .5*(electrons[0]-electrons[1])
    spin_square = float(spin_z*(spin_z+1)+electrons[1]
                        - np.trace(spin_density[0]@metric@spin_density[1]@metric))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    row = dict(name=name, method=method, atom_bohr=SYSTEMS[name], symbols=list(mol.to_spec().symbols),
               coordinates_bohr=np.asarray(coordinates).tolist(), basis="6-31g*", cart=True, spin=1,
               energy_hartree=float(value), gradient_hartree_per_bohr=gradient.tolist(),
               spin_density=spin_density.tolist(), spin_electrons=electrons.tolist(), spin_square=spin_square,
               reference_energy_hartree=float(mf.e_tot),
               spin_diagnostic="UHF reference determinant" if correlated else "SCF determinant",
               internal_minimum_curvature=float(check.minimum_curvature),
               initial_energy_hartree=initial_energy, stability_following=following,
               stability_residual_norms=np.asarray(check.residual_norms).tolist(),
               stability_spectrum_certified=bool(check.spectrum_certified),
               stability_options=stability_options, scf_controls=controls,
               orbital_gradient_norm=float(check.gradient_norm),
               grid_level=3 if method == "PBE" else None, grid_response=method == "PBE",
               direction=np.asarray(direction).tolist(), finite_difference_checks=checks,
               cold_energy_gradient_seconds=cold, warm_energy_gradient_seconds=warm_gradient,
               cold_forward_seconds=cold_forward, warm_forward_seconds=warm_forward,
               hostname=platform.node(), python_version=platform.python_version(),
               jax_version=jax.__version__, devices=[str(device) for device in jax.devices()],
               dtype="float64", runtime_source_hashes={path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                   for path in ("src/gradscf_tools/molecular_response.py",
                                "src/gradscf_tools/correlated_response.py",
                                "src/gradscf/scf/orbital_optimization.py",
                                "src/gradscf/scf/stability.py",
                                "src/gradscf/solvers/nonlinear/root.py", __file__)})
    (OUTPUT / f"{name}_{method.lower()}.json").write_text(json.dumps(row, indent=2)+"\n")
    print(f"{name}/{method}: E={float(value):.12f}, <S^2>={spin_square:.8f}, "
          f"bond gradient={np.sum(gradient*np.asarray(direction)):.8f}, "
          f"FD error={checks[-1]['error']:.3e}", flush=True)
    return row


if __name__ == "__main__":
    for name in SYSTEMS:
        for method in METHODS:
            run(name, method)

# Measured on jy_gpu_1, CPU float64, 2026-10-15; dE/dR / Eh Bohr^-1:
# OH/HF  +0.020708807931; OH/PBE -0.011179973529
# NO/HF  +0.058040664960; NO/PBE -0.035350763953
# Independent PySCF max gradient errors in that order:
# [6.08e-13, 7.07e-15, 1.40e-11, 1.26e-14] Eh Bohr^-1.
# Dense NO/PBE minimum curvature = -1.362e-6 Eh; retained explicitly,
# within stability_tol=1e-5, not a strictly positive spectrum.
# Warm energy+gradient medians: [0.194, 4.003, 2.169, 27.805] s.
# Correlated runs: integration snapshot, distinct from the four mean-field
# stationary-snapshot runs above; selected runtime source hashes are in JSON.
# OH/MP2 +0.000014409093; OH/CCSD -0.004092082817
# NO/MP2 +0.115508670067; NO/CCSD -0.026092878617 Eh Bohr^-1.
# Smallest-step FD errors: [3.05e-9, 3.00e-9, 8.15e-8, 6.53e-9].
# Independent SciPy physical-pi-gauge/PySCF gradient assembly errors:
# [5.79e-14, 1.40e-12, 9.33e-11, 5.26e-11] Eh Bohr^-1.
# Correlated warm energy+gradient medians: [0.644, 6.666, 3.325, 34.955] s.
# These first nuclear derivatives do not establish arbitrary pi-orbital
# response, correlated-state spin/stability, or higher response orders.
