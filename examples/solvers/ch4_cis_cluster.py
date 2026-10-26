"""Fully relaxed CH4 geometry response of a bright, triply degenerate singlet.

Run forward discovery first, then the response stage in a fresh bounded
process. Numerical implementation lives in gradscf_tools.cis_cluster.
"""
import hashlib
import json
from pathlib import Path
import platform
import resource
from time import perf_counter
from dataclasses import asdict

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from gradscf import gto, scf
from gradscf.solvers import LinearSolverConfig
from gradscf_tools.cis_cluster import CISCluster


OUTPUT = Path("reproducibility/ad_showcase/A03_ch4_cluster_geometry")
SYMBOLS = ("C", "H", "H", "H", "H")
BASIS = "6-31g*"
BOND_BOHR = 1.09/.529177210903
COORDINATES = np.vstack([np.zeros(3), BOND_BOHR/np.sqrt(3)*np.array(
    [[1., 1., 1.], [1., -1., -1.], [-1., 1., -1.], [-1., -1., 1.]])])
WEIGHTS = np.array([[1., .25, 0.], [.25, -1., 0.], [0., 0., 0.]])
DIRECTION = COORDINATES*np.array([1., -1., 0.])
DIRECTION /= np.linalg.norm(DIRECTION)


def _rhf():
    molecule = gto.M(atom=list(zip(SYMBOLS, COORDINATES)), unit="Bohr", basis=BASIS, cart=True)
    return scf.RHF(molecule, integral_backend="native", conv_tol=1e-12,
        conv_tol_density=1e-11, conv_tol_grad=1e-10, max_cycle=180).run()


def run(stage="forward", *, output=OUTPUT):
    if stage not in ("forward", "response"):
        raise ValueError("stage must be forward or response")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output/("forward.json" if stage == "forward" else "gradscf.json")
    if path.exists():
        raise FileExistsError("Preserve prior evidence; use a fresh output folder")
    began = perf_counter()
    report = dict(complete=False, passed=False, stage=stage, symbols=SYMBOLS,
        coordinates_bohr=COORDINATES.tolist(), basis=BASIS, cart=True,
        direction=DIRECTION.tolist(), tensor_weights=WEIGHTS.tolist(),
        scope="Fully relaxed HF/native geometry and dipole response of a complete singlet CIS cluster; not a spin triplet or individual-root derivative.",
        units=dict(energy="Eh", coordinates="bohr", tensor="e^2 bohr^2", tensor_geometry_derivative="e^2 bohr"),
        hostname=platform.node(), backend=jax.default_backend(), jax=jax.__version__,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__), *sorted(Path("src/gradscf_tools").glob("*response.py")),
             Path("src/gradscf_tools/cis_cluster.py"), Path("src/gradscf/scf/orbital_optimization.py"),
             Path("src/gradscf/tddft/response.py"), Path("src/gradscf/solvers/eigen/response.py")]})
    def save():
        report.update(elapsed_seconds=perf_counter()-began,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        path.write_text(json.dumps(report, indent=2)+"\n")
    save()
    try:
        mf = _rhf()
        if not mf.converged:
            raise RuntimeError("RHF did not converge")
        report["rhf"] = dict(energy_hartree=float(mf.e_tot), converged=True, nao=int(mf.mo_coeff.shape[0]))
        stability = mf.stability(solver="dense", channel="internal", eigensolver_tol=1e-9)
        report["rhf"]["minimum_internal_curvature"] = float(stability.minimum_curvature)
        report["rhf"]["internal_stable"] = bool(stability.stable)
        if stability.stable is not True or stability.minimum_curvature <= 1e-5:
            raise RuntimeError("RHF internal stability gate failed")
        linear = LinearSolverConfig(method="gmres", rtol=1e-10, atol=1e-12, restart=40, maxiter=100)
        model = CISCluster(mf, method="davidson", max_subspace=64, response_config=linear)
        report.update(eigensolver_template=asdict(model.config), response_solver=asdict(linear))
        report["discovery"] = model.discover(COORDINATES, rank=3)
        chosen = report["discovery"]["selected"]
        report["cumulative_ranks"] = [chosen["stop"]]+([chosen["start"]] if chosen["start"] else [])
        tensor = np.asarray(chosen["transition_tensor"])
        isotropy = float(np.max(abs(tensor-np.eye(3)*np.trace(tensor)/3)))
        report["center_isotropy_error"] = isotropy
        if model.dimension != 90 or isotropy > 2e-7 or chosen["splitting_hartree"] > 1e-7:
            raise RuntimeError("CH4 complete bright-cluster/isotropy gate failed")
        save()
        print("Discovered complete bright singlet cluster:", chosen, flush=True)
        if stage == "response":
            result = model.geometry_response(COORDINATES, WEIGHTS, DIRECTION)
            jax.block_until_ready(result)
            gradient = np.asarray(result.gradient)
            report.update(transition_tensor=np.asarray(result.state.transition_tensor).tolist(),
                excitation_energy_sum_hartree=float(result.state.excitation_energy_sum),
                geometry_vjp=gradient.tolist(), tensor_jvp=np.asarray(result.tensor_jvp).tolist(),
                directional_vjp=float(np.sum(gradient*DIRECTION)),
                jvp_vjp_error=float(result.jvp_vjp_error),
                translation_sum_error=float(np.max(abs(gradient.sum(axis=0)))),
                response_valid=bool(result.state.response_valid),
                ritz_residual=float(result.state.max_ritz_residual),
                external_gaps=np.asarray(result.state.boundary_gaps).tolist(),
                adjoint_residual=None, adjoint_iterations=None)
            save()
            if (not result.state.response_valid or not np.isfinite(gradient).all()
                    or abs(report["directional_vjp"]) < 1e-5 or report["jvp_vjp_error"] > 2e-8
                    or report["translation_sum_error"] > 2e-6):
                raise RuntimeError("Complete original-frame geometry response gate failed")
        report.update(complete=True, passed=True)
        save()
        return report
    except Exception as error:
        report.update(passed=False, error=f"{type(error).__name__}: {error}")
        save()
        raise


def check_frame(directory=OUTPUT, *, method="davidson", max_subspace=90):
    """One explicit forward configuration; no automatic dense fallback."""
    directory = Path(directory)
    source = directory/"gradscf.json"
    native = json.loads(source.read_text())
    if not native.get("complete") or not native.get("passed"):
        raise ValueError("Require the passing original-frame geometry response")
    path = directory/f"frame_{method}_{max_subspace}.json"
    if path.exists():
        raise FileExistsError("Preserve earlier frame records")
    began = perf_counter()
    report = dict(complete=False, passed=False, method=method, max_subspace=max_subspace,
        native_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    try:
        model = CISCluster(_rhf(), method=method, max_subspace=max_subspace,
            response_config=LinearSolverConfig(method="gmres", rtol=1e-10, atol=1e-12, restart=40, maxiter=100))
        selection = model.discover(COORDINATES)
        if (selection["selected"]["start"], selection["selected"]["stop"]) != tuple(
                native["discovery"]["selected"][key] for key in ("start", "stop")):
            raise RuntimeError("Gauge comparison selected a different cluster")
        rng = np.random.default_rng(20261016)
        frame = np.zeros((23, 23))
        frame[:5, :5] = np.linalg.qr(rng.normal(size=(5, 5)))[0]
        frame[5:, 5:] = np.linalg.qr(rng.normal(size=(18, 18)))[0]
        result = model.geometry_response(COORDINATES, WEIGHTS, DIRECTION, frame=frame)
        jax.block_until_ready(result)
        tensor_error = float(np.max(abs(np.asarray(result.state.transition_tensor)-native["transition_tensor"])))
        gradient_error = float(np.max(abs(np.asarray(result.gradient)-native["geometry_vjp"])))
        passed = (bool(result.state.response_valid) and np.isfinite(result.gradient).all()
                  and np.isfinite(result.tensor_jvp).all() and tensor_error <= 2e-7
                  and gradient_error <= 2e-6 and float(result.jvp_vjp_error) <= 2e-8)
        report.update(complete=True, passed=bool(passed), frame_seed=20261016,
            transition_tensor=np.asarray(result.state.transition_tensor).tolist(),
            geometry_vjp=np.asarray(result.gradient).tolist(), tensor_error=tensor_error,
            vjp_error=gradient_error, jvp_vjp_error=float(result.jvp_vjp_error),
            response_valid=bool(result.state.response_valid), ritz_residual=float(result.state.max_ritz_residual))
        if not passed:
            raise RuntimeError("Frame gauge comparison failed")
        return report
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report.update(elapsed_seconds=perf_counter()-began,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        path.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    run()

# Verified on c20 CPU/float64, 2026-10-16: complete singlet roots[0:3],
# splitting3.77e-15 Eh, external gap0.04123055 Eh; geometry VJP.u=2.9799479012
# e^2 bohr, JVP/VJP mismatch2.58e-10. Independent two-step PySCF loss-FD
# errors4.55e-6 ->4.08e-7 e^2 bohr. Full90 frame-gauge VJP error2.49e-10;
# the strict Ritz failures at capacities32/64 remain archived, without a
# tolerance change or a Davidson efficiency claim.
