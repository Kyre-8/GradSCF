"""Checkpointed raw LiH force constants on a compatible rectangular q mesh.

The fixed P03 geometry/basis/XC is retained. k=(2,2,3) permits genuinely
distinct q=(0,0,+/-1/3). Every requested q is calculated independently; no
conjugacy or ASR repair is applied. Run one process per q on jy_gpu_1 with
PYTHONPATH=src:. and an external RSS/time watchdog.
"""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import resource
from time import perf_counter


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_digest():
    digest = hashlib.sha256()
    for path in sorted(Path("src").rglob("*.py")):
        digest.update(path.as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def save_checkpoint(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2)+"\n")
    temporary.replace(path)


def open_checkpoint(path, identity):
    path = Path(path)
    if path.exists():
        record = json.loads(path.read_text())
        if record.get("identity") != identity:
            raise ValueError("Checkpoint identity changed; preserve the previous result and choose a new folder")
        return record
    return dict(identity=identity, complete=False, passed=False, columns=[],
                started_utc=datetime.now(timezone.utc).isoformat())


def run_point(q, *, kmesh=(2, 2, 3), mesh=31, output=None,
              max_grid=30000, max_pair_elements=40000000):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from gradscf import dfpt
    from gradscf.dfpt.force_constants import primitive_force_response
    from gradscf.pbc.momentum import momentum_transfer
    from gradscf.solvers import LinearSolverConfig
    from examples.pbc.lih_forces_stress import make_mean_field
    from examples.pbc.lih_gamma_phonons import SETTINGS, MASSES_AMU, _record

    jax.config.update("jax_enable_x64", True)
    began = perf_counter()
    kmesh = np.asarray(kmesh, dtype=int)
    q = np.asarray(q, dtype=float)
    if kmesh.shape != (3,) or np.any(kmesh < 1) or mesh < 3 or mesh % 2 != 1:
        raise ValueError("Use a positive rectangular k mesh and an odd FFT dimension")
    if q.shape != (3,) or not np.isfinite(q).all() or not np.allclose(q*kmesh, np.rint(q*kmesh), atol=1e-10, rtol=0):
        raise ValueError("Requested q is not commensurate with the declared k mesh")
    indices = np.rint(q*kmesh).astype(int) % kmesh
    signed = np.where(2*indices >= kmesh, indices-kmesh, indices)
    canonical = signed/kmesh
    key = "q_"+"_".join(str(int(i)).replace("-", "m") for i in signed)
    settings = dict(SETTINGS, kmesh=kmesh.tolist(), mesh=[mesh]*3)
    folder = Path(output) if output is not None else Path("reproducibility/ad_showcase/P03_lih_qmesh")/(
        "k"+"x".join(map(str, kmesh))+f"_fft{mesh}")
    folder.mkdir(parents=True, exist_ok=True)
    path, state_path = folder/(key+".json"), folder/(key+"_reference.npz")
    config = LinearSolverConfig(method="gmres", rtol=1e-10, atol=1e-12, restart=40, maxiter=100)
    gates = dict(linear_residual=1e-9, hermiticity_hartree_per_bohr2=1e-8)
    helpers = {name:fingerprint(Path(__file__).with_name(name))
               for name in ("lih_forces_stress.py", "lih_gamma_phonons.py")}
    identity = dict(source_python_sha256=source_digest(), script_sha256=fingerprint(__file__),
        helper_sha256=helpers, settings=settings, q_evaluated=canonical.tolist(),
        masses_amu=MASSES_AMU.tolist(), linear_solver=asdict(config), gates=gates,
        phase="Phi_ab(q)=sum_R C_ab(R)*exp(2*pi*i*q.R); lattice translation only")
    record = open_checkpoint(path, identity)
    if record["complete"]:
        if not record["passed"]:
            raise RuntimeError("The preserved completed q record failed its raw gates")
        print("Reusing complete q checkpoint:", path, flush=True)
        return record
    record.update(stage="SCF", q_requested=q.tolist(), q_evaluated=canonical.tolist(),
        q_indices=indices.tolist(), settings=settings, asr_projected=False,
        source_python_sha256=identity["source_python_sha256"],
        script_sha256=identity["script_sha256"], hostname=platform.node(),
        python=platform.python_version(), jax=jax.__version__, dtype="float64/complex128")
    save_checkpoint(path, record)
    mean_field = make_mean_field(settings)
    predicted_pairs = mesh**3*mean_field.cell.topology.nao**2*len(mean_field.kpts)**2
    record["workspace_limits"] = dict(max_grid=max_grid, max_pair_elements=max_pair_elements)
    record["workspace_pair_elements"] = predicted_pairs
    save_checkpoint(path, record)
    if mesh**3 > max_grid or predicted_pairs > max_pair_elements:
        raise ValueError("The declared grid/AO-pair workspace limit is too small; assess resources before increasing it")
    if "reference_sha256" in record:
        if not state_path.exists() or fingerprint(state_path) != record["reference_sha256"]:
            raise ValueError("The reference-density checkpoint changed")
        saved_density = np.load(state_path)["density"]
        mean_field.kernel(dm0=saved_density)
        delta = float(np.max(np.abs(np.asarray(mean_field.make_rdm1())-saved_density)))
        energy_delta = abs(float(mean_field.e_tot)-record["reference"]["energy_hartree"])
        record["resumed_reference_difference"] = dict(density_max=delta, energy_hartree=energy_delta)
        if delta > 1e-8 or energy_delta > 2e-10:
            raise RuntimeError("Re-converged checkpoint reference changed beyond the declared tolerance")
    else:
        mean_field.run()
    actual = _record(mean_field, settings)
    if "reference_sha256" not in record:
        temporary = state_path.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, density=np.asarray(mean_field.make_rdm1()))
        temporary.replace(state_path)
        record["reference_sha256"] = fingerprint(state_path)
        record["reference"] = actual
    record["current_reference"] = actual
    grid = mesh**3
    nk, nao = len(mean_field.kpts), actual["nao"]
    transfer = momentum_transfer(actual["fractional_kpoints"], canonical)
    record.update(stage="raw force-constant columns", reference_seconds=perf_counter()-began,
        workspace_grid_points=grid, workspace_pair_elements=grid*nao**2*nk**2,
        primary_ao_shape=[nk, 4, grid, nao], k_partners=transfer.partners.tolist(),
        reciprocal_shifts=transfer.shifts.tolist(), response_dimension=(2 if not np.any(signed) else 4)*nk*nao**2)
    save_checkpoint(path, record)
    if not np.any(signed):
        problem = dfpt.atomic_problem(mean_field)
        compiled = jax.jit(lambda direction: problem.response(tangent=direction, config=config))
        def respond(direction):
            answer = compiled(direction)
            return -answer.values["forces"].ravel(), answer.residual_norm, answer.converged
    else:
        _, respond = primitive_force_response(mean_field, canonical, config)
    existing = {int(row["index"]):row for row in record["columns"]}
    if len(existing) != len(record["columns"]) or any(index not in range(6) for index in existing):
        raise ValueError("Malformed or duplicate column checkpoint")
    for index, direction in enumerate(np.eye(6).reshape(6, 2, 3)):
        if index in existing and existing[index]["converged"]:
            continue
        tick = perf_counter()
        column, residual, converged = respond(jnp.asarray(direction))
        column = np.asarray(column).reshape(6)
        row = dict(index=index, direction=direction.tolist(), real=column.real.tolist(),
            imag=column.imag.tolist(), residual=float(residual), converged=bool(converged),
            seconds=perf_counter()-tick)
        existing[index] = row
        record["columns"] = [existing[i] for i in sorted(existing)]
        record.update(completed_columns=len(existing), elapsed_seconds_this_invocation=perf_counter()-began,
                      peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        save_checkpoint(path, record)
        print(key, "column", index, "residual", row["residual"], "seconds", row["seconds"], flush=True)
        if not row["converged"] or not np.isfinite(column).all() or row["residual"] > gates["linear_residual"]:
            raise RuntimeError("A raw q-response column failed its finite/convergence/residual gate")
    matrix = np.stack([np.asarray(existing[i]["real"])+1j*np.asarray(existing[i]["imag"])
                       for i in range(6)], axis=1)
    errors = dict(linear_residual=max(row["residual"] for row in existing.values()),
                  hermiticity_hartree_per_bohr2=float(np.max(np.abs(matrix-matrix.conj().T))))
    passed = bool(np.isfinite(matrix).all() and all(errors[name] <= limit for name, limit in gates.items()))
    record.update(complete=True, passed=passed, stage="complete", errors=errors, gates=gates,
        matrix_real_hartree_per_bohr2=matrix.real.tolist(), matrix_imag_hartree_per_bohr2=matrix.imag.tolist(),
        max_imaginary_hartree_per_bohr2=float(np.max(np.abs(matrix.imag))),
        raw_asr_residual_hartree_per_bohr2=float(np.linalg.norm(matrix@np.tile(np.eye(3), (2, 1)))) if not np.any(signed) else None,
        elapsed_seconds_this_invocation=perf_counter()-began,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope="One independently computed raw q matrix of a fixed finite model; no ASR/conjugacy repair or material-convergence claim.")
    save_checkpoint(path, record)
    print("Completed raw q matrix:", path, passed, errors, flush=True)
    if not passed:
        raise RuntimeError("Raw q matrix failed Hermiticity; preserve it before any refinement")
    return record


def check_pair(paths, output):
    import numpy as np
    records = [json.loads(Path(path).read_text()) for path in paths]
    if not all(row.get("complete") and row.get("passed") for row in records):
        raise ValueError("Both independently computed q records must be complete and passing")
    identities = [{key:value for key, value in row["identity"].items() if key != "q_evaluated"} for row in records]
    if identities[0] != identities[1]:
        raise ValueError("q-pair model/source identities differ")
    qs = np.asarray([row["q_evaluated"] for row in records])
    if not np.allclose(qs.sum(axis=0), np.rint(qs.sum(axis=0)), atol=1e-12, rtol=0):
        raise ValueError("The two supplied q points are not opposite reciprocal representatives")
    if np.allclose(2*qs[0], np.rint(2*qs[0]), atol=1e-12, rtol=0):
        raise ValueError("A self-inverse point is not an independent complex q/-q test")
    matrices = [np.asarray(row["matrix_real_hartree_per_bohr2"])+1j*np.asarray(row["matrix_imag_hartree_per_bohr2"])
                for row in records]
    error = float(np.max(np.abs(matrices[1]-matrices[0].conj())))
    report = dict(complete=True, passed=bool(error <= 1e-8), qpoints=qs.tolist(),
        conjugacy_error_hartree_per_bohr2=error, conjugacy_gate_hartree_per_bohr2=1e-8,
        max_imaginary_hartree_per_bohr2=max(row["max_imaginary_hartree_per_bohr2"] for row in records),
        input_sha256={str(path):fingerprint(path) for path in paths}, identity=identities[0],
        asr_projected=False, scope="Raw independent positive/negative q matrices; no averaged or repaired pair.")
    save_checkpoint(output, report)
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise RuntimeError("Raw q/-q conjugacy failed; do not apply ASR to hide the defect")
    return report


def run(q, *, kmesh=(2, 2, 3), mesh=31, output=None,
        max_grid=30000, max_pair_elements=40000000):
    """Compute one q point; use check_pair(paths, output) separately for a pair."""
    return run_point(q, kmesh=kmesh, mesh=mesh, output=output,
                     max_grid=max_grid, max_pair_elements=max_pair_elements)


if __name__ == "__main__":
    run((0., 0., 1/3), mesh=41, max_grid=70000, max_pair_elements=90000000,
        output=Path("artifacts/ad_showcase/P03_qmesh_example"))
