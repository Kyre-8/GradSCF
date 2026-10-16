"""Compare raw IFC interpolation with direct, genuinely held-out DFPT points.

Electronic model and source must match exactly. No numerical interpolation
accuracy is presumed: errors are reported separately from provenance/raw
response checks, without ASR or an asserted material-convergence verdict.
"""
import hashlib
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np


def validate_holdout(reference, direct_reference, samples, point):
    for name in ("settings", "lattice_bohr", "coordinates_bohr", "fractional_kpoints",
                 "basis_data", "pseudopotential_data", "masses_amu", "source_python_sha256"):
        if reference[name] != direct_reference[name]:
            raise ValueError(f"Held-out electronic-model/source {name} differs")
    difference = np.asarray(samples)-np.asarray(point)
    if np.any(np.max(abs(difference-np.rint(difference)), axis=1) <= 1e-10):
        raise ValueError("A sampled point or reciprocal copy is not held out")


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(input_path, direct, output):
    """Run this example with ordinary Python parameters; no command-line parser."""
    args = SimpleNamespace(input=Path(input_path), direct=[Path(path) for path in direct], output=Path(output))
    import jax
    from scipy.constants import physical_constants, h, c
    from gradscf.pbc.ifc import ForceConstants

    jax.config.update("jax_enable_x64", True)
    began = perf_counter()
    coarse_path, arrays_path = args.input/"results.json", args.input/"ifc_data.npz"
    coarse = json.loads(coarse_path.read_text())
    if not coarse.get("complete") or not coarse.get("passed") or coarse.get("asr_projected"):
        raise ValueError("Require completed raw coarse IFC data")
    if fingerprint(arrays_path) != coarse["arrays_sha256"]:
        raise ValueError("Coarse IFC array checksum changed")
    for name, expected in coarse["interpolation_source_sha256"].items():
        if fingerprint(Path("src/gradscf/pbc")/name) != expected:
            raise ValueError("Interpolation source changed")
    arrays = np.load(arrays_path)
    records = [json.loads(path.read_text()) for path in args.direct]
    qpoints = np.asarray([record["q_evaluated"] for record in records])
    if (np.max(abs(qpoints.sum(axis=0)-np.rint(qpoints.sum(axis=0)))) > 1e-10
            or np.max(abs(2*qpoints[0]-np.rint(2*qpoints[0]))) <= 1e-10):
        raise ValueError("Require an independently evaluated non-self-inverse q/-q pair")
    matrices = []
    for record in records:
        if not record.get("complete") or not record.get("passed") or record.get("asr_projected"):
            raise ValueError("A direct raw q record is incomplete/failed/projected")
        validate_holdout(coarse["reference"], record["reference"], arrays["qpoints"], record["q_evaluated"])
        if abs(record["reference"]["energy_hartree"]-coarse["reference"]["energy_hartree"]) > 2e-10:
            raise ValueError("Direct and coarse reference energies differ")
        matrices.append(np.asarray(record["matrix_real_hartree_per_bohr2"])
                        +1j*np.asarray(record["matrix_imag_hartree_per_bohr2"]))
    matrices = np.asarray(matrices)
    raw_errors = dict(hermiticity=float(np.max(abs(matrices-matrices.conj().transpose(0, 2, 1)))),
                     conjugacy=float(np.max(abs(matrices[1]-matrices[0].conj()))))
    if not np.isfinite(matrices).all() or max(raw_errors.values()) > 1e-8:
        raise ValueError(f"Direct pair fails raw checks: {raw_errors}")
    model = ForceConstants.from_qmesh(SimpleNamespace(lattice=arrays["lattice_bohr"],
        coords=arrays["coordinates_bohr"]), arrays["qpoints"], arrays["matrices"],
        masses=arrays["masses_amu"], asr=False, tolerance=1e-8)
    interpolated = np.asarray(model.matrix(qpoints))
    conversion = physical_constants["Hartree energy"][0]/(h*c*100)
    ratio = physical_constants["atomic mass constant"][0]/physical_constants["electron mass"][0]
    scale = np.sqrt(np.repeat(arrays["masses_amu"], 3)*ratio)
    def frequencies(matrix):
        eigenvalues = np.linalg.eigvalsh(matrix/scale[:, None]/scale[None, :])
        return np.sign(eigenvalues)*np.sqrt(abs(eigenvalues))*conversion
    points = []
    for q, direct, prediction in zip(qpoints, matrices, interpolated):
        actual_frequencies, predicted_frequencies = frequencies(direct), frequencies(prediction)
        points.append(dict(q=q.tolist(), max_matrix_error_hartree_per_bohr2=float(np.max(abs(prediction-direct))),
            relative_frobenius_matrix_error=float(np.linalg.norm(prediction-direct)/np.linalg.norm(direct)),
            direct_frequencies_cm1=actual_frequencies.tolist(), interpolated_frequencies_cm1=predicted_frequencies.tolist(),
            max_frequency_error_cm1=float(np.max(abs(actual_frequencies-predicted_frequencies))),
            direct_matrix_real=direct.real.tolist(), direct_matrix_imag=direct.imag.tolist(),
            interpolated_matrix_real=prediction.real.tolist(), interpolated_matrix_imag=prediction.imag.tolist()))
    report = dict(complete=True, finite_model_checks_passed=True, interpolation_convergence_established=False,
        settings=coarse["settings"], coarse_qmesh=coarse["qmesh"], points=points,
        raw_direct_errors_hartree_per_bohr2=raw_errors, asr_projected=False, polar_correction=False,
        source_python_sha256=coarse["source_python_sha256"], script_sha256=fingerprint(__file__),
        input_sha256={str(path):fingerprint(path) for path in [coarse_path, arrays_path, *args.direct]},
        elapsed_seconds=perf_counter()-began,
        scope="Same electronic model: direct held-out raw matrices versus finite coarse-q interpolation. Errors are measured, not presumed small; no bulk/convergence claim.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print("Held-out interpolation errors:", [(p["q"], p["max_matrix_error_hartree_per_bohr2"],
          p["max_frequency_error_cm1"]) for p in points], flush=True)


if __name__ == "__main__":
    run(Path("reproducibility/ad_showcase/P03_lih_qmesh/k2x2x6_fft41/ifc_raw"),
        [Path("reproducibility/ad_showcase/P03_lih_qmesh/k2x2x6_fft41/q_0_0_1.json"),
         Path("reproducibility/ad_showcase/P03_lih_qmesh/k2x2x6_fft41/q_0_0_m1.json")],
        Path("reproducibility/ad_showcase/P03_lih_qmesh/k2x2x6_fft41/heldout_zsixth.json"))
