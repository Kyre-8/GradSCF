"""A01 independent HF/PBE energy/dipole finite differences; no CLI."""
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import pyscf

from pyscf_field_reference import energy_and_dipole


OUTPUT = Path("reproducibility/ad_showcase/A01_explicit_implicit")


def main():
    native_path = OUTPUT/"gradscf.json"
    native = json.loads(native_path.read_text())
    if not native["complete"]:
        raise RuntimeError("Complete the matched native cases before independent comparison.")
    report = dict(complete=False, passed=False, cases=[], pyscf_version=pyscf.__version__,
        hostname=platform.node(), native_sha256=hashlib.sha256(native_path.read_bytes()).hexdigest(),
        reference_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        field_reference_sha256=hashlib.sha256(Path(__file__).with_name("pyscf_field_reference.py").read_bytes()).hexdigest(),
        value_tolerances=dict(energy=1e-8, dipole_z=1e-7), gradient_theta_atol=2e-6,
        energy_definition="Electronic finite-field SCF plus Enuc minus F dot sum(ZR); no common GradSCF arrays.")
    (OUTPUT/"pyscf.json").write_text(json.dumps(report, indent=2)+"\n")
    start = perf_counter()
    for method in native["methods"]:
        cases = [case for case in native["cases"] if case["method"] == method]
        data = cases[0]
        point = np.asarray(data["point_theta"])
        base = np.asarray(data["seed_coordinates_bohr"])
        stretch = np.asarray(data["coordinate_direction_bohr"])
        field = np.asarray(data["field_direction_au"])

        def observable(theta):
            return energy_and_dipole(data["symbols"], base+theta[0]*stretch,
                data["basis"], theta[1]*field, method=method, grid_level=data["grid_level"] or 0)

        values = observable(point)
        checks = []
        for step in [check["step_theta"] for check in data["finite_difference_checks"]]:
            jacobian = np.column_stack([(observable(point+step*axis)-observable(point-step*axis))/(2*step)
                                         for axis in np.eye(2)])
            errors = {}
            for case in cases:
                row = 0 if case["observable"] == "energy" else 3
                key = f"{case['mode']}_{case['observable']}"
                errors[key] = float(np.max(np.abs(jacobian[row]-case["gradient_theta"])))
            checks.append(dict(step_theta=step, jacobian_theta=jacobian.tolist(), errors=errors))
        value_errors = {}
        for case in cases:
            row = 0 if case["observable"] == "energy" else 3
            value_errors[f"{case['mode']}_{case['observable']}"] = abs(float(values[row])-case["value"])
            np.testing.assert_allclose(values[row], case["value"],
                                       atol=report["value_tolerances"][case["observable"]], rtol=0)
            np.testing.assert_allclose(jacobian[row], case["gradient_theta"], atol=2e-6, rtol=2e-5)
        report["cases"].append(dict(method=method, values=values.tolist(),
            value_errors=value_errors, finite_difference_checks=checks, passed=True))
        report["elapsed_seconds"] = perf_counter()-start
        (OUTPUT/"pyscf.json").write_text(json.dumps(report, indent=2)+"\n")
        print("A01 independent", method, "PASS; last-step errors", checks[-1]["errors"], flush=True)
    report["complete"] = report["passed"] = True
    (OUTPUT/"pyscf.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()

# jy_gpu_1/PySCF 2.13.0: HF and PBE gates PASS in 1.783 s;
# smallest-step energy/dipole theta-gradient error <=8.10e-10.
