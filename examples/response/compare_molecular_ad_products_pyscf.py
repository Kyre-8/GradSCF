"""A02 independent water finite differences of energies and dipoles.

The dense Jacobian uses independently relaxed scalar observables. H*v uses
mixed central energy differences in theta, with four energies per component;
no GradSCF gradient/Hessian or PySCF AD is used to construct that oracle.
"""
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import pyscf

from pyscf_field_reference import energy_and_dipole


OUTPUT = Path("reproducibility/ad_showcase/A02_molecular_products")


def main():
    native_path = OUTPUT/"gradscf.json"
    data = json.loads(native_path.read_text())
    if not data["complete"]:
        raise RuntimeError("Finish the native product/dense/adjoint gates first.")
    report = dict(complete=False, passed=False, pyscf_version=pyscf.__version__,
        hostname=platform.node(), native_sha256=hashlib.sha256(native_path.read_bytes()).hexdigest(),
        reference_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        field_reference_sha256=hashlib.sha256(Path(__file__).with_name("pyscf_field_reference.py").read_bytes()).hexdigest(),
        jacobian_checks=[], hvp_checks=[], jacobian_theta_atol=2e-6, hvp_theta_atol=2e-6,
        oracle="Independent PySCF finite-field E and dipole; HVP from four-energy mixed central differences.")
    (OUTPUT/"pyscf.json").write_text(json.dumps(report, indent=2)+"\n")
    base = np.asarray(data["base_coordinates_bohr"])
    point, direction = np.asarray(data["point_theta"]), np.asarray(data["direction_theta"])
    cotangent = np.asarray(data["cotangent"])

    def observable(theta):
        coordinates = base+data["coordinate_scale_bohr"]*theta[:9].reshape(3, 3)
        field = data["field_scale_au"]*theta[9:]
        return energy_and_dipole(data["symbols"], coordinates, data["basis"], field, method="HF")

    start = perf_counter()
    values = observable(point)
    np.testing.assert_allclose(values[0], data["values"][0], atol=1e-8, rtol=0)
    np.testing.assert_allclose(values[1:], data["values"][1:], atol=1e-7, rtol=0)
    report["values"] = values.tolist()
    report["value_errors"] = np.abs(values-data["values"]).tolist()
    for step in (1e-2, 3e-3, 1e-3):
        jacobian = np.column_stack([(observable(point+step*axis)-observable(point-step*axis))/(2*step)
                                     for axis in np.eye(12)])
        errors = dict(jacobian=float(np.max(np.abs(jacobian-data["jacobian_theta"]))),
            jvp=float(np.max(np.abs(jacobian@direction-data["jvp"]))),
            vjp=float(np.max(np.abs(jacobian.T@cotangent-data["vjp"]))))
        report["jacobian_checks"].append(dict(step_theta=step, jacobian_theta=jacobian.tolist(), errors=errors))
        report["elapsed_seconds"] = perf_counter()-start
        (OUTPUT/"pyscf.json").write_text(json.dumps(report, indent=2)+"\n")
        print("A02 independent Jacobian", step, errors, flush=True)
    np.testing.assert_allclose(jacobian, data["jacobian_theta"], atol=2e-6, rtol=2e-5)
    for step in (2e-2, 1e-2, 5e-3):
        product = []
        for axis in np.eye(12):
            pp = observable(point+step*(direction+axis))[0]
            pm = observable(point+step*(direction-axis))[0]
            mp = observable(point-step*(direction-axis))[0]
            mm = observable(point-step*(direction+axis))[0]
            product.append((pp-pm-mp+mm)/(4*step**2))
        error = float(np.max(np.abs(np.asarray(product)-data["hvp"])))
        report["hvp_checks"].append(dict(step_theta=step, hvp=product, max_error=error))
        report["elapsed_seconds"] = perf_counter()-start
        (OUTPUT/"pyscf.json").write_text(json.dumps(report, indent=2)+"\n")
        print("A02 independent four-energy HVP", step, error, flush=True)
    np.testing.assert_allclose(product, data["hvp"], atol=2e-6, rtol=2e-5)
    report["complete"] = report["passed"] = True
    (OUTPUT/"pyscf.json").write_text(json.dumps(report, indent=2)+"\n")
    print("A02 independent E/mu, dense Jacobian and HVP gates PASS", flush=True)


if __name__ == "__main__":
    main()

# jy_gpu_1/PySCF 2.13.0: E, dipole, dense Jacobian and HVP gates PASS in 44.477 s.
# Four-energy HVP errors at h=.02,.01,.005: 8.05e-10, 1.10e-9, 5.77e-9.
