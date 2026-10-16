"""Independent SciPy linear algebra for the saved full BSE optical kernel.

The A/B kernel and AO integrals are native inputs, not an independent GW
calculation. This checks the complete spectral sum, normalization, signs,
spin factor and retarded convention against a direct 140-dimensional solve.
"""
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.linalg import eigvalsh, solve
from scipy.integrate import simpson


OUTPUT = Path("reproducibility/ad_showcase/V06_water_bse_dynamic_raman")


def main():
    started = perf_counter()
    data = np.load(OUTPUT/"response.npz")
    a, b, d = (data[key] for key in ("a", "b", "position"))
    n = len(a)
    np.testing.assert_allclose(a, a.T, atol=2e-11, rtol=0)
    np.testing.assert_allclose(b, b.T, atol=2e-11, rtol=0)
    margins = [float(eigvalsh(a-b)[0]), float(eigvalsh(a+b)[0])]
    assert min(margins) > 1e-5
    x, y = (data[key].reshape(n, n).T for key in ("x", "y"))
    np.testing.assert_allclose(x.T@x-y.T@y, np.eye(n), atol=2e-10, rtol=0)
    np.testing.assert_allclose(x@x.T-y@y.T, np.eye(n), atol=2e-10, rtol=0)
    np.testing.assert_allclose(x@y.T-y@x.T, 0., atol=2e-10, rtol=0)
    operator = np.block([[a, b], [-b, -a]])
    rhs = np.concatenate((d, -d))
    identity = np.eye(2*n)
    maxima, residuals = [], []
    for eta, native in zip(data["dampings_hartree"], data["polarizability"]):
        errors, norms = [], []
        for omega, expected in zip(data["frequencies_hartree"], native):
            matrix = operator-(omega+1j*eta)*identity
            response = solve(matrix, rhs, assume_a="gen")
            alpha = 2*d.T@(response[:n]+response[n:])
            errors.append(np.max(abs(alpha-expected)))
            norms.append(np.max(abs(matrix@response-rhs)))
        maxima.append(float(max(errors))); residuals.append(float(max(norms)))
        assert maxima[-1] < 2e-8 and residuals[-1] < 2e-10
    static = 4*d.T@solve(a+b, d, assume_a="pos")
    static_error = float(np.max(abs(static-data["static_polarizability"])))
    assert static_error < 2e-9
    activity_checks = []
    for eta, modes in zip(data["dampings_hartree"], data["activities_angstrom4_amu"]):
        for mode, curve in enumerate(modes, 1):
            areas = [float(simpson(curve[::stride], x=data["frequencies_hartree"][::stride]))
                     for stride in (4, 2, 1)]
            change = abs(areas[-1]-areas[-2])/abs(areas[-1])
            activity_checks.append(dict(eta_hartree=float(eta), mode=mode,
                                        integrated_activities=areas, last_relative_change=change))
            assert change < 2e-4
    report = dict(complete=True, oracle="SciPy direct complex full-BSE resolvent; shared native A/B input",
        samples_per_damping=len(data["frequencies_hartree"]), dampings_hartree=data["dampings_hartree"].tolist(),
        maximum_polarizability_errors_bohr3=maxima, maximum_linear_residuals=residuals,
        static_error_bohr3=static_error, stability_margins_hartree=margins,
        activity_grid_checks=activity_checks,
        input_sha256=hashlib.sha256((OUTPUT/"response.npz").read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), elapsed_seconds=perf_counter()-started)
    (OUTPUT/"scipy.json").write_text(json.dumps(report, indent=2)+"\n")
    print("Independent full-BSE resolvent PASS", report, flush=True)


if __name__ == "__main__":
    main()

# Recorded on jy_gpu_1 (2026-10-16):
# All 5283 complex tensors vs independent 140-dimensional resolvents: PASS.
# Maximum alpha difference: 4.98e-12 bohr^3; true residual: 2.91e-15.
# Static identity difference: 1.36e-13 bohr^3.
# Maximum final-grid activity integral change: 2.32e-8 (gate 2e-4).
