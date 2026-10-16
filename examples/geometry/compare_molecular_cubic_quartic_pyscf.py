"""Independent oracle: differences of fully relaxed analytic PySCF RHF Hessians.

This reference uses finite differences only for orders 3/4. The native example
obtains those orders by AD. Both use the same Cartesian basis and straight path.
"""
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pyscf
from pyscf import gto, scf


def run(case="h2"):
    output = Path("reproducibility/ad_showcase/G08_molecular_cubic_quartic") / case
    data = json.loads((output / "gradscf.json").read_text())
    coordinates, direction = np.array(data["coordinates_bohr"]), np.array(data["direction"])
    direction_flat = direction.ravel()
    points = []

    def curvature(q):
        mol = gto.M(atom=list(zip(data["symbols"], coordinates+q*direction)),
                    basis=data["basis"], cart=True, unit="Bohr", verbose=0)
        mf = scf.RHF(mol)
        mf.conv_tol, mf.conv_tol_grad, mf.conv_tol_cpscf = 1e-13, 1e-11, 1e-12
        mf.max_cycle = 200
        mf.kernel()
        orbital_gradient = np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ))
        if not mf.converged or orbital_gradient > 1e-10:
            raise RuntimeError(f"Reference SCF failed at q={q}: orbital gradient={orbital_gradient}")
        gradient = mf.nuc_grad_method().kernel()
        hessian = mf.Hessian().kernel().transpose(0, 2, 1, 3).reshape(coordinates.size, -1)
        value = float(direction_flat @ hessian @ direction_flat)
        points.append(dict(q_bohr=q, energy=float(mf.e_tot),
                           first_derivative=float(np.sum(gradient*direction)), curvature=value,
                           orbital_gradient_norm=float(orbital_gradient)))
        return value

    start = perf_counter()
    center = curvature(0.)
    checks = []
    for step in (3e-3, 1e-3, 3e-4):
        plus, minus = curvature(step), curvature(-step)
        cubic, quartic = (plus-minus)/(2*step), (plus-2*center+minus)/step**2
        check = dict(step_bohr=step, cubic=cubic, quartic=quartic,
                     cubic_error=abs(cubic-data["derivatives"][3]),
                     quartic_error=abs(quartic-data["derivatives"][4]))
        checks.append(check)
        print(f"{case} PySCF curvature FD h={step:g}: cubic error={check['cubic_error']:.3e}, "
              f"quartic error={check['quartic_error']:.3e}", flush=True)
    assert checks[-1]["cubic_error"] < 3e-5
    assert checks[-1]["quartic_error"] < 3e-4
    np.testing.assert_allclose([points[0]["energy"], points[0]["first_derivative"], center],
                               data["derivatives"][:3], atol=2e-7, rtol=0)
    reference = dict(case=case, pyscf_version=pyscf.__version__, points=points,
                     checks=checks, seconds=perf_counter()-start,
                     method="central differences of independent PySCF analytic relaxed Hessians",
                     scf_energy_tolerance=1e-13, scf_gradient_tolerance=1e-11,
                     cpscf_tolerance=1e-12, actual_orbital_gradient_gate=1e-10)
    (output / "pyscf.json").write_text(json.dumps(reference, indent=2)+"\n")
    return reference


if __name__ == "__main__":
    run("h2")

# jy_gpu_1, PySCF2.13.0 CPU float64, 2026-10-16. Errors at h=3e-4 bohr:
# case             |d3_AD-d3_FD|    |d4_AD-d4_FD|
# h2                 2.478e-7         6.155e-8
# water_sto3g        2.483e-7         8.113e-6
# water_631gstar     1.585e-8         4.244e-7
# Full three-step tables are saved; smaller h can amplify Hessian/SCF roundoff.
