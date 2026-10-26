"""Independent PySCF DF-HF energies and log-exponent finite differences for A05."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pyscf
from pyscf import gto, scf

OUTPUT = Path("reproducibility/ad_showcase/A05_basis_exponents")


def main():
    start = perf_counter()
    native_path = OUTPUT/"gradscf.json"
    data = json.loads(native_path.read_text())
    assert data["complete"]

    def energy(theta, exact=False):
        basis = deepcopy(data["basis"])
        for scale, shell in zip(np.exp(theta), basis["H"]):
            for primitive in shell[1:]:
                primitive[0] *= scale
        mol = gto.M(atom=list(zip(data["symbols"], data["coordinates_bohr"])),
            basis=basis, unit="Bohr", cart=data["cart"], verbose=0)
        mf = scf.RHF(mol)
        if not exact:
            mf = mf.density_fit(auxbasis=data["auxiliary_basis"])
        mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-13, 1e-10, 200
        mf.kernel()
        assert mf.converged
        np.testing.assert_allclose(np.diag(mf.get_ovlp()), 1., atol=2e-12, rtol=0)
        return float(mf.e_tot)

    checks = []
    for native in data["derivative_checks"]:
        point = np.asarray(native["theta"])
        value = energy(point)
        target = data["initial_energy"] if native["point"] == "initial" else data["final_energy"]
        np.testing.assert_allclose(value, target, atol=2e-9, rtol=0)
        differences = []
        for step in (5e-3, 1e-3, 2e-4):
            fd = [(energy(point+step*axis)-energy(point-step*axis))/(2*step) for axis in np.eye(2)]
            differences.append(dict(step=step, gradient_theta=fd,
                max_error=float(np.max(np.abs(np.asarray(fd)-native["gradient_theta"])))))
        np.testing.assert_allclose(differences[-1]["gradient_theta"], native["gradient_theta"], atol=2e-7, rtol=2e-5)
        exact = energy(point, exact=True)
        checks.append(dict(point=native["point"], energy=value, native_energy_error=abs(value-target),
            exact_four_center_energy=exact, df_minus_exact=value-exact, finite_differences=differences))
        print("A05 independent", native["point"], checks[-1], flush=True)
    report = dict(passed=True, checks=checks, pyscf=pyscf.__version__, elapsed_seconds=perf_counter()-start,
        native_sha256=hashlib.sha256(native_path.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope="Same fixed auxiliary basis and normalized main basis; exact four-center energies reported separately, not used as DF gradient oracle.")
    (OUTPUT/"pyscf.json").write_text(json.dumps(report, indent=2)+"\n")
    print("A05 independent normalized DF energy and exponent FD gates PASS", flush=True)


if __name__ == "__main__":
    main()

# jy_gpu_1/PySCF 2.13.0: independent DF energy/FD gates PASS in 1.49 s.
# Exact 4c energies differ from DF by about 9.4e-6 Eh; they are separate models.
