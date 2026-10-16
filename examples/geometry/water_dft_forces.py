"""G02: PBE/PBE0/B3LYP forces at the existing matched water geometry.

The common geometry and seed-23 direction come from the saved G02 input.
All three methods use Cartesian 6-31G*, a level-3 moving grid and the shared
MolecularEnergy/implicit orbital solver. No SCF or response solver is copied.
"""
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from gradscf import dft, gto
from gradscf_tools.molecular_response import MolecularEnergy


METHODS = ("pbe", "pbe0", "b3lyp")
output = Path("reproducibility/ad_showcase/G02_water_correlated_forces")
common = json.loads((output / "gradscf.json").read_text())
coordinates = np.asarray(common["coordinates_bohr"])
direction = np.asarray(common["direction"])
np.testing.assert_allclose(np.linalg.norm(direction), 1., atol=1e-12, rtol=0)
mol = gto.M(atom=list(zip(common["symbols"], coordinates)), basis="6-31g*",
            unit="Bohr", cart=True)
records = []
for method in METHODS:
    mf = dft.RKS(mol, xc=method, grids_level=3, conv_tol=1e-12,
                 conv_tol_density=1e-11, conv_tol_grad=1e-10, max_cycle=150,
                 integral_backend="native").run()
    energy = MolecularEnergy(mf)
    value_and_grad = jax.jit(jax.value_and_grad(energy))
    start = perf_counter()
    value, derivative = value_and_grad(coordinates)
    derivative = np.asarray(derivative)
    seconds = perf_counter()-start
    if not np.isfinite(value) or not np.isfinite(derivative).all():
        raise RuntimeError(f"{method}: electronic state or nuclear gradient failed.")
    np.testing.assert_allclose(derivative.sum(axis=0), 0., atol=1e-8, rtol=0)
    np.testing.assert_allclose(np.cross(coordinates, derivative).sum(axis=0), 0., atol=1e-8, rtol=0)
    print(f"{method}: E={float(value):.12f} Eh; complete AD gradient available", flush=True)
    forward = jax.jit(energy)
    checks = []
    for step in (1e-3, 3e-4, 1e-4):
        numerical = float((forward(coordinates+step*direction)
                           - forward(coordinates-step*direction))/(2*step))
        if not np.isfinite(numerical):
            raise RuntimeError(f"{method}: displaced state failed at step {step:.1e}.")
        error = abs(numerical-float(np.sum(derivative*direction)))
        checks.append(dict(step_bohr=step, derivative=numerical, error=error))
        print(f"{method}: step={step:.1e} Bohr; directional AD-FD={error:.3e} Eh/Bohr", flush=True)
    assert checks[-1]["error"] < 1e-6
    records.append(dict(basis=mol.basis, method=method, energy_hartree=float(value),
                         gradient_hartree_per_bohr=derivative.tolist(), cold_seconds=seconds,
                         directional_checks=checks, grid_level=3, grid_response=True,
                         net_force=(-derivative.sum(axis=0)).tolist(),
                         torque=(-np.cross(coordinates, derivative).sum(axis=0)).tolist()))
    data = dict(symbols=common["symbols"], coordinates_bohr=coordinates.tolist(),
                cart=True, direction=direction.tolist(), records=records, grid_level=3,
                complete=len(records) == len(METHODS), requested_methods=list(METHODS),
                b3lyp_convention="VWN-RPA (Gaussian B3LYP)",
                source_input=str(output / "gradscf.json"), dtype="float64",
                integral_backend="native", grid_ao_backend="jax",
                scf_energy_tolerance=1e-12, scf_density_tolerance=1e-11,
                scf_orbital_tolerance=1e-10, orbital_gradient_tolerance=1e-9,
                implicit_tolerance=1e-10, jax_version=jax.__version__,
                numpy_version=np.__version__, python_version=platform.python_version(),
                hostname=platform.node(), devices=[str(device) for device in jax.devices()],
                runtime_source_hashes={name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                    for name in ("src/gradscf_tools/molecular_response.py",
                                 "src/gradscf/scf/orbital_optimization.py",
                                 "src/gradscf/scf/uks.py", "src/gradscf/scf/rsh.py",
                                 "src/gradscf/integrals/grids/ao.py")})
    (output / "dft_gradscf.json").write_text(json.dumps(data, indent=2) + "\n")


# Measured on jy_gpu_1, CPU float64, 2026-10-15, Cartesian 6-31G*, level 3:
# method    E / Hartree         directional AD-FD at 1e-3 / 3e-4 / 1e-4 Bohr
# PBE      -76.321861744206     3.166e-8 / 2.145e-9 / 8.682e-9 Hartree/Bohr
# PBE0     -76.325587311744     3.199e-8 / 2.059e-9 / 5.695e-10 Hartree/Bohr
# B3LYP    -76.408739551093     3.257e-8 / 2.459e-9 / 3.344e-9 Hartree/Bohr
# Cold value+gradient calls: 122.326 / 102.530 / 151.705 s, respectively.
# Concurrent jobs were active; these timings are not a throughput benchmark.
