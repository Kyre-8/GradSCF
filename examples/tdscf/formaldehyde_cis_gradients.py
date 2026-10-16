"""X02: separate CIS excitation and total excited-state nuclear gradients.

The chosen C2v formaldehyde geometry and Cartesian 6-31+G* basis match X01.
Native full integrals, implicit relaxed RHF orbitals, shared AO-to-MO and CIS
solvers supply d(omega)/dR and d(E_HF+omega)/dR. Only first derivatives are
claimed. Run on jy_gpu_1 from the repository root; no CLI or PySCF import.
"""
import csv
import hashlib
import json
from pathlib import Path
import platform
import resource
import subprocess
from time import perf_counter

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from gradscf import gto, scf
from gradscf_tools.excited import CISGeometry


basis = "6-31+g*"
nroots = 6
root_index = 0
ch, half_angle = 1.10, np.deg2rad(58.)
atom = [("C", (0., 0., 0.)), ("O", (0., 0., 1.21)),
        ("H", (0., ch*np.sin(half_angle), -ch*np.cos(half_angle))),
        ("H", (0., -ch*np.sin(half_angle), -ch*np.cos(half_angle)))]
output = Path("reproducibility/ad_showcase/X02_formaldehyde_cis_gradients")
output.mkdir(parents=True, exist_ok=True)
started = perf_counter()
mol = gto.M(atom=atom, basis=basis, unit="Angstrom", cart=True)
mf = scf.RHF(mol, integral_backend="native", conv_tol=1e-12,
             conv_tol_density=1e-11, conv_tol_grad=1e-10, max_cycle=180).run()
if not mf.converged:
    raise RuntimeError("The initial RHF reference did not converge.")
seed_seconds = perf_counter()-started
model = CISGeometry(mf, nroots=nroots, gradient_tolerance=1e-10)
coordinates = np.asarray(model.energy.coordinates)
initial = model.state(coordinates)
anchor = model.anchor(coordinates, initial, root_index)
initial_match = model.match(coordinates, initial, anchor)

start = perf_counter()
values, gradients, state = model.derivatives(coordinates, root_index)
cold_seconds = perf_counter()-start
start = perf_counter()
warm_values, warm_gradients, _ = model.derivatives(coordinates, root_index)
warm_seconds = perf_counter()-start
np.testing.assert_allclose(warm_values, values, atol=1e-12, rtol=0)
np.testing.assert_allclose(warm_gradients, gradients, atol=1e-11, rtol=0)
ground_gradient = gradients[1]-gradients[0]
print(f"omega = {values[0]:.12f} Eh; E_HF+omega = {values[1]:.12f} Eh", flush=True)
print("domega/dR / (Eh/Bohr):\n", gradients[0], flush=True)
print("d(E_HF+omega)/dR / (Eh/Bohr):\n", gradients[1], flush=True)
print(f"First/warm paired gradient: {cold_seconds:.3f}/{warm_seconds:.3f} s", flush=True)

# All +/- displacements are independently relaxed and matched to one anchor.
seed = 1022
directions = np.random.default_rng(seed).normal(size=(3, *coordinates.shape))
directions /= np.linalg.norm(directions, axis=(1, 2))[:, None, None]
directional_ad = np.einsum("sij,dij->sd", gradients, directions)
steps = (1e-3, 3e-4, 1e-4)
finite_differences, tracking = [], []
for step in steps:
    columns = []
    for direction_index, direction in enumerate(directions):
        displaced_values = []
        for sign in (1, -1):
            moved = coordinates+sign*step*direction
            candidate = model.state(moved)
            match = model.match(moved, candidate, anchor)
            omega = float(candidate.excitation_energies[match["root"]])
            displaced_values.append(np.array([omega, float(candidate.ground_energy)+omega]))
            tracking.append(dict(step_bohr=step, direction=direction_index,
                                 sign=sign, **match))
        columns.append((displaced_values[0]-displaced_values[1])/(2*step))
    derivative = np.array(columns).T
    finite_differences.append(derivative)
    error = float(np.max(np.abs(derivative-directional_ad)))
    print(f"step={step:.1e} Bohr; max paired AD-FD error={error:.3e} Eh/Bohr", flush=True)
for derivative in finite_differences[-2:]:
    np.testing.assert_allclose(derivative, directional_ad, atol=1e-6, rtol=0)
net_gradient = gradients.sum(axis=1)
torque = np.cross(coordinates[None], gradients).sum(axis=1)
np.testing.assert_allclose(net_gradient, 0., atol=1e-7, rtol=0)
np.testing.assert_allclose(torque, 0., atol=1e-7, rtol=0)

repository = Path(__file__).resolve().parents[2]
revision_file = repository / "SOURCE_REVISION"
revision = (revision_file.read_text().strip() if revision_file.exists() else
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip())
source = hashlib.sha256()
for path in sorted((repository / "src").rglob("*.py")):
    source.update(path.relative_to(repository).as_posix().encode())
    source.update(path.read_bytes())
report = dict(
    method="RHF/CIS singlet", basis=basis, cart=True, charge=0, spin=0,
    symbols=[item[0] for item in atom], atom_angstrom=atom,
    coordinates_bohr=coordinates.tolist(), nroots=nroots, selected_root=root_index,
    ground_energy_hartree=float(state.ground_energy), excitation_hartree=float(values[0]),
    total_excited_energy_hartree=float(values[1]),
    excitation_gradient_hartree_per_bohr=gradients[0].tolist(),
    total_excited_gradient_hartree_per_bohr=gradients[1].tolist(),
    ground_gradient_hartree_per_bohr=ground_gradient.tolist(),
    all_excitation_energies_hartree=np.asarray(state.excitation_energies).tolist(),
    ci_residual_norms=np.asarray(state.ci_residual_norms).tolist(),
    seed_scf_seconds=seed_seconds, first_paired_gradient_seconds=cold_seconds,
    warm_paired_gradient_seconds=warm_seconds, elapsed_seconds=perf_counter()-started,
    peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    finite_difference_steps_bohr=list(steps), directions=directions.tolist(), direction_seed=seed,
    directional_ad=directional_ad.tolist(), finite_differences=np.asarray(finite_differences).tolist(),
    derivative_order=["excitation", "total_excited"], net_gradient=net_gradient.tolist(), torque=torque.tolist(),
    tracking_metric="Normalized absolute transition-density overlap in the cross-geometry AO metric; not a many-electron wavefunction overlap.",
    minimum_overlap=.85, minimum_matching_margin=.20, minimum_gap_hartree=1e-4,
    initial_match=initial_match, displaced_tracking=tracking,
    scf_energy_tolerance=1e-12, orbital_gradient_tolerance=1e-10,
    implicit_tolerance=1e-10, ci_tolerance=1e-10, independent_gradient_tolerance=1e-6,
    dtype="float64", integral_backend="native", scf_response="implicit orbital stationarity",
    ci_solver="shared dense Hermitian; first-order eigenvalue response",
    source_revision=revision, source_python_sha256=source.hexdigest(),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    hostname=platform.node(), machine=platform.machine(), python=platform.python_version(),
    jax=jax.__version__, numpy=np.__version__, devices=[str(device) for device in jax.devices()],
    complete=True,
)
(output / "gradscf.json").write_text(json.dumps(report, indent=2)+"\n")
with (output / "gradients.csv").open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["atom", "axis", "excitation_gradient_hartree_per_bohr",
                     "total_excited_gradient_hartree_per_bohr", "ground_gradient_hartree_per_bohr"])
    for atom_index, symbol in enumerate(report["symbols"]):
        for axis, label in enumerate("xyz"):
            writer.writerow([f"{symbol}{atom_index+1}", label, gradients[0, atom_index, axis],
                             gradients[1, atom_index, axis], ground_gradient[atom_index, axis]])
print(f"Accepted tracked CIS gradients; elapsed {report['elapsed_seconds']:.2f} s", flush=True)

# Measured on jy_gpu_1, CPU float64, 2026-10-15:
# omega = 0.168280654334 Eh; E_HF+omega = -113.701633304838 Eh.
# First/warm paired gradient: 28.865/12.777 s.
# Step 1e-3 / 3e-4 / 1e-4 Bohr: max AD-FD error 5.096e-8 / 4.466e-9 / 1.055e-9 Eh/Bohr.
# Accepted tracked CIS gradients; elapsed 132.11 s.
