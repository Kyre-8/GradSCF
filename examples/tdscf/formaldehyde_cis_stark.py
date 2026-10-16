"""X08: clamped-nuclei RHF/CIS total-energy polarizability and Stark shifts.

At the validated X10 RHF minimum, mu=-dE/dF and alpha=-d2E/dF2 are evaluated
for E0 and E1=E0+omega. Nuclear field energy is retained. Electronic response
is included; nuclear relaxation, vibrational response and linewidths are not.
"""
import hashlib
import json
from pathlib import Path
import platform
import resource
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from gradscf import gto, scf
from gradscf.solvers import EigenResponseConfig, LinearSolverConfig
from gradscf_tools.excited import CISGeometry


input_path = Path("reproducibility/ad_showcase/X10_formaldehyde_cis_emission/emission.json")
paired = json.loads(input_path.read_text())
checked = json.loads((input_path.parent/"emission_pyscf.json").read_text())
input_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
if not paired["complete"] or not checked["passed"] or checked["native_sha256"] != input_hash:
    raise RuntimeError("Use the independently validated X10 ground minimum.")
output = Path("reproducibility/ad_showcase/X08_formaldehyde_cis_stark")
output.mkdir(parents=True, exist_ok=True)
(output/"gradscf.json").write_text('{"complete": false}\n')
started = perf_counter()
coordinates = np.asarray(paired["ground_coordinates_bohr"])
mol = gto.M(atom=list(zip(paired["symbols"], coordinates)), basis=paired["basis"], unit="Bohr", cart=True)
mf = scf.RHF(mol, integral_backend="native", conv_tol=1e-12,
             conv_tol_density=1e-11, conv_tol_grad=1e-11, max_cycle=180).run()
if not mf.converged:
    raise RuntimeError("The RHF seed did not converge.")
policy = EigenResponseConfig(linear_config=LinearSolverConfig(method="direct", max_dense=512, rtol=1e-11))
model = CISGeometry(mf, nroots=6, gradient_tolerance=1e-11, response=policy)
excited_dipole, excited_alpha, state = model.electric_response(coordinates, 0)
ground_dipole = np.asarray(model.energy.dipole(jnp.asarray(coordinates)))
ground_alpha = np.asarray(jax.jit(jax.jacfwd(lambda field: model.energy.dipole(jnp.asarray(coordinates), field)))(jnp.zeros(3)))
dipoles = np.asarray([ground_dipole, excited_dipole])
alpha = np.asarray([ground_alpha, excited_alpha])
asymmetry = float(np.max(np.abs(alpha-alpha.transpose(0, 2, 1))))
np.testing.assert_allclose(alpha, alpha.transpose(0, 2, 1), atol=1e-7, rtol=0)
# The planar C2v geometry makes both tensors diagonal and dipoles z-directed.
np.testing.assert_allclose(dipoles[:, :2], 0., atol=1e-8, rtol=0)
np.testing.assert_allclose(alpha-np.asarray([np.diag(np.diag(value)) for value in alpha]), 0., atol=1e-7, rtol=0)
baseline = np.asarray([state.ground_energy, state.ground_energy+state.excitation_energies[0]], dtype=float)
np.testing.assert_allclose(baseline, [paired["electronic_energies_hartree"]["ground_at_ground"],
                                    paired["electronic_energies_hartree"]["excited_at_ground"]], atol=1e-8, rtol=0)
anchor = model.anchor(coordinates, state, 0)
tracking, cache = [], {}


def energies(field):
    field = np.asarray(field, dtype=float)
    key = tuple(field)
    if key not in cache:
        current = model.state(coordinates, field=field)
        match = model.match(coordinates, current, anchor)
        value = np.asarray([current.ground_energy,
                            current.ground_energy+current.excitation_energies[match["root"]]], dtype=float)
        tracking.append(dict(field_au=field.tolist(), **match))
        cache[key] = value
    return cache[key]


zero = energies(np.zeros(3))
directions = list(np.eye(3))
directions.extend((np.eye(3)[i]+np.eye(3)[j])/np.sqrt(2) for i, j in ((0, 1), (0, 2), (1, 2)))
direction = np.random.default_rng(108).normal(size=3)
directions.append(direction/np.linalg.norm(direction))
directions = np.asarray(directions)
checks = []
for step in (1e-3, 3e-4, 1e-4):
    moment_fd, curvature_fd = [], []
    for direction in directions:
        plus, minus = energies(step*direction), energies(-step*direction)
        moment_fd.append(-(plus-minus)/(2*step))
        curvature_fd.append(-(plus+minus-2*zero)/step**2)
    moment_fd, curvature_fd = np.asarray(moment_fd), np.asarray(curvature_fd)
    expected_moments = directions@dipoles.T
    expected_curvatures = np.einsum("di,sij,dj->ds", directions, alpha, directions)
    error_mu = float(np.max(np.abs(moment_fd-expected_moments)))
    error_alpha = float(np.max(np.abs(curvature_fd-expected_curvatures)))
    checks.append(dict(step_au=step, directional_dipoles=moment_fd.tolist(),
                        directional_polarizabilities=curvature_fd.tolist(),
                        dipole_error_au=error_mu, polarizability_error_au=error_alpha))
    print("Field step", step, "mu / alpha errors", error_mu, error_alpha, flush=True)
np.testing.assert_allclose(checks[-1]["directional_dipoles"], expected_moments, atol=2e-6, rtol=0)
np.testing.assert_allclose(checks[-1]["directional_polarizabilities"], expected_curvatures, atol=2e-4, rtol=0)
curves = []
for axis in (0, 2):
    for magnitude in np.linspace(-.003, .003, 13):
        field = np.eye(3)[axis]*magnitude
        value = energies(field)
        predicted = baseline-dipoles@field-.5*np.einsum("i,sij,j->s", field, alpha, field)
        curves.append(dict(axis="xyz"[axis], field_au=float(magnitude), energies_hartree=value.tolist(),
                            quadratic_energies_hartree=predicted.tolist(),
                            excitation_shift_hartree=float((value[1]-value[0])-(baseline[1]-baseline[0])),
                            quadratic_excitation_shift_hartree=float((predicted[1]-predicted[0])-(baseline[1]-baseline[0]))))
repository = Path(__file__).resolve().parents[2]
digest = hashlib.sha256()
for path in sorted((repository/"src").rglob("*.py")):
    digest.update(path.relative_to(repository).as_posix().encode())
    digest.update(path.read_bytes())
report = dict(
    complete=True, method="real singlet RHF/CIS", basis=paired["basis"], cart=True,
    symbols=paired["symbols"], coordinates_bohr=coordinates.tolist(), selected_root=0, nroots=6,
    baseline_energies_hartree=baseline.tolist(), excitation_energies_hartree=np.asarray(state.excitation_energies).tolist(),
    dipoles_au=dipoles.tolist(), polarizabilities_au=alpha.tolist(),
    dipole_difference_au=(dipoles[1]-dipoles[0]).tolist(), polarizability_difference_au=(alpha[1]-alpha[0]).tolist(),
    raw_polarizability_asymmetry=asymmetry, field_directions=directions.tolist(),
    finite_field_checks=checks, curves=curves, tracking=tracking,
    baseline_coefficients_ao=np.asarray(state.coefficients).tolist(),
    baseline_amplitudes=np.asarray(state.amplitudes[0]).tolist(),
    input_minimum_sha256=input_hash, source_python_sha256=digest.hexdigest(),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    elapsed_seconds=perf_counter()-started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    hostname=platform.node(), python=platform.python_version(), jax=jax.__version__, dtype="float64",
    scope="Clamped nuclei; total excited E1 derivatives. Nuclear field term retained; no positivity requirement for excited alpha.",
)
(output/"gradscf.json").write_text(json.dumps(report, indent=2)+"\n")
print("X08 native total excited-state field gates passed", report["elapsed_seconds"], "s", flush=True)

# Measured jy_gpu_1 CPU float64, OMP/BLAS=2: native gates passed in 412.78 s.
# Total S1 alpha diagonal = [17.962602986, 12.403895682, 20.850449970] au.
# Native final-step mu / alpha FD errors <=2.50e-7 / 2.62e-5 au; no tensor symmetrization.
