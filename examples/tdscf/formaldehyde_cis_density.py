"""X04: relaxed CIS dipole and response density at the fixed X01 geometry.

D_relaxed = d(E_RHF+omega_CIS)/dV uses a symmetric AO one-body perturbation.
The unrelaxed CIS wavefunction density and its attachment/detachment operators
are reported separately. Native SCF, integrals, AO2MO and CIS solvers are reused.
"""
import csv
import hashlib
import json
from pathlib import Path
import platform
import resource
from time import perf_counter

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from gradscf import gto, scf
from gradscf.solvers import EigenResponseConfig, LinearSolverConfig
from gradscf_tools.excited import CISGeometry, singlet_cis_densities


input_folder = Path("reproducibility/ad_showcase/X02_formaldehyde_cis_gradients")
input_data = json.loads((input_folder/"gradscf.json").read_text())
input_check = json.loads((input_folder/"pyscf.json").read_text())
input_hash = hashlib.sha256((input_folder/"gradscf.json").read_bytes()).hexdigest()
if not input_data["complete"] or not input_check["passed"] or input_check["native_sha256"] != input_hash:
    raise RuntimeError("Use the independently validated X02 fixed-geometry record.")
output = Path("reproducibility/ad_showcase/X04_formaldehyde_cis_density")
output.mkdir(parents=True, exist_ok=True)
(output/"gradscf.json").write_text(json.dumps(dict(complete=False, method="RHF/CIS singlet"))+"\n")
started = perf_counter()
coordinates = np.asarray(input_data["coordinates_bohr"])
symbols, basis = input_data["symbols"], input_data["basis"]
root, nroots = 0, 6
mol = gto.M(atom=list(zip(symbols, coordinates)), basis=basis, unit="Bohr", cart=True)
mf = scf.RHF(mol, integral_backend="native", conv_tol=1e-12,
             conv_tol_density=1e-11, conv_tol_grad=1e-10, max_cycle=180).run()
if not mf.converged:
    raise RuntimeError("The RHF seed did not converge.")
policy = EigenResponseConfig(linear_config=LinearSolverConfig(method="direct", max_dense=512, rtol=1e-10))
model = CISGeometry(mf, nroots=nroots, gradient_tolerance=1e-10, response=policy)
field = np.zeros(3)
density, dipole, state = model.density_and_dipole(coordinates, root, field=field)
amplitude = np.asarray(state.amplitudes[root])
np.testing.assert_allclose(np.sum(amplitude**2), 1., atol=1e-12, rtol=0)
one_body = singlet_cis_densities(state.coefficients, amplitude, nocc=model.nocc)
ground, unrelaxed, attachment, detachment, transition = map(np.asarray, one_body)
overlap = np.asarray(model.energy.plan.evaluate("overlap", model.energy.parameters))
position = np.asarray(model.energy.plan.evaluate("dipole", model.energy.parameters, origin=np.zeros(3)))
nuclear_dipole = np.asarray(model.energy.spec.charges)@coordinates
ground_dipole = nuclear_dipole-np.einsum("pq,xqp->x", ground, position)
unrelaxed_dipole = nuclear_dipole-np.einsum("pq,xqp->x", unrelaxed, position)
np.testing.assert_allclose(dipole, nuclear_dipole-np.einsum("pq,xqp->x", density, position), atol=2e-9, rtol=0)
np.testing.assert_allclose(density, density.T, atol=1e-10, rtol=0)
charges = {name: float(np.einsum("pq,qp", value, overlap)) for name, value in (
    ("ground", ground), ("relaxed", density), ("unrelaxed", unrelaxed),
    ("attachment", attachment), ("detachment", detachment), ("relaxed_difference", density-ground))}
for name in ("ground", "relaxed", "unrelaxed"):
    np.testing.assert_allclose(charges[name], mol.nelectron, atol=1e-7, rtol=0)
np.testing.assert_allclose([charges["attachment"], charges["detachment"]], 1., atol=1e-10, rtol=0)
np.testing.assert_allclose(charges["relaxed_difference"], 0., atol=1e-7, rtol=0)
np.testing.assert_allclose([ground_dipole[:2], dipole[:2], unrelaxed_dipole[:2]], 0., atol=1e-8, rtol=0)
values, vectors = np.linalg.eigh(overlap)
if values[0] <= 0:
    raise RuntimeError("The AO overlap is not positive definite.")
sqrt_overlap = (vectors*np.sqrt(values))@vectors.T
occupations = {name: np.linalg.eigvalsh(sqrt_overlap@value@sqrt_overlap).tolist()
               for name, value in (("relaxed", density), ("unrelaxed", unrelaxed))}
if min(occupations["unrelaxed"]) < -1e-9 or max(occupations["unrelaxed"]) > 2+1e-9:
    raise RuntimeError("The unrelaxed singlet CIS 1-RDM has invalid occupations.")
# The energy-derivative density is not subjected to a positivity gate.
relaxation_mo = np.asarray(state.coefficients).T@overlap@(density-unrelaxed)@overlap@np.asarray(state.coefficients)
same_block_error = max(np.max(np.abs(relaxation_mo[:model.nocc, :model.nocc])),
                       np.max(np.abs(relaxation_mo[model.nocc:, model.nocc:])))
np.testing.assert_allclose(same_block_error, 0., atol=2e-7, rtol=0)
anchor = model.anchor(coordinates, state, root)
tracking = []


def energies(current_field, potential=None):
    current = model.state(coordinates, field=current_field, potential=potential)
    match = model.match(coordinates, current, anchor)
    tracking.append(match)
    omega = float(current.excitation_energies[match["root"]])
    return np.array([float(current.ground_energy), float(current.ground_energy)+omega])


steps = (1e-4, 3e-5, 1e-5)
field_checks = []
expected_dipoles = np.array([ground_dipole, dipole])
for step in steps:
    finite = np.column_stack([-(energies(step*axis)-energies(-step*axis))/(2*step)
                              for axis in np.eye(3)])
    error = float(np.max(np.abs(finite-expected_dipoles)))
    field_checks.append(dict(step_au=step, dipoles_au=finite.tolist(), max_error_au=error))
    print(f"Field step {step:.1e} au: max dipole error {error:.3e} e Bohr", flush=True)
np.testing.assert_allclose(field_checks[-1]["dipoles_au"], expected_dipoles, atol=2e-6, rtol=0)

rng = np.random.default_rng(104)
directions = []
for _ in range(6):
    direction = rng.normal(size=overlap.shape)
    direction = .5*(direction+direction.T)
    direction /= np.linalg.norm(direction)
    # Unit Frobenius norm in the orthonormal AO frame controls perturbation size.
    directions.append(sqrt_overlap@direction@sqrt_overlap)
potential_checks = []
for number, direction in enumerate(directions):
    expected = np.array([np.sum(ground*direction), np.sum(density*direction)])
    for step in steps:
        finite = (energies(field, step*direction)-energies(field, -step*direction))/(2*step)
        error = float(np.max(np.abs(finite-expected)))
        potential_checks.append(dict(direction=number, step_hartree=step,
                                      derivative=finite.tolist(), expected=expected.tolist(), max_error=error))
        if step == steps[-1]:
            np.testing.assert_allclose(finite, expected, atol=2e-6, rtol=0)
    print(f"AO direction {number}: final density contraction error {error:.3e}", flush=True)
step = 1e-4
shift = (energies(field, step*overlap)-energies(field, -step*overlap))/(2*step)
np.testing.assert_allclose(shift, mol.nelectron, atol=1e-7, rtol=0)

repository = Path(__file__).resolve().parents[2]
source = hashlib.sha256()
for path in sorted((repository/"src").rglob("*.py")):
    source.update(path.relative_to(repository).as_posix().encode())
    source.update(path.read_bytes())
report = dict(
    complete=True, method="RHF/CIS singlet", basis=basis, cart=True, symbols=symbols,
    coordinates_bohr=coordinates.tolist(), field_au=field.tolist(), selected_root=root, nroots=nroots,
    nocc=model.nocc, nelectron=mol.nelectron, ground_energy_hartree=float(state.ground_energy),
    excitation_energies_hartree=np.asarray(state.excitation_energies).tolist(),
    total_excited_energy_hartree=float(state.ground_energy+state.excitation_energies[root]),
    coefficients_ao=np.asarray(state.coefficients).tolist(), amplitudes=np.asarray(state.amplitudes).tolist(),
    cis_residual_norms=np.asarray(state.ci_residual_norms).tolist(), overlap=overlap.tolist(),
    position_integrals_bohr=position.tolist(), ground_density_ao=ground.tolist(),
    relaxed_response_density_ao=density.tolist(), unrelaxed_cis_density_ao=unrelaxed.tolist(),
    attachment_density_ao=attachment.tolist(), detachment_density_ao=detachment.tolist(),
    transition_density_ao=transition.tolist(), nuclear_dipole_au=nuclear_dipole.tolist(),
    ground_dipole_au=ground_dipole.tolist(), excited_relaxed_dipole_au=dipole.tolist(),
    excited_unrelaxed_dipole_au=unrelaxed_dipole.tolist(), dipole_difference_au=(dipole-ground_dipole).tolist(),
    metric_electron_counts=charges, natural_occupations=occupations,
    same_occupation_block_relaxation_error=float(same_block_error),
    relaxation_density_ov_norm=float(np.linalg.norm(relaxation_mo[:model.nocc, model.nocc:])),
    field_finite_differences=field_checks, potential_directions_ao=np.asarray(directions).tolist(),
    potential_direction_seed=104, potential_finite_differences=potential_checks,
    constant_overlap_shift_slopes=shift.tolist(), tracking=tracking,
    density_definition="Relaxed total-energy derivative with respect to sym(V); no positivity assertion",
    unrelaxed_definition="Normalized real singlet CIS wavefunction 1-RDM; one-electron attachment/detachment",
    transition_definition="AO T[p,q]=<excited|a_q^dagger a_p|ground>; occupied-virtual block sqrt(2)*X",
    source_revision=input_data["source_revision"], source_python_sha256=source.hexdigest(),
    input_geometry_sha256=input_hash, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    elapsed_seconds=perf_counter()-started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    hostname=platform.node(), python=platform.python_version(), jax=jax.__version__,
    dtype="float64", devices=[str(device) for device in jax.devices()],
)
(output/"gradscf.json").write_text(json.dumps(report, indent=2)+"\n")
with (output/"dipoles.csv").open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["quantity", "mu_x_au", "mu_y_au", "mu_z_au"])
    for label, vector in (("ground", ground_dipole), ("excited_relaxed", dipole),
                          ("excited_unrelaxed", unrelaxed_dipole), ("relaxed_difference", dipole-ground_dipole)):
        writer.writerow([label, *vector])
print("Metric electron counts:", charges, flush=True)
print("Ground / relaxed excited / unrelaxed excited dipoles / au:",
      ground_dipole, dipole, unrelaxed_dipole, flush=True)
print(f"Native X04 response-density and dipole gates passed in {report['elapsed_seconds']:.2f} s", flush=True)

# Measured on jy_gpu_1 (CPU float64, two threads, JAX 0.10.0): 239.45 s.
# mu_z ground / relaxed S1 / unrelaxed S1 = -1.1818182201 / -0.5841030989 / -0.2967762330 au.
# Final-step energy-FD dipole error 4.27e-9 au; exact metric counts 16 / 16 / 16.
# Relaxed response occupations span [-0.00338231, 2.01019254]; positivity is not claimed.
# Executed script/source hashes and all finite differences are stored in the X04 JSON.
