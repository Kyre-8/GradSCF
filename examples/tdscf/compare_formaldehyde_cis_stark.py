"""Independent PySCF total-energy finite fields for the X08 CIS Stark example."""
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import pyscf
from pyscf import gto, scf
from scipy.linalg import eigh


output = Path("reproducibility/ad_showcase/X08_formaldehyde_cis_stark")
native = json.loads((output/"gradscf.json").read_text())
if not native["complete"]:
    raise RuntimeError("Complete native X08 checks first.")
started = perf_counter()
coordinates = np.asarray(native["coordinates_bohr"])


def reference(field):
    mol = gto.M(atom=list(zip(native["symbols"], coordinates)), basis=native["basis"], unit="Bohr", cart=True, verbose=0)
    mf = scf.RHF(mol)
    hcore = mf.get_hcore()+np.einsum("x,xpq->pq", field, mol.intor("int1e_r", comp=3))
    enuc = mol.energy_nuc()-field@(mol.atom_charges()@coordinates)
    mf.get_hcore = lambda mol=None: hcore
    mf.energy_nuc = lambda *args, **kwargs: enuc
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-13, 1e-12, 180
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("Independent field RHF failed to converge.")
    matrix, _ = mf.TDA().get_ab()
    no, nv = matrix.shape[:2]
    matrix = matrix.reshape(no*nv, no*nv)
    values, vectors = eigh(matrix, subset_by_index=(0, native["nroots"]-1))
    residual = float(np.linalg.norm(matrix@vectors-vectors*values, axis=0).max())
    if residual > 1e-9:
        raise RuntimeError("Independent CIS eigenstate residual failed.")
    transitions = np.sqrt(2)*np.einsum("pi,sia,qa->spq", mf.mo_coeff[:, :no],
        vectors.T.reshape(native["nroots"], no, nv), mf.mo_coeff[:, no:])
    return dict(ground_energy=float(mf.e_tot), excitations=values, transitions=transitions,
                overlap=mf.get_ovlp(), residual=residual)


baseline = reference(np.zeros(3))
overlap = baseline["overlap"]
anchor = baseline["transitions"][native["selected_root"]]
norm0 = np.trace(anchor.T@overlap@anchor@overlap)
c, x = np.asarray(native["baseline_coefficients_ao"]), np.asarray(native["baseline_amplitudes"])
no = x.shape[0]
target = np.sqrt(2)*c[:, :no]@x@c[:, no:].T
initial_match = abs(np.trace(target.T@overlap@anchor@overlap))/np.sqrt(
    np.trace(target.T@overlap@target@overlap)*norm0)
np.testing.assert_allclose(initial_match, 1., atol=1e-8, rtol=0)
tracking, cache = [], {}


def energies(field):
    field = np.asarray(field, dtype=float)
    key = tuple(field)
    if key not in cache:
        current = baseline if not np.any(field) else reference(field)
        inner = np.einsum("pq,pr,nrs,qs->n", anchor, overlap, current["transitions"], overlap)
        norms = np.einsum("npq,pr,nrs,qs->n", current["transitions"], overlap, current["transitions"], overlap)
        matches = np.abs(inner)/np.sqrt(norm0*norms)
        order = np.argsort(matches)[::-1]
        root = int(order[0])
        gap = float(np.min(np.abs(np.delete(current["excitations"], root)-current["excitations"][root])))
        if matches[root] < .85 or matches[root]-matches[order[1]] < .2 or gap < 1e-4:
            raise RuntimeError("Independent field state identity is unresolved.")
        tracking.append(dict(field_au=field.tolist(), root=root, overlap=float(matches[root]),
                             gap_hartree=gap, residual=current["residual"]))
        cache[key] = np.asarray([current["ground_energy"], current["ground_energy"]+current["excitations"][root]])
    return cache[key]


zero = energies(np.zeros(3))
directions = np.asarray(native["field_directions"])
expected_mu = directions@np.asarray(native["dipoles_au"]).T
expected_alpha = np.einsum("di,sij,dj->ds", directions, native["polarizabilities_au"], directions)
checks = []
for step in (1e-3, 3e-4, 1e-4):
    moments, curvature = [], []
    for direction in directions:
        plus, minus = energies(step*direction), energies(-step*direction)
        moments.append(-(plus-minus)/(2*step))
        curvature.append(-(plus+minus-2*zero)/step**2)
    moments, curvature = np.asarray(moments), np.asarray(curvature)
    checks.append(dict(step_au=step, directional_dipoles=moments.tolist(), directional_polarizabilities=curvature.tolist(),
                        dipole_error_au=float(np.max(np.abs(moments-expected_mu))),
                        polarizability_error_au=float(np.max(np.abs(curvature-expected_alpha)))))
    print("Independent field step", step, "mu / alpha errors", checks[-1]["dipole_error_au"],
          checks[-1]["polarizability_error_au"], flush=True)
tensor = np.zeros((2, 3, 3))
for axis in range(3):
    tensor[:, axis, axis] = curvature[axis]
for index, (i, j) in enumerate(((0, 1), (0, 2), (1, 2)), start=3):
    tensor[:, i, j] = tensor[:, j, i] = curvature[index]-.5*(curvature[i]+curvature[j])
errors = dict(
    baseline_energies_hartree=float(np.max(np.abs(zero-native["baseline_energies_hartree"]))),
    dipole_finite_field_au=checks[-1]["dipole_error_au"],
    polarizability_directions_au=checks[-1]["polarizability_error_au"],
    polarizability_tensor_au=float(np.max(np.abs(tensor-native["polarizabilities_au"]))),
)
curves = []
for point in native["curves"]:
    field = np.eye(3)["xyz".index(point["axis"])]*point["field_au"]
    value = energies(field)
    error = float(np.max(np.abs(value-point["energies_hartree"])))
    errors["curve_energies_hartree"] = max(errors.get("curve_energies_hartree", 0.), error)
    curves.append(dict(axis=point["axis"], field_au=point["field_au"], energies_hartree=value.tolist(),
                        excitation_shift_hartree=float((value[1]-value[0])-(zero[1]-zero[0])), max_error_hartree=error))
tolerances = dict(baseline_energies_hartree=2e-8, dipole_finite_field_au=2e-6,
                  polarizability_directions_au=2e-4, polarizability_tensor_au=2e-4, curve_energies_hartree=2e-8)
passed = all(errors[key] <= gate for key, gate in tolerances.items())
report = dict(passed=bool(passed), reference="Independent PySCF RHF/get_ab + SciPy dense tracked CIS total energies; first and second finite fields",
    errors=errors, tolerances=tolerances, baseline_energies_hartree=zero.tolist(),
    dipoles_au=moments[:3].T.tolist(), polarizability_tensors_au=tensor.tolist(),
    finite_field_checks=checks, curves=curves, tracking=tracking, initial_transition_overlap=float(initial_match),
    native_sha256=hashlib.sha256((output/"gradscf.json").read_bytes()).hexdigest(),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    elapsed_seconds=perf_counter()-started, hostname=platform.node(), pyscf=pyscf.__version__)
(output/"pyscf.json").write_text(json.dumps(report, indent=2)+"\n")
if not passed:
    raise RuntimeError(f"Independent X08 checks failed: {errors}")
print("Independent X08 field gates passed", report["elapsed_seconds"], "s", flush=True)

# Measured jy_gpu_1 / PySCF 2.13.0, OMP/BLAS=2: all gates passed in 56.00 s.
# Independent alpha directions / reconstructed tensor errors <=6.82e-5 / 7.25e-5 au.
# Finite-energy subtraction limits this oracle's off-diagonal precision; raw values are retained.
