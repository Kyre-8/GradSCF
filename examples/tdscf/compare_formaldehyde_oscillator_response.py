"""Independent PySCF tracked transition-moment/strength differences for X07."""
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import pyscf
from pyscf import gto, scf
from scipy.linalg import eigh


output = Path("reproducibility/ad_showcase/X07_formaldehyde_oscillator_response")
native = json.loads((output/"gradscf.json").read_text())
if not native["complete"]:
    raise RuntimeError("Complete native X07 checks first.")
started = perf_counter()
coordinates, modes = np.asarray(native["coordinates_bohr"]), np.asarray(native["modes_bohr_per_Q"])


def reference(r):
    mol = gto.M(atom=list(zip(native["symbols"], r)), basis=native["basis"], unit="Bohr", cart=True, verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-12, 1e-12, 180
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("Independent RHF did not converge.")
    td = mf.TDA()
    a, _ = td.get_ab()
    no, nv = a.shape[:2]
    matrix = a.reshape(no*nv, no*nv)
    energies, vectors = eigh(matrix, subset_by_index=(0, native["nroots"]-1))
    residual = float(np.linalg.norm(matrix@vectors-vectors*energies, axis=0).max())
    if residual > 1e-9:
        raise RuntimeError("Independent CIS residual failed.")
    x = vectors.T.reshape(native["nroots"], no, nv)
    td.e = energies
    td.xy = [(value.copy()/np.sqrt(2), np.zeros_like(value)) for value in x]
    # Public PySCF contracts the position operator with +r. Use -r for the
    # physical electronic moment, then align the electronic state phase via T.
    values = np.column_stack([-td.transition_dipole(), td.oscillator_strength()])
    transitions = np.sqrt(2)*np.einsum("pi,sia,qa->spq", mf.mo_coeff[:, :no], x, mf.mo_coeff[:, no:])
    return dict(mol=mol, overlap=mf.get_ovlp(), transitions=transitions,
                values=values, energies=energies, residual=residual)


def match(anchor, old, current):
    cross = gto.intor_cross("int1e_ovlp", old["mol"], current["mol"])
    inner = np.einsum("pq,pr,nrs,qs->n", anchor, cross, current["transitions"], cross)
    norm0 = np.trace(anchor.T@old["overlap"]@anchor@old["overlap"])
    norms = np.einsum("npq,pr,nrs,qs->n", current["transitions"], current["overlap"],
                      current["transitions"], current["overlap"])
    overlaps = np.abs(inner)/np.sqrt(norm0*norms)
    order = np.argsort(overlaps)[::-1]
    root = int(order[0])
    gap = float(np.min(np.abs(np.delete(current["energies"], root)-current["energies"][root])))
    if overlaps[root] < .85 or overlaps[root]-overlaps[order[1]] < .2 or gap < 1e-4:
        raise RuntimeError("Independent state tracking is unresolved.")
    record = dict(root=root, overlap=float(overlaps[root]), gap_hartree=gap,
                   residual=current["residual"])
    return root, (1. if inner[root] >= 0 else -1.), record


r0 = np.asarray(native["initial_coordinates_bohr"])
previous = reference(r0)
initial_roots = [int(label[1:])-1 for label in native["original_state_labels"]]
anchors = [previous["transitions"][root] for root in initial_roots]
continuation = []
for point in native["continuation"]:
    fraction = point["fraction"]
    current = reference(r0+fraction*(coordinates-r0))
    matches = [match(anchor, previous, current) for anchor in anchors]
    roots = [item[0] for item in matches]
    if len(set(roots)) != 2:
        raise RuntimeError("Independent continuation assigned duplicate roots.")
    continuation.append(dict(fraction=fraction, matches=[item[2] for item in matches]))
    anchors = [current["transitions"][root] for root in roots]
    previous = current
baseline = previous
np.testing.assert_array_equal(roots, native["selected_roots"])
phases, baseline_values, overlaps = [], [], []
for slot, root in enumerate(roots):
    target = np.asarray(native["baseline_transition_densities_ao"])[slot]
    transition, s = baseline["transitions"][root], baseline["overlap"]
    inner = np.trace(target.T@s@transition@s)
    overlap = abs(inner)/np.sqrt(np.trace(target.T@s@target@s)*np.trace(transition.T@s@transition@s))
    np.testing.assert_allclose(overlap, 1., atol=1e-8, rtol=0)
    phase = 1. if inner >= 0 else -1.
    value = baseline["values"][root].copy()
    value[:3] *= phase
    phases.append(phase)
    baseline_values.append(value)
    overlaps.append(float(overlap))
baseline_values = np.asarray(baseline_values)
tracking = []


def evaluate(r):
    current = reference(r)
    values = []
    for anchor, phase in zip(anchors, phases):
        root, sign, record = match(anchor, baseline, current)
        value = current["values"][root].copy()
        value[:3] *= sign*phase
        values.append(value)
        tracking.append(record)
    return np.asarray(values)


errors = dict(
    excitation_hartree=float(np.max(np.abs(baseline["energies"]-native["excitation_energies_hartree"]))),
    baseline_moments_strengths=float(np.max(np.abs(baseline_values-native["values"]))),
)
checks = []
for mode in range(6):
    expected = np.asarray(native["mode_derivatives"])[:, :, mode]
    for step in (2e-3, 6e-4, 2e-4):
        plus, minus = evaluate(coordinates+step*modes[:, :, mode]), evaluate(coordinates-step*modes[:, :, mode])
        finite = (plus-minus)/(2*step)
        quadratic = float((plus[0, 3]+minus[0, 3]-2*baseline_values[0, 3])/(2*step**2))
        error = float(np.max(np.abs(finite-expected)))
        quadratic_error = abs(quadratic-native["dark_quadratic_coefficients"][mode])
        checks.append(dict(mode=mode, step_bohr_sqrt_amu=step, derivative=finite.tolist(), max_error=error,
                            dark_quadratic_coefficient=quadratic, dark_quadratic_error=quadratic_error))
        if step == 2e-4:
            errors["mode_derivatives"] = max(errors.get("mode_derivatives", 0.), error)
            errors["dark_quadratic"] = max(errors.get("dark_quadratic", 0.), quadratic_error)
    print("Mode", mode+1, "independent final derivative error", error, flush=True)
scans = []
for point in native["scans"]:
    mode, displacement = point["mode"], point["displacement_bohr_sqrt_amu"]
    values = evaluate(coordinates+displacement*modes[:, :, mode])
    error = float(np.max(np.abs(values-point["values"])))
    errors["scan_moments_strengths"] = max(errors.get("scan_moments_strengths", 0.), error)
    scans.append(dict(mode=mode, displacement_bohr_sqrt_amu=displacement, values=values.tolist(), max_error=error))
tolerances = dict(excitation_hartree=1e-8, baseline_moments_strengths=2e-7,
                  mode_derivatives=2e-6, dark_quadratic=2e-6, scan_moments_strengths=2e-7)
passed = all(errors[key] <= value for key, value in tolerances.items())
report = dict(passed=bool(passed), reference="Independent PySCF RHF/get_ab + SciPy dense roots; public transition moments and strengths",
    errors=errors, tolerances=tolerances, baseline_values=baseline_values.tolist(),
    native_reference_transition_overlaps=overlaps, continuation=continuation,
    finite_difference_checks=checks, scans=scans, tracking=tracking,
    elapsed_seconds=perf_counter()-started, pyscf=pyscf.__version__, hostname=platform.node(),
    native_sha256=hashlib.sha256((output/"gradscf.json").read_bytes()).hexdigest(),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(output/"pyscf.json").write_text(json.dumps(report, indent=2)+"\n")
if not passed:
    raise RuntimeError(f"Independent X07 checks failed: {errors}")
print("Independent X07 gates passed", report["elapsed_seconds"], "s", flush=True)

# Measured jy_gpu_1 / PySCF 2.13.0, OMP/BLAS=2: all gates passed in 72.21 s.
# Max mode derivative error 1.023e-6; dark gamma error 8.39e-11; scan moment/f error 4.27e-12.
