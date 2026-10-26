"""Independent PySCF directional checks for the three X06 water CIS probes.

Geometry points rebuild RHF/public CIS matrices and track the same state.
Amplitude/reference partials hold the recorded common electronic state fixed
and use PySCF's public transition-moment contraction, with independent AO r.
No off-stationary HF reference is passed to a CIS energy solver.
"""
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import pyscf
from pyscf import gto, scf
from scipy.linalg import eigh, expm


output = Path("reproducibility/ad_showcase/X06_water_cis_probes")
native = json.loads((output/"gradscf.json").read_text())
if not native["complete"]:
    raise RuntimeError("Complete the native water probe checks first.")
started = perf_counter()
coordinates = np.asarray(native["coordinates_bohr"])


def reference(r):
    mol = gto.M(atom=list(zip(native["symbols"], r)), basis=native["basis"],
                unit="Bohr", cart=True, verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-13, 1e-12, 180
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("Independent displaced RHF did not converge.")
    td = mf.TDA()
    a, _ = td.get_ab()
    no, nv = a.shape[:2]
    matrix = a.reshape(no*nv, no*nv)
    energies, vectors = eigh(matrix, subset_by_index=(0, native["nroots"]-1))
    residual = np.linalg.norm(matrix@vectors-vectors*energies, axis=0).max()
    if residual > 1e-9:
        raise RuntimeError("Independent CIS roots did not pass residual checks.")
    x = vectors.T.reshape(native["nroots"], no, nv)
    td.e = energies
    td.xy = [(value.copy()/np.sqrt(2), np.zeros_like(value)) for value in x]
    dipoles = td.transition_dipole()
    transitions = np.sqrt(2)*np.einsum("pi,sia,qa->spq", mf.mo_coeff[:, :no], x, mf.mo_coeff[:, no:])
    return dict(mol=mol, mf=mf, energies=energies, dipoles=dipoles, transition=transitions,
                overlap=mf.get_ovlp(), residual=float(residual), ground_energy=float(mf.e_tot))


baseline = reference(coordinates)
s = baseline["overlap"]
np.testing.assert_allclose(s, native["overlap"], atol=1e-10, rtol=0)
np.testing.assert_allclose(baseline["mol"].intor("int1e_r", comp=3), native["position_integrals_bohr"], atol=1e-9, rtol=0)
anchor = baseline["transition"][native["selected_root"]]
anchor_norm = np.trace(anchor.T@s@anchor@s)
c = np.asarray(native["coefficients_ao"])
x = np.asarray(native["amplitudes"])
native_transition = np.sqrt(2)*c[:, :native["nocc"]]@x@c[:, native["nocc"]:].T
initial_match = abs(np.trace(native_transition.T@s@anchor@s))/np.sqrt(
    np.trace(native_transition.T@s@native_transition@s)*anchor_norm)
np.testing.assert_allclose(initial_match, 1., atol=1e-8, rtol=0)
tracking = []


def geometry_probe(r):
    current = reference(r)
    cross = gto.intor_cross("int1e_ovlp", baseline["mol"], current["mol"])
    inner = np.einsum("pq,pr,nrs,qs->n", anchor, cross, current["transition"], cross)
    norms = np.einsum("npq,pr,nrs,qs->n", current["transition"], current["overlap"],
                      current["transition"], current["overlap"])
    overlaps = np.abs(inner)/np.sqrt(anchor_norm*norms)
    order = np.argsort(overlaps)[::-1]
    root = int(order[0])
    gap = float(np.min(np.abs(np.delete(current["energies"], root)-current["energies"][root])))
    if overlaps[root] < .85 or overlaps[root]-overlaps[order[1]] < .2 or gap < 1e-4:
        raise RuntimeError("Independent tracked state is unresolved.")
    tracking.append(dict(root=root, overlap=float(overlaps[root]), gap_hartree=gap,
                         residual=current["residual"]))
    return float(current["dipoles"][root]@current["dipoles"][root])


def wavefunction_probe(coefficients, amplitude):
    # This is only a one-body contraction on the specified normalized states.
    # It neither solves SCF nor labels these arbitrary orbitals stationary.
    custom = baseline["mf"].copy()
    custom.mo_coeff = coefficients
    driver = custom.TDA()
    normalized = amplitude/np.linalg.norm(amplitude)/np.sqrt(2)
    driver.xy = [(normalized, np.zeros_like(normalized))]
    moment = driver.transition_dipole()[0]
    return float(moment@moment)


def reference_probe(theta):
    no, nmo = native["nocc"], native["nmo"]
    k = np.zeros((nmo, nmo))
    k[:no, no:], k[no:, :no] = theta, -theta.T
    return wavefunction_probe(c@expm(k), x)


errors = dict(
    excitation_hartree=float(np.max(np.abs(baseline["energies"]-native["excitation_energies_hartree"]))),
    ground_energy_hartree=abs(baseline["ground_energy"]-native["ground_energy_hartree"]),
    geometry_probe=abs(float(baseline["dipoles"][0]@baseline["dipoles"][0])-native["probe"]),
    common_state_probe=abs(wavefunction_probe(c, x)-native["probe"]),
)
checks = []
for check in native["finite_difference_checks"]:
    kind, direction = check["kind"], np.asarray(check["direction"])
    if kind == "geometry":
        point, function = coordinates, geometry_probe
    elif kind == "amplitude":
        point, function = x, lambda amplitude: wavefunction_probe(c, amplitude)
    else:
        point, function = np.zeros_like(direction), reference_probe
    finite = []
    for step in (1e-3, 3e-4, 1e-4):
        derivative = (function(point+step*direction)-function(point-step*direction))/(2*step)
        finite.append(dict(step=step, derivative=derivative, absolute_error=abs(derivative-check["derivative"])))
    errors[kind+"_derivative"] = max(errors.get(kind+"_derivative", 0.), finite[-1]["absolute_error"])
    checks.append(dict(kind=kind, index=check["index"], expected_derivative=check["derivative"],
                       finite_differences=finite))
    print(kind, check["index"], "independent final-step error", finite[-1]["absolute_error"], flush=True)
tolerances = {name: 2e-6 if name.endswith("_derivative") else 2e-8 for name in errors}
passed = all(errors[key] <= value for key, value in tolerances.items())
report = dict(passed=bool(passed), errors=errors, tolerances=tolerances,
    finite_difference_checks=checks, tracking=tracking, initial_transition_overlap=float(initial_match),
    reference="Fresh PySCF RHF/get_ab + SciPy dense geometry states; public transition_dipole on common states for amplitude/reference partials",
    elapsed_seconds=perf_counter()-started, pyscf=pyscf.__version__, hostname=platform.node(),
    native_sha256=hashlib.sha256((output/"gradscf.json").read_bytes()).hexdigest(),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(output/"pyscf.json").write_text(json.dumps(report, indent=2)+"\n")
if not passed:
    raise RuntimeError(f"Independent X06 checks failed: {errors}")
print("Independent X06 gates passed", report["elapsed_seconds"], "s", flush=True)

# Measured jy_gpu_1, PySCF 2.13.0, two CPU threads: all gates passed in 4.60 s.
# Independent geometry/amplitude/reference directional errors <=2.49e-9 / 9.19e-10 / 3.09e-10.
