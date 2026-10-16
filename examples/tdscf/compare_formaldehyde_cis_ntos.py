"""Independent PySCF public NTO/transition-moment checks for X05.

Fresh RHF integrals and public get_ab are diagonalized by SciPy. Each public
get_nto call receives a fresh amplitude copy: PySCF normalizes X in place.
Compare weighted projectors and phase-invariant moments, not raw NTO columns.
"""
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import pyscf
from pyscf import gto, scf
from scipy.linalg import eigh


output = Path("reproducibility/ad_showcase/X05_formaldehyde_cis_ntos")
native = json.loads((output/"gradscf.json").read_text())
if not native["complete"]:
    raise RuntimeError("Complete X05 native normalization checks first.")
started = perf_counter()
mol = gto.M(atom=list(zip(native["symbols"], native["coordinates_bohr"])),
            basis=native["basis"], unit="Bohr", cart=True, verbose=0)
mf = scf.RHF(mol)
mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-12, 1e-12, 180
mf.kernel()
if not mf.converged:
    raise RuntimeError("Independent RHF did not converge.")
td = mf.TDA()
td.singlet = True
a, _ = td.get_ab()
no, nv = a.shape[:2]
a = a.reshape(no*nv, no*nv)
energies, vectors = eigh(a, subset_by_index=(0, native["nroots"]-1))
residuals = np.linalg.norm(a@vectors-vectors*energies, axis=0)
if residuals.max() > 1e-9:
    raise RuntimeError("Independent dense CIS residual failed.")
amplitudes = vectors.T.reshape(native["nroots"], no, nv)
td.e = energies
td.xy = [(x.copy()/np.sqrt(2), np.zeros_like(x)) for x in amplitudes]
# Public transition observables use PySCF's spin-adapted norm of one half.
dipoles = td.transition_dipole()
strengths = td.oscillator_strength(gauge="length")
overlap = mf.get_ovlp()
position = mol.intor("int1e_r", comp=3)
transitions = np.sqrt(2)*np.einsum("pi,sia,qa->spq", mf.mo_coeff[:, :no], amplitudes, mf.mo_coeff[:, no:])
errors = dict(
    excitations_hartree=float(np.max(np.abs(energies-native["excitation_energies_hartree"]))),
    oscillator_strengths=float(np.max(np.abs(strengths-native["oscillator_strengths"]))),
    dipole_outer_products=float(np.max(np.abs(np.einsum("sx,sy->sxy", dipoles, dipoles)
        -np.einsum("sx,sy->sxy", native["transition_dipoles_au"], native["transition_dipoles_au"])))),
)
records = []
for state in native["states"]:
    root = state["root"]
    # Isolate get_nto's in-place normalization and independent phase choices.
    nto_driver = mf.TDA()
    nto_driver.e = energies.copy()
    nto_driver.xy = [(x.copy()/np.sqrt(2), np.zeros_like(x)) for x in amplitudes]
    weights, coefficients = nto_driver.get_nto(state=root+1, verbose=0)
    holes, particles = coefficients[:, :no], coefficients[:, no:no+len(weights)]
    attachment, detachment = (particles*weights)@particles.T, (holes*weights)@holes.T
    transition = transitions[root]
    # The public routine separately changes hole/particle phases. Retain the
    # full coupling to reconstruct T, including any degenerate-block rotations.
    coupling = holes.T@overlap@transition@overlap@particles/np.sqrt(2)
    reconstructed = np.sqrt(2)*holes@coupling@particles.T
    native_transition = np.asarray(state["transition_density_ao"])
    inner = float(np.trace(native_transition.T@overlap@transition@overlap))
    norms = [float(np.trace(value.T@overlap@value@overlap)) for value in (native_transition, transition)]
    normalized_overlap = abs(inner)/np.sqrt(np.prod(norms))
    electron = np.einsum("pq,xqp->x", attachment, position)
    hole = np.einsum("pq,xqp->x", detachment, position)
    local_errors = dict(
        weights=float(np.max(np.abs(weights-state["weights"]))),
        attachment=float(np.max(np.abs(attachment-state["attachment_density_ao"]))),
        detachment=float(np.max(np.abs(detachment-state["detachment_density_ao"]))),
        phase_invariant_transition_overlap=abs(normalized_overlap-1),
        reference_transition_reconstruction=float(np.max(np.abs(reconstructed-transition))),
        centroid_bohr=float(np.max(np.abs(np.array([electron, hole])
            -np.array([state["electron_centroid_bohr"], state["hole_centroid_bohr"]])))),
    )
    for name, value in local_errors.items():
        errors[name] = max(errors.get(name, 0.), value)
    records.append(dict(root=root, weights=weights.tolist(), coupling=coupling.tolist(),
                        transition_metric_overlap=normalized_overlap, errors=local_errors))
np.testing.assert_allclose(np.sum(td.xy[0][0]**2), .5, atol=1e-14, rtol=0)
tolerances = {name: 2e-7 for name in errors}
tolerances["excitations_hartree"] = 1e-8
tolerances["reference_transition_reconstruction"] = 1e-10
passed = all(errors[name] <= gate for name, gate in tolerances.items())
report = dict(
    passed=bool(passed), reference="Independent PySCF RHF/get_ab + SciPy dense roots; public get_nto, transition_dipole and oscillator_strength",
    excitation_energies_hartree=energies.tolist(), transition_dipoles_au=dipoles.tolist(),
    oscillator_strengths=strengths.tolist(), max_eigen_residual=float(residuals.max()),
    states=records, errors=errors, tolerances=tolerances,
    native_sha256=hashlib.sha256((output/"gradscf.json").read_bytes()).hexdigest(),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    elapsed_seconds=perf_counter()-started, hostname=platform.node(), pyscf=pyscf.__version__,
)
(output/"pyscf.json").write_text(json.dumps(report, indent=2)+"\n")
if not passed:
    raise RuntimeError(f"Independent X05 gates failed: {errors}")
print("Independent X05 errors:", errors, flush=True)
print(f"Independent X05 checks passed in {report['elapsed_seconds']:.2f} s", flush=True)

# Measured jy_gpu_1 PySCF 2.13.0, two CPU threads: all gates passed in .76 s.
# NTO weight / weighted attachment errors: 2.10e-12 / 3.20e-12; six f errors <4.08e-13.
