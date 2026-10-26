"""Independent PySCF finite-field and AO-potential checks for X04.

RHF and the public CIS A matrix are rebuilt at every perturbation. SciPy dense
roots are matched through AO transition operators. The relaxed density is
checked through directional energy derivatives; no external full relaxed 1-RDM
API is claimed. PySCF CISD's public RDM routine checks the unrelaxed CIS density.
"""
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import pyscf
from pyscf import gto, scf, ci
from scipy.linalg import eigh


output = Path("reproducibility/ad_showcase/X04_formaldehyde_cis_density")
native = json.loads((output/"gradscf.json").read_text())
if not native["complete"]:
    raise RuntimeError("Complete the native X04 checks first.")
started = perf_counter()
coordinates = np.asarray(native["coordinates_bohr"])
field0 = np.asarray(native["field_au"])


def reference(field, potential=None):
    mol = gto.M(atom=list(zip(native["symbols"], coordinates)), basis=native["basis"],
                unit="Bohr", cart=True, verbose=0)
    mf = scf.RHF(mol)
    position = mol.intor("int1e_r", comp=3)
    core = mf.get_hcore()+np.einsum("x,xpq->pq", field, position)
    if potential is not None:
        core = core+.5*(potential+potential.T)
    nuclear_dipole = mol.atom_charges()@coordinates
    nuclear_energy = mol.energy_nuc()-field@nuclear_dipole
    mf.get_hcore = lambda mol=None: core
    mf.energy_nuc = lambda *args, **kwargs: nuclear_energy
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-12, 1e-12, 180
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("Independent perturbed RHF did not converge.")
    td = mf.TDA()
    td.singlet = True
    a, _ = td.get_ab()
    no, nv = a.shape[:2]
    a = a.reshape(no*nv, no*nv)
    values, vectors = eigh(a, subset_by_index=(0, native["nroots"]-1))
    residuals = np.linalg.norm(a@vectors-vectors*values, axis=0)
    if np.max(residuals) > 1e-9:
        raise RuntimeError("Independent CIS root residual failed.")
    amplitudes = vectors.T.reshape(native["nroots"], no, nv)
    transitions = np.sqrt(2)*np.einsum("pi,sia,qa->spq", mf.mo_coeff[:, :no], amplitudes,
                                      mf.mo_coeff[:, no:])
    return dict(mol=mol, mf=mf, ground_energy=float(mf.e_tot), energies=values,
                amplitudes=amplitudes, transitions=transitions, overlap=mf.get_ovlp(),
                position=position, nuclear_dipole=nuclear_dipole,
                max_eigen_residual=float(np.max(residuals)))


baseline = reference(field0)
overlap = baseline["overlap"]
np.testing.assert_allclose(overlap, native["overlap"], atol=1e-10, rtol=0)
np.testing.assert_allclose(baseline["position"], native["position_integrals_bohr"], atol=1e-9, rtol=0)
anchor = baseline["transitions"][native["selected_root"]]
anchor_norm = np.einsum("pq,pr,rs,qs", anchor, overlap, anchor, overlap)
tracking = []


def energies(field, potential=None):
    current = reference(field, potential)
    transition = current["transitions"]
    inner = np.einsum("pq,pr,nrs,qs->n", anchor, overlap, transition, overlap)
    norms = np.einsum("npq,pr,nrs,qs->n", transition, overlap, transition, overlap)
    matches = np.abs(inner)/np.sqrt(anchor_norm*norms)
    order = np.argsort(matches)[::-1]
    root = int(order[0])
    gap = float(np.min(np.abs(np.delete(current["energies"], root)-current["energies"][root])))
    if matches[root] < .85 or matches[root]-matches[order[1]] < .20 or gap < 1e-4:
        raise RuntimeError("Independent root tracking is unresolved.")
    tracking.append(dict(root=root, overlap=float(matches[root]), gap_hartree=gap,
                         max_eigen_residual=current["max_eigen_residual"]))
    return np.array([current["ground_energy"], current["ground_energy"]+current["energies"][root]])


ground_density = baseline["mf"].make_rdm1()
ground_dipole = baseline["nuclear_dipole"]-np.einsum("pq,xqp->x", ground_density, baseline["position"])
no = native["nocc"]
nv = baseline["mf"].mo_coeff.shape[1]-no
x = baseline["amplitudes"][native["selected_root"]]
cisd_vector = ci.cisd.amplitudes_to_cisdvec(0., x/np.sqrt(2), np.zeros((no, no, nv, nv)))
unrelaxed = ci.cisd.make_rdm1(ci.CISD(baseline["mf"]), cisd_vector, ao_repr=True)
unrelaxed_dipole = baseline["nuclear_dipole"]-np.einsum("pq,xqp->x", unrelaxed, baseline["position"])
steps = (1e-4, 3e-5, 1e-5)
expected_dipoles = np.array([native["ground_dipole_au"], native["excited_relaxed_dipole_au"]])
field_checks = []
for step in steps:
    finite = np.column_stack([-(energies(field0+step*axis)-energies(field0-step*axis))/(2*step)
                              for axis in np.eye(3)])
    error = float(np.max(np.abs(finite-expected_dipoles)))
    field_checks.append(dict(step_au=step, dipoles_au=finite.tolist(), max_error_au=error))
    print(f"Independent field step {step:.1e}: max dipole difference {error:.3e} au", flush=True)
potential_checks = []
for number, direction in enumerate(np.asarray(native["potential_directions_ao"])):
    expected = np.array([np.sum(np.asarray(native["ground_density_ao"])*direction),
                         np.sum(np.asarray(native["relaxed_response_density_ao"])*direction)])
    for step in steps:
        finite = (energies(field0, step*direction)-energies(field0, -step*direction))/(2*step)
        potential_checks.append(dict(direction=number, step_hartree=step, derivative=finite.tolist(),
                                      expected=expected.tolist(), max_error=float(np.max(np.abs(finite-expected)))))
step = 1e-4
constant_shift = (energies(field0, step*overlap)-energies(field0, -step*overlap))/(2*step)
errors = dict(
    ground_energy_hartree=abs(baseline["ground_energy"]-native["ground_energy_hartree"]),
    excitation_hartree=abs(float(baseline["energies"][native["selected_root"]])
                           -native["excitation_energies_hartree"][native["selected_root"]]),
    ground_density=float(np.max(np.abs(ground_density-native["ground_density_ao"]))),
    unrelaxed_density=float(np.max(np.abs(unrelaxed-native["unrelaxed_cis_density_ao"]))),
    unrelaxed_dipole_au=float(np.max(np.abs(unrelaxed_dipole-native["excited_unrelaxed_dipole_au"]))),
    relaxed_dipole_finite_field_au=field_checks[-1]["max_error_au"],
    relaxed_density_directions=max(item["max_error"] for item in potential_checks if item["step_hartree"] == steps[-1]),
    constant_overlap_shift=float(np.max(np.abs(constant_shift-native["nelectron"]))),
)
tolerances = dict(ground_energy_hartree=1e-8, excitation_hartree=1e-7,
                  ground_density=2e-7, unrelaxed_density=2e-7, unrelaxed_dipole_au=2e-7,
                  relaxed_dipole_finite_field_au=2e-6, relaxed_density_directions=2e-6,
                  constant_overlap_shift=1e-7)
passed = all(errors[key] <= value for key, value in tolerances.items())
report = dict(
    passed=bool(passed), reference="Independent PySCF RHF/get_ab plus SciPy dense CIS; finite-field/AO-potential energies",
    ground_energy_hartree=baseline["ground_energy"], excitation_energies_hartree=baseline["energies"].tolist(),
    ground_density_ao=ground_density.tolist(), unrelaxed_cis_density_ao=unrelaxed.tolist(),
    ground_dipole_au=ground_dipole.tolist(), excited_unrelaxed_dipole_au=unrelaxed_dipole.tolist(),
    field_finite_differences=field_checks, potential_finite_differences=potential_checks,
    constant_overlap_shift_slopes=constant_shift.tolist(), tracking=tracking, errors=errors, tolerances=tolerances,
    relaxed_density_validation="Six controlled symmetric one-body directions plus constant-overlap shift; not full componentwise external relaxed RDM",
    elapsed_seconds=perf_counter()-started, pyscf=pyscf.__version__, hostname=platform.node(),
    native_sha256=hashlib.sha256((output/"gradscf.json").read_bytes()).hexdigest(),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
)
(output/"pyscf.json").write_text(json.dumps(report, indent=2)+"\n")
if not passed:
    raise RuntimeError(f"Independent X04 checks failed: {errors}")
print("Independent X04 errors:", errors, flush=True)
print(f"Independent X04 checks passed in {report['elapsed_seconds']:.2f} s", flush=True)

# Measured on jy_gpu_1 (PySCF 2.13.0, CPU, two threads): 61.24 s, all gates passed.
# Relaxed dipole finite-field error 1.50e-8 au; six AO-potential contractions 7.22e-9.
# Unrelaxed full AO density error 4.07e-13; no external full relaxed-RDM claim.
