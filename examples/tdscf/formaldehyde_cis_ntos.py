"""X05: normalized CIS transition density, NTOs and attachment/detachment.

Reuse the independently validated X04 native orbitals and six CIS roots.
Analyze S1 and the brightest transition within that finite six-root window.
NTO display phases and degenerate SVD frames are not physical observables.
"""
import csv
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from gradscf_tools.excited import singlet_cis_densities, singlet_cis_ntos


input_folder = Path("reproducibility/ad_showcase/X04_formaldehyde_cis_density")
native = json.loads((input_folder/"gradscf.json").read_text())
reference = json.loads((input_folder/"pyscf.json").read_text())
input_hash = hashlib.sha256((input_folder/"gradscf.json").read_bytes()).hexdigest()
if not native["complete"] or not reference["passed"] or reference["native_sha256"] != input_hash:
    raise RuntimeError("Use the independently validated X04 native state record.")
output = Path("reproducibility/ad_showcase/X05_formaldehyde_cis_ntos")
output.mkdir(parents=True, exist_ok=True)
(output/"gradscf.json").write_text('{"complete": false}\n')
started = perf_counter()
coefficients = np.asarray(native["coefficients_ao"])
amplitudes = np.asarray(native["amplitudes"])
overlap = np.asarray(native["overlap"])
position = np.asarray(native["position_integrals_bohr"])
energies = np.asarray(native["excitation_energies_hartree"])
nocc = native["nocc"]
np.testing.assert_allclose(coefficients.T@overlap@coefficients, np.eye(coefficients.shape[1]), atol=1e-10, rtol=0)
np.testing.assert_allclose(np.sum(amplitudes**2, axis=(1, 2)), 1., atol=1e-12, rtol=0)
properties = [singlet_cis_densities(coefficients, x, nocc=nocc) for x in amplitudes]
transitions = np.asarray([value.transition for value in properties])
dipoles = -np.einsum("spq,xqp->sx", transitions, position)
strengths = (2/3)*energies*np.sum(dipoles**2, axis=1)
brightest = int(np.argmax(strengths))
if brightest == 0 or strengths[brightest] < 1e-4:
    raise RuntimeError("The retained roots do not resolve a distinct bright example.")
selected = [0, brightest]
records = []
for root in selected:
    one = properties[root]
    nto = singlet_cis_ntos(coefficients, amplitudes[root], nocc=nocc)
    weights, holes, particles = map(np.asarray, nto[:3])
    counts = {label: float(np.trace(np.asarray(value)@overlap)) for label, value in (
        ("attachment", one.attachment), ("detachment", one.detachment), ("transition", one.transition))}
    transition = np.asarray(one.transition)
    norm2 = float(np.trace(transition.T@overlap@transition@overlap))
    reconstructed = np.sqrt(2)*(holes*np.sqrt(weights))@particles.T
    errors = dict(
        weight_sum=abs(float(weights.sum())-1),
        transition_reconstruction=float(np.max(np.abs(reconstructed-transition))),
        attachment_reconstruction=float(np.max(np.abs((particles*weights)@particles.T-one.attachment))),
        detachment_reconstruction=float(np.max(np.abs((holes*weights)@holes.T-one.detachment))),
        hole_orthonormality=float(np.max(np.abs(holes.T@overlap@holes-np.eye(len(weights))))),
        particle_orthonormality=float(np.max(np.abs(particles.T@overlap@particles-np.eye(len(weights))))),
    )
    if max(errors.values()) > 1e-10:
        raise RuntimeError(f"NTO reconstruction failed: {errors}")
    np.testing.assert_allclose([counts["attachment"], counts["detachment"], counts["transition"], norm2],
                               [1., 1., 0., 2.], atol=1e-10, rtol=0)
    electron = np.einsum("pq,xqp->x", one.attachment, position)
    hole = np.einsum("pq,xqp->x", one.detachment, position)
    unrelaxed_dipole_change = -np.einsum("pq,xqp->x", one.unrelaxed-one.ground, position)
    np.testing.assert_allclose(unrelaxed_dipole_change, -(electron-hole), atol=1e-10, rtol=0)
    records.append(dict(
        root=root, state_label=f"S{root+1}", excitation_hartree=float(energies[root]),
        oscillator_strength=float(strengths[root]), transition_dipole_au=dipoles[root].tolist(),
        transition_density_ao=transition.tolist(), attachment_density_ao=np.asarray(one.attachment).tolist(),
        detachment_density_ao=np.asarray(one.detachment).tolist(), weights=weights.tolist(),
        hole_ntos_ao=holes.tolist(), particle_ntos_ao=particles.tolist(),
        degenerate_weight_groups=nto.degenerate_groups, degeneracy_tolerance=1e-8,
        metric_counts=counts, transition_metric_norm_squared=norm2, errors=errors,
        electron_centroid_bohr=electron.tolist(), hole_centroid_bohr=hole.tolist(),
        centroid_separation_bohr=float(np.linalg.norm(electron-hole)),
        unrelaxed_dipole_change_au=unrelaxed_dipole_change.tolist(),
    ))
    print(f"S{root+1}: f={strengths[root]:.9g}, leading NTO weight={weights[0]:.9f}, "
          f"centroid separation={np.linalg.norm(electron-hole):.9f} Bohr", flush=True)
report = dict(
    complete=True, method="real singlet RHF/CIS", basis=native["basis"], cart=True,
    symbols=native["symbols"], coordinates_bohr=native["coordinates_bohr"],
    nocc=nocc, nroots=native["nroots"], selected_roots=selected,
    selection="S1 and the largest oscillator strength among the first six singlets",
    excitation_energies_hartree=energies.tolist(), transition_dipoles_au=dipoles.tolist(),
    oscillator_strengths=strengths.tolist(), states=records,
    input_native_sha256=input_hash, input_source_python_sha256=native["source_python_sha256"],
    helper_sha256=hashlib.sha256(Path("src/gradscf_tools/excited.py").read_bytes()).hexdigest(),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    elapsed_seconds=perf_counter()-started, hostname=platform.node(), jax=jax.__version__,
    dtype="float64", scope="Analysis of the validated X04 native states; no new SCF or CIS solve",
)
(output/"gradscf.json").write_text(json.dumps(report, indent=2)+"\n")
with (output/"states.csv").open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["root_1_based", "excitation_hartree", "oscillator_strength", "mu_x_au", "mu_y_au", "mu_z_au"])
    for root, (omega, strength, dipole) in enumerate(zip(energies, strengths, dipoles)):
        writer.writerow([root+1, omega, strength, *dipole])
with (output/"nto_weights.csv").open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["root_1_based", "pair_1_based", "weight"])
    for state in records:
        for pair, weight in enumerate(state["weights"]):
            writer.writerow([state["root"]+1, pair+1, weight])
print(f"Native X05 normalization/reconstruction gates passed in {report['elapsed_seconds']:.2f} s", flush=True)

# Measured jy_gpu_1 CPU float64 analysis of validated X04 data: 1.53 s.
# S1 leading weight .997578885155, f=1.8473e-31; S4 weight .732594907632, f=.324693185513.
# Each attachment/detachment metric trace is one; transition metric norm squared is two.
