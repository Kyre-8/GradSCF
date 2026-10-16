"""Build raw LiH IFC bands, signed DOS and one positive-mode animation.

Consumes only a complete, source-consistent q mesh that passes raw Hermiticity
and conjugacy. Reuses ForceConstants without ASR or long-range correction.
All numerical postprocessing/plotting runs on jy_gpu_1.
"""
import hashlib
import itertools
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from scipy.constants import physical_constants, h, c

from gradscf.pbc.ifc import ForceConstants


PATH = [
    [[0., 0., 0.], [.5, 0., .5], [.625, .25, .625]],
    [[.375, .375, .75], [0., 0., 0.], [.5, .5, .5], [.5, .25, .75]],
]
PATH_SOURCE = "https://phonopy.github.io/phonopy/phonopy-module.html#band-structure"
TO_CM1 = physical_constants["Hartree energy"][0]/(h*c*100)


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(q):
    return tuple(np.mod(np.round(np.mod(q, 1.), 12), 1.))


def load_mesh(folder, mesh):
    found = {}
    for path in folder.glob("q_*.json"):
        value = json.loads(path.read_text())
        if "q_evaluated" not in value:
            continue
        point = key(value["q_evaluated"])
        if point in found:
            raise ValueError("Duplicate canonical q point")
        found[point] = (path, value)
    qpoints = np.array(list(itertools.product(*[np.arange(n)/n for n in mesh])))
    selected = []
    for point in qpoints:
        if key(point) not in found:
            raise ValueError(f"Incomplete q mesh: missing {point}")
        path, value = found[key(point)]
        if not value.get("complete") or not value.get("passed") or value.get("asr_projected"):
            raise ValueError(f"Raw q record is incomplete/failed/projected: {path}")
        selected.append((path, value))
    identities = [{k:v for k,v in value["identity"].items() if k != "q_evaluated"} for _,value in selected]
    if any(identity != identities[0] for identity in identities):
        raise ValueError("The q mesh mixes source or physical-model identities")
    references = [value["reference"] for _,value in selected]
    for field in ("settings", "lattice_bohr", "coordinates_bohr", "fractional_kpoints", "basis_data", "pseudopotential_data"):
        if any(reference[field] != references[0][field] for reference in references):
            raise ValueError(f"The q mesh mixes reference {field}")
    energies = [reference["energy_hartree"] for reference in references]
    if max(energies)-min(energies) > 2e-10:
        raise ValueError("Re-converged references differ beyond the energy gate")
    matrices = np.array([np.asarray(value["matrix_real_hartree_per_bohr2"])
                        +1j*np.asarray(value["matrix_imag_hartree_per_bohr2"]) for _,value in selected])
    indices = np.rint(qpoints*np.asarray(mesh)).astype(int)
    negative = np.ravel_multi_index(((-indices)%mesh).T, mesh)
    errors = dict(hermiticity=float(np.max(abs(matrices-matrices.conj().transpose(0, 2, 1)))),
                  conjugacy=float(np.max(abs(matrices[negative]-matrices.conj()))))
    if max(errors.values()) > 1e-8:
        raise ValueError(f"Raw mesh fails before interpolation or ASR: {errors}")
    return qpoints, matrices, selected, errors


def band_path(lattice, count=41):
    reciprocal = 2*np.pi*np.linalg.inv(lattice).T
    segments, distances, ticks = [], [], [0.]
    offset = 0.
    for block in PATH:
        for start, end in zip(block[:-1], block[1:]):
            points = np.linspace(start, end, count)
            length = np.linalg.norm((np.asarray(end)-start)@reciprocal)
            segments.append(points)
            distances.append(offset+np.linspace(0., length, count))
            offset += length
            ticks.append(offset)
    return np.concatenate(segments), np.asarray(distances), ticks


def draw(report, arrays, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [3.4, 1.]}, sharey=True)
    frequency = arrays["band_frequencies_hartree"]*TO_CM1
    for distance, values in zip(arrays["band_distances"], frequency.reshape(5, -1, 6)):
        axes[0].plot(distance, values, color="#246b8e", lw=1.)
    energy = arrays["dos_energies_hartree"]*TO_CM1
    density = arrays["dos_states_per_hartree"]/TO_CM1
    axes[1].plot(density, energy, color="#246b8e", lw=1.2)
    axes[1].fill_betweenx(energy, 0., density, where=energy < 0., color="#b44738", alpha=.35)
    for axis in axes:
        axis.axhline(0., color=".35", lw=.8)
        axis.axhspan(energy.min(), 0., color="#b44738", alpha=.055)
        axis.grid(axis="y", alpha=.2)
    for tick in report["path_ticks"]:
        axes[0].axvline(tick, color=".8", lw=.6)
    axes[0].set(xticks=report["path_ticks"], xticklabels=[r"$\Gamma$", "X", "U | K", r"$\Gamma$", "L", "W"],
                xlim=(0., report["path_ticks"][-1]), ylabel="Signed frequency / cm⁻¹", title="Raw shortest-image IFC interpolation")
    axes[1].set(xlabel="States / cm⁻¹ / cell", title="Signed DOS", xlim=(0., 1.1*density.max()))
    axes[1].xaxis.set_major_locator(MaxNLocator(nbins=3))
    settings = report["settings"]
    fig.suptitle(f"LiH PBE / GTH-Pade / SZV · k={settings['kmesh']} · FFT={settings['mesh']}\n"
                 f"Complete q mesh {report['qmesh']}; raw negative curvatures retained", fontsize=12)
    fig.subplots_adjust(left=.09, right=.97, bottom=.21, top=.83, wspace=.09)
    fig.text(.09, .11, f"Raw Γ ASR norm: {report['raw_asr_residual_hartree_per_bohr2']:.3e} Eh/bohr²; no ASR or polar correction. "
             f"DOS integral: {report['dos_integral_states_per_cell']:.8f} states/cell.\n"
             "Negative values denote imaginary curvatures, not stable negative-frequency excitations.\n"
             "This is the specified finite electronic/q-grid model; basis, k, FFT and interpolation convergence are separate checks.",
             fontsize=8.5, va="top", linespacing=1.4)
    for suffix in ("png", "pdf"):
        fig.savefig(output/("lih_ifc_bands_dos."+suffix), dpi=220, bbox_inches="tight")
    plt.close(fig)


def animate(model, report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    q = np.array([0., 0., 1/3])
    result = model.bands(q[None])
    frequencies, modes = np.asarray(result.frequencies[0]), np.asarray(result.modes[0])
    index = int(np.argmax(frequencies))
    if frequencies[index] <= 0:
        raise ValueError("A positive mode is required for an oscillatory illustration")
    vector = modes[:, index]/np.sqrt(np.repeat(report["masses_amu"], 3))
    vector *= np.exp(-1j*np.angle(vector[np.argmax(abs(vector))]))
    vector = vector.reshape(2, 3)
    amplitude = .25
    vector *= amplitude/np.linalg.norm(vector, axis=1).max()
    lattice, positions = np.asarray(report["lattice_bohr"]), np.asarray(report["coordinates_bohr"])
    repeats = np.array([[0, 0, z] for z in range(3)])
    base = (positions[None]+repeats[:, None]@lattice).reshape(6, 3)
    complex_displacement = (np.exp(2j*np.pi*(repeats@q))[:, None, None]*vector[None]).reshape(6, 3)
    fig = plt.figure(figsize=(6.8, 5.2))
    fig.subplots_adjust(left=.03, right=.83, bottom=.14, top=.87)
    axis = fig.add_subplot(111, projection="3d")
    colors = ["#9467bd", "#c9c9c9"]*3
    sizes = [110, 55]*3
    points = axis.scatter(*base.T, c=colors, s=sizes, edgecolors=".3")
    for start, end in itertools.combinations(np.array(list(itertools.product([0, 1], repeat=3))), 2):
        if np.sum(abs(start-end)) == 1:
            for repeat in repeats:
                line = (np.stack((start, end))+repeat)@lattice
                axis.plot(*line.T, color=".8", lw=.5)
    extent = np.concatenate((base, (repeats+1)@lattice, repeats@lattice))
    lower, upper = extent.min(axis=0)-1, extent.max(axis=0)+1
    axis.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), zlim=(lower[2], upper[2]),
             xlabel="x / bohr", ylabel="y / bohr", zlabel="z / bohr")
    axis.set_box_aspect(upper-lower)
    label = axis.text2D(.03, .98, "", transform=axis.transAxes, va="top", fontsize=9)
    def update(frame):
        phase = 2*np.pi*frame/40
        positions = base+np.real(complex_displacement*np.exp(-1j*phase))
        points._offsets3d = tuple(positions.T)
        label.set_text(f"q=(0,0,1/3), mode {index+1}, {frequencies[index]*TO_CM1:.1f} cm⁻¹\n"
                       f"Illustrative amplitude ≤{amplitude} bohr; phase {phase/(2*np.pi):.2f} cycles")
        return points, label
    update(0)
    fig.suptitle("LiH positive-mode phase animation · three primitive cells", fontsize=11)
    fig.text(.04, .035, "Purple: Li; gray: H. Mass-unweighted complex mode; real displacement shown.\n"
             "Illustration of the fixed harmonic eigenmode, not a molecular-dynamics trajectory.", fontsize=8)
    fig.savefig(output/"lih_positive_mode.png", dpi=180, bbox_inches="tight")
    animation = FuncAnimation(fig, update, frames=40, interval=80, blit=False)
    animation.save(output/"lih_positive_mode.gif", writer=PillowWriter(fps=12))
    plt.close(fig)
    return dict(q=q.tolist(), mode=index, frequency_signed_hartree=float(frequencies[index]),
        frequency_cm1=float(frequencies[index]*TO_CM1), illustrative_amplitude_bohr=amplitude,
        primitive_mode_real=vector.real.tolist(), primitive_mode_imag=vector.imag.tolist(),
        repeat_translations=repeats.tolist(), scope="Real phase evolution of one positive numerical mode; no mode AD or MD claim.")


def run(input_path, *, qmesh=(2, 2, 3), output=None, dos_mesh=10):
    """Run this example with ordinary Python parameters; no command-line parser."""
    args = SimpleNamespace(input=Path(input_path), qmesh=list(qmesh),
        output=None if output is None else Path(output), dos_mesh=dos_mesh)
    if min(args.qmesh) < 1 or args.dos_mesh < 1:
        raise ValueError("q/DOS mesh dimensions must be positive")
    jax.config.update("jax_enable_x64", True)
    started = perf_counter()
    output = args.output or args.input/"ifc_raw"
    output.mkdir(parents=True, exist_ok=True)
    path = output/"results.json"
    path.write_text('{"complete": false, "passed": false}\n')
    qpoints, matrices, records, raw_errors = load_mesh(args.input, tuple(args.qmesh))
    reference = records[0][1]["reference"]
    lattice, positions = np.asarray(reference["lattice_bohr"]), np.asarray(reference["coordinates_bohr"])
    model = ForceConstants.from_qmesh(SimpleNamespace(lattice=lattice, coords=positions), qpoints,
        matrices, masses=reference["masses_amu"], asr=False, tolerance=1e-8)
    reconstruction = float(np.max(abs(np.asarray(model.matrix(qpoints))-matrices)))
    if reconstruction > 2e-10:
        raise RuntimeError("IFC sampled-matrix reconstruction failed")
    check_q = np.array([[.13, .17, .21], [.22, -.11, .19], [0., 0., .1], [.5, .2, .1]])
    check_matrices = np.asarray(model.matrix(check_q))
    periodicity = float(np.max(abs(np.asarray(model.matrix(check_q+[1., -1., 2.]))-check_matrices)))
    if periodicity > 2e-10:
        raise RuntimeError("Interpolated lattice-phase reciprocal periodicity failed")
    bands_q, distances, ticks = band_path(lattice)
    bands = model.bands(bands_q)
    dos_q = np.array(list(itertools.product(np.arange(args.dos_mesh)/args.dos_mesh, repeat=3)))
    dos_frequencies = np.asarray(model.bands(dos_q).frequencies)
    width = 25./TO_CM1
    energy = np.linspace(dos_frequencies.min()-8*width, dos_frequencies.max()+8*width, 2001)
    density = np.asarray(model.dos(dos_q, energy, width=width))
    integral = float(np.trapezoid(density, energy))
    if abs(integral-6.) > 1e-5:
        raise RuntimeError("Signed DOS finite-window normalization failed")
    arrays = dict(qpoints=qpoints, matrices=matrices, lattice_bohr=lattice, coordinates_bohr=positions,
        masses_amu=np.asarray(reference["masses_amu"]), translations=np.asarray(model.translations),
        pairs=np.asarray(model.pairs), realspace_values=np.asarray(model.values),
        check_qpoints=check_q, check_matrices=check_matrices, band_qpoints=bands_q,
        band_distances=distances, band_frequencies_hartree=np.asarray(bands.frequencies),
        dos_qpoints=dos_q, dos_frequencies_hartree=dos_frequencies, dos_energies_hartree=energy,
        dos_states_per_hartree=density)
    np.savez_compressed(output/"ifc_data.npz", **arrays)
    report = dict(complete=False, passed=False, settings=reference["settings"], qmesh=args.qmesh,
        source_python_sha256=records[0][1]["source_python_sha256"], script_sha256=fingerprint(__file__),
        interpolation_source_sha256={name:fingerprint(Path("src/gradscf/pbc")/name)
                                     for name in ("ifc.py", "phonon.py")},
        input_sha256={str(p):fingerprint(p) for p,_ in records}, arrays_sha256=fingerprint(output/"ifc_data.npz"),
        lattice_bohr=lattice.tolist(), coordinates_bohr=positions.tolist(), masses_amu=reference["masses_amu"],
        reference=reference, raw_errors_hartree_per_bohr2=raw_errors,
        sample_reconstruction_error_hartree_per_bohr2=reconstruction,
        reciprocal_periodicity_error_hartree_per_bohr2=periodicity,
        raw_asr_residual_hartree_per_bohr2=float(model.raw_asr_residual), asr_projected=False,
        asr_correction_norm=float(model.asr_correction_norm), polar_correction=False,
        path=PATH, path_labels=["Gamma", "X", "U|K", "Gamma", "L", "W"], path_source=PATH_SOURCE, path_ticks=ticks,
        dos_sampling=[args.dos_mesh]*3, dos_width_cm1=25., dos_integral_states_per_cell=integral,
        negative_dos_mode_samples=int(np.count_nonzero(dos_frequencies < 0)),
        scope="Complete finite-model raw IFC interpolation and signed DOS; no converged material-stability or fine-grid claim.")
    path.write_text(json.dumps(report, indent=2)+"\n")
    draw(report, arrays, output)
    report["animation"] = animate(model, report, output)
    report.update(complete=True, passed=True, elapsed_seconds=perf_counter()-started)
    path.write_text(json.dumps(report, indent=2)+"\n")
    print("Complete raw IFC bands/DOS/animation:", output, reconstruction, integral, flush=True)


if __name__ == "__main__":
    run(Path("reproducibility/ad_showcase/P03_lih_qmesh/k2x2x3_fft41"))
