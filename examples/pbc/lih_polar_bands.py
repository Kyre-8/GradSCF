"""Matched LiH raw/polar IFC bands and signed DOS, without ASR projection.

Consumes validated GradSCF mechanical, Born, dielectric and finite-field
records from one finite electronic model. Uses the existing polar IFC and
numerical mode routines; source SCF/IFC construction is not differentiated.
"""
import hashlib
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np


def validate_models(reference, born, dielectric):
    if not (born.get("complete") and born.get("full_tensor") and born.get("validation_passed")
            and dielectric.get("complete") and dielectric.get("full_tensor")):
        raise ValueError("Require validated complete Born and dielectric tensors")
    settings = {key:value for key,value in reference["settings"].items() if key != "displacement_bohr"}
    for name, value in (("Born", born), ("dielectric", dielectric)):
        if value["settings"] != settings:
            raise ValueError(f"{name} settings differ from the mechanical model")
        for field in ("lattice_bohr", "coordinates_bohr", "fractional_kpoints"):
            if np.shape(reference[field]) != np.shape(value[field]) or not np.allclose(reference[field], value[field], atol=1e-12, rtol=0):
                raise ValueError(f"{name} {field} differs from the mechanical model")
        if abs(reference["energy_hartree"]-value["energy_hartree"]) > 2e-10:
            raise ValueError(f"{name} reference energy differs from the mechanical model")
    for field in ("basis_data", "pseudopotential_data"):
        if reference[field] != born[field]:
            raise ValueError(f"Born {field} differs from the mechanical model")


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def broaden(frequencies, energy, width):
    values = frequencies.ravel()
    result = np.zeros(len(energy))
    for start in range(0, len(values), 256):
        offset = (energy[:, None]-values[None, start:start+256])/width
        result += np.exp(-.5*offset**2).sum(axis=1)/(len(frequencies)*width*np.sqrt(2*np.pi))
    return result


def draw(report, arrays, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.3), gridspec_kw={"width_ratios": [3.4, 1.]}, sharey=True)
    for x, raw, polar in zip(arrays["band_distances"], arrays["raw_band_cm1"].reshape(5, -1, 6),
                             arrays["polar_band_cm1"].reshape(5, -1, 6)):
        axes[0].plot(x, raw, color=".55", ls="--", lw=.8, alpha=.8)
        axes[0].plot(x, polar, color="#246b8e", lw=1.05)
    axes[1].plot(arrays["raw_dos_states_per_cm1"], arrays["dos_energies_cm1"], color=".55", ls="--", lw=1.)
    axes[1].plot(arrays["polar_dos_states_per_cm1"], arrays["dos_energies_cm1"], color="#246b8e", lw=1.2)
    for axis in axes:
        axis.axhline(0., color=".35", lw=.8)
        axis.axhspan(arrays["dos_energies_cm1"].min(), 0., color="#b44738", alpha=.055)
        axis.grid(axis="y", alpha=.2)
    for tick in report["path_ticks"]:
        axes[0].axvline(tick, color=".8", lw=.6)
    axes[0].set(xticks=report["path_ticks"], xticklabels=[r"$\Gamma$", "X", "U | K", r"$\Gamma$", "L", "W"],
                xlim=(0., report["path_ticks"][-1]), ylabel="Signed frequency / cm⁻¹", title="Matched raw and polar interpolation")
    axes[1].set(xlabel="States / cm⁻¹ / cell", title="Signed DOS", xlim=(0., None))
    axes[1].xaxis.set_major_locator(MaxNLocator(nbins=3))
    axes[0].legend(handles=[Line2D([], [], color=".55", ls="--", label="Raw IFC"),
                           Line2D([], [], color="#246b8e", label="Subtract / interpolate / restore")], fontsize=8, loc="lower right")
    settings = report["settings"]
    fig.suptitle(f"LiH PBE / GTH-Pade / SZV · k={settings['kmesh']} · FFT={settings['mesh']}\n"
                 f"q mesh {report['qmesh']}; matched full Born and dielectric tensors", fontsize=12)
    fig.subplots_adjust(left=.09, right=.97, bottom=.23, top=.82, wspace=.09)
    fig.text(.09, .13, "Γ band endpoints use the Cartesian tangent of each path segment; DOS Γ uses the short-circuit value.\n"
             f"No ASR/charge projection. Raw total ASR norm {report['raw_total_asr_residual']:.3e} Eh/bohr². "
             f"Gaussian σ = {report['dos_width_cm1']:g} cm⁻¹.\n"
             "Negative values denote imaginary modes. Finite-model result; decomposition-width and q-grid convergence are separate.",
             fontsize=8.5, va="top", linespacing=1.4)
    for suffix in ("png", "pdf"):
        fig.savefig(output/("lih_polar_bands_dos."+suffix), dpi=220, bbox_inches="tight")
    plt.close(fig)


def run(raw, born, dielectric, field, *, output=Path("reproducibility/ad_showcase/P05_lih_polar_bands")):
    """Run this example with ordinary Python parameters; no command-line parser."""
    args = SimpleNamespace(raw=Path(raw), born=Path(born), dielectric=Path(dielectric),
        field=Path(field), output=Path(output))
    import jax
    from scipy.constants import physical_constants, h, c
    from gradscf.pbc.polar_ifc import DipoleCorrection, PolarForceConstants
    from gradscf.pbc.phonon import _modes

    jax.config.update("jax_enable_x64", True)
    began = perf_counter()
    raw_path, array_path = args.raw/"results.json", args.raw/"ifc_data.npz"
    raw, born, dielectric, field = [json.loads(path.read_text()) for path in (raw_path, args.born, args.dielectric, args.field)]
    if not raw.get("complete") or not raw.get("passed") or raw.get("asr_projected") or fingerprint(array_path) != raw["arrays_sha256"]:
        raise ValueError("Require the complete unchanged raw IFC capsule")
    validate_models(raw["reference"], born, dielectric)
    if (not field.get("complete") or not field.get("validation_passed")
            or field["settings"] != dielectric["settings"]
            or field["linear_result_sha256"] != fingerprint(args.dielectric)):
        raise ValueError("Require the completed matching finite-field dielectric validation")
    for record in (born, dielectric):
        for path, expected in record["source_sha256"].items():
            if fingerprint(path) != expected:
                raise ValueError(f"Electric tensor source changed: {path}")
    for name, expected in raw["interpolation_source_sha256"].items():
        if fingerprint(Path("src/gradscf/pbc")/name) != expected:
            raise ValueError("Raw IFC source changed")
    arrays = dict(np.load(array_path))
    lattice, positions, masses = arrays["lattice_bohr"], arrays["coordinates_bohr"], arrays["masses_amu"]
    charge, epsilon = np.asarray(born["born_charge_e"]), np.asarray(dielectric["epsilon_infinity"])
    cell = SimpleNamespace(lattice=lattice, coords=positions)
    width, cutoff = .35, 5
    correction = DipoleCorrection(lattice, positions, charge, epsilon, smearing=width, cutoff=cutoff)
    model = PolarForceConstants.from_qmesh(cell, arrays["qpoints"], arrays["matrices"], masses=masses,
                                          correction=correction, tolerance=1e-8)
    if model.sample_reconstruction_error > 2e-10:
        raise RuntimeError("Material polar sampled reconstruction failed")
    targets = arrays["check_qpoints"]
    reference = DipoleCorrection(lattice, positions, charge, epsilon, smearing=width, cutoff=6)
    reference_values = np.array([reference.matrix(q) for q in targets])
    cutoff_rows = []
    for size in (1, 2, 3, 4, 5):
        current = DipoleCorrection(lattice, positions, charge, epsilon, smearing=width, cutoff=size)
        values = np.array([current.matrix(q) for q in targets])
        row = dict(cutoff=size, max_error_hartree_per_bohr2=float(np.max(abs(values-reference_values))),
            reciprocal_periodicity_error=float(np.max(abs(current.matrix(targets[0]+[1., -1., 2.])-values[0]))),
            self_inverse_imaginary=float(np.max(abs(current.matrix([.5, 0., 0.]).imag))))
        cutoff_rows.append(row)
    if max(cutoff_rows[-1][key] for key in ("max_error_hartree_per_bohr2", "reciprocal_periodicity_error", "self_inverse_imaginary")) > 2e-11:
        raise RuntimeError("Material reciprocal dipole cutoff is unresolved")
    native_values = np.array([model.matrix(q) for q in targets])
    width_rows = []
    for alternate_width in (.25, .35, .45):
        alternate = DipoleCorrection(lattice, positions, charge, epsilon, smearing=alternate_width, cutoff=cutoff)
        varied = PolarForceConstants.from_qmesh(cell, arrays["qpoints"], arrays["matrices"], masses=masses,
                                              correction=alternate, tolerance=1e-8)
        values = np.array([varied.matrix(q) for q in targets])
        width_rows.append(dict(smearing_inverse_bohr=alternate_width,
            change_from_035_hartree_per_bohr2=float(np.max(abs(values-native_values))),
            sample_reconstruction_error=varied.sample_reconstruction_error))
    conversion = physical_constants["Hartree energy"][0]/(h*c*100)
    matrix_at = jax.jit(model.matrix)
    def frequencies(q, direction=None):
        value = model.gamma(direction) if np.linalg.norm(q-np.rint(q)) < 1e-12 else matrix_at(q)
        if np.max(abs(np.asarray(value)-np.asarray(value).conj().T)) > 1e-8:
            raise RuntimeError("Polar interpolation lost Hermiticity")
        return np.asarray(_modes(value, masses, False)[1])*conversion
    reciprocal = 2*np.pi*np.linalg.inv(lattice).T
    band_values, gamma_limits = [], []
    for segment in arrays["band_qpoints"].reshape(5, -1, 3):
        direction = (segment[-1]-segment[0])@reciprocal
        for q in segment:
            band_values.append(frequencies(q, direction))
            if np.linalg.norm(q-np.rint(q)) < 1e-12:
                gamma_limits.append(dict(cartesian_direction=direction.tolist(), frequencies_cm1=band_values[-1].tolist()))
    dos_values = np.array([frequencies(q) for q in arrays["dos_qpoints"]])
    raw_dos = arrays["dos_frequencies_hartree"]*conversion
    energy = np.linspace(min(dos_values.min(), raw_dos.min())-200., max(dos_values.max(), raw_dos.max())+200., 2001)
    spectra = [broaden(values, energy, 25.) for values in (raw_dos, dos_values)]
    integrals = [float(np.trapezoid(density, energy)) for density in spectra]
    if max(abs(value-6.) for value in integrals) > 1e-5:
        raise RuntimeError("Polar signed DOS normalization failed")
    direction = np.array([1., .2, .3])
    saved = dict(band_distances=arrays["band_distances"], band_qpoints=arrays["band_qpoints"],
        raw_band_cm1=arrays["band_frequencies_hartree"]*conversion, polar_band_cm1=np.array(band_values),
        dos_energies_cm1=energy, raw_dos_states_per_cm1=spectra[0], polar_dos_states_per_cm1=spectra[1],
        long_range_check_qpoints=targets, long_range_check_matrices=np.array([correction.matrix(q) for q in targets]),
        long_range_gamma=np.asarray(correction.gamma()), long_range_gamma_direction=direction,
        long_range_directional_gamma=np.asarray(correction.gamma(direction)), polar_check_matrices=native_values)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output/"material.npz", **saved)
    report = dict(complete=False, passed=False, settings=dielectric["settings"], qmesh=raw["qmesh"],
        lattice_bohr=lattice.tolist(), positions_bohr=positions.tolist(), masses_amu=masses.tolist(),
        born_charges=charge.tolist(), epsilon_infinity=epsilon.tolist(), smearing_inverse_bohr=width, cutoff=cutoff,
        cutoff_records=cutoff_rows, width_sensitivity_records=width_rows, gamma_limits=gamma_limits,
        gamma_dos_convention="G=0 excluded short-circuit value at the single sampled Gamma point",
        sample_reconstruction_error=model.sample_reconstruction_error, raw_total_asr_residual=float(model.raw_asr_residual),
        short_range_asr_residual=float(model.short_range.raw_asr_residual), raw_born_charge_sum_e=charge.sum(axis=0).tolist(),
        asr_projected=False, charge_projected=False, path_ticks=raw["path_ticks"], path_source=raw["path_source"],
        dos_sampling=raw["dos_sampling"], dos_width_cm1=25., dos_integrals_states_per_cell=integrals,
        source_python_sha256=raw["source_python_sha256"], source_sha256=fingerprint("src/gradscf/pbc/polar_ifc.py"),
        script_sha256=fingerprint(__file__), arrays_sha256=fingerprint(args.output/"material.npz"),
        input_sha256={str(path):fingerprint(path) for path in (raw_path, array_path, args.born, args.dielectric, args.field)},
        interpolation_convergence_established=False,
        scope="Matched GradSCF finite-model raw/polar interpolation. Negative modes and anisotropic tensors retained; no ASR/charge projection or bulk-convergence claim.")
    path = args.output/"gradscf.json"
    path.write_text(json.dumps(report, indent=2)+"\n")
    draw(report, saved, args.output)
    report.update(complete=True, passed=True, elapsed_seconds=perf_counter()-began)
    path.write_text(json.dumps(report, indent=2)+"\n")
    print("Matched material polar interpolation:", model.sample_reconstruction_error, integrals, width_rows, flush=True)


if __name__ == "__main__":
    run(Path("reproducibility/ad_showcase/P03_lih_qmesh/k2x2x3_fft41/ifc_raw"),
        Path("reproducibility/ad_showcase/P04_lih_born/k223_fft41.json"),
        Path("reproducibility/ad_showcase/P04_lih_dielectric/k223_fft41.json"),
        Path("reproducibility/ad_showcase/P04_lih_dielectric/k223_fft41_finite_field.json"))
