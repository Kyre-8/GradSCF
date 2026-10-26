"""Water: QP removal spectrum and its complete nuclear directional response.

Compose the already verified native GW geometry JVP with the shared Lorentzian
routine. Reconverged displaced QP data, including their actual residues, come
from water_geometry_response.py. This is the occupied-QP part of a spectral
trace: it omits satellites and photoionization matrix elements. The imposed
width is a resolution, not a computed lifetime or electron-vibration effect.
"""
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

from gradscf.tools.spectra import HARTREE_TO_EV, lorentzian_spectrum


SOURCE = Path("reproducibility/ad_showcase/X02_water_gw_bse_response/primary/g0w0_full_nw200_continued.json")
OUTPUT = Path("reproducibility/ad_showcase/X14_water_qp_removal")


def run():
    started = perf_counter()
    source = json.loads(SOURCE.read_text())
    assert source["validation_passed"] and source["method"] == "g0w0"
    nmo, nocc = source["nmo"], source["nocc"]
    def parameters(values):
        return jnp.concatenate((-HARTREE_TO_EV*jnp.asarray(values[:nocc]),
                                2*jnp.asarray(values[nmo:nmo+nocc])))
    point = parameters(source["values"])
    tangent = parameters(source["derivative"])
    assert np.all(np.asarray(point[:nocc]) > 0) and np.all(np.asarray(point[nocc:]) > 0)
    lower, upper = 0., 600.  # eV; the core and all four valence QP peaks are included.
    hwhm = .15  # eV; do not normalize the broadened or finite-window spectrum.
    def area(p):
        centers, weights = p[:nocc], p[nocc:]
        return jnp.sum(weights*(jnp.arctan((upper-centers)/hwhm)
                               -jnp.arctan((lower-centers)/hwhm))/jnp.pi)
    exact_area, exact_area_derivative = map(float, jax.jvp(area, (point,), (tangent,)))
    full_area = float(jnp.sum(point[nocc:]))
    checks = []
    for count in (60001, 120001):
        grid = jnp.linspace(lower, upper, count)
        spectrum = jax.jit(lambda p: lorentzian_spectrum(p[:nocc], p[nocc:], grid, eta=hwhm))
        value, derivative = map(np.asarray, jax.jvp(spectrum, (point,), (tangent,)))
        numerical = float(np.trapezoid(value, np.asarray(grid)))
        numerical_derivative = float(np.trapezoid(derivative, np.asarray(grid)))
        checks.append(dict(points=count, spacing_ev=(upper-lower)/(count-1),
            integrated_weight=numerical, integrated_derivative=numerical_derivative,
            area_error=abs(numerical-exact_area), derivative_area_error=abs(numerical_derivative-exact_area_derivative)))
    finite_checks = []
    for row in source["finite_difference_checks"]:
        step = row["step_bohr"]
        finite = np.asarray((spectrum(parameters(row["plus_values"]))
                             -spectrum(parameters(row["minus_values"])))/(2*step))
        finite_checks.append(dict(step_bohr=step, max_spectrum_derivative_error=float(np.max(abs(finite-derivative)))))
    accepted = (max(checks[-1]["area_error"], checks[-1]["derivative_area_error"]) < 2e-8
                and finite_checks[-1]["max_spectrum_derivative_error"] < 2e-5)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = dict(validation_passed=bool(accepted), source=str(SOURCE),
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        hostname=platform.node(), python=platform.python_version(), jax=jax.__version__,
        backend=jax.default_backend(), dtype="float64", postprocessing_seconds=perf_counter()-started,
        binding_energy_ev=np.asarray(point[:nocc]).tolist(), weights_2Z=np.asarray(point[nocc:]).tolist(),
        binding_energy_derivative_ev_per_bohr=np.asarray(tangent[:nocc]).tolist(),
        weight_derivative_per_bohr=np.asarray(tangent[nocc:]).tolist(),
        imposed_hwhm_ev=hwhm, window_ev=[lower, upper], model_all_real_axis_weight=full_area,
        exact_window_weight=exact_area, missing_model_tails=full_area-exact_area,
        exact_window_weight_derivative_per_bohr=exact_area_derivative,
        quadrature_checks=checks, geometry_checks=finite_checks,
        scope="Occupied QP spectral trace with spin factor two, no satellites or photoionization matrix elements; not a normalized experimental PES.")
    (OUTPUT/"response.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print("Model all-axis/window area:", full_area, exact_area, flush=True)
    print("Geometry derivative checks:", finite_checks, flush=True)
    assert accepted, "Saved spectral weight/derivative gates did not pass."

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.8), sharex="col", layout="constrained")
    for column, (bounds, title) in enumerate((((5., 45.), "Valence QP peaks"),
                                             ((float(point[0])-6, float(point[0])+6), "Core QP peak"))):
        mask = (np.asarray(grid) >= bounds[0]) & (np.asarray(grid) <= bounds[1])
        x = np.asarray(grid)[mask]
        axes[0, column].plot(x, value[mask], color="#137c9a")
        axes[0, column].set(title=title, ylabel="QP spectral density / eV⁻¹")
        axes[1, column].plot(x, derivative[mask], color="#88539b", label="Full geometry AD")
        axes[1, column].plot(x, finite[mask], "--", color="#c46236", label="Rebuilt ±1e−5 bohr")
        axes[1, column].axhline(0., color=".6", linewidth=.6)
        axes[1, column].set(xlabel="Binding energy / eV", ylabel="Directional response / (eV bohr)⁻¹")
        np.savetxt(OUTPUT/f"{'valence' if column == 0 else 'core'}.csv",
                   np.column_stack((x, value[mask], derivative[mask], finite[mask])), delimiter=",",
                   header="binding_eV,QP_density_per_eV,AD_per_eV_bohr,FD_per_eV_bohr", comments="")
    axes[1, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("Water / Cartesian 6-31G*: G₀W₀ occupied-QP removal model\n"
                 "Actual 2Z weights; imposed HWHM 0.15 eV; satellites and matrix elements omitted")
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT/f"photoelectron_response.{suffix}", dpi=180)
    plt.close(fig)
    return report


if __name__ == "__main__":
    run()

# jy_gpu_1 CPU float64, from the recorded full native GW geometry response:
# Binding energies/eV: 546.787137, 32.563251, 18.611438, 14.456528, 12.178484
# All-real-axis QP model weight: 8.540629619965
# Weight within [0,600] eV:      8.518137821035
# Unplotted model tails:        0.022491798930 (no renormalization)
# At 0.005-eV grid spacing: area / area-derivative errors 4.02e-10 / 2.22e-11.
# At 1e-5-bohr central step: max spectrum AD-FD error 7.66e-7 /(eV bohr).
