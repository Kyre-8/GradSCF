"""V07: 0 K single-electronic-state vibronic Raman in the frozen X11/X12 model.

Reuse X12's signed Cartesian transition moment and complete common-Eckart
harmonic model. Final states are the six ground fundamentals. The two KHD
time orderings use the scattering S-matrix (constant-sign damping) convention.
Raw atomic-unit amplitudes/strengths only: no photon prefactor, absolute cross
section, thermal population, other electronic states, or experimental claim.
"""
import csv
import hashlib
import json
import platform
from pathlib import Path
import resource
from time import perf_counter

import jax
import numpy as np
from scipy.integrate import trapezoid

jax.config.update("jax_enable_x64", True)
from gradscf_tools.khd_raman import linear_dipole_vertices, vertex_closure, khd_tensor, isotropic_strengths

BASE = Path("reproducibility/ad_showcase")
OUTPUT = BASE/"V07_formaldehyde_khd"
INPUT = BASE/"X12_formaldehyde_herzberg_teller/gradscf.json"
HARTREE_TO_EV = 27.211386245988
HARTREE_TO_CM1 = 219474.6313632
MAX_QUANTA = 24
CUTOFFS = (8, 12, 16, 20, 24)
GAMMA_EV = (.015, .03, .06)


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def closure_record(a, b, expected_a, expected_b):
    measured_a = a.conj().T@a
    measured_b = np.stack([v.conj().T@v for v in b.transpose(1, 0, 2)])
    target_a = float(np.trace(expected_a).real)
    target_b = np.trace(expected_b, axis1=-2, axis2=-1).real
    missing_a = target_a-float(np.trace(measured_a).real)
    missing_b = target_b-np.trace(measured_b, axis1=-2, axis2=-1).real
    return dict(absorption_missing_trace=missing_a, emission_missing_traces=missing_b,
        absorption_relative_remainder=missing_a/target_a, emission_relative_remainders=missing_b/target_b,
        absorption_gram_max_error=float(np.max(abs(measured_a-expected_a))),
        emission_gram_max_error=float(np.max(abs(measured_b-expected_b))))


def tail_bound(record, cutoff, minimum_frequency, zero_zero, final, laser, gamma):
    """Uniform Frobenius bound from exact vertex closures and omitted energy floor.

    Nonnegative tails come from completeness. Roundoff-scale negative deficits
    are not silently set to zero: the caller retains raw values and this bound
    includes a 64-epsilon absolute Gram-trace budget declared in its metadata.
    """
    error_budget = 64*np.finfo(float).eps
    da = record["absorption_missing_trace"]+error_budget
    db = record["emission_missing_traces"]+error_budget
    assert da >= 0 and np.all(db >= 0)
    lower = zero_zero+(cutoff+1)*minimum_frequency
    resonant_distance = np.hypot(max(0., lower-laser.max()), gamma)
    anti_distance = np.hypot(lower+laser.min()-final, gamma)
    return np.sqrt(da*db)*(1/resonant_distance+1/anti_distance)


def main():
    started = perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT/"gradscf.json").write_text('{"complete": false}\n')
    data = json.loads(INPUT.read_text())
    assert data["complete"] and data["selected_root"] == 0
    model = data["harmonic_model"]
    wi, wf, rotation, shift = [np.asarray(model[key]) for key in (
        "initial_frequencies_hartree", "final_frequencies_hartree", "canonical_rotation", "shift")]
    mu = np.asarray(data["dipole_au"])
    derivative = np.asarray(data["derivatives_mu_per_Qme"])
    zero_zero = model["model_zero_zero_hartree"]
    # These are X12's exact values, including the excited quadratic constant,
    # minimum offset and ZPE already present in its 0-0. No extra ZPE is added.
    print("V07 frozen X12", hashlib.sha256(INPUT.read_bytes()).hexdigest(), flush=True)
    states, a, b = linear_dipole_vertices(wi, wf, rotation, shift, mu, derivative, max_quanta=MAX_QUANTA)
    expected_a, expected_b = vertex_closure(wi, mu, derivative)
    energies = zero_zero+states@wf
    quanta = states.sum(axis=1)
    axis_ev = zero_zero*HARTREE_TO_EV+np.linspace(-.15, .85, 321)
    axis = axis_ev/HARTREE_TO_EV
    coarse = axis[::4]
    convergence, previous = [], None
    for cutoff in CUTOFFS:
        count = int(np.count_nonzero(quanta <= cutoff))
        aa, bb = a[:count], b[:count]
        closure = closure_record(aa, bb, expected_a, expected_b)
        tensors = np.stack([sum(khd_tensor(energies[:count], aa, bb, wi, coarse, gamma/HARTREE_TO_EV)) for gamma in GAMMA_EV])
        bound = np.stack([tail_bound(closure, cutoff, min(wf), zero_zero, wi, axis, gamma/HARTREE_TO_EV) for gamma in GAMMA_EV])
        scale = float(np.max(np.linalg.norm(tensors, axis=(-2, -1))))
        change = None if previous is None else float(np.max(np.linalg.norm(tensors-previous, axis=(-2, -1)))/scale)
        convergence.append(dict(max_quanta=cutoff, states=count, **closure, uniform_tensor_tail_bound=bound,
            maximum_tensor_norm=scale, max_tail_bound_relative_to_global_peak=float(np.max(bound)/scale),
            maximum_tensor_change_relative_to_global_peak=change))
        previous = tensors
        print("V07 cutoff", cutoff, "a tail", closure["absorption_relative_remainder"],
              "b tails", closure["emission_relative_remainders"], "tensor change", change,
              "bound/global peak", np.max(bound)/scale, flush=True)
    assert convergence[-1]["absorption_relative_remainder"] < 1e-6
    assert max(convergence[-1]["emission_relative_remainders"]) < 2e-5
    assert convergence[-1]["maximum_tensor_change_relative_to_global_peak"] < 2e-5
    assert convergence[-1]["max_tail_bound_relative_to_global_peak"] < 1e-4
    resonant, anti, profiles, grid_checks = [], [], [], []
    for gamma in GAMMA_EV:
        r, ar = khd_tensor(energies, a, b, wi, axis, gamma/HARTREE_TO_EV)
        total = r+ar
        strength = isotropic_strengths(total).transpose(1, 2, 0)
        # Gamma variation is model sensitivity; only frequency-grid and state
        # cutoff variations test numerical convergence at the same Gamma.
        integrals = [trapezoid(strength[::stride], axis_ev[::stride], axis=0) for stride in (4, 2, 1)]
        relative_change = float(np.max(abs(integrals[-1]-integrals[-2]))/np.max(integrals[-1]))
        grid_checks.append(dict(gamma_ev=gamma, grid_points=[81, 161, 321], window_integrals=integrals,
                               relative_last_integral_change=relative_change))
        assert relative_change < 1e-4
        resonant.append(r); anti.append(ar); profiles.append(strength)
        print("V07 gamma", gamma, "grid integral change", relative_change, flush=True)
    resonant, anti, profiles = map(np.asarray, (resonant, anti, profiles))
    total = resonant+anti
    # Preserve every signed Cartesian vertex remotely and every complex tensor
    # locally; neither state amplitudes nor plotted channels are normalized.
    np.savez_compressed(OUTPUT/"intermediate_vertices.npz", occupations=states, energies_hartree=energies,
                        absorption_au=a, emission_au=b)
    np.savez_compressed(OUTPUT/"khd_tensors.npz", laser_ev=axis_ev, gamma_ev=GAMMA_EV, final_gaps_hartree=wi,
                        resonant=resonant, antiresonant=anti, total=total, parallel_perpendicular=profiles)
    with (OUTPUT/"excitation_profiles.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["laser_ev", "gamma_ev", "ground_mode", "raman_shift_cm1", "parallel_au2", "perpendicular_au2"])
        for g, gamma in enumerate(GAMMA_EV):
            for l, laser in enumerate(axis_ev):
                for mode in range(len(wi)):
                    writer.writerow([laser, gamma, mode+1, wi[mode]*HARTREE_TO_CM1, *profiles[g, l, mode]])
    strongest = np.unravel_index(np.argmax(profiles.sum(axis=-1)), profiles.shape[:-1])
    norm = np.linalg.norm(total, axis=(-2, -1))
    anti_norm = np.linalg.norm(anti, axis=(-2, -1))
    antisym = np.linalg.norm(total-total.swapaxes(-1, -2), axis=(-2, -1))
    incoherent = isotropic_strengths(resonant).transpose(1, 2, 3, 0)+isotropic_strengths(anti).transpose(1, 2, 3, 0)
    source_hashes = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in (
        "src/gradscf_tools/franck_condon.py", "src/gradscf_tools/khd_raman.py", str(Path(__file__).relative_to(Path.cwd())))}
    report = dict(complete=True, model="Frozen X11/X12 common-Eckart single-well harmonic/linear-HT; one intermediate electronic state",
        symbols=data["symbols"], basis=data["basis"], temperature_kelvin=0, electronic_root=0,
        input_path=str(INPUT), input_sha256=hashlib.sha256(INPUT.read_bytes()).hexdigest(), inherited_input_checks=data["input_checks"],
        harmonic_model=model, dipole_au=mu, derivatives_mu_per_Qme=derivative,
        ground_fundamentals_cm1=wi*HARTREE_TO_CM1, gamma_ev=GAMMA_EV, gamma_definition="Phenomenological positive energy HWHM; not a calculated lifetime",
        denominator_convention="Scattering S-matrix: Delta-omegaL-iGamma and Delta+omegaS-iGamma; omegaS=omegaL-ground_gap",
        amplitude_units="(e bohr)^2/Eh", strength_units="[(e bohr)^2/Eh]^2; photon prefactors omitted; not absolute cross sections",
        cartesian_order="scattered i, incident j; no symmetrization", closure_exact_a=expected_a, closure_exact_b=expected_b,
        tail_bound_roundoff_budget="64*float64 epsilon added to each raw missing vertex norm squared; not an interval-arithmetic certificate",
        convergence=convergence, grid_convergence=grid_checks, strongest_gamma_laser_mode_indices=strongest,
        strongest_laser_ev=float(axis_ev[strongest[1]]), strongest_mode=int(strongest[2])+1,
        maximum_antisymmetric_norm=float(np.max(antisym)), maximum_total_tensor_norm=float(np.max(norm)),
        maximum_antiresonant_norm=float(np.max(anti_norm)),
        max_time_order_interference_strength=float(np.max(abs(profiles-incoherent))),
        source_sha256=source_hashes, wall_seconds=perf_counter()-started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        python=platform.python_version(), jax=jax.__version__, numpy=np.__version__, backend=jax.default_backend(),
        scope="AD electronic input is reused from X07. This host sum is not full-spectrum AD. No other electronic states, anharmonicity, global parity/tunneling, thermal/rotational effects, or experimental accuracy claim.",
        sources=["https://doi.org/10.1103/PhysRevA.76.021803", "https://arxiv.org/abs/1809.04498"])
    (OUTPUT/"gradscf.json").write_text(json.dumps(report, default=jsonable, indent=2)+"\n")
    print("V07 COMPLETE", report["wall_seconds"], "s; peak RSS", report["peak_rss_kib"], "KiB", flush=True)


if __name__ == "__main__":
    main()

# Measured on jy_gpu_1, CPU float64, using the frozen X11/X12 model:
# total occupation cutoff = 24; intermediate states = 593775
# absorption closure remainder = 1.35388e-7 (relative)
# largest emission closure remainder = 1.80903e-6 (relative)
# N20 -> N24 tensor change / global peak = 2.04200e-7
# uniform omitted-tensor bound / global peak = 1.33908e-7
# 161 -> 321 frequency-point integral changes at Gamma = 0.015/0.030/0.060 eV:
# 2.27521e-5, 4.22897e-6, 1.14623e-5 (all below the declared 1e-4 gate)
# wall time = 355.69 s; peak RSS = 1496436 KiB
# These are raw single-electronic-state model strengths, not cross sections.
