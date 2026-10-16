"""Independent finite Hermitian bath realization of the one-phonon Fan model.

Eliminating auxiliary blocks H_X +/- omega reproduces its exact rational
self-energy. This is a linear-algebra oracle for that approximation, not a
multiphonon Hamiltonian or an independent electronic-structure calculation.
"""
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.constants import physical_constants
from scipy.integrate import simpson
from scipy.linalg import eigh


OUTPUT = Path("reproducibility/ad_showcase/X15_water_bse_temperature")
EV = physical_constants["Hartree energy in eV"][0]
KB = physical_constants["Boltzmann constant in eV/K"][0]/EV


def main():
    started = perf_counter()
    data = np.load(OUTPUT/"temperature.npz")
    energies, modes, g, quadratic, dipoles = (data[key] for key in
        ("energies_hartree", "mode_energies_hartree", "couplings", "quadratic", "dipoles"))
    n = len(energies)
    axis = data["photon_energy_ev"]/EV
    records = []
    for ti, temperature in enumerate(data["temperatures_kelvin"]):
        beta = 1e8 if temperature == 0 else 1/(KB*temperature)
        thermal = np.exp(-beta*modes)/(-np.expm1(-beta*modes))
        for mi, name in enumerate(("fan", "fan_dw")):
            h = np.diag(energies)
            if mi:
                h += np.einsum("l,lij->ij", thermal+.5, quadratic)
            size = n*(1+2*len(modes))
            matrix = np.zeros((size, size))
            matrix[:n, :n] = h
            block = 1
            for frequency, occupation, vertex in zip(modes, thermal, g):
                for shift, weight in ((frequency, occupation+1), (-frequency, occupation)):
                    section = slice(block*n, (block+1)*n)
                    matrix[section, section] = np.diag(energies+shift)
                    matrix[:n, section] = vertex*np.sqrt(weight)
                    matrix[section, :n] = matrix[:n, section].T
                    block += 1
            levels, vectors = eigh(matrix)
            assert levels.min() > 0
            projected = vectors[:n]
            np.testing.assert_allclose(projected@projected.T, np.eye(n), atol=2e-13, rtol=0)
            moments = projected.T@dipoles
            strengths = np.sum(moments**2, axis=1)
            strength_error = float(abs(strengths.sum()-np.sum(dipoles**2)))
            assert strength_error < 2e-13
            errors = []
            for wi, width in enumerate(data["widths_ev"]):
                eta = width/EV
                z = axis[:, None]+1j*eta
                alpha_trace = ((1/(levels-z)+1/(levels+z))@strengths)/3
                spectrum = 4*np.pi*axis*alpha_trace.imag/137.035999084*(.529177210903**2*100)
                error = float(np.max(abs(spectrum-data["cross_sections_mb"][ti, mi, wi])))
                errors.append(error)
                assert error < 2e-7
            # Trace weights sum to the electronic dimension. A finite-window
            # numerical integral is compared with its analytic Lorentzian area;
            # missing tails remain explicit and no curve is renormalized.
            eta = data["widths_ev"][0]/EV
            weights = np.sum(projected**2, axis=0)
            np.savetxt(OUTPUT/f"sticks_{name}_{temperature:g}K.csv",
                np.column_stack((levels*EV, weights, strengths, (2/3)*levels*strengths)),
                delimiter=",", header="energy_ev,central_trace_weight,dipole_strength_bohr2,oscillator_strength",
                comments="")
            area = float(np.sum(weights*(np.arctan((axis[-1]-levels)/eta)
                                          -np.arctan((axis[0]-levels)/eta)))/np.pi)
            numerical = float(simpson(data["spectral_trace_per_hartree"][ti, mi], x=axis))
            assert abs(numerical-area) < 2e-6
            records.append(dict(temperature_kelvin=float(temperature), model=name,
                auxiliary_dimension=size, minimum_auxiliary_eigenvalue_hartree=float(levels.min()),
                pole_energies_hartree=levels.tolist(), central_trace_weights=weights.tolist(),
                dipole_strengths_bohr2=strengths.tolist(),
                cross_section_errors_mb=errors, dipole_strength_closure_error=strength_error,
                full_trace_weight=float(weights.sum()), analytic_window_trace_weight=area,
                integrated_window_trace_weight=numerical, outside_window_trace_weight=float(weights.sum()-area)))
    report = dict(complete=True, checks=records, elapsed_seconds=perf_counter()-started,
        input_sha256=hashlib.sha256((OUTPUT/"temperature.npz").read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope="Independent SciPy 21-state auxiliary representation of the same fixed one-phonon self-energy; not a thermal multiphonon model.")
    (OUTPUT/"scipy.json").write_text(json.dumps(report, indent=2)+"\n")
    print("Temperature spectra and raw spectral-weight closure PASS", flush=True)


if __name__ == "__main__":
    main()

# Recorded on jy_gpu_1 (2026-10-16):
# All temperatures, both models and both widths: PASS.
# Maximum cross-section difference: 1.005e-9 Mb; dipole closure: 3.34e-16.
# Full central trace weight: 3. Window tails: .002857-.002864, not renormalized.
# Eight raw-stick CSV files retain even zero-residue auxiliary states.
