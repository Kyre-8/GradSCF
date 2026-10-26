"""X15: 0 K spectra from the two validated, matched diagonal vertex models.

The existing harmonic-overlap helper supplies all Franck-Condon amplitudes.
Independent Poisson formulas check raw weights and tails, without renormalizing
them. Frequencies, vertical energies and Condon dipoles are identical in both
models. Common imposed Lorentzian widths display the discrete progressions.
"""
import hashlib
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import gammaln, xlogy
from scipy.stats import poisson

jax.config.update("jax_enable_x64", True)

from gradscf.tools.spectra import HARTREE_TO_EV, lorentzian_spectrum
from gradscf_tools.franck_condon import ground_state_overlaps


output = Path("reproducibility/ad_showcase/X15_water_matched_vertices")
validation = json.loads((output/"finite_differences.json").read_text())
assert validation["complete"] and validation["validation_passed"]
for name, expected in validation["input_sha256"].items():
    assert hashlib.sha256((output/name).read_bytes()).hexdigest() == expected
data = np.load(output/"vertices.npz")
omega, electronic, dipoles = data["omega"], data["excitation_energy"], data["dipoles"]
couplings = np.stack((data["frozen_g"], data["relaxed_g"]))  # model, mode, state
names = ("frozen", "relaxed")
huang_rhys = (couplings/omega[None, :, None])**2
reorganization = np.sum(couplings**2/omega[None, :, None], axis=1)
zero_zero = electronic[None, :]-reorganization
assert np.all(zero_zero > 0), "This local harmonic excited surface falls below the reference ground state."
shifts = np.sqrt(2.)*couplings/omega[None, :, None]**1.5
# The helper uses Q_final=Q_initial+K. The potential minimum is Q_eq=-K.
bare_strengths = (2/3)*electronic*np.sum(dipoles**2, axis=1)
grid = np.linspace(0., 25., 5001)
widths = (.05, .025)


@jax.jit
def broaden_block(energies_ev, strengths, width):
    return lorentzian_spectrum(energies_ev, strengths, jnp.asarray(grid), eta=width)


def broaden(energies, strengths, width):
    """Reuse the existing line shape with at most 512 lines in a grid block."""
    energies, strengths = np.asarray(energies).ravel()*HARTREE_TO_EV, np.asarray(strengths).ravel()
    curve = np.zeros_like(grid)
    for start in range(0, len(energies), 512):
        count = min(512, len(energies)-start)
        e = np.pad(energies[start:start+count], (0, 512-count))
        f = np.pad(strengths[start:start+count], (0, 512-count))
        curve += np.asarray(broaden_block(jnp.asarray(e), jnp.asarray(f), width))
    return curve


# Zero coupling returns precisely the electronic sticks, with no sidebands.
zero_occupations, zero_amplitudes = ground_state_overlaps(
    omega, omega, np.eye(3), np.zeros(3), max_quanta=4)
np.testing.assert_allclose(zero_amplitudes[0], 1., atol=2e-13)
np.testing.assert_allclose(zero_amplitudes[1:], 0., atol=2e-13)
bare_curves = np.asarray([broaden(electronic, bare_strengths, width) for width in widths])
zero_lines = electronic[:, None]+zero_occupations @ omega
zero_strengths = (2/3)*zero_lines*zero_amplitudes[None, :]**2*np.sum(dipoles**2, axis=1)[:, None]
np.testing.assert_allclose(broaden(zero_lines, zero_strengths, widths[0]), bare_curves[0], atol=2e-12)

start_time = perf_counter()
history, previous = [], None
for cutoff in (8, 16, 24, 32, 48, 64, 96, 128):
    probabilities, line_energies, strengths, model_checks = [], [], [], []
    for model, name in enumerate(names):
        model_p, model_e, model_f, state_checks = [], [], [], []
        for state in range(len(electronic)):
            occupations, amplitude = ground_state_overlaps(
                omega, omega, np.eye(3), shifts[model, :, state], max_quanta=cutoff)
            probability = amplitude**2
            s = huang_rhys[model, :, state]
            independent = np.exp(-s.sum()+np.sum(xlogy(occupations, s)-gammaln(occupations+1), axis=1))
            np.testing.assert_allclose(probability, independent, atol=3e-12, rtol=3e-10)
            for mode in range(3):
                one = np.flatnonzero(np.all(occupations == np.eye(3, dtype=int)[mode], axis=1))[0]
                np.testing.assert_allclose(amplitude[one],
                    couplings[model, mode, state]/omega[mode]*amplitude[0], atol=2e-12)
            tail = float(poisson.sf(cutoff, s.sum()))
            np.testing.assert_allclose(probability.sum(), 1-tail, atol=3e-12, rtol=0)
            energy = zero_zero[model, state]+occupations @ omega
            strength = (2/3)*energy*probability*np.sum(dipoles[state]**2)
            partial_moment = electronic[state]*(1-tail)-reorganization[model, state]*poisson.pmf(cutoff, s.sum())
            expected_strength = (2/3)*np.sum(dipoles[state]**2)*partial_moment
            np.testing.assert_allclose(strength.sum(), expected_strength, atol=3e-12, rtol=3e-10)
            state_checks.append(dict(state=state, retained_probability=float(probability.sum()),
                raw_missing_probability=float(1-probability.sum()), poisson_tail=tail,
                max_poisson_weight_error=float(np.max(np.abs(probability-independent))),
                oscillator_strength=float(strength.sum()), expected_partial_strength=float(expected_strength)))
            model_p.append(probability)
            model_e.append(energy)
            model_f.append(strength)
        probabilities.append(model_p)
        line_energies.append(model_e)
        strengths.append(model_f)
        model_checks.append(dict(model=name, states=state_checks))
    probabilities, line_energies, strengths = map(np.asarray, (probabilities, line_energies, strengths))
    curves = np.asarray([[broaden(line_energies[model], strengths[model], width)
                          for width in widths] for model in range(2)])
    change = None if previous is None else np.sum(np.abs(curves-previous), axis=2)/np.sum(np.abs(curves), axis=2)
    maximum_tail = max(state["poisson_tail"] for item in model_checks for state in item["states"])
    history.append(dict(max_quanta=cutoff, lines_per_state=len(occupations), models=model_checks,
        relative_l1_change=None if change is None else change.tolist(), maximum_poisson_tail=maximum_tail))
    print("FC cutoff", cutoff, "max raw Poisson tail", maximum_tail,
          "largest curve change", None if change is None else np.max(change), flush=True)
    if previous is not None and maximum_tail < 1e-10 and np.max(change) < 1e-6:
        break
    previous = curves
else:
    raise RuntimeError("Increase the physical FC cutoff; no retained-weight renormalization is allowed.")

# Compare numerical finite-window areas with the exact Lorentzian integrals.
area_checks = []
for model, name in enumerate(names):
    e, f = line_energies[model].ravel()*HARTREE_TO_EV, strengths[model].ravel()
    for index, width in enumerate(widths):
        exact = np.sum(f*(np.arctan((grid[-1]-e)/width)-np.arctan((grid[0]-e)/width))/np.pi)
        numeric = np.trapezoid(curves[model, index], grid)
        assert abs(numeric-exact) < 1e-8
        area_checks.append(dict(model=name, hwhm_ev=width, finite_window_area=float(exact),
            numerical_area_error=float(abs(numeric-exact)), outside_window_strength=float(f.sum()-exact)))

np.savez_compressed(output/"diagonal_vibronic.npz", occupations=occupations,
    line_energies_hartree=line_energies, probabilities=probabilities, oscillator_strengths=strengths,
    energy_ev=grid, curves=curves, bare_curves=bare_curves, widths_ev=widths)
np.savetxt(output/"diagonal_spectra.csv", np.column_stack((grid, bare_curves[0], curves[0, 0], curves[1, 0])),
    delimiter=",", comments="", header="energy_eV,bare_strength_per_eV,frozen_strength_per_eV,relaxed_strength_per_eV")
report = dict(validation_passed=True, temperature_kelvin=0, diagonal_only=True,
    models=names, max_quanta=cutoff, huang_rhys=huang_rhys.tolist(),
    reorganization_hartree=reorganization.tolist(), zero_zero_hartree=zero_zero.tolist(),
    helper_shift_K=shifts.tolist(), equilibrium_Q=(-shifts).tolist(),
    bare_oscillator_strengths=bare_strengths.tolist(), cutoffs=history, area_checks=area_checks,
    elapsed_seconds=perf_counter()-start_time, hwhm_ev=widths,
    input_sha256={name: hashlib.sha256((output/name).read_bytes()).hexdigest()
                  for name in ("vertices.npz", "finite_differences.json")},
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    scope="Matched local diagonal, equal-frequency, zero-temperature Condon surfaces; host FC enumeration; imposed line smoothing, no lifetime prediction.")
(output/"diagonal_vibronic.json").write_text(json.dumps(report, indent=2)+"\n")

# Measured total-quanta cutoffs 8 / 16 / 24 / 32:
# max raw Poisson tails 3.316e-2 / 3.271e-6 / 8.579e-12 / 1.832e-18.
# Largest final spectrum change at either width: 1.101e-11.
# Maximum finite-window area integration error: 9.592e-12.
# Program wall time 6.78 s; FC/cutoff loop 0.929 s. No probabilities rescaled.
