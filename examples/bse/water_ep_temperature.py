"""Water: finite-temperature one-phonon absorption and its temperature JVP.

First run water_ep_coupling.py. Reuse that explicitly frozen electronic
model; only the Bose bath occupations change here. This is not a thermal
multiphonon Franck-Condon calculation or full nuclear GW/BSE response.
"""
from dataclasses import replace
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from scipy.constants import physical_constants
from scipy.integrate import simpson

jax.config.update("jax_enable_x64", True)

from gradscf.bse import ep_coupling as ep
from gradscf.gw.ep_coupling.phonon import bose_occupation


SOURCE = Path("reproducibility/gw_bse/water_bse_phonons")
OUTPUT = Path("reproducibility/ad_showcase/X15_water_bse_temperature")
HARTREE_TO_EV = physical_constants["Hartree energy in eV"][0]
KB = physical_constants["Boltzmann constant in eV/K"][0]/HARTREE_TO_EV
TEMPERATURES = (0., 300., 600., 1200.)
AXIS_EV = np.linspace(4., 18., 28001)
WIDTHS_EV = (.01, .003)


def main():
    started = perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT/"gradscf.json").write_text('{"complete": false}\n')
    data = np.load(SOURCE/"inputs.npz")
    energy, dipoles = (jnp.asarray(data[key]) for key in ("excitation_energies", "dipoles"))
    model = ep.PhononModel(jnp.asarray(data["mode_energies"]), jnp.asarray(data["couplings"]),
        jnp.asarray(data["quadratic"]), reference="Screened RHF vertices; frozen GW correction/BSE kernel")
    models = (replace(model, quadratic=None), model)
    axis = jnp.asarray(AXIS_EV/HARTREE_TO_EV)
    widths = jnp.asarray(WIDTHS_EV)/HARTREE_TO_EV
    calculate = jax.jit(lambda value, beta, width: ep.absorption_cross_section(
        energy, value, dipoles, axis, beta=beta, eta=width, unit="Mb"))
    spectral_trace = jax.jit(lambda value, beta: jnp.trace(ep.spectral_function(
        energy, value, axis, beta=beta, eta=widths[0]), axis1=-2, axis2=-1).real)
    start = perf_counter()
    np.asarray(calculate(model, 1/(KB*600), widths[0]))
    cold_forward = perf_counter()-start
    start = perf_counter()
    np.asarray(calculate(model, 1/(KB*600), widths[0]))
    warm_forward = perf_counter()-start
    curves, occupations, traces = [], [], []
    for temperature in TEMPERATURES:
        beta = 1e8 if temperature == 0 else 1/(KB*temperature)
        occupations.append(np.asarray(bose_occupation(model.energies, beta)))
        curves.append([[np.asarray(calculate(value, beta, width)) for width in widths] for value in models])
        traces.append([np.asarray(spectral_trace(value, beta)) for value in models])
        print("One-phonon bath temperature / K", temperature, flush=True)
    curves = np.array(curves)
    assert np.isfinite(curves).all() and curves.min() >= -1e-12

    # T=0 is the exact underflowed Bose vacuum, not beta=inf (which the API
    # deliberately rejects). One kelvin must recover it at these mode energies.
    cold = np.asarray(calculate(model, 1/KB, widths[0]))
    np.testing.assert_allclose(cold, curves[0, 1, 0], atol=1e-12, rtol=1e-12)
    zero = replace(model, couplings=jnp.zeros_like(model.couplings), quadratic=jnp.zeros_like(model.quadratic))
    zero_curve = np.asarray(calculate(zero, 1/(KB*1200), widths[0]))
    z = np.asarray(axis)[:, None]+1j*float(widths[0])
    bare_alpha = np.einsum("ws,si,sj->wij", 1/(np.asarray(energy)-z)+1/(np.asarray(energy)+z),
                           np.asarray(dipoles), np.asarray(dipoles))
    bare = 4*np.pi*np.asarray(axis)*np.trace(bare_alpha.imag, axis1=-2, axis2=-1)/3/137.035999084*(.529177210903**2*100)
    np.testing.assert_allclose(zero_curve, bare, atol=2e-9, rtol=2e-11)

    def spectrum(temperature):
        return ep.absorption_cross_section(energy, model, dipoles, axis,
            beta=1/(KB*temperature), eta=widths[0], unit="Mb")
    differentiate = jax.jit(lambda t: jax.jvp(spectrum, (t,), (jnp.ones_like(t),))[1])
    start = perf_counter()
    derivative = np.asarray(differentiate(600.))
    cold_derivative = perf_counter()-start
    start = perf_counter()
    np.asarray(differentiate(600.))
    warm_derivative = perf_counter()-start
    checks = []
    for step in (10., 3., 1., .3, .1, .03):
        difference = (np.asarray(calculate(model, 1/(KB*(600+step)), widths[0]))
                     -np.asarray(calculate(model, 1/(KB*(600-step)), widths[0])))/(2*step)
        error = float(np.max(abs(derivative-difference)))
        scaled = float(np.max(abs(derivative-difference)/(2e-7+2e-4*abs(derivative))))
        checks.append(dict(step_kelvin=step, max_error_mb_per_kelvin=error, max_scaled_error=scaled))
        print("Temperature FD", checks[-1], flush=True)
    (OUTPUT/"temperature_fd.json").write_text(json.dumps(checks, indent=2)+"\n")
    assert checks[-1]["max_scaled_error"] < 1.
    integration = []
    for ti, temperature in enumerate(TEMPERATURES):
        for mi, name in enumerate(("fan", "fan_dw")):
            for wi, width in enumerate(WIDTHS_EV):
                curve = curves[ti, mi, wi]
                areas = [float(simpson(curve[::stride], x=AXIS_EV[::stride])) for stride in (4, 2, 1)]
                relative = abs(areas[-1]-areas[-2])/abs(areas[-1])
                integration.append(dict(temperature_kelvin=temperature, model=name, eta_ev=width,
                    samples=[7001, 14001, 28001], integrated_cross_section_mb_ev=areas, last_relative_change=relative))
                assert relative < 2e-3
    np.savez_compressed(OUTPUT/"temperature.npz", photon_energy_ev=AXIS_EV, temperatures_kelvin=TEMPERATURES,
        widths_ev=WIDTHS_EV, cross_sections_mb=curves, spectral_trace_per_hartree=traces,
        temperature_derivative_mb_per_kelvin=derivative,
        zero_coupling_mb=zero_curve, energies_hartree=energy, dipoles=dipoles,
        mode_energies_hartree=model.energies, couplings=model.couplings, quadratic=model.quadratic)
    report = dict(complete=True, model="Frozen RHF/3-21G vertices and three-state TDA-BSE; one-phonon Fan or Fan+DW",
        temperatures_kelvin=TEMPERATURES, widths_ev=WIDTHS_EV, bose_occupations=np.asarray(occupations).tolist(),
        zero_temperature_max_error_mb=float(np.max(abs(cold-curves[0, 1, 0]))),
        zero_coupling_max_error_mb=float(np.max(abs(zero_curve-bare))),
        temperature_derivative_at_kelvin=600., derivative_units="Mb/K", temperature_fd_checks=checks,
        frequency_grid_checks=integration, elapsed_seconds=perf_counter()-started,
        timings_seconds=dict(cold_forward=cold_forward, warm_forward=warm_forward, cold_derivative=cold_derivative,
                             warm_derivative=warm_derivative),
        input_sha256={str(SOURCE/name):hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()
                      for name in ("inputs.npz", "results.json")},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        hostname=platform.node(), jax=jax.__version__, backend=jax.default_backend(), dtype="float64",
        scope="Temperature changes only the fixed Bose bath. No electronic thermal SCF, mode relaxation, thermal multiphonon FC, full GW/BSE geometry response or experimental lifetime.")
    (OUTPUT/"gradscf.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print("Water one-phonon temperature/zero limits/AD gates PASS", checks, flush=True)


if __name__ == "__main__":
    main()

# Recorded on jy_gpu_1, CPU float64, OMP=2 / BLAS=1 (2026-10-16):
# Temperatures 0/300/600/1200 K; Fan and Fan+DW: all gates PASS.
# At 600 K, final temperature-JVP/FD difference: 1.47508e-8 Mb/K (step .03 K).
# The initial 1 K step failed; its error decreases quadratically with refinement.
# Zero-temperature / zero-coupling spectrum errors: 0 / 1.14e-13 Mb.
# Cold/warm forward: 1.2425 / .4184 s; cold/warm JVP: 1.4723 / .7721 s.
# Body: 21.228 s; maximum final frequency-grid integral change: 5.04e-5.
