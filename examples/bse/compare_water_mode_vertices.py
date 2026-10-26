"""Fully reconverged finite differences for both named X15 vertex models.

GW/BSE is recalculated for the relaxed model. The frozen model recalculates HF,
transports its Fock operator into the stored QP MO frame, and applies the stored
X0 projection. Comparing either derivative with the other model's displacement
would test a different physical approximation.
"""
import hashlib
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from gradscf import bse
from gradscf_tools.excited import transition_density_overlaps
from water_mode_vertices import OUTPUT, load_modes, reference, configuration, gw_model


data, coordinates, modes, omega = load_modes()
stored = np.load(OUTPUT/"vertices.npz")
record = json.loads((OUTPUT/"gradscf.json").read_text())
mf, engine, hf, _ = reference(data, coordinates)
model = gw_model(mf, jnp.asarray(stored["qp_energy"]))
cfg = configuration()
assert record["nstates"] == cfg.nroots and record["nw"] == model.nw
assert record["eta_hartree"] == model.eta and record["qp_tolerance_hartree"] == model.qp_tol
for name, expected in (("coordinates", coordinates), ("modes", modes), ("omega", omega)):
    np.testing.assert_array_equal(stored[name], expected)
forward = jax.jit(lambda r: model.optical(r, config=cfg)[:3])
space = bse.make_bse_space(19, 5)
c0, x0, f0 = stored["qp_coeff"], stored["amplitudes"], stored["hf_fock"]
e0 = stored["excitation_energy"]
qp0, bse0, _ = forward(jnp.asarray(coordinates))
np.testing.assert_allclose(qp0.mo_energy, stored["qp_energy"], atol=1e-10, rtol=0)
np.testing.assert_allclose(bse0.excitation_energies, e0, atol=1e-10, rtol=0)


def displaced(vector, step):
    point = coordinates+step*vector
    qp, excited, _ = forward(jnp.asarray(point))
    assert bool(qp.converged) and np.asarray(qp.converged_mask & qp.qp_computed_mask).all()
    assert np.asarray(excited.converged & excited.stable & excited.response_valid).all()
    assert bool(excited.screening_valid)
    cross = np.asarray(engine.cross_overlap(coordinates, point))
    c = np.asarray(qp.mo_coeff)
    scores = np.stack([transition_density_overlaps(c0, amplitude, c,
        np.asarray(excited.x_amplitudes), cross, nocc=5) for amplitude in x0])
    assert np.array_equal(np.argmax(scores, axis=1), np.arange(len(e0)))
    assert np.min(np.diag(scores)) > .99
    assert np.min(np.abs(np.diag(c0.T @ cross @ c))) > .99

    moving_hf, _ = engine.evaluate(point, init_density=hf.density_matrix, gradient=False)
    fock, minimum_overlap = engine.transport(moving_hf, c0, coordinates, point)
    change = bse.ep_coupling.project_couplings(x0, space, (fock-f0)[None])[0]
    frozen_energy = e0+np.diag(np.asarray(change))
    return dict(qp_energy=np.asarray(qp.mo_energy), relaxed=np.asarray(excited.excitation_energies),
                frozen=frozen_energy, fock=fock,
                qp_residual=float(np.max(np.abs(qp.qp_residual))),
                bse_residual=float(np.max(excited.residual_norms)),
                minimum_orbital_overlap=float(np.min(np.abs(np.diag(c0.T @ cross @ c)))),
                minimum_transition_overlap=float(np.min(np.diag(scores))),
                minimum_transport_overlap=minimum_overlap)


start = perf_counter()
checks = []
for mode in range(3):
    for step in (.05, .02, .01, .005, .002):
        plus, minus = (displaced(modes[:, :, mode], sign*step) for sign in (1., -1.))
        numerical = {name: (plus[name]-minus[name])/(2*step)
                     for name in ("qp_energy", "relaxed", "frozen", "fock")}
        errors = dict(
            qp=float(np.max(np.abs(numerical["qp_energy"]-stored["qp_slopes"][mode]))),
            relaxed=float(np.max(np.abs(numerical["relaxed"]-stored["relaxed_slopes"][mode]))),
            frozen=float(np.max(np.abs(numerical["frozen"]-stored["frozen_slopes"][mode]))),
            transported_fock=float(np.max(np.abs(numerical["fock"]-stored["frozen_fock_slopes"][mode]))))
        diagnostics = {name: [float(plus[name]), float(minus[name])] for name in (
            "qp_residual", "bse_residual", "minimum_orbital_overlap", "minimum_transition_overlap",
            "minimum_transport_overlap")}
        checks.append(dict(mode=mode, step_Q=step,
            maximum_cartesian_displacement_bohr=float(step*np.max(np.abs(modes[:, :, mode]))),
            errors=errors, derivative_relaxed=numerical["relaxed"].tolist(),
            derivative_frozen=numerical["frozen"].tolist(),
            plus_excitation=plus["relaxed"].tolist(), minus_excitation=minus["relaxed"].tolist(),
            diagnostics=diagnostics))
        print(f"mode={mode} dQ={step:.3g}: {errors}", flush=True)
        (OUTPUT/"finite_differences.json").write_text(json.dumps(dict(
            validation_passed=False, complete=False, records=checks), indent=2)+"\n")

final = [checks[5*mode+4] for mode in range(3)]
passed = all(row["errors"]["relaxed"] < 2e-7 and row["errors"]["frozen"] < 2e-7
             and row["errors"]["qp"] < 2e-6 and row["errors"]["transported_fock"] < 2e-6
             for row in final)
result = dict(validation_passed=passed, complete=True, steps_are_mass_weighted=True, records=checks,
    sweep_seconds=perf_counter()-start, vertex_source="gradscf.json",
    input_sha256={name: hashlib.sha256((OUTPUT/name).read_bytes()).hexdigest()
                  for name in ("vertices.npz", "gradscf.json")},
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    scope="Independent reconverged finite differences of each named approximation; not an external GW nuclear-gradient benchmark.")
(OUTPUT/"finite_differences.json").write_text(json.dumps(result, indent=2)+"\n")
assert passed, "Refine or diagnose the displaced branches before constructing a vibronic spectrum."

# Measured maximum errors at dQ=.002, in Eh/Q:
# Relaxed TDA 4.558e-12; frozen projection 6.701e-11;
# transported HF Fock 1.391e-10; all 19 QP energies 5.831e-9.
# All 15 positive/negative displacement comparisons passed. Wall time 49.54 s.
