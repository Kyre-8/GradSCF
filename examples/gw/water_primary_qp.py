"""Water: native main-peak QP seeds, finite-eta G0W0, and complete static BSE.

The eta=0 arrowhead problem supplies numerical starting points only. Actual
energies and residues come from the finite-eta contour-deformation equation.
All 19 QP/G orbitals and 70 screening/optical transitions are retained. No
external reference enters the calculation. Compare saved output separately
with compare_water_molgw.py; nuclear derivatives use water_geometry_response.py.
"""
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from gradscf import bse
from gradscf.solvers import LinearSolverConfig
from gradscf.tools.spectra import HARTREE_TO_EV
from gradscf_tools.gw_response import MolecularGW
from water_geometry_response import COORDINATES, SYMBOLS, water_reference


OUTPUT = Path("reproducibility/ad_showcase/X02_water_gw_bse_response/primary")


def run(*, nw=200):
    output = OUTPUT/f"nw{nw}"
    output.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    mf = water_reference()
    hf_seconds = perf_counter()-start
    selector = MolecularGW(mf, nw=nw, eta=1e-5, qp_solver="newton")
    start = perf_counter()
    seeds = selector.pole_seeds(max_dense=2048)
    seed_seconds = perf_counter()-start
    assert seeds.valid.all(), "Ambiguous or nonconverged seed groups must not be continued."

    model = MolecularGW(mf, nw=nw, eta=1e-5, qp_solver="newton", qp_initial=seeds.seeds)
    forward = jax.jit(model)
    start = perf_counter()
    qp = forward(model.coordinates)
    np.asarray(qp.mo_energy)
    qp_seconds = perf_counter()-start
    assert bool(qp.converged) and np.asarray(qp.converged_mask).all()
    assert np.isfinite(np.asarray(qp.qp_weight)).all()
    record = dict(method="g0w0", basis="6-31g*", cart=True, symbols=SYMBOLS,
        coordinates_bohr=COORDINATES.tolist(), nmo=model.nmo, nocc=model.nocc,
        hf_energy_hartree=float(mf.e_tot), hf_orbital_hartree=np.asarray(mf.mo_energy).tolist(),
        qp_orbitals=list(range(model.nmo)), g_orbitals=list(range(model.nmo)),
        screening_occupied=list(range(model.nocc)), screening_virtual=list(range(model.nocc, model.nmo)),
        nw=nw, eta_gw_hartree=1e-5, qp_tolerance_hartree=1e-11, qp_solver="newton",
        factorization="symmetric AO-pair square root; no auxiliary basis; tolerance 1e-12",
        eta0_seed_hartree=seeds.seeds.tolist(), eta0_seed_weight=seeds.weights.tolist(),
        eta0_weight_gap=seeds.weight_gap.tolist(), eta0_bath_pole_gap_hartree=seeds.pole_gap.tolist(),
        eta0_eigensolver_residual_hartree=seeds.residual_norms.tolist(),
        qp_energy_hartree=np.asarray(qp.mo_energy).tolist(), qp_weight=np.asarray(qp.qp_weight).tolist(),
        qp_residual_hartree=np.asarray(qp.qp_residual).tolist(),
        screening_energy_hartree=np.asarray(qp.screening_energy).tolist(),
        timings_seconds=dict(hf=hf_seconds, seeds=seed_seconds, qp_cold=qp_seconds), optical=[],
        hostname=platform.node(), python=platform.python_version(), jax=jax.__version__,
        backend=jax.default_backend(), dtype="float64", hartree_to_ev=HARTREE_TO_EV,
        scope="Eta=0 main-weight seeds followed by finite-eta roots, not a proof of finite-eta global maximum residue.")
    # An imposed width illustrates discrete electronic transitions. It is not
    # a calculated vibrational, environmental or electronic lifetime.
    energy_ev = np.linspace(0., 50., 1000)
    for tda in (True, False):
        label = "tda" if tda else "full"
        configuration = bse.BSEConfig(nroots=model.nocc*(model.nmo-model.nocc),
            tda=tda, solver="dense", conv_tol=1e-10,
            response_config=LinearSolverConfig(method="direct", rtol=1e-10, max_dense=256))
        start = perf_counter()
        repeated_qp, excited, dipole = jax.jit(
            lambda r: model.optical(r, config=configuration)[:3])(model.coordinates)
        space = bse.make_bse_space(model.nmo, model.nocc)
        np.asarray(excited.excitation_energies)
        elapsed = perf_counter()-start
        np.testing.assert_allclose(repeated_qp.mo_energy, qp.mo_energy, atol=1e-12, rtol=0)
        assert np.asarray(excited.converged).all() and np.asarray(excited.stable).all()
        strength = np.asarray(bse.oscillator_strengths(excited, dipole, space))
        alpha = np.asarray(bse.polarizability(excited, dipole, space)).real
        sigma = np.asarray(bse.absorption_cross_section(excited, dipole, space,
                           energy_ev/HARTREE_TO_EV, eta=.01, unit="au"))
        assert np.isfinite(sigma).all() and np.min(sigma) >= 0
        arrays = {field.name: np.asarray(getattr(excited, field.name))
                  for field in fields(excited) if getattr(excited, field.name) is not None}
        np.savez_compressed(output/f"{label}_response.npz", **arrays, dipole_mo=dipole)
        np.savetxt(output/f"{label}_spectrum.csv", np.column_stack((energy_ev, sigma)),
                   delimiter=",", header="energy_eV,cross_section_bohr2", comments="")
        record["optical"].append(dict(tda=tda, singlet=True, nroots=configuration.nroots,
            excitation_hartree=np.asarray(excited.excitation_energies).tolist(),
            oscillator_strength=strength.tolist(), static_polarizability_au=alpha.tolist(),
            residual_hartree=np.asarray(excited.residual_norms).tolist(),
            response_valid=np.asarray(excited.response_valid).tolist(),
            imposed_hwhm_hartree=.01, cold_gw_bse_seconds=elapsed))
        print(label, "lowest excitation / eV", float(excited.excitation_energies[0])*HARTREE_TO_EV,
              "max residual / Eh", float(np.max(excited.residual_norms)), flush=True)

    paths = [Path("src/gradscf_tools/gw_response.py"), Path("src/gradscf_tools/molecular_response.py"),
             Path("src/gradscf/integrals/molecular/factorization.py"), Path(__file__)]
    for module in ("gw", "bse", "solvers/eigen", "solvers/nonlinear", "solvers/linear"):
        paths.extend(Path("src/gradscf", module).glob("*.py"))
    record["source_sha256"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in sorted(set(paths))}
    (output/"native.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
    print("Saved", output, "max QP residual / Eh", np.max(abs(np.asarray(qp.qp_residual))), flush=True)
    return record


if __name__ == "__main__":
    run()

# Measured on jy_gpu_1, CPU float64, 200 imaginary-axis points:
# RHF energy: -76.010492818245 Eh
# Maximum residual of all 19 refined QP roots: 6.25e-16 Eh
#                         TDA                  full BSE
# Lowest excitation/eV    8.9072377826         8.8696124579
# Max 70-root residual/Eh 6.92e-14             2.15e-13
# MolGW energies/optics passed the separate original gates. Its first
# 5e-6-Eh graphical-grid residue comparison failed (max dZ=1.65e-4);
# that reference precision check is retained separately from native convergence.
