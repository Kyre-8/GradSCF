"""Fixed-geometry formaldehyde singlet absorption, using native GradSCF only.

C=O 1.21 Angstrom, C-H 1.10 Angstrom, H-C-H 116 degrees; this is a chosen
C2v geometry, not an optimized or experimental structure. Cartesian 6-31+G*
includes diffuse functions but basis/grid convergence is not established.
HF/TDA is singlet CIS; HF/full TD is TDHF. No CLI or PySCF calculation.
omegaB97X-D includes both exact-exchange ranges and original CHG dispersion;
the geometry-only CHG term does not shift vertical excitation energies.
Run on jy_gpu_1 from the repository root with PYTHONPATH=src and CPU float64.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
from time import perf_counter

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from gradscf import dft, gto, scf, tdscf
from gradscf.tools.spectra import HARTREE_TO_EV


functional_names = ("hf", "pbe", "pbe0", "b3lyp", "wb97xd")
basis = "6-31+g*"
nstates = 6
grid_level = 3
ch = 1.10
half_angle = np.deg2rad(58.0)
atom = [("C", (0., 0., 0.)), ("O", (0., 0., 1.21)),
        ("H", (0., ch*np.sin(half_angle), -ch*np.cos(half_angle))),
        ("H", (0., -ch*np.sin(half_angle), -ch*np.cos(half_angle)))]
output = Path("reproducibility/ad_showcase/X01_formaldehyde_absorption")
output.mkdir(parents=True, exist_ok=True)
root = Path(__file__).resolve().parents[2]
revision_file = root / "SOURCE_REVISION"
revision = (revision_file.read_text().strip() if revision_file.exists() else
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip())
source = hashlib.sha256()
for path in sorted((root / "src").rglob("*.py")):
    source.update(path.relative_to(root).as_posix().encode())
    source.update(path.read_bytes())
cpu_info = Path("/proc/cpuinfo")
cpu_model = (next(line.split(":", 1)[1].strip() for line in cpu_info.read_text().splitlines()
                  if line.startswith("model name")) if cpu_info.exists() else platform.processor())
report = dict(
    system="formaldehyde", charge=0, spin=0, atom_angstrom=atom,
    geometry="chosen C2v: CO=1.21 Angstrom, CH=1.10 Angstrom, HCH=116 degrees",
    basis=basis, cart=True, nstates=nstates, functional_names=list(functional_names),
    grid_level=grid_level, grid_radial="Treutler-Ahlrichs",
    grid_angular="Lebedev / NWChem pruning",
    grid_partition="original Becke / Treutler radii adjustment",
    integral_backend="native", jk_backend="full", grid_ao_backend="jax",
    scf_energy_tolerance=1e-12, scf_density_tolerance=1e-10,
    scf_gradient_tolerance=1e-9, scf_max_cycle=180,
    td_tolerance=1e-9, td_max_cycle=150, td_max_subspace=80, td_initial_guess_count=32,
    hartree_to_ev=HARTREE_TO_EV, transition_dipole_units="Bohr",
    oscillator_strength_units="dimensionless", dtype="float64", backend="cpu",
    source_revision=revision, source_python_sha256=source.hexdigest(),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    hostname=platform.node(), cpu=cpu_model, machine=platform.machine(),
    python=platform.python_version(), jax=jax.__version__, numpy=np.__version__,
    devices=[str(device) for device in jax.devices()],
    threads={key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
    scope="Vertical clamped-nuclei first-six-singlet spectra; no basis/grid convergence, experimental comparison or derivative claim.",
    assignment_convention="Top normalized (X^2+Y^2) contributions; MO indices are one-based. Root indices are local to each method, not cross-method state tracking.",
    complete=False, cases=[],
    wb97xd_definition="Original fixed omega=0.2 bohr^-1, SR HF=.222036, LR HF=1, local LibXC471 and original CHG dispersion; no D3/D4 or excitation shift.",
)
started = perf_counter()
(output / "gradscf.json").write_text(json.dumps(report, indent=2)+"\n")

for functional in functional_names:
    mol = gto.M(atom=atom, basis=basis, unit="Angstrom", cart=True, charge=0, spin=0)
    settings = dict(grids_level=grid_level, conv_tol=1e-12, conv_tol_density=1e-10,
                    conv_tol_grad=1e-9, max_cycle=180, integral_backend="native",
                    grid_ao_backend="jax", execution_device="cpu")
    mf = scf.RHF(mol, **settings) if functional == "hf" else dft.RKS(mol, xc=functional, **settings)
    start = perf_counter()
    mf.run()
    scf_seconds = perf_counter()-start
    if not mf.converged:
        raise RuntimeError(f"{functional} SCF did not converge; no spectrum accepted.")
    # The TD adapter owns complete RSH response preparation and reuses SCF.
    reference = tdscf.TDA(mf).molecule
    dispersion = getattr(mf.scf_result, "dispersion_energy", None)
    dispersion = 0. if dispersion is None else float(dispersion)
    electronic = float(mf.scf_result.electronic_energy)
    nuclear = float(mf.scf_result.nuclear_repulsion)
    np.testing.assert_allclose(mf.e_tot, electronic+nuclear+dispersion, atol=1e-10, rtol=0)
    nocc = int(np.count_nonzero(np.asarray(mf.mo_occ) > 1e-8))
    print(f"{functional:6s} SCF: {float(mf.e_tot):.12f} Eh, {scf_seconds:.2f} s", flush=True)

    for label, factory in (("TDA", tdscf.TDA), ("TDDFT", tdscf.TDDFT)):
        td = factory(mf, nstates=nstates, eigensolver="davidson", davidson_tol=1e-9,
                     davidson_max_iter=150, davidson_max_subspace=80,
                     davidson_initial_guess_count=32)
        start = perf_counter()
        result = td.kernel()
        energies = np.asarray(td.e)
        first_seconds = perf_counter()-start
        if not bool(np.all(np.asarray(result.converged))):
            raise RuntimeError(f"{functional}/{label} first solve did not converge.")
        # This repeats the same public solve; it does not reuse stored eigenpairs.
        start = perf_counter()
        result = td.kernel()
        warm_energies = np.asarray(td.e)
        warm_seconds = perf_counter()-start
        if not bool(np.all(np.asarray(result.converged))):
            raise RuntimeError(f"{functional}/{label} warm solve did not converge.")
        np.testing.assert_allclose(warm_energies, energies, atol=1e-9, rtol=0)
        dipoles = np.asarray(td.transition_dipole())
        strengths = np.asarray(td.oscillator_strength())
        x, y = ((np.asarray(td.xy), np.zeros_like(np.asarray(td.xy))) if label == "TDA"
                else tuple(np.asarray(value) for value in td.xy))
        norm = 2*np.sum(x*x-y*y, axis=(1, 2))
        np.testing.assert_allclose(norm, 1., atol=2e-7, rtol=0)
        if energies.shape != (nstates,) or not np.isfinite(energies).all() or not np.all(energies > 0):
            raise RuntimeError(f"Invalid excitation energies for {functional}/{label}.")
        if not np.isfinite(dipoles).all() or not np.isfinite(strengths).all() or np.any(strengths < 0):
            raise RuntimeError(f"Invalid optical observables for {functional}/{label}.")
        states = []
        for state, energy in enumerate(energies):
            weights = x[state]**2+y[state]**2
            weights /= weights.sum()
            configurations = []
            for index in np.argsort(weights.ravel())[-3:][::-1]:
                occupied, virtual = np.unravel_index(index, weights.shape)
                hole = "HOMO" if occupied == nocc-1 else f"HOMO-{nocc-1-occupied}"
                particle = "LUMO" if virtual == 0 else f"LUMO+{virtual}"
                configurations.append(dict(occupied_mo=int(occupied+1), virtual_mo=int(nocc+virtual+1),
                                           label=f"{hole} -> {particle}", weight=float(weights[occupied, virtual])))
            states.append(dict(root=state+1, excitation_hartree=float(energy),
                               excitation_ev=float(energy*HARTREE_TO_EV),
                               transition_dipole_bohr=dipoles[state].tolist(),
                               dipole_outer_bohr2=np.outer(dipoles[state], dipoles[state]).tolist(),
                               oscillator_strength=float(strengths[state]), xy_norm=float(norm[state]),
                               dominant_configurations=configurations))
        case = dict(functional=functional, response=label, scf_converged=True, td_converged=True,
                    ground_energy_hartree=float(mf.e_tot), scf_cycles=int(mf.cycles), nocc=nocc,
                    electronic_energy_hartree=electronic, nuclear_repulsion_hartree=nuclear,
                    dispersion_energy_hartree=dispersion,
                    nao=int(np.shape(mf.mo_coeff)[0]), grid_points=int(len(reference.grid.weights)),
                    mo_energy_hartree=np.asarray(mf.mo_energy).tolist(),
                    scf_first_seconds=scf_seconds, td_first_seconds=first_seconds,
                    td_warm_seconds=warm_seconds, states=states)
        report["cases"].append(case)
        report["elapsed_seconds"] = perf_counter()-started
        report["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["complete"] = len(report["cases"]) == 2*len(functional_names)
        (output / "gradscf.json").write_text(json.dumps(report, indent=2)+"\n")
        print(f"{functional:6s} {label:5s}: first/warm {first_seconds:.2f}/{warm_seconds:.2f} s", flush=True)
        for state in states:
            main = state["dominant_configurations"][0]
            print(f"  S{state['root']} {state['excitation_ev']:10.6f} eV f={state['oscillator_strength']:.8f}  "
                  f"{main['label']} ({main['weight']:.3f})", flush=True)

with (output / "sticks.csv").open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["functional", "response", "root", "excitation_hartree", "excitation_ev",
                     "oscillator_strength", "mu_x_bohr", "mu_y_bohr", "mu_z_bohr", "dominant_configuration", "weight"])
    for case in report["cases"]:
        for state in case["states"]:
            main = state["dominant_configurations"][0]
            writer.writerow([case["functional"], case["response"], state["root"], state["excitation_hartree"],
                             state["excitation_ev"], state["oscillator_strength"], *state["transition_dipole_bohr"],
                             main["label"], main["weight"]])
print(f"Saved {len(report['cases'])*nstates} converged singlet transitions; elapsed {report['elapsed_seconds']:.2f} s", flush=True)

# Measured on jy_gpu_1, CPU float64, 2026-10-15; 32 initial vectors:
# hf     SCF: -113.869913959172 Eh
# pbe    SCF: -114.375353292335 Eh
# pbe0   SCF: -114.374748797564 Eh
# b3lyp  SCF: -114.508753082088 Eh
# wb97xd SCF: -114.466945076331 Eh (includes original CHG).
# Full-TD S1 / eV: HF 4.410553; PBE 3.863236; PBE0 3.979480;
# B3LYP 3.978222; omegaB97X-D 3.995408.
# Saved 60 converged singlet transitions; elapsed 202.87 s.
# First-call and warm timings for every response are saved in gradscf.json.
