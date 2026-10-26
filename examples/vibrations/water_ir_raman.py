"""V01/V02: harmonic water IR strengths and nonresonant static Raman activities.

Run the matching HF/PBE optimization and Hessian examples first. JVPs differentiate
the relaxed dipole and polarizability along each normal mode. The stationary
dipole identity retains the density's implicit field and geometry response.
The finite displacements below validate AD; they do not generate the spectra.
"""
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from scipy.constants import physical_constants, Avogadro, alpha as fine_structure

jax.config.update("jax_enable_x64", True)

from gradscf import dft, gto, scf
from gradscf_tools.molecular_response import MolecularEnergy
from gradscf_tools.vibrations import AMU_TO_ELECTRON_MASS


METHODS = ("HF", "PBE")
RESULTS = Path("reproducibility/ad_showcase")
OUTPUTS = {"HF": RESULTS / "V01_V02_water_rhf_spectra",
           "PBE": RESULTS / "V01_V02_water_pbe_spectra"}


def run(method, output=None):
    """Run a method at its own optimized geometry; output may isolate a check."""
    method = method.upper()
    if method not in METHODS:
        raise ValueError("Choose HF or PBE.")
    tag = "rhf" if method == "HF" else "pbe"
    harmonic = json.loads((RESULTS / f"G05_water_{tag}_hessian/gradscf.json").read_text())
    mol = gto.M(atom=list(zip(harmonic["symbols"], harmonic["coordinates_bohr"])),
                basis=harmonic["basis"], unit="Bohr", cart=True)
    if method == "HF":
        mf = scf.RHF(mol, conv_tol=1e-12, conv_tol_density=1e-11,
                     integral_backend="native").run()
    else:
        mf = dft.RKS(mol, xc="pbe", grids_level=harmonic["grid_level"],
                     conv_tol=1e-12, conv_tol_density=1e-11, conv_tol_grad=1e-10,
                     max_cycle=150, integral_backend="native").run()
    energy = MolecularEnergy(mf)
    output = OUTPUTS[method] if output is None else Path(output)
    return calculate_spectrum(energy, harmonic, output,
                              source_input=RESULTS / f"G05_water_{tag}_hessian/gradscf.json")


def calculate_spectrum(energy, harmonic, output, *, source_input):
    """Shared relaxed IR/Raman physics for a supplied energy and water Hessian."""
    method = harmonic.get("xc", "HF").upper()
    label = harmonic.get("label", method)
    coordinates, field = energy.coordinates, jnp.zeros(3)
    # energy.dipole supplies the complete relaxed dipole: the stationary
    # density identity for HF/DFT, or -dE_total/dF for a correlated model.
    dipole = lambda r: energy.dipole(r, field)
    alpha = lambda r: jax.jacfwd(energy.dipole, argnums=1)(r, field)
    dipole_jvp = jax.jit(lambda r, v: jax.jvp(dipole, (r,), (v,))[1])
    alpha_jvp = jax.jit(lambda r, v: jax.jvp(alpha, (r,), (v,))[1])
    calculate_dipole, calculate_alpha = jax.jit(dipole), jax.jit(alpha)
    directions = np.moveaxis(np.array(harmonic["isotopes"]["H2O"]["displacements"]), -1, 0)
    frequencies = harmonic["isotopes"]["H2O"]["frequencies_cm1"]
    mu_derivatives, alpha_derivatives, checks = [], [], []
    start = perf_counter()
    for index, direction in enumerate(directions):
        print(f"{label}: mode {index+1}, computing complete IR/Raman AD", flush=True)
        dmu = np.asarray(dipole_jvp(coordinates, jnp.asarray(direction)))
        dalpha = np.asarray(alpha_jvp(coordinates, jnp.asarray(direction)))
        if not np.isfinite(dmu).all() or not np.isfinite(dalpha).all():
            raise RuntimeError(f"Mode {index+1} electronic response is nonfinite.")
        np.testing.assert_allclose(dalpha, dalpha.T, atol=1e-9, rtol=0)
        mu_derivatives.append(dmu)
        alpha_derivatives.append(dalpha)
        for cartesian_step in (1e-3, 3e-4):
            step = cartesian_step / np.max(np.abs(direction))
            plus, minus = coordinates+step*direction, coordinates-step*direction
            mu_fd = (calculate_dipole(plus)-calculate_dipole(minus))/(2*step)
            alpha_fd = (calculate_alpha(plus)-calculate_alpha(minus))/(2*step)
            if not np.isfinite(mu_fd).all() or not np.isfinite(alpha_fd).all():
                raise RuntimeError(f"Mode {index+1} displaced response did not converge.")
            checks.append(dict(mode=index+1, max_cartesian_step_bohr=cartesian_step,
                               normal_coordinate_step=step,
                               max_dipole_derivative_error=float(np.max(np.abs(dmu-mu_fd))),
                               max_alpha_derivative_error=float(np.max(np.abs(dalpha-alpha_fd)))))
        np.testing.assert_allclose(dmu, mu_fd, atol=1e-6, rtol=1e-4)
        np.testing.assert_allclose(dalpha, alpha_fd, atol=3e-5, rtol=3e-4)
        print(f"Mode {index+1}: dmu/dQ={dmu}; max Raman FD error="
              f"{checks[-1]['max_alpha_derivative_error']:.3e}", flush=True)

    mu_derivatives, alpha_derivatives = np.array(mu_derivatives), np.array(alpha_derivatives)
    # Q is mass-weighted with masses in electron masses. The IR conversion is
    # N_A*pi*alpha_fs^2*a0/3000; see Psi4 driver/qcdb/vib.py's amu convention,
    # with its electron-mass-in-u factor removed for this electron-mass Q.
    bohr_m = physical_constants["Bohr radius"][0]
    ir_conversion = Avogadro * np.pi * fine_structure**2 * bohr_m / 3000
    ir_strengths = np.sum(mu_derivatives**2, axis=1) * ir_conversion
    isotropic = np.trace(alpha_derivatives, axis1=1, axis2=2) / 3
    traceless = alpha_derivatives - isotropic[:, None, None]*np.eye(3)
    anisotropy_squared = 1.5 * np.sum(traceless**2, axis=(1, 2))
    raman_conversion = (bohr_m*1e10)**4 * AMU_TO_ELECTRON_MASS
    activities = (45*isotropic**2 + 7*anisotropy_squared) * raman_conversion
    depolarization = 3*anisotropy_squared / (45*isotropic**2 + 4*anisotropy_squared)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data = dict(method=harmonic.get("method", "RHF" if method == "HF" else "RKS"),
                xc=method, label=label, basis=harmonic["basis"], cart=True,
                symbols=harmonic["symbols"], coordinates_bohr=harmonic["coordinates_bohr"],
                frequencies_cm1=frequencies, displacements=directions.tolist(),
                normal_coordinate_units="Bohr * sqrt(electron mass)",
                dipole_derivatives=mu_derivatives.tolist(), alpha_derivatives=alpha_derivatives.tolist(),
                ir_intensities_km_mol=ir_strengths.tolist(), raman_activities_angstrom4_amu=activities.tolist(),
                depolarization_ratios=depolarization.tolist(), finite_difference_checks=checks,
                ir_conversion=ir_conversion, raman_conversion=raman_conversion,
                elapsed_seconds=perf_counter()-start, approximation="harmonic IR and static nonresonant Raman")
    data.update(grid_level=harmonic.get("grid_level"), grid_response=harmonic.get("grid_response", method != "HF"),
                dipole_expression=harmonic.get("dipole_expression",
                    "stationary identity; implicit relaxed density response retained"),
                source_input=str(source_input), dispersion=harmonic.get("dispersion", "none"),
                raw_rotation_residual_hartree_per_bohr=harmonic.get("raw_rotation_residual_hartree_per_bohr"),
                jax_version=jax.__version__, numpy_version=np.__version__,
                python_version=platform.python_version(), hostname=platform.node(),
                devices=[str(device) for device in jax.devices()], dtype="float64",
                runtime_source_hashes={name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                                      for name in ("src/gradscf_tools/molecular_response.py",
                                                   "src/gradscf/scf/orbital_optimization.py",
                                                   "src/gradscf/scf/uks.py",
                                               "src/gradscf/scf/rsh.py",
                                               "src/gradscf/integrals/grids/ao.py") if Path(name).exists()})
    (output / "gradscf.json").write_text(json.dumps(data, indent=2) + "\n")
    print("IR intensities / (km/mol):", ir_strengths)
    print("Raman activities / (Angstrom^4/amu):", activities)
    print("Depolarization ratios:", depolarization)
    return data


if __name__ == "__main__":
    for method in METHODS:
        run(method)


# RHF original direct-energy run, jy_gpu_1, CPU float64, 2026-10-15:
# IR intensities / (km/mol): [107.27054398 18.20900311 58.11052885]
# Raman activities / (Angstrom^4/amu): [5.72357771 75.53654975 39.08640435]
# Depolarization ratios: [0.52998310 0.18301674 0.75000000]
# Mode-wise max Raman FD errors: [2.079e-9, 7.523e-9, 2.660e-11]

# PBE stationary-dipole run, jy_gpu_1, CPU float64, 2026-10-15:
# IR intensities / (km/mol): [67.17110961 0.31577233 14.11962595]
# Raman activities / (Angstrom^4/amu): [8.54600005 77.41465215 38.09181347]
# Depolarization ratios: [0.55486634 0.17983741 0.75000000]
# Mode-wise max Raman FD errors at 3e-4 Bohr: [1.450e-9, 3.629e-9, 2.278e-9]
# Three AD modes plus both displaced checks per mode: 1764.087 s, excluding seed SCF.
# Water HF/PBE stationary-identity alpha agrees with direct E-AD within 2.381e-13;
# all three RHF mode alpha derivatives agree with the saved E-AD values within 4.053e-15.

# M2 caller reuse verified on jy_gpu_1, 2026-10-15, for PBE0/B3LYP/WB97XD.
# Three water modes plus both displaced-response checks passed for each method.
# Their measured strengths, times and grid limits are in water_hybrid_spectra.py
# and the separate M2_water_<xc>_spectra records; no solver or conversion was copied.
