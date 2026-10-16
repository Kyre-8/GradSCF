"""G08: relaxed RHF cubic/quartic energy derivatives along one fixed direction.

All orders use AD of native moving integrals and the shared implicit SCF root.
Differences of the relaxed curvature validate the derivatives, and the separate
compare_molecular_cubic_quartic_pyscf.py supplies an independent Hessian oracle.
This is a directional derivative demonstration, not a full anharmonic model.
"""
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from gradscf import gto, scf
from gradscf_tools.molecular_response import MolecularEnergy


def case_inputs(case):
    if case == "h2":
        return dict(symbols=["H", "H"], basis="6-31g",
                    coordinates_bohr=[[0., 0., -.7], [0., 0., .7]],
                    direction=[[0., 0., -.5], [0., 0., .5]],
                    coordinate_definition="q is the H-H bond-length change in bohr")
    if case == "water_sto3g":
        direction = np.array([[0., 0., 0.], [0., 1., .7], [0., -1., .7]])
        direction -= direction.mean(axis=0)
        direction /= np.linalg.norm(direction)
        return dict(symbols=["O", "H", "H"], basis="sto-3g",
                    coordinates_bohr=[[0., 0., 0.], [0., 1.43, 1.11], [0., -1.43, 1.11]],
                    direction=direction.tolist(),
                    coordinate_definition="q is a bohr displacement along a fixed unit-norm A1 direction")
    if case == "water_631gstar":
        path = Path("reproducibility/ad_showcase/G05_water_rhf_hessian/gradscf.json")
        harmonic = json.loads(path.read_text())
        displacement = np.array(harmonic["isotopes"]["H2O"]["displacements"])[..., 0]
        norm = np.linalg.norm(displacement)
        return dict(symbols=harmonic["symbols"], basis=harmonic["basis"],
                    coordinates_bohr=harmonic["coordinates_bohr"],
                    direction=(displacement/norm).tolist(),
                    coordinate_definition="q is a bohr displacement along the unit-norm H2O bend mode",
                    source_harmonic=str(path), source_mode_index=0,
                    source_mass_normalized_direction=displacement.tolist(),
                    dq_dQ_inverse_sqrt_electron_mass=float(norm))
    raise ValueError(case)


def run(case="h2"):
    data = case_inputs(case)
    coordinates = jnp.asarray(data["coordinates_bohr"])
    direction = jnp.asarray(data["direction"])
    mol = gto.M(atom=list(zip(data["symbols"], data["coordinates_bohr"])),
                basis=data["basis"], cart=True, unit="Bohr")
    mf = scf.RHF(mol, integral_backend="native", conv_tol=1e-13,
                 conv_tol_density=1e-12, conv_tol_grad=1e-11).run()
    stability = mf.stability(solver="dense", eigensolver_tol=1e-10)
    if stability.stable is not True or not stability.spectrum_certified:
        raise RuntimeError("The restricted orbital stability gate is unresolved or negative.")
    energy = MolecularEnergy(mf, gradient_tolerance=1e-11)
    center_state = energy.solve(coordinates)
    assert bool(center_state.stationary) and float(center_state.gradient_norm) < 1e-10
    scalar = lambda q: energy(coordinates+q*direction)
    derivatives = [scalar, jax.grad(scalar)]
    for order in range(2, 5):
        previous = derivatives[-1]
        derivatives.append(lambda q, previous=previous: jax.jvp(
            previous, (q,), (jnp.ones_like(q),))[1])
    compiled = [jax.jit(function) for function in derivatives]
    values, seconds = [], []
    for order, function in enumerate(compiled):
        start = perf_counter()
        value = float(function(jnp.asarray(0.)))
        seconds.append(perf_counter()-start)
        if not np.isfinite(value):
            raise RuntimeError(f"Nonfinite relaxed derivative of order {order}.")
        values.append(value)
        print(f"{case}: d^{order}E/dq^{order} = {value:.13g}; {seconds[-1]:.2f} s", flush=True)

    checks = []
    for step in (3e-3, 1e-3, 3e-4):
        plus, minus = float(compiled[2](step)), float(compiled[2](-step))
        cubic = (plus-minus)/(2*step)
        quartic = (plus-2*values[2]+minus)/step**2
        checks.append(dict(step_bohr=step, cubic=cubic, quartic=quartic,
                           cubic_error=abs(cubic-values[3]), quartic_error=abs(quartic-values[4])))
        print(f"curvature FD h={step:g}: cubic error={checks[-1]['cubic_error']:.3e}, "
              f"quartic error={checks[-1]['quartic_error']:.3e}", flush=True)
    assert checks[-1]["cubic_error"] < 3e-5
    assert checks[-1]["quartic_error"] < 3e-4
    # Explicitly verify that the electronic state responds to this displacement.
    density_derivative = jax.jit(lambda q: jax.jvp(
        lambda x: energy.solve(coordinates+x*direction).density_matrix,
        (q,), (jnp.ones_like(q),))[1])(jnp.asarray(0.))
    assert np.isfinite(density_derivative).all() and np.linalg.norm(density_derivative) > 1e-5
    data.update(case=case, method="RHF", cart=True, field=None,
                derivative_method="reverse first derivative, then nested forward JVP; native integrals and implicit SCF",
                derivative_units=["Eh", "Eh/bohr", "Eh/bohr^2", "Eh/bohr^3", "Eh/bohr^4"],
                derivatives=values, seconds_including_compilation=seconds,
                native_curvature_differences=checks,
                density_directional_derivative_norm=float(np.linalg.norm(density_derivative)),
                center_orbital_gradient_norm=float(center_state.gradient_norm),
                orbital_stability_eigenvalues=np.asarray(stability.eigenvalues).tolist(),
                orbital_stability_residuals=np.asarray(stability.residual_norms).tolist(),
                scf_energy_tolerance=1e-13, scf_gradient_tolerance=1e-11,
                implicit_response_tolerance=1e-10,
                backend="CPU float64", threads=2)
    output = Path("reproducibility/ad_showcase/G08_molecular_cubic_quartic") / case
    output.mkdir(parents=True, exist_ok=True)
    (output / "gradscf.json").write_text(json.dumps(data, indent=2)+"\n")
    return data


if __name__ == "__main__":
    run("h2")
    # The larger stages can be run individually with run("water_sto3g") and
    # run("water_631gstar"); each stage writes its own result directory.

# jy_gpu_1, CPU float64, two threads, 2026-10-16. d3/d4 units Eh/bohr^3, Eh/bohr^4:
# h2:              -1.319474877751,  4.428751185659
# water_sto3g:     -2.259645261691,  5.412656625653
# water_631gstar:  -0.090363618759, -0.068701627807
# Native curvature FD errors at h=3e-4 bohr, respectively:
# (2.478e-7, 5.760e-7), (2.481e-7, 4.561e-7), (1.061e-8, 1.006e-7).
# Water/6-31G* d1=4.1074e-8 Eh/bohr; the stored bend direction has unit Euclidean norm.
# The negative bend quartic is a local derivative, not a globally bounded potential.
