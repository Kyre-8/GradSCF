"""V03: CO2 harmonic IR/Raman mutual exclusion at a linear RHF minimum.

Only the symmetric bond length is optimized; the full Cartesian Hessian then
tests bending stability. Two degenerate bends are interpreted as a subspace.
Frequencies/strengths are finite-basis predictions, not experimental fits.
"""
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import root_scalar
from scipy.constants import physical_constants, Avogadro, alpha as fine_structure

jax.config.update("jax_enable_x64", True)

from gradscf import gto, scf
from gradscf_tools.molecular_ep import NativeRHF
from gradscf_tools.molecular_response import MolecularEnergy
from gradscf_tools.vibrations import normal_modes, AMU_TO_ELECTRON_MASS


basis = "6-31g*"
start = perf_counter()
engine = NativeRHF("C 0 0 0; O 0 0 2.2; O 0 0 -2.2", basis=basis)
bond_direction = np.array([[0., 0., 0.], [0., 0., 1.], [0., 0., -1.]])


def objective(length):
    state, gradient = engine.evaluate(length[0]*bond_direction)
    print(f"R={length[0]:.9f} Bohr, E={float(state.total_energy):.12f} Eh, "
          f"dE/dR={np.sum(gradient*bond_direction):.3e} Eh/Bohr", flush=True)
    return float(state.total_energy), np.array([np.sum(gradient*bond_direction)])


# In this one-dimensional symmetric subspace, solve the analytic force for
# zero directly. Energy line searches reach roundoff before the force target.
optimum = root_scalar(lambda length: objective([length])[1][0],
                      bracket=(2.0, 2.4), method="brentq", xtol=1e-10)
assert optimum.converged
coordinates = optimum.root*bond_direction
state, gradient = engine.evaluate(coordinates)
print("Bond-force root:", optimum.flag, "max gradient", np.max(np.abs(gradient)), flush=True)
assert np.max(np.abs(gradient)) < 1e-7
response = engine.response(coordinates, state)
hessian = np.column_stack([response(v)[0].ravel()
                            for v in np.eye(9).reshape(9, 3, 3)])
modes = normal_modes(hessian, coordinates, [12., 15.99491461957, 15.99491461957])
assert modes.rigid_rank == 5 and np.all(modes.frequencies_cm1 > 0)
assert modes.rigid_residual < 1e-9

mol = gto.M(atom=list(zip(["C", "O", "O"], coordinates)), basis=basis, unit="Bohr", cart=True)
mf = scf.RHF(mol, conv_tol=1e-12, conv_tol_density=1e-11,
             integral_backend="native").run()
energy = MolecularEnergy(mf)
zero = jnp.zeros(3)
# The stationary identity avoids redundant third-energy graphs. The relaxed
# density remains differentiable, so neither field nor geometry response is frozen.
dipole = lambda r: energy.dipole(r, zero)
alpha = lambda r: jax.jacfwd(energy.dipole, argnums=1)(r, zero)
dipole_jvp = jax.jit(lambda r, v: jax.jvp(dipole, (r,), (v,))[1])
alpha_jvp = jax.jit(lambda r, v: jax.jvp(alpha, (r,), (v,))[1])
calculate_dipole, calculate_alpha = jax.jit(dipole), jax.jit(alpha)
directions = np.moveaxis(modes.displacements, -1, 0)
mu_derivatives, alpha_derivatives, checks = [], [], []
for index, direction in enumerate(directions):
    dmu = np.asarray(dipole_jvp(jnp.asarray(coordinates), jnp.asarray(direction)))
    dalpha = np.asarray(alpha_jvp(jnp.asarray(coordinates), jnp.asarray(direction)))
    if not np.isfinite(dmu).all() or not np.isfinite(dalpha).all():
        raise RuntimeError(f"Mode {index+1} electronic response failed.")
    mu_derivatives.append(dmu)
    alpha_derivatives.append(dalpha)
    np.testing.assert_allclose(dalpha, dalpha.T, atol=1e-9, rtol=0)
    for cartesian_step in (1e-3, 3e-4):
        step = cartesian_step / np.max(np.abs(direction))
        plus, minus = coordinates+step*direction, coordinates-step*direction
        mu_fd = np.asarray((calculate_dipole(plus)-calculate_dipole(minus))/(2*step))
        alpha_fd = np.asarray((calculate_alpha(plus)-calculate_alpha(minus))/(2*step))
        if not np.isfinite(mu_fd).all() or not np.isfinite(alpha_fd).all():
            raise RuntimeError(f"Mode {index+1} displaced response failed.")
        checks.append(dict(mode=index+1, max_cartesian_step_bohr=cartesian_step,
                           normal_coordinate_step=step,
                           max_dipole_derivative_error=float(np.max(np.abs(dmu-mu_fd))),
                           max_alpha_derivative_error=float(np.max(np.abs(dalpha-alpha_fd)))))
    np.testing.assert_allclose(dmu, mu_fd, atol=1e-6, rtol=1e-4)
    np.testing.assert_allclose(dalpha, alpha_fd, atol=3e-5, rtol=3e-4)
    print(f"Mode {index+1}, {modes.frequencies_cm1[index]:.3f} cm^-1: "
          f"|dmu/dQ|={np.linalg.norm(dmu):.4e}, |dalpha/dQ|={np.linalg.norm(dalpha):.4e}", flush=True)

dmu, dalpha = np.array(mu_derivatives), np.array(alpha_derivatives)
bohr_m = physical_constants["Bohr radius"][0]
ir = np.sum(dmu**2, axis=1) * Avogadro*np.pi*fine_structure**2*bohr_m/3000
average = np.trace(dalpha, axis1=1, axis2=2) / 3
anisotropy = 1.5*np.sum((dalpha-average[:, None, None]*np.eye(3))**2, axis=(1, 2))
raman = (45*average**2 + 7*anisotropy) * (bohr_m*1e10)**4*AMU_TO_ELECTRON_MASS

# Inversion swaps the oxygen sites and reverses Cartesian displacement.
# The middle-frequency stretch is gerade; bends and antisymmetric stretch
# are ungerade. Determine parity from vectors, never from sorted frequencies.
inverted = -directions[:, [0, 2, 1], :]
mass = np.array([12., 15.99491461957, 15.99491461957])*AMU_TO_ELECTRON_MASS
parity = np.einsum("mai,a,mai->m", directions, mass, inverted)
gerade = parity > .99
ungerade = parity < -.99
assert np.all(gerade | ungerade) and gerade.sum() == 1
assert ir[gerade].max() < 1e-6
assert raman[ungerade].max() < 1e-6

# Rotate the degenerate bending basis: its summed strengths are invariant.
bends = np.flatnonzero(np.sum(directions[:, :, :2]**2, axis=(1, 2)) > 1e-6)
assert len(bends) == 2
np.testing.assert_allclose(modes.frequencies_cm1[bends[0]], modes.frequencies_cm1[bends[1]],
                           atol=1e-3, rtol=0)
angle = .43
rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
np.testing.assert_allclose(np.sum((rotation@dmu[bends])**2), np.sum(dmu[bends]**2), atol=1e-14)
rotated_alpha = np.einsum("ab,bij->aij", rotation, dalpha[bends])
np.testing.assert_allclose(np.sum(rotated_alpha**2), np.sum(dalpha[bends]**2), atol=1e-14)

output = Path("reproducibility/ad_showcase/V03_co2_selection_rules")
output.mkdir(parents=True, exist_ok=True)
data = dict(method="RHF", basis=basis, cart=True, symbols=["C", "O", "O"],
            coordinates_bohr=coordinates.tolist(), energy_hartree=float(state.total_energy),
            gradient_hartree_per_bohr=np.asarray(gradient).tolist(),
            hessian_hartree_per_bohr2=hessian.tolist(), frequencies_cm1=modes.frequencies_cm1.tolist(),
            displacements=directions.tolist(), inversion_parity=parity.tolist(),
            ir_intensities_km_mol=ir.tolist(), raman_activities_angstrom4_amu=raman.tolist(),
            dipole_derivatives=dmu.tolist(), alpha_derivatives=dalpha.tolist(),
            finite_difference_checks=checks,
            response_construction="stationary dipole identity with implicit density response",
            normal_coordinate_units="Bohr * sqrt(electron mass)",
            rigid_rank=modes.rigid_rank, rigid_residual=modes.rigid_residual,
            hessian_asymmetry=modes.hessian_asymmetry, elapsed_seconds=perf_counter()-start,
            hostname=platform.node(), python_version=platform.python_version(),
            jax_version=jax.__version__, dtype="float64", devices=[str(d) for d in jax.devices()],
            runtime_source_hashes={name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                                   for name in ("src/gradscf_tools/molecular_response.py",
                                                "src/gradscf_tools/molecular_ep.py",
                                                "src/gradscf_tools/vibrations.py",
                                                "src/gradscf/scf/orbital_optimization.py")})
(output / "gradscf.json").write_text(json.dumps(data, indent=2) + "\n")
print("IR / km mol^-1:", ir)
print("Raman activities / Angstrom^4 amu^-1:", raman)

# Measured on jy_gpu_1, CPU float64, RHF/Cartesian 6-31G*, 2026-10-15:
# Frequencies / cm^-1: [745.80627644, 745.80627646, 1518.49578170, 2585.03147902]
# IR / km mol^-1: [68.93186191, 68.93186191, <1e-20, 988.58974613]
# Raman / Angstrom^4 amu^-1: [<1e-20, <1e-20, 12.33115265, <1e-20]
# Native multi-step mode FD errors: dmu <= 2.054e-9, dalpha <= 9.155e-10 a.u.
# Independent PySCF: max Hessian error 4.020e-8 Eh Bohr^-2;
# max dmu/dalpha errors 2.054e-9 / 1.034e-7 a.u. (same mass-weighted coordinates).
# Selection rules use inversion parity and the summed degenerate bend strength.
