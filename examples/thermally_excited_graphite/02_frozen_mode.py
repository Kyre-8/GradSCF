"""Local cubic/quartic energy coefficients from relaxed frozen displacements.

Every displaced calculation reconverges SCF at the SAME electronic temperature.
This is an explicit finite-displacement fit, not an all-order SCF Taylor rule.
The resulting Gamma-mode coefficients are not a bulk thermal frequency shift:
the missing finite-q mixed-mode vertices must still be integrated over the BZ.
"""
import json
import time
import jax
import jax.numpy as jnp
import numpy as np
from gradscf.integrals.periodic.fft import build_inputs
from gradscf.pbc.scf import run_kpoint_scf
from gradscf.pbc.anharmonic import fit_frozen_mode
from settings import OUTPUT, make_cell, mean_field, optical_directions, calculation_settings, SCF_CONTROLS

jax.config.update('jax_enable_x64', True)
OUTPUT.mkdir(exist_ok=True)
cell = make_cell()
mf = mean_field(cell, 300.).run()
assert mf.converged

# u_atom = s * direction_atom. The scalar s is in Bohr; no mass weighting.
direction = jnp.asarray(optical_directions()[0])
# Coarse k sampling at low electronic temperature can make the local Taylor
# radius small. Check order/window dependence rather than fitting large motion.
displacements = np.linspace(-.0012, .0012, 13)


@jax.jit
def displaced_state(s):
    coordinates = cell.coords+s*direction
    inputs = build_inputs(cell, parameters=cell.parameters_at(coordinates), kpts=mf.kpts)
    result = run_kpoint_scf(inputs, mesh=cell.mesh, nelec=cell.nelec, xc='pbe',
        exxdiv=None, temperature=mf.temperature, init_density=mf.result.density_spin,
        **SCF_CONTROLS)
    return result.free_energy, result.converged, result.cycles


energies = []
start = time.perf_counter()
for amplitude in displacements:
    free_energy, converged, iterations = displaced_state(amplitude)
    assert bool(converged), f'SCF failed at displacement {amplitude} Bohr'
    energies.append(float(free_energy))
    print(f's = {amplitude:+.6f} Bohr; F = {float(free_energy):.12f} Eh; '
          f'iterations = {int(iterations)}', flush=True)

# Two orders and a smaller displacement window expose truncation sensitivity.
# Never enforce evenness simply because the sample grid is symmetric.
fit4 = fit_frozen_mode(displacements, energies, order=4)
fit6 = fit_frozen_mode(displacements, energies, order=6)
inner = abs(displacements) <= .0008001
fit_inner = fit_frozen_mode(displacements[inner], np.asarray(energies)[inner], order=4)
print('Fourth-order derivatives (Eh/Bohr^n):', np.asarray(fit4.derivatives)[1:])
print('Sixth-order fit, same derivatives:', np.asarray(fit6.derivatives)[1:5])
print('Inner-window fit, same derivatives:', np.asarray(fit_inner.derivatives)[1:])
print('Fourth-order RMS fit residual (Eh):', float(fit4.rms_residual))

np.savez(OUTPUT/'frozen_mode.npz', displacements=displacements, energies=energies,
         direction=np.asarray(direction), coefficients4=np.asarray(fit4.coefficients),
         coefficients6=np.asarray(fit6.coefficients), derivatives4=np.asarray(fit4.derivatives),
         derivatives6=np.asarray(fit6.derivatives), derivatives_inner=np.asarray(fit_inner.derivatives),
         settings=json.dumps(calculation_settings(), sort_keys=True))
(OUTPUT/'frozen_mode.json').write_text(json.dumps({
    'electron_temperature_K': 300., 'coordinate_unit': 'Bohr',
    'energy': 'electronic free energy per primitive cell',
    'method': 'reconverged finite displacement; scaled polynomial least squares',
    'derivatives4_Eh_Bohr_power': np.asarray(fit4.derivatives).tolist(),
    'derivatives6_Eh_Bohr_power': np.asarray(fit6.derivatives).tolist(),
    'rms_fit_residual_Eh': float(fit4.rms_residual),
    'seconds_including_compilation': time.perf_counter()-start,
    'scope': 'one Gamma direction; no BZ-integrated anharmonic self-energy or linewidth',
    'settings': calculation_settings(),
}, indent=2)+'\n')

# CPU float64, |s| <= 0.0012 Bohr:
# d2F/ds2 from order 4 / order 6 / inner-window fits (Eh/Bohr^2):
# 0.942300915569, 0.942511917669, 0.942469654
# d4F/ds4 from order 4 / order 6 fits (Eh/Bohr^4):
# -185839.841162, -190678.468446
# Order-4 RMS residual: 6.45349e-12 Eh. Order/window sensitivity is still visible.
