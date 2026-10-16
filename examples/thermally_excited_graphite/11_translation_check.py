"""Diagnose grid translation error without enforcing an acoustic sum rule.

A rigid displacement should cost no energy in the continuum. Compare the raw
DFPT curvature to independently reconverged free energies on the fixed FFT grid.
Agreement of nonzero curvatures diagnoses the discretized energy, not an
accurate physical acoustic mode. Run 10 first. Coordinates/steps are in Bohr.
"""
import json
import time
import jax
import jax.numpy as jnp
import numpy as np
from gradscf.integrals.periodic.fft import build_inputs
from gradscf.pbc.scf import run_kpoint_scf
from settings import OUTPUT, make_cell, mean_field, SCF_CONTROLS

jax.config.update('jax_enable_x64', True)
output = OUTPUT/'gamma_precision'/'r2scan_szv_fft33_k3'
with np.load(output/'gamma_300K.npz') as archive:
    directions = archive['directions'].reshape(8, 12)
    force_constants = archive['force_constants']
cell = make_cell(mesh=(33, 33, 63))
start = time.perf_counter()
mf = mean_field(cell, 300., xc='r2scan').run()
assert mf.converged
direction = jnp.zeros((4, 3)).at[:, 0].set(.5)  # unit Euclidean rigid-x vector
components = directions@np.asarray(direction).ravel()
ad_curvature = float((components@force_constants@components).real)


@jax.jit
def translated(s):
    coordinates = cell.coords+s*direction
    data = build_inputs(cell, parameters=cell.parameters_at(coordinates), kpts=mf.kpts)
    result = run_kpoint_scf(data, mesh=cell.mesh, nelec=cell.nelec, xc='r2scan',
        exxdiv=None, temperature=mf.temperature, init_density=mf.result.density_spin,
        **SCF_CONTROLS)
    return result.free_energy, result.converged


step = .0004
amplitudes = np.array([-step, -step/2, 0., step/2, step])
energies = []
for amplitude in amplitudes:
    value, converged = translated(amplitude)
    assert bool(converged)
    energies.append(float(value))
    print('Rigid-x amplitude / free energy:', amplitude, float(value), flush=True)
coarse = (energies[0]-2*energies[2]+energies[4])/step**2
fine = (energies[1]-2*energies[2]+energies[3])/(step/2)**2
record = dict(direction=np.asarray(direction).tolist(), amplitudes_bohr=amplitudes.tolist(),
              free_energies_Eh=energies, dfpt_curvature_Eh_Bohr2=ad_curvature,
              fd_curvature_Eh_Bohr2=coarse, half_step_curvature_Eh_Bohr2=fine,
              seconds=time.perf_counter()-start,
              interpretation='No ASR applied; a rigid translation should have zero curvature in the continuum.')
(output/'translation_check.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record), flush=True)
