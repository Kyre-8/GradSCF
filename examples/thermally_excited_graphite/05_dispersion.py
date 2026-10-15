"""Direct in-plane phonon samples along Gamma--K--M--Gamma.

This is the more expensive extension of the Gamma example. A 6x6x1 electronic
mesh is required to include both thirds (K) and halves (M) without an inexact
momentum mapping. The plotted lines only connect computed samples; they are
not a dense interpolation or a converged bulk graphite dispersion.
Lines connect frequency-sorted samples, without eigenvector-character tracking.
"""
import json
import time
import jax
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gradscf.pbc.phonon import Phonons
from settings import (OUTPUT, MASSES, HARTREE_TO_MEV, make_cell, mean_field,
                      inplane_directions, calculation_settings)

jax.config.update('jax_enable_x64', True)
OUTPUT.mkdir(exist_ok=True)
cell = make_cell()
kmesh = (6, 6, 1)
mf = mean_field(cell, 300., kmesh=kmesh).run()
assert mf.converged
driver = Phonons(mf, masses=MASSES, max_grid=100_000, max_pair_elements=4_000_000_000)

# Retain all in-plane displacements, including acoustic and interlayer modes.
directions = inplane_directions()
qpoints = np.array([[0., 0., 0.], [1/6, 1/6, 0.], [1/3, 1/3, 0.],
                    [.5, 0., 0.], [1/3, 0., 0.], [1/6, 0., 0.]])
frequencies, modes, leakage, timings = [], [], [], []
for q in qpoints:
    start = time.perf_counter()
    print('Starting q:', q, flush=True)
    result = driver.project(directions, q=q)
    relative_residual = float(result.subspace_residual_norm)/np.linalg.norm(np.asarray(result.frequencies)**2)
    assert relative_residual < 1e-3, 'In-plane subspace is not invariant at this discretization.'
    frequencies.append(np.asarray(result.frequencies))
    modes.append(np.asarray(result.modes))
    leakage.append(relative_residual)
    timings.append(time.perf_counter()-start)
    print('q:', q, 'signed phonon energies (meV):', frequencies[-1]*HARTREE_TO_MEV,
          'seconds:', timings[-1], flush=True)
    metadata = dict(calculation_settings(), k_mesh=list(kmesh), electron_temperature_K=300.)
    np.savez(OUTPUT/'dispersion_samples.npz', qpoints=qpoints[:len(frequencies)],
             energies=frequencies, modes=modes, relative_subspace_residuals=leakage,
             seconds=timings, settings=json.dumps(metadata, sort_keys=True))

closed_q = np.vstack((qpoints, qpoints[:1]))
cartesian = closed_q @ np.asarray(cell.reciprocal_vectors())
distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(cartesian, axis=0), axis=1))]
values = np.vstack((frequencies, frequencies[:1]))*HARTREE_TO_MEV
fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
ax.plot(distance, values, 'o-', markersize=3, linewidth=.8)
ax.axhline(0., color='black', linewidth=.6)
ax.set_xticks(distance[[0, 2, 3, 6]], [r'$\Gamma$', 'K', 'M', r'$\Gamma$'])
ax.set(ylabel='Signed phonon energy (meV)', title='AB graphite: direct in-plane samples, k=6x6x1')
fig.savefig(OUTPUT/'dispersion_samples.png', dpi=180)
fig.savefig(OUTPUT/'dispersion_samples.pdf')

# CPU float64: six q points completed in 3496.63 s including per-q compilation.
# Gamma optical energies: [219.537818, 219.820982, 225.097071, 225.377444] meV.
# Two Gamma modes are imaginary; their signed energies are -48.215290 and
# -46.960694 meV. A low mode at q=(1/6,0,0) is also negative (-25.688598 meV).
# These are raw coarse-grid results, not a stable/converged material prediction.
