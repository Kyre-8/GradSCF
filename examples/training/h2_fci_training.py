"""H2/6-31G*: 100 Adam steps in each mode, with FCI energy supervision.

All three runs start from the same four-layer density-matrix MLP, seed and
optimizer. Loss = delta_E**2 + abs(delta_E), using numerical energies in Ha
(equivalently delta_E normalized by 1 Ha). FCI density is diagnostic only:
activating density supervision would request SCF even in fixed-density mode.
No reference correlation energy is mislabeled as an XC energy.

Run: PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 python examples/training/h2_fci_training.py
"""
from pathlib import Path
from time import perf_counter
import csv
import json
import platform

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)

from gradscf import gto, scf, fci
from gradscf.integrals.molecular.ao2mo import transform_integrals
from gradscf.model.neural_xc import training
from density_matrix_mlp import functional


mol = gto.M(atom='H 0 0 0; H 0 0 .74', basis='6-31g*')
mf = scf.RHF(mol, grids_level=0, integral_backend='native').run()
reference = mf.to_reference()
coeff = reference.mo_coeff[0]
solver = fci.FCI(mf, solver='dense', conv_tol=1e-12).run()
# Export the small Hamiltonian only for the independent FCI cross-check.
h1, eri = transform_integrals(reference.h1e, coeff, eri_pair_matrix=reference.eri_pair_matrix)
if not solver.converged:
    raise RuntimeError('FCI reference did not converge')
target = jax.lax.stop_gradient(solver.e_tot)
fci_dm = coeff @ solver.make_rdm1() @ coeff.T
fci_rho = jnp.einsum('pq,gp,gq->g', fci_dm, reference.ao, reference.ao)
np.testing.assert_allclose(jnp.sum(fci_dm*reference.overlap_matrix), 2., atol=1e-10)
print('FCI total energy / Ha:', float(target), flush=True)

steps, learning_rate, seed = 100, .002, 0
scf_settings = dict(max_cycle=80, damping=.2, require_converged=True,
                    conv_tol_energy=1e-11, conv_tol_density=1e-9, eigenvalue_jitter=0.)
adjoint_settings = dict(tolerance=1e-9, max_iter=40)
data = [training.Sample(mf, energy=target)]
folder = Path(__file__).with_name('h2_fci_results')
folder.mkdir(exist_ok=True)
report = dict(atom='H 0 0 0; H 0 0 .74', unit='Angstrom', basis='6-31g*',
    grid_level=0, backend=jax.default_backend(), dtype='float64', jax=jax.__version__,
    python=platform.python_version(), platform=platform.platform(),
    architecture=[16,16,16,16,1], dense_layers=4, activation='tanh', output_scale=.05,
    seed=seed, steps=steps, optimizer='Adam', learning_rate=learning_rate,
    loss='delta_E**2 + abs(delta_E), delta_E=(E-E_FCI)/(1 Ha)',
    fixed_density='initial GradSCF HF density, held fixed',
    supervision='FCI total energy only; FCI density is diagnostic',
    fci_energy_hartree=float(target), fci_determinants=solver.space.size,
    nuclear_repulsion_hartree=float(reference.nuclear_repulsion),
    scf=dict(max_cycle=80,damping=.2,energy_tolerance=1e-11,density_tolerance=1e-9),
    implicit=dict(tolerance=1e-9,max_iter=40),
    command='PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 python examples/training/h2_fci_training.py',
    row_semantics='step k is loss after k completed attempts; next_update_accepted refers to k -> k+1',
    runs={})
np.savez_compressed(folder/'reference.npz', h1_mo=np.asarray(h1), eri_mo=np.asarray(eri),
    ecore=np.asarray(reference.nuclear_repulsion), fci_rdm1_mo=np.asarray(solver.make_rdm1()),
    fci_rdm1_ao=np.asarray(fci_dm), fci_density_grid=np.asarray(fci_rho),
    grid_weights=np.asarray(reference.grid.weights), overlap=np.asarray(reference.overlap_matrix))

for name in ('fixed_density', 'explicit', 'implicit'):
    trainer = training.Trainer(functional)
    trainer.seed = seed
    trainer.mode = name
    trainer.loss = {'energy': {'mse': 1., 'mae': 1.}}
    trainer.learning_rate = learning_rate
    trainer.scf = scf_settings
    trainer.adjoint = adjoint_settings
    start = perf_counter()
    trainer.run(data, steps=steps)
    elapsed = perf_counter()-start
    history, rows = trainer.history, []
    for step in range(steps+1):
        predicted = history['energy'][step][0]
        error = predicted-float(target)
        row = dict(step=step, energy_hartree=predicted, energy_error_hartree=error,
            mse=history['energy_mse'][step], mae=history['energy_mae'][step],
            loss=history['loss'][step],
            next_update_accepted=history['update_accepted'][step+1] if step<steps else None,
            scf_converged=history['scf_converged'][step], scf_cycles=int(history['scf_cycles'][step]))
        np.testing.assert_allclose(row['loss'], row['mse']+row['mae'], rtol=1e-10, atol=1e-14)
        rows.append(row)
        if step % 20 == 0:
            print(name, step, 'loss', row['loss'], 'error / Ha', error, flush=True)
    with (folder/f'{name}.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)

    # Evaluate every final model self-consistently as an additional diagnostic.
    scf_metrics = trainer.evaluate(data, mode='implicit')
    scf_loss = scf_metrics['loss']
    scf_energy, scf_state = trainer.predict(mf, mode='implicit')
    rho = training.density_on_grid(scf_state)
    density_l2 = jnp.sqrt(jnp.sum(reference.grid.weights*(rho-fci_rho)**2)
                          /jnp.sum(reference.grid.weights*fci_rho**2))
    report['runs'][name] = dict(initial=rows[0], final=rows[-1],
        accepted_updates=sum(bool(r['next_update_accepted']) for r in rows[:-1]),
        optimizer_step=history['optimizer_step'][-1], seconds_including_compile=elapsed,
        self_consistent_final_energy_hartree=float(scf_energy),
        self_consistent_final_loss=float(scf_loss),
        self_consistent_final_converged=bool(np.all(scf_metrics['scf_converged'])),
        self_consistent_density_relative_l2_to_fci=float(density_l2))
    from flax.serialization import to_bytes
    (folder/f'{name}_params.msgpack').write_bytes(to_bytes(trainer.params))
    (folder/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(name, 'accepted', report['runs'][name]['accepted_updates'], 'seconds', elapsed, flush=True)

print('Results:', folder, flush=True)

# CPU float64, seed 0, 100 Adam updates, lr=0.002, loss=MSE+MAE:
# FCI energy / Ha: -1.1516725449612413
# mode           initial loss       final loss          final MAE / Ha
# fixed_density  1.112662199200644  0.437609924132773  0.329222481685569
# explicit       1.021108762032138  0.374749471532906  0.290410951045661
# implicit       1.021108762032138  0.374749471527732  0.290410951042388
# All three runs accepted 100/100 updates. This is not yet an accurate FCI fit.
