"""H2 dissociation: FCI-supervised neural_xc training and implicit GPU memory.

Reuses the molecular loss and Adam step of the H2 ground-state training workflow.
All molecular preparation, HF and FCI references use GradSCF public APIs.
Only network parameters are differentiated; geometry, integrals and FCI labels
are fixed. No SCF, adjoint solver or optimizer is reimplemented here.

Run in a GPU allocation with the existing neural_xc dependencies and nvidia-ml-py:
  PYTHONPATH=src JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
    python examples/training/neural_xc/h2_implicit_memory.py

Settings are below. Environment overrides allow isolated benchmark processes:
NEURAL_XC_BASIS, NEURAL_XC_GRID_LEVEL, NEURAL_XC_WIDTH, NEURAL_XC_DEPTH, NEURAL_XC_STEPS,
NEURAL_XC_MEMORY_OUTPUT, and NEURAL_XC_REQUIRE_GPU (set 0 for a CPU smoke test).
NVML reports sampled process residency; JAX peak_bytes_in_use is cumulative.
Neither quantity is a resettable, exact per-phase temporary-memory peak.
"""
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
import json
import os
import platform
import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax

jax.config.update('jax_enable_x64', True)

from gradscf import gto, scf, fci
from gradscf.model.neural_xc import training
from gradscf.model import neural_xc
from gpu_memory import MemoryMonitor, nvml_process_reader


DISTANCES = (.50, .74, 1.00, 1.50, 2.20)  # Angstrom, training points only
BASIS = os.environ.get('NEURAL_XC_BASIS', '6-31g*')
GRID_LEVEL = int(os.environ.get('NEURAL_XC_GRID_LEVEL', '0'))
WIDTH = int(os.environ.get('NEURAL_XC_WIDTH', '64'))
DEPTH = int(os.environ.get('NEURAL_XC_DEPTH', '2'))
STEPS = int(os.environ.get('NEURAL_XC_STEPS', '3'))
OUTPUT = Path(os.environ.get('NEURAL_XC_MEMORY_OUTPUT',
                            'outputs/neural_xc_implicit_memory'))
SEED, LEARNING_RATE = 0, 1e-4


def check_forward(loss, metrics):
    if not np.isfinite(float(loss)):
        raise RuntimeError('Nonfinite self-consistent loss.')
    if not np.all(np.asarray(metrics['scf_converged']) == 1):
        raise RuntimeError('A neural-XC forward SCF did not converge.')


def check_gradients(gradients):
    leaves = [np.asarray(x) for x in jax.tree.leaves(gradients)]
    if not all(np.isfinite(x).all() for x in leaves):
        raise RuntimeError('Nonfinite raw implicit gradient; measurement is invalid.')
    return float(np.sqrt(sum(np.sum(x*x) for x in leaves)))


def main():
    if min(WIDTH, DEPTH, STEPS) < 1 or GRID_LEVEL < 0:
        raise ValueError('Width/depth/steps must be positive and grid level nonnegative.')
    device = jax.devices()[0]
    gpu = device.platform == 'gpu'
    if os.environ.get('NEURAL_XC_REQUIRE_GPU', '1') != '0' and not gpu:
        raise RuntimeError('GPU required: a CPU run cannot measure GPU memory.')
    if len(jax.devices()) != 1:
        raise RuntimeError('This example requires exactly one visible JAX device.')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    monitor = MemoryMonitor(nvml_process_reader() if gpu else lambda: None)
    report = dict(status='preparing', command=' '.join(sys.argv),
        backend=jax.default_backend(), device=str(device), gpu_model=device.device_kind,
        hostname=platform.node(), python=platform.python_version(), jax=jax.__version__,
        dtype='float64 arrays; model parameter dtypes recorded below', seed=SEED,
        slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        preallocate=os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE', 'default'),
        allocator=os.environ.get('XLA_PYTHON_CLIENT_ALLOCATOR', 'default'),
        gradscf_source=str(Path(training.__file__).resolve()),
        basis=BASIS, grid_level=GRID_LEVEL, distances_angstrom=list(DISTANCES),
        architecture='graddft_residual', hidden_dims=[WIDTH]*DEPTH,
        learning_rate=LEARNING_RATE, steps=STEPS,
        loss='mean((E-E_FCI)**2 + abs(E-E_FCI)), numerical energies in Ha',
        memory='NVML PID residency sampled every 10 ms; JAX peaks are cumulative; '
               'phases share one process and retain prior executables/allocator pools',
        reference=[], phases=[], history=[])

    def save():
        (OUTPUT/'summary.json').write_text(json.dumps(report, indent=2)+'\n')

    def measure(name, operation, synchronize=jax.block_until_ready):
        print('phase:', name, flush=True)
        before = device.memory_stats()
        result, record = monitor.measure(name, operation, synchronize)
        record.update(jax_allocator_before=before, jax_allocator_after=device.memory_stats())
        report['phases'].append(record)
        save()
        peak = record['process_sampled_peak_bytes']
        print(name, 'seconds', round(record['seconds'], 4), 'sampled peak MiB',
              None if peak is None else round(peak/2**20, 2), flush=True)
        return result

    started = perf_counter()
    data = []
    # The native integral FFI has CPU targets. Prepare fixed references there,
    # then explicitly transfer the complete training PyTrees onto the GPU.
    with jax.default_device(jax.devices('cpu')[0]):
        for distance in DISTANCES:
            mol = gto.M(atom=f'H 0 0 0; H 0 0 {distance}', basis=BASIS, unit='Angstrom')
            mf = scf.RHF(mol, grids_level=GRID_LEVEL, integral_backend='native').run()
            if not mf.converged:
                raise RuntimeError(f'Initial HF did not converge at R={distance} Angstrom.')
            solver = fci.FCI(mf, solver='dense', conv_tol=1e-12).run()
            if not solver.converged:
                raise RuntimeError(f'FCI did not converge at R={distance} Angstrom.')
            reference = mf.to_reference()
            target = jax.lax.stop_gradient(solver.e_tot)
            data.append(training.Sample(reference, energy=target))
            report['reference'].append(dict(distance_angstrom=distance,
                fci_energy_hartree=float(target), nao=int(reference.ao.shape[1]),
                grid_points=int(reference.ao.shape[0])))
            print('reference:', distance, 'Angstrom, FCI', float(target), 'Ha', flush=True)
    data = tuple(jax.device_put(datum, device) for datum in data)
    jax.block_until_ready(data)
    report['preparation_backend'] = 'native CPU integrals/HF/FCI, then device_put to training GPU'
    report['preparation_seconds'] = perf_counter()-started

    functional = neural_xc.Functional(hidden_dims=(WIDTH,)*DEPTH,
        architecture='graddft_residual', input_feature_mode='canonical',
        # The existing HF nograd channel intentionally stops density derivatives.
        # Use semilocal channels for a full converged-loss finite-difference test.
        include_hfx_channel=False, ground_state_hf_mode='off',
        include_pt2_channel=False, name='h2_implicit_memory')
    config = training.MolecularTrainingConfig(mode='self_consistent',
        scf_gradient_mode='implicit', e0_total_mse_weight=1., e0_total_mae_weight=1.,
        scf_require_converged=True, scf_max_cycle=128, scf_damping=.25,
        scf_conv_tol_energy=1e-10, scf_conv_tol_density=1e-8,
        scf_eigenvalue_jitter=0., scf_implicit_diff_tolerance=1e-9,
        scf_implicit_diff_max_iter=40, scf_implicit_diff_regularization=0.)
    state = training.trainer.create_train_state_from_molecule(functional,
        jax.random.PRNGKey(SEED), data[0].molecule, optax.adam(LEARNING_RATE))
    # Flax initializes Dense parameters in float32 by default. Promote both
    # parameters and Adam moments for a float64 adjoint/finite-difference check.
    promote = lambda x: x.astype(jnp.float64) if jnp.issubdtype(x.dtype, jnp.inexact) else x
    state = state.replace(step=jnp.asarray(state.step),
                          params=jax.tree.map(promote, state.params),
                          opt_state=jax.tree.map(promote, state.opt_state))
    # Match the explicit placement of data. Otherwise the first JIT update
    # commits previously uncommitted arrays and changes the next dispatch key.
    state = jax.device_put(state, device)
    jax.block_until_ready(state)
    report.update(config=asdict(config),
        semilocal_channels=list(functional.semilocal_xc), ground_state_hf_mode='off',
        parameter_count=sum(x.size for x in jax.tree.leaves(state.params)),
        parameter_dtypes=sorted({str(x.dtype) for x in jax.tree.leaves(state.params)}),
        status='measuring')
    save()

    def objective(params, local_data):
        return training.molecular_loss(params, functional, local_data, training_config=config)

    forward = jax.jit(objective)
    for name in ('forward_first_call', 'forward_warm'):
        loss, metrics = measure(name, lambda: forward(state.params, data))
        check_forward(loss, metrics)
    report['initial_loss'] = float(loss)

    # has_aux excludes diagnostic metrics from the cotangent. The forward below
    # retains the residuals needed for backward, unlike value-only evaluation.
    def loss_for_params(params):
        return forward(params, data)

    def linearize():
        return jax.vjp(loss_for_params, state.params, has_aux=True)

    for name in ('vjp_forward_first_call', 'vjp_forward_warm'):
        if name == 'vjp_forward_warm':
            del pullback  # Do not retain two independent VJP tapes during warm-up.
        loss, pullback, metrics = measure(name, linearize,
            jax.block_until_ready)
        check_forward(loss, metrics)
    # VJP objects are registered PyTrees. Pass their residual arrays as runtime
    # arguments instead of embedding a particular pullback as JIT constants.
    backward = jax.jit(lambda local_pullback, cotangent: local_pullback(cotangent))
    cotangent = jnp.ones_like(loss)
    for name in ('backward_first_call', 'backward_warm'):
        gradients, = measure(name, lambda: backward(pullback, cotangent))
        report['raw_gradient_norm'] = check_gradients(gradients)

    # Compare the raw VJP with a directional finite difference of the same
    # converged loss; reject invalid adjoints before the sanitized train step.
    direction = jax.tree.map(jnp.ones_like, state.params)
    size = report['parameter_count']
    direction = jax.tree.map(lambda x: x/jnp.sqrt(size), direction)
    analytic = float(sum(jnp.vdot(g, v) for g, v in
                         zip(jax.tree.leaves(gradients), jax.tree.leaves(direction))))
    epsilon = 1e-4
    shifted = []
    for sign in (-1., 1.):
        params = jax.tree.map(lambda p, v: p+sign*epsilon*v, state.params, direction)
        value, diagnostic = forward(params, data)
        check_forward(value, diagnostic)
        shifted.append(float(value))
    finite_difference = (shifted[1]-shifted[0])/(2*epsilon)
    report['gradient_check'] = dict(direction='unit normalized all-ones parameter tree',
        epsilon=epsilon, implicit=analytic, finite_difference=finite_difference,
        absolute_error=abs(analytic-finite_difference))
    save()
    np.testing.assert_allclose(analytic, finite_difference, rtol=2e-3, atol=2e-6)
    del gradients, pullback, backward

    train_step = jax.jit(training.trainer.make_molecular_train_step(functional, config))
    for step in range(STEPS):
        name = 'train_first_call' if step == 0 else f'train_warm_{step}'
        state, metrics = measure(name, lambda: train_step(state, data))
        check_forward(metrics['total_loss'], metrics)
        if not bool(metrics['update_accepted']):
            raise RuntimeError('Training update rejected; inspect SCF/adjoint convergence.')
        report['history'].append(dict(step=step+1, pre_update_loss=float(metrics['total_loss']),
            grad_norm=float(metrics['grad_norm'][0]),
            scf_cycles=np.asarray(metrics['scf_cycles']).tolist(), update_accepted=True))
    loss, metrics = measure('final_forward', lambda: forward(state.params, data))
    check_forward(loss, metrics)
    report.update(final_loss=float(loss), optimizer_step=int(state.step),
        final_energies_hartree=np.asarray(metrics['predicted_e0_total_h']).tolist())
    if float(loss) >= report['initial_loss']:
        raise RuntimeError('Expected a loss reduction in this short training test.')
    training.save_params_checkpoint(OUTPUT/'params.msgpack', state.params,
        metadata=dict(basis=BASIS, distances_angstrom=list(DISTANCES), config=asdict(config)))
    report['status'] = 'complete'
    save()
    print('Loss:', report['initial_loss'], '->', report['final_loss'], flush=True)
    print('Results:', OUTPUT.resolve(), flush=True)


if __name__ == '__main__':
    main()
