"""Measure the MP2 molecular response adapter in isolated old/new processes.

All energies and derivatives come from src. This runner records compile/warm
costs and process RSS separately; XLA buffer estimates are not measured RSS.
Input JSON uses symbols/coords_angstrom/charge/spin, with a fixed basis per run.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
from time import perf_counter

import jax
jax.config.update('jax_enable_x64', True)
import numpy as np
from gradscf import gto, scf
from gradscf_tools.correlated_response import CorrelatedEnergy


def fingerprint(array):
    array = np.ascontiguousarray(array)
    metadata = dict(shape=array.shape, dtype=array.dtype.str)
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
    digest.update(array.tobytes())
    return dict(**metadata, sha256=digest.hexdigest())


def memory():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    current = None
    if Path('/proc/self/status').exists():
        current = next(int(line.split()[1]) for line in
                       Path('/proc/self/status').read_text().splitlines()
                       if line.startswith('VmRSS:'))
    return dict(current_kib=current, peak_kib=peak/1024 if sys.platform == 'darwin' else peak)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--basis', default='6-31g*')
    parser.add_argument('--mode', choices=('legacy', 'full', 'direct'), required=True)
    parser.add_argument('--derivative', choices=('energy', 'gradient'), default='gradient')
    parser.add_argument('--max-memory', type=float, default=60000.)
    parser.add_argument('--direct-workers', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compile-only', action='store_true')
    parser.add_argument('--max-xla-temp-gib', type=float, default=16.)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    if args.direct_workers < 1:
        parser.error('--direct-workers must be positive')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(args.geometry.read_text())
    atom = list(zip(data['symbols'], data['coords_angstrom']))
    report = dict(complete=False, command=sys.argv, geometry=data,
        geometry_sha256=hashlib.sha256(args.geometry.read_bytes()).hexdigest(),
        basis=args.basis, cartesian=True, unit='Angstrom', method='noncanonical MP2',
        energy_unit='Hartree', gradient_unit='Hartree/Bohr',
        mode=args.mode, derivative=args.derivative, max_memory_mb=args.max_memory,
        direct_workers=args.direct_workers,
        jax=jax.__version__, python=platform.python_version(), platform=platform.platform(),
        affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
        environment={key: os.environ.get(key) for key in ('JAX_PLATFORMS',
            'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
        source_sha256={}, memory_before=memory())
    import gradscf.mp.integrals as mp_integrals
    import gradscf_tools.correlated_response as response_module
    for module in (mp_integrals, response_module):
        path = Path(module.__file__)
        report['source_sha256'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    package = Path(mp_integrals.__file__).parents[1]
    for relative in ('ao2mo/native.py', 'scf/orbital_optimization.py',
                     'solvers/nonlinear/minimize.py', 'solvers/linear/gmres.py',
                     'solvers/linear/tensor_sum.py'):
        path = package / relative
        if path.exists():
            report['source_sha256'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()

    def save(phase):
        report.update(phase=phase, memory_latest=memory())
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(phase, report['memory_latest'], flush=True)

    def save_outputs(result):
        arrays = [np.asarray(x) for x in jax.tree.leaves(result)]
        if not all(np.isfinite(x).all() for x in arrays):
            raise RuntimeError('Nonfinite correlated energy or derivative')
        temporary = args.output.with_suffix('.tmp.npz')
        np.savez(temporary, *arrays)
        temporary.replace(args.output.with_suffix('.npz'))
        report.update(energy_hartree=float(arrays[0]),
                      outputs=[fingerprint(x) for x in arrays])

    save('scf')
    mol = gto.M(atom=atom, basis=args.basis, charge=data.get('charge', 0),
                 spin=data.get('spin', 0), unit='Angstrom', cart=True)
    factory = scf.UHF if mol.spin else scf.RHF
    if args.mode == 'direct':
        from gradscf.integrals.plan import IntegralPlan
        original_evaluate = IntegralPlan.evaluate

        def checked_evaluate(self, operator, *values, **options):
            if operator == 'eri':
                raise RuntimeError('Direct benchmark requested a full AO ERI')
            return original_evaluate(self, operator, *values, **options)

        IntegralPlan.evaluate = checked_evaluate
    t0 = perf_counter()
    mf = factory(mol, integral_backend='native', execution_device='cpu',
        jk_backend='direct' if args.mode == 'direct' else 'full', direct_scf_tol=0.,
        direct_scf_workers=args.direct_workers,
        conv_tol=1e-12, conv_tol_density=1e-11, conv_tol_grad=1e-10,
        max_cycle=250, max_memory=args.max_memory).run()
    jax.block_until_ready(mf.e_tot)
    report.update(scf_seconds=perf_counter()-t0, scf_converged=bool(mf.converged),
        scf_energy_hartree=float(mf.e_tot), coefficients=fingerprint(mf.mo_coeff),
        occupations=fingerprint(mf.mo_occ), memory_after_scf=memory())
    save('adapter')
    if not mf.converged:
        raise RuntimeError('SCF did not converge')
    options = {} if args.mode == 'legacy' else dict(jk_backend=args.mode, max_memory=args.max_memory)
    model = CorrelatedEnergy(mf, 'mp2', gradient_tolerance=1e-10, **options)
    coordinates = model.coordinates
    del mf
    function = model if args.derivative == 'energy' else jax.value_and_grad(model)
    function = jax.jit(function)
    save('compile')
    t0 = perf_counter()
    executable = function.lower(coordinates).compile()
    report['compile_seconds'] = perf_counter()-t0
    analysis = executable.memory_analysis()
    report['xla_memory_bytes'] = {key: getattr(analysis, key, None) for key in
        ('argument_size_in_bytes', 'output_size_in_bytes', 'alias_size_in_bytes',
         'temp_size_in_bytes', 'host_temp_size_in_bytes')}
    if args.compile_only:
        report['compile_only'] = True
        save('compile-only')
        return
    temporary = report['xla_memory_bytes'].get('temp_size_in_bytes')
    if temporary is not None and temporary > args.max_xla_temp_gib*1024**3:
        save('allocation-guard')
        raise MemoryError('Compiled temporary estimate exceeds the runner execution limit')
    save('first')
    t0 = perf_counter()
    result = jax.block_until_ready(executable(coordinates))
    report['first_seconds'] = perf_counter()-t0
    if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(result)):
        save('invalid-first')
        raise RuntimeError('Nonfinite correlated energy or derivative')
    save_outputs(result)
    save('warm')
    durations = []
    for _ in range(args.repeats):
        t0 = perf_counter()
        result = jax.block_until_ready(executable(coordinates))
        durations.append(perf_counter()-t0)
        report['warm_seconds'] = durations
        save_outputs(result)
        save('warm')
    report.update(complete=True, warm_seconds=durations,
                  warm_median_seconds=float(np.median(durations)))
    save('complete')


if __name__ == '__main__':
    main()
