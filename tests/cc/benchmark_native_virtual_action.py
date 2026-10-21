"""Matched native/dense physical-frame action and converged CC probes.

Without --input the action is an algebraic overcomplete stress probe, not a
physical molecular virtual-space benchmark. Arrays stay on the measurement host.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from gradscf import cc, integrals
from gradscf.ao2mo import VirtualERI, transform_eri_blocks
from gradscf.ao2mo.actions import contract_vvvv
from gradscf.cc.integrals import from_orbitals
from gradscf.cc.rccsd import residual as cc_residual
from gradscf.integrals.backends.native.jk import NativeDirectBasis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('dense', 'direct'), required=True)
    parser.add_argument('--case', choices=('action', 'cc'), default='action')
    parser.add_argument('--input', type=Path, help='Existing converged RHF frame JSON; no new SCF')
    parser.add_argument('--max-memory', type=float, default=512.)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    rss_events = []

    def snapshot(phase):
        current = next(int(line.split()[1]) for line in Path('/proc/self/status').read_text().splitlines()
                       if line.startswith('VmRSS:'))
        rss_events.append(dict(phase=phase, seconds=time.perf_counter()-started,
            current_rss_kib=current, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

    snapshot('imports')
    rng = np.random.default_rng(6414)
    atom = 'O 0 0 0; H 0 -.757 .587; H 0 .757 .587'
    input_metadata = None
    frame = None
    if args.input is not None:
        if args.case != 'action':
            parser.error('--input is reserved for the frozen-frame action probe')
        input_metadata = json.loads(args.input.read_text())
        assert input_metadata['scf_converged']
        frame = args.input.with_suffix('.npz')
        assert hashlib.sha256(frame.read_bytes()).hexdigest() == input_metadata['input_npz_sha256']
        with np.load(frame) as saved:
            c = jnp.asarray(saved['coefficients'][:, input_metadata['nocc']:])
            centers = saved['coordinates_bohr']
            topology, parameters = integrals.prepare_basis(list(zip(('O', 'H', 'H'), centers)),
                input_metadata['basis'], unit='Bohr', cart=input_metadata['cart'])
            np.testing.assert_allclose(parameters.nuclear_coords, centers, atol=2e-14, rtol=0.)
        basis_name = input_metadata['basis']
        atom = input_metadata['geometry']
    else:
        basis_name = 'cc-pvdz' if args.case == 'action' else 'sto-3g'
        topology, parameters = integrals.prepare_basis(atom, basis_name)
    basis = NativeDirectBasis(integrals.make_plan(topology, backend='native'), parameters)
    snapshot('input')
    start = time.perf_counter()
    if args.case == 'action':
        # Rectangular real frames are valid for the action. A larger virtual
        # frame makes its v^4 ownership visible without changing either mode.
        if frame is None:
            c = jnp.asarray(rng.normal(scale=.1, size=(basis.nao, 48)))
        x = jnp.asarray(rng.normal(scale=.03, size=(4, 4, c.shape[1], c.shape[1])))
        provider = (VirtualERI(c, basis, max_memory=args.max_memory) if args.mode == 'direct'
                    else transform_eri_blocks((c,), ((0, 0, 0, 0),), direct_basis=basis,
                        max_memory=args.max_memory)[0])
        jax.block_until_ready(provider)
        prepare_seconds = time.perf_counter()-start
        snapshot('prepared')
        executable = jax.jit(contract_vvvv).lower(provider, x).compile()
        snapshot('compiled')
        result = executable(provider, x)
        jax.block_until_ready(result)
        expected_converged = None
    else:
        from pyscf import gto, scf
        mol = gto.M(atom=atom, basis=basis_name, verbose=0)
        mf = scf.RHF(mol).run(conv_tol=1e-13)
        assert mf.converged
        c = jnp.asarray(mf.mo_coeff)
        prepared = from_orbitals(c, jnp.asarray(c.T@mf.get_fock()@c), mol.nelectron//2,
            jnp.asarray(mf.e_tot), direct_basis=basis, max_memory=args.max_memory, vvvv=args.mode)
        jax.block_until_ready(prepared)
        prepare_seconds = time.perf_counter()-start
        snapshot('prepared')
        config = cc.CCConfig(conv_tol=1e-11, residual_tol=1e-10)
        executable = jax.jit(lambda p: cc.run_cc_from_integrals(p, config=config)).lower(prepared).compile()
        snapshot('compiled')
        result = executable(prepared)
        jax.block_until_ready(result)
        assert result.converged
        expected_converged = bool(result.converged)
    first_seconds = time.perf_counter()-start
    snapshot('first_result')
    times = []
    for _ in range(5):
        tick = time.perf_counter()
        result = executable(provider, x) if args.case == 'action' else executable(prepared)
        jax.block_until_ready(result)
        times.append(time.perf_counter()-tick)
    primary_peak_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    snapshot('warm_results')
    components = None
    if args.case == 'action' and args.mode == 'direct':
        operations = (
            ('density', jax.jit(lambda coeff, rhs: jnp.einsum(
                'pc,...cd,qd->...pq', coeff, rhs, coeff, precision='highest')), (c, x)),
        )
        density = operations[0][1](*operations[0][2])
        exchange_function = jax.jit(lambda d: basis.get_jk(d, with_j=False, with_k=True,
                                                         screening_threshold=0., workers=1)[1])
        exchange = exchange_function(density)
        projection = jax.jit(lambda coeff, k: jnp.einsum(
            'pa,...pq,qb->...ab', coeff, k, coeff, precision='highest'))
        operations += (('native_exchange', exchange_function, (density,)),
                       ('projection', projection, (c, exchange)))
        components = {}
        for name, operation, arguments in operations:
            jax.block_until_ready(operation(*arguments))
            samples = []
            for _ in range(5):
                tick = time.perf_counter()
                jax.block_until_ready(operation(*arguments))
                samples.append(time.perf_counter()-tick)
            components[name] = statistics.median(samples)
        np.testing.assert_allclose(result, projection(c, exchange), atol=2e-11, rtol=2e-10)
    if args.case == 'cc':
        residual = cc_residual(result.t1, result.t2, prepared)
        residual_norm = float(jnp.sqrt(sum(jnp.vdot(r, r) for r in residual)))
        values = jnp.asarray([result.total_energy, result.correlation_energy])
        ground = dict(total_energy_hartree=float(result.total_energy),
            correlation_energy_hartree=float(result.correlation_energy),
            reference_energy_hartree=float(result.reference_energy), iterations=int(result.iterations),
            conv_tol=1e-11, residual_tol=1e-10)
    else:
        residual_norm = None
        values = result
        ground = None
    output_arrays = Path(args.output).with_suffix('.npz')
    np.savez(output_arrays, values=np.asarray(values))
    memory = executable.memory_analysis()
    cpu = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                if line.startswith('model name')), 'unknown')
    report = dict(mode=args.mode, case=args.case, atom=atom, basis=basis_name,
        backend=jax.default_backend(), dtype=str(values.dtype), jax=jax.__version__,
        command=[sys.executable, *sys.argv], cpu=cpu, affinity=sorted(os.sched_getaffinity(0)),
        omp_threads=os.environ.get('OMP_NUM_THREADS'), blas_threads=os.environ.get('OPENBLAS_NUM_THREADS'),
        seed=6414, coefficient_shape=list(c.shape), rhs_shape=list(x.shape) if args.case == 'action' else None,
        frame_kind=('physical_frozen_rhf_virtuals' if frame is not None else
                    'algebraic_overcomplete_stress' if args.case == 'action' else 'physical_rhf_ccsd'),
        input_metadata=input_metadata, input_npz_sha256=hashlib.sha256(frame.read_bytes()).hexdigest() if frame else None,
        input_json_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest() if args.input else None,
        max_memory_mb=args.max_memory, prepare_seconds=prepare_seconds, first_seconds=first_seconds,
        warm_seconds=times, warm_median_seconds=statistics.median(times),
        component_warm_median_seconds=components,
        compiled_temp_bytes=memory.temp_size_in_bytes,
        compiled_argument_bytes=memory.argument_size_in_bytes,
        compiled_output_bytes=memory.output_size_in_bytes,
        peak_rss_kib=primary_peak_rss_kib, rss_snapshot_before_optional_component_profile=True,
        rss_timeline=rss_events, geometry_unit='Angstrom', energy_unit='Hartree', cart=True,
        ground=ground, output_npz_sha256=hashlib.sha256(output_arrays.read_bytes()).hexdigest(),
        sum=float(jnp.sum(values)), norm=float(jnp.linalg.norm(values)),
        converged=expected_converged, residual_norm=residual_norm,
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in map(Path,
            ('src/gradscf/ao2mo/actions.py', 'src/gradscf/cc/integrals.py',
             'src/gradscf/cc/_intermediates.py', 'tests/cc/benchmark_native_virtual_action.py'))})
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
