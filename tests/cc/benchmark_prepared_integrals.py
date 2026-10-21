"""Matched full-MO->CC versus selected-CC preparation in isolated processes.

Generate one small RHF frame with --mode setup. Full/selected runs load only
the same frame arrays, so SCF's integral allocations do not contaminate their
process RSS. Energies are Hartree; input geometry is Angstrom, Cartesian AOs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
from time import perf_counter


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('setup', 'full', 'selected'), required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--basis', default='6-31g*')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--max-memory', type=float, default=60000.)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1 or (args.mode != 'setup' and args.output is None):
        parser.error('Positive repeats and an output path are required for measurement')
    import numpy as np
    started = perf_counter()
    if args.mode == 'setup':
        import pyscf
        geometry = 'O 0 0 0; H 0 -.757 .587; H 0 .757 .587'
        mol = pyscf.gto.M(atom=geometry, basis=args.basis, unit='Angstrom', cart=True, verbose=0)
        mf = mol.RHF().run(conv_tol=1e-12)
        if not mf.converged:
            raise RuntimeError('RHF input preparation did not converge')
        c = mf.mo_coeff
        args.input.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.input.with_suffix('.npz'), coefficients=c, hcore=mf.get_hcore(),
            fock=c.T@mf.get_fock()@c, reference_energy=np.asarray(mf.e_tot),
            nuclear_repulsion=np.asarray(mol.energy_nuc()), coordinates_bohr=mol.atom_coords())
        report = dict(geometry=geometry, geometry_unit='Angstrom', cart=True,
            basis=args.basis, nocc=int(mol.nelectron//2), nao=int(mol.nao_nr()),
            energy_unit='Hartree', pyscf=pyscf.__version__, scf_converged=bool(mf.converged),
            setup_seconds=perf_counter()-started,
            input_npz_sha256=digest(args.input.with_suffix('.npz')))
        args.input.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2), flush=True)
        return

    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    from gradscf import ao2mo, integrals
    from gradscf.cc import integrals as cc_integrals
    from gradscf.integrals.backends.native.jk import NativeDirectBasis

    data = json.loads(args.input.read_text())
    assert digest(args.input.with_suffix('.npz')) == data['input_npz_sha256']
    with np.load(args.input.with_suffix('.npz')) as saved:
        c, h, f, energy, enuc = (jnp.asarray(saved[key]) for key in
            ('coefficients', 'hcore', 'fock', 'reference_energy', 'nuclear_repulsion'))
        atom = list(zip(('O', 'H', 'H'), saved['coordinates_bohr']))
        top, parameters = integrals.prepare_basis(atom, basis=data['basis'], unit='Bohr', cart=True)
        np.testing.assert_allclose(parameters.nuclear_coords, saved['coordinates_bohr'],
                                   atol=2e-14, rtol=0)
    basis = NativeDirectBasis(integrals.make_plan(top, backend='native'), parameters)
    nocc = data['nocc']

    def function(coefficients):
        if args.mode == 'selected':
            return cc_integrals.from_orbitals(coefficients, f, nocc, energy,
                direct_basis=basis, max_memory=args.max_memory)
        hmo, gmo = ao2mo.transform_integrals(h, coefficients,
            direct_basis=basis, max_memory=args.max_memory)
        return cc_integrals.prepare_integrals(hmo, gmo, nocc=nocc, nuclear_repulsion=enuc)

    report = dict(complete=False, command=sys.argv, mode=args.mode, input=data,
        input_json_sha256=digest(args.input), initialization_seconds=perf_counter()-started,
        max_memory_mb=args.max_memory, dtype=str(c.dtype), jax=jax.__version__,
        affinity=sorted(os.sched_getaffinity(0)), device=str(c.device),
        environment={key: os.getenv(key) for key in ('JAX_PLATFORMS', 'CUDA_VISIBLE_DEVICES',
            'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}, source_sha256={})
    package = Path(cc_integrals.__file__).parents[1]
    for path in (package/'cc/integrals.py', package/'ao2mo/blocks.py',
                 package/'ao2mo/native.py', package/'_memory.py', Path(__file__)):
        report['source_sha256'][str(path)] = digest(path)
    report['native_sha256'] = digest(package/'integrals/_native/lib/libgradscf_integrals.so')
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        report['peak_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        args.output.write_text(json.dumps(report, indent=2)+'\n')

    save()
    started = perf_counter()
    executable = jax.jit(function).lower(c).compile()
    report['compile_seconds'] = perf_counter()-started
    analysis = executable.memory_analysis()
    report['xla_estimated_bytes'] = {key: getattr(analysis, key, None) for key in
        ('argument_size_in_bytes', 'output_size_in_bytes', 'alias_size_in_bytes',
         'temp_size_in_bytes', 'host_temp_size_in_bytes')}
    save()
    started = perf_counter()
    values = jax.block_until_ready(executable(c))
    report['first_seconds'] = perf_counter()-started
    durations = []
    for _ in range(args.repeats):
        started = perf_counter()
        values = jax.block_until_ready(executable(c))
        durations.append(perf_counter()-started)
    arrays = tuple(np.asarray(x) for x in values)
    if not all(np.isfinite(x).all() for x in arrays):
        raise RuntimeError('Nonfinite CC integral output')
    np.savez(args.output.with_suffix('.npz'), *arrays)
    report.update(complete=True, warm_seconds=durations,
        warm_median_seconds=float(np.median(durations)),
        output_shapes=[x.shape for x in arrays], output_sha256=[
            hashlib.sha256(x.tobytes()).hexdigest() for x in arrays])
    save()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
