"""Fixed-density native J/K geometry-gradient timings, with no SCF iteration.

Run explicitly on an allocated CPU. Input coordinates are in Angstrom; the
geometry gradient is with respect to Bohr. The seeded positive density is a
fixed contraction input, not a self-consistent molecular density. This isolates
native worker costs; it does not measure an MP2 gradient or physical accuracy.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from gradscf import gto, integrals
from gradscf.integrals.backends.native.jk import NativeDirectBasis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--workers', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    atom = list(zip(data['symbols'], data['coords_angstrom']))
    molecule = gto.M(atom=atom, basis='6-31g*', unit='Angstrom', cart=True,
                    charge=data['charge'], spin=data['spin'])
    topology, parameters = integrals.prepare_basis(molecule.to_spec(), molecule.basis, cart=True)
    plan = integrals.make_plan(topology, backend='native')
    coordinates = jnp.asarray(molecule.to_spec().coords_bohr)
    owners = np.argmin(np.linalg.norm(np.asarray(parameters.centers)[:, None]
                       - np.asarray(coordinates)[None], axis=-1), axis=1)
    rng = np.random.default_rng(807)
    factor = rng.normal(size=(topology.nao, 25))
    density = factor @ factor.T
    density = jnp.asarray(50.*density/np.trace(density))

    def energy(r):
        moving = replace(parameters, centers=r[owners], nuclear_coords=r)
        provider = NativeDirectBasis(plan, moving)
        j, k = provider.get_jk(density, workers=args.workers, screening_threshold=0.)
        return .5*jnp.sum(density*j)-.25*jnp.sum(density*k)

    start = time.perf_counter()
    lowered = jax.jit(jax.value_and_grad(energy)).lower(coordinates)
    lower_seconds = time.perf_counter()-start
    start = time.perf_counter()
    compiled = lowered.compile()
    compile_seconds = time.perf_counter()-start
    start = time.perf_counter()
    value, gradient = jax.block_until_ready(compiled(coordinates))
    first_seconds = time.perf_counter()-start
    assert np.isfinite(value) and np.isfinite(gradient).all()
    times = []
    for _ in range(3):
        start = time.perf_counter()
        jax.block_until_ready(compiled(coordinates))
        times.append(time.perf_counter()-start)
    analysis = compiled.memory_analysis()
    report = dict(input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        basis='6-31g*', cart=True, nao=topology.nao, seed=807, density_rank=25,
        workers=args.workers, screening_threshold=0., dtype=str(coordinates.dtype),
        lower_seconds=lower_seconds, compile_seconds=compile_seconds,
        first_seconds=first_seconds, warm_seconds=times,
        warm_median_seconds=float(np.median(times)), value=float(value),
        gradient=np.asarray(gradient).tolist(),
        xla_temporary_bytes=analysis.temp_size_in_bytes if analysis else None,
        scope='fixed-density J/K geometry response; no SCF or correlated response')
    if args.reference:
        reference = json.loads(args.reference.read_text())
        for key in ('input_sha256', 'basis', 'cart', 'nao', 'seed', 'density_rank', 'dtype'):
            assert report[key] == reference[key], key
        np.testing.assert_allclose(value, reference['value'], atol=2e-9, rtol=2e-12)
        np.testing.assert_allclose(gradient, reference['gradient'], atol=2e-9, rtol=2e-11)
        report['reference_max_gradient_difference'] = float(np.max(np.abs(
            np.asarray(gradient)-reference['gradient'])))
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
