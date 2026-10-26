"""Inspect existing streamed-(T) owners; no alternative triples equations."""
import json
import time
import sys
from pathlib import Path

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

from gradscf._memory import (array_storage_bytes, tree_storage_bytes,
                             jaxpr_live_bytes, workspace_bytes)
from gradscf.cc._memory import triples_budget, _streamed_response_workspace
from gradscf.cc.integrals import CCIntegrals
from gradscf.cc.types import CCResult
from gradscf.cc import _streamed_triples as streamed

TOLERANCE = 1e-10
output = []


def save():
    Path('streamed-triples-owners.json').write_text(json.dumps(output, indent=2))


def arguments(occupied, virtual):
    spec = lambda shape: jax.ShapeDtypeStruct(shape, jnp.float64)
    return tuple(map(spec, ((occupied+virtual,)*2, (occupied, virtual, virtual, virtual),
        (occupied, occupied, occupied, virtual), (occupied, occupied, virtual, virtual),
        (occupied, virtual), (occupied, occupied, virtual, virtual))))


def budget_window(o, v):
    spec = lambda shape: jax.ShapeDtypeStruct(shape, jnp.float64)
    ints = CCIntegrals(*map(spec, ((o+v,)*2, (o+v,), (o,)*4, (o,v,o,o),
        (o,v,o,v), (o,o,v,v), (o,v,v,o), (o,v,v,v), (v,)*4, ())))
    state = CCResult(spec(()), spec(()), spec(()), spec((o,v)), spec((o,o,v,v)),
                     spec(()), spec(()), 1, True, spec(()), 2)
    resident = tree_storage_bytes(ints, state)
    def fits(total_bytes):
        try:
            triples_budget(ints, state, total_bytes/1e6, 0, o, v, 'ccsd',
                            'semicanonical', 'streamed')
        except ValueError as error:
            assert 'memory' in str(error) or 'budget' in str(error)
            return False
        return True
    start = time.perf_counter_ns()
    assert fits(1_024_000_000)
    first = (time.perf_counter_ns()-start)/1e9
    start = time.perf_counter_ns()
    assert fits(1_024_000_000)
    cached = (time.perf_counter_ns()-start)/1e9
    start = time.perf_counter_ns()
    for _ in range(100):
        assert fits(1_024_000_000)
    repeated = (time.perf_counter_ns()-start)/1e9/100
    lower, upper = 1, 1_000_000
    while not fits(upper):
        upper *= 2
    while lower < upper:
        middle = (lower+upper)//2
        if fits(middle):
            upper = middle
        else:
            lower = middle+1
    caller = (ints.fock, ints.ovvv, ints.ovoo, ints.ovov, state.t1, state.t2)
    profiled = _streamed_response_workspace(tuple(a.shape for a in caller),
                                            tuple(a.dtype.name for a in caller), False)
    return dict(resident_bytes=resident, minimum_total_MB=upper/1e6,
        profiled_caller_working_bound_bytes=profiled,
        planner_first_seconds=first, planner_cached_seconds=cached,
        planner_repeat_mean_seconds=repeated,
        working_window_bytes=workspace_bytes(upper/1e6, resident_bytes=resident))


def scalar(*args):
    return jnp.sum(streamed._components(TOLERANCE, *args)[:2])


for o, v in ((2, 8), (2, 16), (2, 32)):
    shapes = arguments(2*o, 2*v)
    start = time.monotonic()
    record = dict(spatial_occupied=o, spatial_virtual=v, **budget_window(o, v))
    record['spin_input_bytes'] = tree_storage_bytes(shapes)
    rotated = jax.eval_shape(lambda *xs: streamed._factor_inputs(*xs)[-1], *shapes)
    record['rotated_input_bytes'] = tree_storage_bytes(rotated)
    rows = jax.eval_shape(lambda *xs: streamed._value_and_gradients(TOLERANCE, *xs), *shapes)[1]
    record['two_returned_gradient_rows_bytes'] = tree_storage_bytes(rows)
    functions = {
        'forward': (lambda *xs: streamed._forward(TOLERANCE, *xs), shapes),
        'gradient_rows': (lambda *xs: streamed._value_and_gradients(TOLERANCE, *xs), shapes),
        'hessian_rows': (lambda *xs: streamed._hessian_product(TOLERANCE, xs[:6], xs[6:]), shapes+shapes),
        'scalar_vjp': (jax.grad(scalar, argnums=tuple(range(6))), shapes),
        'scalar_hvp': (lambda *xs: jax.jvp(jax.grad(scalar, argnums=tuple(range(6))),
                                          xs[:6], xs[6:])[1], shapes+shapes),
    }
    record['kernel_only_reference_bounds'] = {}
    for label, (function, inputs) in functions.items():
        graph = jax.make_jaxpr(function)(*inputs)
        record['kernel_only_reference_bounds'][label] = jaxpr_live_bytes(graph)
    record['metadata_seconds'] = time.monotonic()-start
    output.append(record)
    save()
    print(json.dumps(record), flush=True)

if '--metadata-only' in sys.argv:
    raise SystemExit(0)

# One concrete VJP exposes retained Jacobian rows independently of any caller
# cotangent. No SCF, CC iteration, AO integrals or numerical HVP is run here.
o, v = 4, 32
rng = np.random.default_rng(8751)
arrays = [jnp.asarray(rng.normal(scale=.02, size=shape.shape)) for shape in arguments(o, v)]
fock = jnp.diag(jnp.concatenate((jnp.linspace(-1.4, -.9, o), jnp.linspace(.3, 1.1, v))))
fock = fock+.1*(arrays[0]+arrays[0].T)
arrays[0] = fock
args = tuple(arrays)
start = time.monotonic()
values, pullback = jax.vjp(lambda *xs: streamed._components(TOLERANCE, *xs)[:2], *args)
jax.block_until_ready((values, jax.tree.leaves(pullback)))
seen = set()
leaves = []
for leaf in jax.tree.leaves(pullback):
    if hasattr(leaf, 'shape') and hasattr(leaf, 'dtype') and id(leaf) not in seen:
        seen.add(id(leaf))
        leaves.append(dict(shape=list(leaf.shape), dtype=str(leaf.dtype), bytes=array_storage_bytes(leaf)))
record = dict(spin_occupied=o, spin_virtual=v,
    actual_saved_bytes_excluding_inputs=tree_storage_bytes(pullback, exclude=args),
    saved_leaves=leaves, values=np.asarray(values).tolist(), seconds=time.monotonic()-start)
for index in range(2):
    gradients = pullback(jax.nn.one_hot(index, 2, dtype=jnp.float64))
    jax.block_until_ready(gradients)
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in gradients)
    record[f'cotangent_{index}_nonzero'] = any(np.any(np.asarray(leaf)!=0) for leaf in gradients)
record['working_window_bytes'] = output[1]['working_window_bytes']
record['saved_alone_exceeds_working'] = record['actual_saved_bytes_excluding_inputs'] > record['working_window_bytes']
output.append(record)
save()
print(json.dumps(record), flush=True)
