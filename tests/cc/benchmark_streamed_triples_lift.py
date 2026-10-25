"""Measure the existing complete RCC lift plus streamed correction owners."""
import json
from pathlib import Path
import time

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

from gradscf._memory import (array_storage_bytes, tree_storage_bytes,
                             jaxpr_live_bytes, workspace_bytes)
from gradscf.cc.integrals import CCIntegrals
from gradscf.cc._memory import triples_budget
from gradscf.cc._streamed_triples import _restricted_inputs, _streamed_general_correction
from test_streamed_triples_memory import _restricted_case


def correction(*args):
    fock, ovvv, ovoo, ovov, t1, t2 = args
    data = CCIntegrals(fock, None, None, ovoo, ovov, None, None, ovvv, None, None)
    spin, singles, doubles = _restricted_inputs(data, t1, t2)
    return jnp.stack(_streamed_general_correction(spin, singles, doubles,
                                                 denominator_tol=1e-10)[:3])


def reverse(args, cotangent):
    value, pullback = jax.vjp(correction, *args)
    return value, pullback(cotangent)


def second(args, cotangent, direction, dcotangent):
    return jax.jvp(reverse, (args, cotangent), (direction, dcotangent))


data, state = _restricted_case()
o, v = data.nocc, data.nvir
args = (data.fock, data.ovvv, data.ovoo, data.ovov, state.t1, state.t2)
shapes = tuple(jax.ShapeDtypeStruct(a.shape, a.dtype) for a in args)
cotangent = jax.ShapeDtypeStruct((3,), jnp.float64)
resident = tree_storage_bytes(data, state)
current_memory = 40.411891
# Keep the former requested budget as a comparison point after the fix;
# rejection is an observation, not a failure of this small ownership probe.
try:
    triples_budget(data, state, current_memory, 0, o, v, 'ccsd', 'semicanonical', 'streamed')
except ValueError as error:
    assert 'memory' in str(error) or 'budget' in str(error)
    admitted = False
else:
    admitted = True
window = workspace_bytes(current_memory, resident_bytes=resident)
start = time.monotonic()
bounds = {
    'value': jaxpr_live_bytes(jax.make_jaxpr(correction)(*shapes)),
    'first': jaxpr_live_bytes(jax.make_jaxpr(reverse)(shapes, cotangent)),
    'directional_second': jaxpr_live_bytes(jax.make_jaxpr(second)(
        shapes, cotangent, shapes, cotangent)),
}
seed = array_storage_bytes(cotangent)
whole_bound = max(bounds['value'], bounds['first']+seed,
                  bounds['directional_second']+tree_storage_bytes(shapes)+2*seed)
record = dict(o=o, v=v, requested_budget_MB=current_memory, requested_budget_admitted=admitted,
    resident_bytes=resident,
    requested_working_window_bytes=window, caller_inputs_bytes=tree_storage_bytes(shapes),
    whole_program_order_bounds=bounds, whole_ad_working_bound_bytes=whole_bound,
    whole_bound_exceeds_requested_window=whole_bound>window,
    metadata_seconds=time.monotonic()-start)
Path('streamed-lift-owners.json').write_text(json.dumps(record, indent=2))
print(json.dumps(record), flush=True)

# A real noncanonical input prevents identity rotations or zero gradients
# from concealing the lift's retained linearization arrays.
rng = np.random.default_rng(8752)
values = [jnp.asarray(rng.normal(scale=.02, size=a.shape)) for a in shapes]
fock = jnp.diag(jnp.concatenate((jnp.linspace(-1.3, -.9, o), jnp.linspace(.3, 1.1, v))))
values[0] = fock+.1*(values[0]+values[0].T)
values[-1] = .5*(values[-1]+values[-1].transpose(1,0,3,2))
values = tuple(values)
start = time.monotonic()
primal, pullback = jax.vjp(lambda *xs: correction(*xs)[:2], *values)
jax.block_until_ready((primal, jax.tree.leaves(pullback)))
seen = set()
leaves = []
for leaf in jax.tree.leaves(pullback):
    if hasattr(leaf, 'shape') and hasattr(leaf, 'dtype') and id(leaf) not in seen:
        seen.add(id(leaf))
        leaves.append(dict(shape=list(leaf.shape), dtype=str(leaf.dtype), bytes=array_storage_bytes(leaf)))
record.update(actual_saved_bytes_excluding_inputs=tree_storage_bytes(pullback, exclude=values),
              actual_saved_leaves=leaves, value=np.asarray(primal).tolist(),
              actual_vjp_seconds=time.monotonic()-start)
for i in range(2):
    gradient = pullback(jax.nn.one_hot(i, 2, dtype=jnp.float64))
    jax.block_until_ready(gradient)
    assert all(np.isfinite(np.asarray(g)).all() for g in gradient)
    record[f'cotangent_{i}_nonzero'] = any(np.any(np.asarray(g)!=0) for g in gradient)
record['actual_saved_exceeds_requested_window'] = record['actual_saved_bytes_excluding_inputs']>window
Path('streamed-lift-owners.json').write_text(json.dumps(record, indent=2))
print(json.dumps(record), flush=True)
