"""Program-order array bounds retain live results and nested loop carries."""
from types import SimpleNamespace as Record

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import _memory


def _live():
    function = getattr(_memory, 'jaxpr_live_bytes', None)
    assert callable(function), 'Missing shared live-array Jaxpr accounting'
    return function


def _var(size=4):
    return Record(aval=Record(shape=(size,), dtype=np.dtype('float64')))


def _equation(inputs, outputs, name='array', **params):
    return Record(invars=inputs, outvars=outputs, primitive=Record(name=name), params=params)


def _graph(inputs, outputs, equations, constants=()):
    return Record(invars=inputs, outvars=outputs, eqns=equations, constvars=constants)


def test_dead_arrays_release_after_last_use_but_results_remain_live():
    live = _live()
    x, a, b, c, d = (_var() for _ in range(5))
    chain = _graph((x,), (c,), (_equation((x,), (a,)),
        _equation((a,), (b,)), _equation((b,), (c,))))
    assert live(chain) == 64
    fork = _graph((x,), (a, d), (_equation((x,), (a,)),
        _equation((x,), (b,)), _equation((a, b), (c,)), _equation((c,), (d,))))
    assert live(fork) == 96


def test_constants_literals_and_unused_primitive_outputs_are_charged():
    live = _live()
    x, constant, literal, result, unused = _var(), _var(), _var(1), _var(), _var()
    graph = _graph((x,), (result,),
        (_equation((x, constant, literal), (result, unused)),), constants=(constant,))
    # Constant32 + scalar literal8 + both produced32-byte arrays coexist.
    assert live(graph) == 104


def test_nested_branches_charge_one_live_body_and_retained_outer_output():
    live = _live()
    x, a, b, result = (_var() for _ in range(4))
    identity = _graph((x,), (x,), ())
    branch = _graph((x,), (b,), (_equation((x,), (a,)), _equation((x, a), (b,))))
    graph = _graph((x,), (result,), (_equation((x,), (result,), name='cond',
        branches=(Record(jaxpr=identity), Record(jaxpr=branch))),))
    # Branch input32 + live a32 + b32; outer result32 is retained separately.
    assert live(graph) == 128


@pytest.mark.parametrize('length', [3, 17])
def test_loop_body_carries_and_full_return_stack_share_the_bound(length):
    live = _live()
    carry, step_input, tmp, update = (_var() for _ in range(4))
    body = _graph((carry, step_input), (update, update), (
        _equation((carry,), (tmp,)), _equation((tmp, step_input), (update,))))
    initial, inputs, final, stack = _var(), _var(length*4), _var(), _var(length*4)
    graph = _graph((initial, inputs), (final, stack), (_equation(
        (initial, inputs), (final, stack), name='scan', jaxpr=Record(jaxpr=body), length=length),))
    assert live(graph) == (length+1)*32 + 96


def test_while_condition_and_body_are_alternative_workspaces():
    live = _live()
    x, predicate, tmp, update, result = _var(), _var(1), _var(), _var(), _var()
    condition = _graph((x,), (predicate,), (_equation((x,), (predicate,)),))
    body = _graph((x,), (update,), (_equation((x,), (tmp,)), _equation((tmp,), (update,))))
    graph = _graph((x,), (result,), (_equation((x,), (result,), name='while',
        cond_jaxpr=condition, body_jaxpr=body),))
    assert live(graph) == 96


def test_known_opaque_workspace_is_charged_inside_nested_calls():
    live = _live()
    x, middle, out = _var(), _var(), _var()
    body = _graph((x,), (middle,), (_equation((x,), (middle,), name='native'),))
    graph = _graph((x,), (out,), (_equation((x,), (out,), name='jit', jaxpr=Record(jaxpr=body)),))
    assert live(graph, primitive_bytes={'native': 4096}) == 4192
    with pytest.raises(ValueError, match='primitive_bytes'):
        live(graph, primitive_bytes={'native': -1})


def test_nested_constant_stores_survive_sibling_calls():
    live = _live()
    x, large, small, a, b = _var(1), _var(20), _var(1), _var(1), _var(1)
    first = _graph((x,), (a,), (_equation((x, large), (a,)),), constants=(large,))
    second = _graph((x,), (b,), (_equation((x, small), (b,), name='scratch'),),
                    constants=(small,))
    graph = _graph((x,), (a, b), (
        _equation((x,), (a,), name='jit', jaxpr=first),
        _equation((x,), (b,), name='jit', jaxpr=second)))
    # Both closed constant stores still exist while the second call uses its
    # opaque work area. Counting just the active call's constants is too small.
    assert live(graph, primitive_bytes={'scratch': 1000}) >= (20+1)*8+1000


def test_real_jaxpr_live_bound_covers_elementwise_compiled_temporary():
    live = _live()
    x = jnp.ones(1024, dtype=jnp.float64)
    function = lambda a: jnp.sin(jnp.cos(a))+a
    graph = jax.make_jaxpr(function)(x)
    assert live(graph) == 2*x.nbytes
    compiled = jax.jit(function).lower(x).compile()
    assert compiled.memory_analysis().temp_size_in_bytes <= live(graph)
