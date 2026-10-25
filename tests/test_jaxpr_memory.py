"""Known primitive array estimates are shared by physical operator callers."""
import jax
import jax.numpy as jnp

from gradscf import _memory


def test_nested_call_outputs_are_counted_without_running_the_operator():
    count = getattr(_memory, 'jaxpr_temporary_bytes', None)
    assert callable(count), 'Missing shared declared-array graph estimate'
    inner = jax.jit(lambda x: x+x)
    value = jax.ShapeDtypeStruct((3, 2), jnp.float64)
    graph = jax.make_jaxpr(lambda x: inner(x)*x)(value)
    # One inner add, its enclosing call output, and the outer multiply.
    # Views/call aliases deliberately receive separate conservative charges.
    assert count(graph) == 3*3*2*8
