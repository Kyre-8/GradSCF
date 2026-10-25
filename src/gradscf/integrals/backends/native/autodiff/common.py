"""Shared batching for native derivative product primitives."""
import jax
import jax.numpy as jnp
import numpy as np
from jax.interpreters import batching, mlir


def batch_product(primitive, args, axes):
    size = next(a.shape[axis] for a, axis in zip(args, axes) if axis is not None)
    arrays = tuple(batching.bdim_at_front(a, axis, size) for a, axis in zip(args, axes))
    return jax.lax.map(lambda xs: primitive.bind(*xs), arrays), 0


def register_eager_dispatch(primitive):
    """Dispatch eager calls through lowerings even inside ``disable_jit``."""
    jit_bind = jax.jit(primitive.bind)

    def dispatch(*args):
        # A primitive still needs compilation; an uncompiled bind recurses
        # into this implementation when the caller disables ordinary JIT.
        with jax.disable_jit(False):
            return jit_bind(*args)

    primitive.def_impl(dispatch)


def register_cpu_product(primitive, call, shape):
    """Lower a bounded CPU shell product on CPU or through a CUDA host call."""
    # Eager calls must also dispatch through the platform-specific lowering.
    register_eager_dispatch(primitive)
    primitive.def_abstract_eval(lambda *args: jax.core.ShapedArray(shape(*args), jnp.float64))
    mlir.register_lowering(primitive, mlir.lower_fun(call, multiple_results=False), platform="cpu")
    host_call = jax.jit(call)

    def callback(*args):
        cpu = jax.local_devices(backend="cpu")[0]
        return np.asarray(host_call(*(jax.device_put(arg, cpu) for arg in args)))

    def cuda(*args):
        return jax.pure_callback(callback, jax.ShapeDtypeStruct(shape(*args), jnp.float64),
                                 *args, vmap_method="sequential")

    mlir.register_lowering(primitive, mlir.lower_fun(cuda, multiple_results=False), platform="cuda")
