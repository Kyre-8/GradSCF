"""Native eager primitives still compile their lowering when JIT is disabled."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import integrals
from gradscf.integrals.backends.native.jk import direct_jk, packed_jk


@pytest.mark.parametrize("packed", [False, True], ids=["direct", "packed"])
def test_native_jk_eager_with_jit_disabled(packed):
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        topology, parameters = integrals.prepare_basis("H 0 0 0; H .2 -.1 .8", "sto-3g")
        plan = integrals.make_plan(topology)
        atm, bas, env = plan.pack(parameters)
        full = plan.evaluate("eri", parameters)
        eri = plan.evaluate("eri", parameters, aosym="s8")
        density = jnp.array([[.8, -.2], [.3, .6]])
        expected = (jnp.einsum("pqrs,rs->pq", full, density),
                    jnp.einsum("prqs,rs->pq", full, density))
        with jax.disable_jit(True):
            actual = (packed_jk(eri, density) if packed else
                      direct_jk(atm, bas, env, density, cart=topology.cart))
            actual = jax.block_until_ready(actual)
        for observed, reference in zip(actual, expected):
            assert observed.devices() == {cpu}
            assert observed.shape == reference.shape and observed.dtype == reference.dtype
            assert np.isfinite(np.asarray(observed)).all()
            assert np.isfinite(np.asarray(reference)).all()
            np.testing.assert_allclose(observed, reference, atol=3e-12, rtol=3e-12,
                                       equal_nan=False)
