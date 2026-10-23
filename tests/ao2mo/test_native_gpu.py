"""Opt-in GPU AO2MO: CPU dense oracle, native bounded CPU shell bridge.

Set GRADSCF_TEST_GPU=1 and explicitly select a permitted CUDA device. Geometry
is in Bohr; all arrays are float64. This tests correctness, not a speedup.
"""
from dataclasses import replace
import os

import pytest

if os.environ.get('GRADSCF_TEST_GPU') != '1':
    pytest.skip('Explicit GPU selection is required.', allow_module_level=True)

import jax
import jax.numpy as jnp
import numpy as np

from gradscf import ao2mo, integrals
from gradscf.integrals.backends.native.jk import NativeDirectBasis


def test_native_ao2mo_cuda_coefficient_and_coordinate_hvp():
    cpu, gpu = jax.devices('cpu')[0], jax.devices('gpu')[0]
    with jax.default_device(cpu):
        top, parameters = integrals.prepare_basis('H 0 0 0; H .2 -.1 .8', basis='3-21g')
        plan = integrals.make_plan(top, backend='native')
        rng = np.random.default_rng(220)
        coefficients = tuple(jnp.asarray(rng.normal(scale=.2, size=(top.nao, n)))
                             for n in (2, 3, 1, 2))
        coordinates = jnp.concatenate((parameters.nuclear_coords, parameters.centers))
        inputs = coordinates, coefficients
        direction = jax.tree.map(lambda a: jnp.full_like(a, .03), inputs)

        def loss(r, cs, params, direct):
            moving = replace(params, nuclear_coords=r[:2], centers=r[2:])
            if direct:
                g = ao2mo.transform_eri_block(cs,
                    direct_basis=NativeDirectBasis(plan, moving), max_memory=.12)
            else:
                g = jnp.einsum('pqrs,pP,qQ,rR,sS->PQRS',
                    plan.evaluate('eri', moving), *cs, precision='highest', optimize='optimal')
            return jnp.sum(jnp.sin(.2*g) + .03*g*g)

        dense = lambda r, cs: loss(r, cs, parameters, False)
        expected = jax.jit(jax.value_and_grad(dense, argnums=(0, 1)))(*inputs)
        expected_hvp = jax.jit(lambda r, cs, d: jax.jvp(
            jax.grad(dense, argnums=(0, 1)), (r, cs), d)[1])(*inputs, direction)
        jax.block_until_ready((expected, expected_hvp))

    with jax.default_device(gpu):
        gpu_parameters = jax.tree.map(lambda a: jax.device_put(a, gpu), parameters)
        gpu_inputs = jax.tree.map(lambda a: jax.device_put(a, gpu), inputs)
        gpu_direction = jax.tree.map(lambda a: jax.device_put(a, gpu), direction)
        direct = lambda r, cs: loss(r, cs, gpu_parameters, True)
        eager = direct(*gpu_inputs)
        actual = jax.jit(jax.value_and_grad(direct, argnums=(0, 1)))(*gpu_inputs)
        actual_hvp = jax.jit(lambda r, cs, d: jax.jvp(
            jax.grad(direct, argnums=(0, 1)), (r, cs), d)[1])(*gpu_inputs, gpu_direction)
        assert eager.devices() == {gpu}
        np.testing.assert_allclose(eager, expected[0], atol=3e-11, rtol=3e-10)
        for a, e in zip(jax.tree.leaves((actual, actual_hvp)),
                        jax.tree.leaves((expected, expected_hvp))):
            assert a.devices() == {gpu}
            np.testing.assert_allclose(a, e, atol=3e-10, rtol=3e-9)


def test_cuda_traced_capacity_is_shared_by_blocks_and_solver(monkeypatch):
    from gradscf import _memory
    from gradscf.solvers import solve_tensor_sum

    cpu, gpu = jax.devices('cpu')[0], jax.devices('gpu')[0]
    rng = np.random.default_rng(225)
    with jax.default_device(cpu):
        factors = jnp.asarray(rng.normal(scale=.1, size=(17, 4, 4)))
        coefficients = tuple(jnp.asarray(rng.normal(scale=.2, size=(4, n)))
                             for n in (2, 3, 2, 1))
        expected = jnp.einsum('Qpq,Qrs,pi,qj,rk,sl->ijkl', factors, factors,
                              *coefficients, precision='highest')
        expected = expected / 4.

    capacity = _memory.available_memory_bytes(gpu)
    assert capacity is not None and capacity > 4500
    observed = []

    def limited_capacity(device=None):
        observed.append(device)
        assert device is gpu, 'A traced planner fell back to host capacity'
        return 4500

    monkeypatch.setattr(_memory, 'available_memory_bytes', limited_capacity)

    @jax.jit
    def action(b, cs):
        block = ao2mo.transform_eri_block(cs, df_factors=b, max_memory=60.)
        return solve_tensor_sum(tuple(jnp.eye(n) for n in block.shape), block,
                                max_memory=60.).solution

    with jax.default_device(gpu):
        inputs = jax.tree.map(lambda x: jax.device_put(x, gpu), (factors, coefficients))
        actual = action(*inputs)
        assert actual.devices() == {gpu}
        assert bool(jnp.all(jnp.isfinite(actual)))
        np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-11)
    assert len(observed) >= 2 and all(device is gpu for device in observed)
