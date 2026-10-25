"""Opt-in CUDA shell slabs use a bounded CPU callback and retain CUDA outputs.

Run with GRADSCF_TEST_GPU=1 and an explicitly selected CUDA_VISIBLE_DEVICES.
The legacy full-ERI oracle is evaluated only on CPU; geometry is in Bohr.
"""
from dataclasses import replace
import os

import pytest

if os.environ.get("GRADSCF_TEST_GPU") != "1":
    pytest.skip("Set GRADSCF_TEST_GPU=1 to enable the explicit CUDA test.",
                allow_module_level=True)

import jax
import jax.numpy as jnp
import numpy as np

from gradscf import integrals
from gradscf.integrals.backends.native.autodiff.geometry import evaluate_eri_slab


def _responses(value, coords, direction, cotangent):
    primal, tangent = jax.jit(lambda r, d: jax.jvp(value, (r,), (d,)))(coords, direction)
    gradient = jax.jit(jax.grad(lambda r: jnp.vdot(value(r), cotangent)))(coords)

    def loss(r):
        block = value(r)
        return jnp.sum(jnp.sin(.3 * block) + .1 * block**2)

    nonlinear_gradient = jax.grad(loss)
    hvp = jax.jit(lambda r, d: jax.jvp(nonlinear_gradient, (r,), (d,))[1])(
        coords, direction)
    return primal, tangent, gradient, hvp


@pytest.mark.parametrize('dynamic', [False, True], ids=['static', 'dynamic-padded'])
def test_native_shell_slice_cuda_value_and_coordinate_response(dynamic):
    cpu = jax.devices("cpu")[0]
    gpu = jax.devices("gpu")[0]
    shell_slice = (1, 3, 2, 4, 1, 2, 3, 4)
    # H2/3-21G has four singly contracted s shells: shell and AO offsets agree.
    ao_slices = (slice(1, 3), slice(2, 4), slice(1, 2), slice(3, 4))

    with jax.default_device(cpu):
        top, params = integrals.prepare_basis("H 0 0 0; H .2 -.1 .8", basis="3-21g")
        assert top.angular_momenta == (0, 0, 0, 0)
        assert top.contraction_counts == (1, 1, 1, 1)
        plan = integrals.make_plan(top, backend="native")
        centers = params.centers + jnp.arange(params.centers.size).reshape(-1, 3) * .003
        coords = jnp.concatenate((params.nuclear_coords, centers))
        rng = np.random.default_rng(20261022)
        direction = jnp.asarray(rng.normal(scale=.15, size=coords.shape))
        shape = (3, 3, 2, 2) if dynamic else (2, 2, 1, 1)
        cotangent = jnp.asarray(rng.normal(size=shape))

        def reference(r):
            p = replace(params, nuclear_coords=r[:2], centers=r[2:])
            block = plan.evaluate("eri", p)[ao_slices]
            return jnp.pad(block, ((0, 1),)*4) if dynamic else block

        expected = _responses(reference, coords, direction, cotangent)
        for result in expected:
            assert result.devices() == {cpu}

    with jax.default_device(gpu):
        gpu_params = jax.tree.map(lambda x: jax.device_put(x, gpu), params)
        gpu_coords, gpu_direction, gpu_cotangent = (
            jax.device_put(x, gpu) for x in (coords, direction, cotangent))
        assert all(x.devices() == {gpu} for x in jax.tree.leaves(
            (gpu_params, gpu_coords, gpu_direction, gpu_cotangent)))

        def value(r):
            p = replace(gpu_params, nuclear_coords=r[:2], centers=r[2:])
            if dynamic:
                fixed = replace(p, nuclear_coords=jnp.zeros_like(p.nuclear_coords),
                                centers=jnp.zeros_like(p.centers))
                atm, bas, env = plan.pack(fixed)
                return evaluate_eri_slab(atm, bas, env, r,
                    jnp.asarray(shell_slice, dtype=jnp.int32), shape, cart=True)
            return plan.evaluate("eri", p, shell_slice=shell_slice)

        eager = value(gpu_coords)
        plain_vjp = jax.grad(lambda r: jnp.vdot(value(r), gpu_cotangent))(gpu_coords)
        for observed, reference in ((eager, expected[0]), (plain_vjp, expected[2])):
            assert observed.devices() == {gpu}
            np.testing.assert_allclose(observed, jax.device_put(reference, gpu),
                                       atol=3e-10, rtol=3e-9)

        actual = _responses(value, gpu_coords, gpu_direction, gpu_cotangent)
        for observed, reference in zip(actual, expected):
            assert observed.devices() == {gpu}
            np.testing.assert_allclose(observed, jax.device_put(reference, gpu),
                                       atol=3e-10, rtol=3e-9)
