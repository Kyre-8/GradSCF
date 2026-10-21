"""Opt-in CUDA component gate for the bounded native virtual ERI action.

The tiny physical ERI oracle and all its responses run on CPU. Native shell
work on CUDA uses the existing bounded CPU bridge; every action/AD output must
remain on the explicitly selected CUDA device. This is not a full CC gate.
"""
from dataclasses import replace
import os

import pytest

if os.environ.get('GRADSCF_TEST_GPU') != '1':
    pytest.skip('Explicit CUDA device selection is required.', allow_module_level=True)

import jax
import jax.numpy as jnp
import numpy as np

from gradscf.integrals.backends.native.jk import NativeDirectBasis, ProjectedNativeDirectBasis
from test_native_virtual_action import _action, _assert_trees, _dense, _problem


@pytest.mark.parametrize('projected', [False, True], ids=['native', 'projected'])
def test_native_virtual_action_cuda_joint_responses(projected, monkeypatch):
    devices = jax.devices('gpu')
    assert len(devices) == 1 and jax.default_backend() == 'gpu'
    cpu, gpu = jax.devices('cpu')[0], devices[0]

    with jax.default_device(cpu):
        primitive, _, c, transform = _problem(projected)
        parameters, plan = primitive.parameters, primitive.plan
        nnuc = len(plan.topology.nuclear_charges)
        coords = jnp.concatenate((parameters.nuclear_coords, parameters.centers))
        rng = np.random.default_rng(6412)
        rhs = jnp.asarray(rng.normal(size=(2, 3, 3)))
        point = (coords, c, rhs, transform)
        direction = tuple(jnp.asarray(rng.normal(scale=.03, size=p.shape)) for p in point)
        probe = jnp.asarray(rng.normal(size=rhs.shape))

    def evaluate(params, values, direct):
        r, coefficients, x, projection = values
        moving = NativeDirectBasis(plan, replace(params,
            nuclear_coords=r[:nnuc], centers=r[nnuc:]))
        if direct:
            provider = ProjectedNativeDirectBasis(moving, projection) if projected else moving
            return _action(coefficients, provider, max_memory=.5).contract(x)
        eri = _dense(moving, coefficients, projection if projected else None)
        return jnp.einsum('acbd,...cd->...ab', eri, x, precision='highest')

    def responses(params, values, tangent, cotangent, direct):
        function = lambda v: evaluate(params, v, direct)
        value, jvp = jax.jvp(function, (values,), (tangent,))
        _, pullback = jax.vjp(function, values)
        vjp, = pullback(cotangent)
        loss = lambda v: jnp.sum(function(v)**2)
        hvp = jax.jvp(jax.grad(loss), (values,), (tangent,))[1]
        return value, jvp, vjp, hvp

    with jax.default_device(cpu):
        expected = jax.jit(lambda values, tangent, cot: responses(
            parameters, values, tangent, cot, False))(point, direction, probe)
        jax.block_until_ready(expected)
        assert all(leaf.devices() == {cpu} for leaf in jax.tree.leaves(expected))

    original = type(plan).evaluate

    def forbid_full_eri(self, operator, *args, **kwargs):
        if operator == 'eri':
            raise AssertionError('CUDA virtual action requested a full AO ERI fallback')
        return original(self, operator, *args, **kwargs)

    monkeypatch.setattr(type(plan), 'evaluate', forbid_full_eri)
    with jax.default_device(gpu):
        gpu_parameters, gpu_point, gpu_direction, gpu_probe = jax.tree.map(
            lambda value: jax.device_put(value, gpu), (parameters, point, direction, probe))
        assert all(leaf.devices() == {gpu} for leaf in jax.tree.leaves(
            (gpu_parameters, gpu_point, gpu_direction, gpu_probe)))
        actual = jax.jit(lambda values, tangent, cot: responses(
            gpu_parameters, values, tangent, cot, True))(gpu_point, gpu_direction, gpu_probe)
        jax.block_until_ready(actual)
        assert all(leaf.devices() == {gpu} for leaf in jax.tree.leaves(actual))
        _assert_trees(actual, expected, atol=3e-9, rtol=3e-8)
