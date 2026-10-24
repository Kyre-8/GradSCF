"""Opt-in CUDA eager/JIT J/K dispatch and bounded coordinate response.

Run with GRADSCF_TEST_GPU=1 and an explicitly selected CUDA device. The
independent full-ERI oracle stays on CPU; coordinates are in Bohr, float64.
"""
from dataclasses import replace
import os
from types import SimpleNamespace

import pytest

if os.environ.get("GRADSCF_TEST_GPU") != "1":
    pytest.skip("Set GRADSCF_TEST_GPU=1 and select the allocated CUDA device.",
                allow_module_level=True)

import jax
import jax.numpy as jnp
import numpy as np

from gradscf import integrals
from gradscf.integrals.backends.native.jk import direct_jk, packed_jk
from test_direct_geometry_response import problem, _forbid_eri


@pytest.fixture(params=[True, False], ids=["cartesian", "spherical"])
def cpu_problem(request):
    with jax.default_device(jax.devices("cpu")[0]):
        return problem.__wrapped__(request)


def _assert_cuda_close(actual, expected, gpu, *, atol=3e-10, rtol=3e-9):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for observed, reference in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        assert observed.shape == reference.shape
        assert observed.dtype == reference.dtype
        assert observed.devices() == {gpu}
        assert reference.devices() == {jax.devices("cpu")[0]}
        assert np.isfinite(np.asarray(observed)).all()
        assert np.isfinite(np.asarray(reference)).all()
        np.testing.assert_allclose(observed, np.asarray(reference), atol=atol, rtol=rtol,
                                   equal_nan=False)


@pytest.mark.parametrize("with_j,with_k", [(True, True), (True, False),
                                          (False, True), (False, False)])
@pytest.mark.parametrize("packed", [False, True], ids=["direct", "packed"])
def test_native_jk_cuda_eager_value_and_density_response(cpu_problem, with_j, with_k, packed):
    plan, parameters, coords, density, _, _, _, oracle = cpu_problem
    cpu, gpu = jax.devices("cpu")[0], jax.devices("gpu")[0]
    with jax.default_device(cpu):
        density = density + .2j*jnp.cos(density)
        direction = .13*jnp.sin(density)
        expected_value, expected_jvp = jax.jvp(
            lambda d: oracle(coords, d, with_j=with_j, with_k=with_k),
            (density,), (direction,))
        loss = lambda d: sum(jnp.real(jnp.vdot(x, x)) for x in
                            oracle(coords, d, with_j=with_j, with_k=with_k))
        expected_gradient = jax.grad(loss)(density)
        expected_hvp = jax.jvp(jax.grad(loss), (density,), (direction,))[1]
        eri = plan.evaluate("eri", parameters, aosym="s8") if packed else None
        atm, bas, env = plan.pack(parameters)

    with jax.default_device(gpu):
        dm, delta = (jax.device_put(x, gpu) for x in (density, direction))
        if packed:
            values = jax.device_put(eri, gpu)
            action = lambda d: packed_jk(values, d, with_j=with_j, with_k=with_k)
        else:
            fixed_env = jax.device_put(env, gpu)
            action = lambda d: direct_jk(atm, bas, fixed_env, d,
                cart=plan.topology.cart, workers=2, with_j=with_j, with_k=with_k)

        def responses(d, tangent):
            value, push = jax.jvp(action, (d,), (tangent,))
            objective = lambda x: sum(jnp.real(jnp.vdot(v, v)) for v in action(x))
            gradient = jax.grad(objective)(d)
            hvp = jax.jvp(jax.grad(objective), (d,), (tangent,))[1]
            return value, push, gradient, hvp

        expected = expected_value, expected_jvp, expected_gradient, expected_hvp
        for evaluate in (responses, jax.jit(responses)):
            _assert_cuda_close(jax.block_until_ready(evaluate(dm, delta)), expected, gpu)


@pytest.mark.parametrize("with_j,with_k", [(True, True), (True, False), (False, True)])
def test_native_direct_cuda_eager_coordinate_density_jvp_vjp_hvp(
        cpu_problem, monkeypatch, with_j, with_k):
    plan, parameters, coords, density, direction, _, _, oracle = cpu_problem
    cpu, gpu = jax.devices("cpu")[0], jax.devices("gpu")[0]

    def responses(operation, r, d, u, v):
        apply = lambda x, y: operation(x, y, with_j=with_j, with_k=with_k)
        value, tangent = jax.jvp(apply, (r, d), (u, v))
        objective = lambda x, y: sum(weight*jnp.sum(z*z) for weight, z in
                                    zip((.7, .3), apply(x, y)))
        gradient = jax.grad(objective, (0, 1))
        return value, tangent, gradient(r, d), jax.jvp(gradient, (r, d), (u, v))[1]

    with jax.default_device(cpu):
        density_direction = .13*jnp.sin(density)
        expected = responses(oracle, coords, density, direction, density_direction)
    _forbid_eri(monkeypatch)
    with jax.default_device(gpu):
        gpu_parameters = jax.tree.map(lambda x: jax.device_put(x, gpu), parameters)
        nnuc = len(plan.topology.nuclear_charges)
        action = lambda r, d, **options: plan.get_jk(replace(gpu_parameters,
            nuclear_coords=r[:nnuc], centers=r[nnuc:]), d, workers=2, **options)
        arrays = tuple(jax.device_put(x, gpu) for x in
                       (coords, density, direction, density_direction))
        evaluate = lambda r, d, u, v: responses(action, r, d, u, v)
        for function in (evaluate, jax.jit(evaluate)):
            observed = jax.block_until_ready(function(*arrays))
            _assert_cuda_close(observed, expected, gpu, atol=2e-9, rtol=3e-9)


def test_native_packed_cuda_joint_eri_density_response():
    cpu, gpu = jax.devices("cpu")[0], jax.devices("gpu")[0]
    with jax.default_device(cpu):
        plan, parameters, _, density, _, _, _, _ = problem.__wrapped__(SimpleNamespace(param=True))
        eri = plan.evaluate("eri", parameters, aosym="s8")
        n = plan.topology.nao
        ids = np.arange(n)
        hi, lo = np.maximum(ids[:, None], ids), np.minimum(ids[:, None], ids)
        pairs = hi*(hi+1)//2 + lo
        hi = np.maximum(pairs[:, :, None, None], pairs[None, None, :, :])
        lo = np.minimum(pairs[:, :, None, None], pairs[None, None, :, :])
        indices = jnp.asarray(hi*(hi+1)//2 + lo)
        density = density + .2j*jnp.cos(density)
        direction = .03*jnp.cos(eri), .13*jnp.sin(density)

        def oracle(e, d):
            full = e[indices]
            return jnp.einsum("pqrs,brs->bpq", full, d), jnp.einsum("prqs,brs->bpq", full, d)

        def responses(operation, e, d, u, v):
            value, tangent = jax.jvp(operation, (e, d), (u, v))
            objective = lambda x, y: sum(jnp.real(jnp.vdot(z, z)) for z in operation(x, y))
            gradient = jax.grad(objective, (0, 1))
            return value, tangent, gradient(e, d), jax.jvp(gradient, (e, d), (u, v))[1]

        expected = responses(oracle, eri, density, *direction)
    with jax.default_device(gpu):
        arrays = tuple(jax.device_put(x, gpu) for x in (eri, density, *direction))
        evaluate = lambda e, d, u, v: responses(packed_jk, e, d, u, v)
        for function in (evaluate, jax.jit(evaluate)):
            _assert_cuda_close(jax.block_until_ready(function(*arrays)), expected, gpu,
                               atol=2e-9, rtol=3e-9)


def test_native_cuda_spherical_d_shell_value_and_density_jvp():
    cpu, gpu = jax.devices("cpu")[0], jax.devices("gpu")[0]
    cart = False
    with jax.default_device(cpu):
        topology, parameters = integrals.prepare_basis("H 0 0 0; H .2 -.1 .8",
            basis={"H": [[0, [1.1, 1.]], [2, [.7, 1.]]]}, cart=cart)
        assert topology.nao == 12
        plan = integrals.make_plan(topology)
        atm, bas, env = plan.pack(parameters)
        full = plan.evaluate("eri", parameters)
        eri = plan.evaluate("eri", parameters, aosym="s8")
        density = jnp.asarray(np.random.default_rng(935).normal(size=(2, topology.nao, topology.nao)))
        density = density + .2j*jnp.cos(density)
        direction = .13*jnp.sin(density)
        oracle = lambda d: (jnp.einsum("pqrs,brs->bpq", full, d),
                            jnp.einsum("prqs,brs->bpq", full, d))
        expected = jax.jvp(oracle, (density,), (direction,))
    with jax.default_device(gpu):
        gpu_env, gpu_eri, dm, delta = (jax.device_put(x, gpu) for x in
                                      (env, eri, density, direction))
        actions = (lambda d: direct_jk(atm, bas, gpu_env, d, cart=cart, workers=2),
                   lambda d: packed_jk(gpu_eri, d))
        for action in actions:
            evaluate = lambda d, v: jax.jvp(action, (d,), (v,))
            for function in (evaluate, jax.jit(evaluate)):
                _assert_cuda_close(jax.block_until_ready(function(dm, delta)), expected, gpu)


def test_native_cuda_eager_values_with_jit_disabled():
    cpu, gpu = jax.devices("cpu")[0], jax.devices("gpu")[0]
    with jax.default_device(cpu):
        plan, parameters, coords, density, _, _, _, oracle = problem.__wrapped__(
            SimpleNamespace(param=True))
        atm, bas, env = plan.pack(parameters)
        eri = plan.evaluate("eri", parameters, aosym="s8")
        expected = oracle(coords, density)
    with jax.default_device(gpu), jax.disable_jit(True):
        gpu_env, gpu_eri, dm = (jax.device_put(x, gpu) for x in (env, eri, density))
        direct = direct_jk(atm, bas, gpu_env, dm, cart=True)
        packed = packed_jk(gpu_eri, dm)
        for actual in (direct, packed):
            _assert_cuda_close(jax.block_until_ready(actual), expected, gpu)
