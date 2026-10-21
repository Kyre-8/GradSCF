"""Opt-in native direct MP2 geometry response on one explicitly visible GPU."""
import os

import pytest

if os.environ.get('GRADSCF_TEST_GPU') != '1':
    pytest.skip('Set GRADSCF_TEST_GPU=1 on an authorized visible GPU', allow_module_level=True)

import jax
import jax.numpy as jnp
import numpy as np


def test_native_direct_mp2_gpu_relaxed_gradient_and_hvp(monkeypatch):
    from gradscf import gto, scf
    from gradscf.integrals.plan import IntegralPlan
    from gradscf_tools.correlated_response import CorrelatedEnergy

    gpu, = jax.devices('gpu')
    cpu = jax.devices('cpu')[0]
    with jax.default_device(cpu):
        molecule = gto.M(atom='H 0 0 -.75; H 0 0 .75', basis='3-21g',
                         unit='Bohr', cart=True)
        source = scf.RHF(molecule, integral_backend='native', execution_device='cpu',
            conv_tol=1e-12, conv_tol_density=1e-11, conv_tol_grad=1e-10).run()
        assert source.converged
        dense = CorrelatedEnergy(source, 'mp2')
        direction = jnp.array([[.1, 0., -.5], [-.1, 0., .5]])
        expected = jax.jit(jax.value_and_grad(dense))(dense.coordinates)
        expected_hvp = jax.jit(lambda r: jax.jvp(jax.grad(dense),
            (r,), (direction,))[1])(dense.coordinates)
        jax.block_until_ready((expected, expected_hvp))
    original = IntegralPlan.evaluate

    def checked(self, operator, *args, **kwargs):
        if operator == 'eri':
            pytest.fail('GPU direct MP2 evaluated a full AO ERI')
        return original(self, operator, *args, **kwargs)

    monkeypatch.setattr(IntegralPlan, 'evaluate', checked)
    with jax.default_device(gpu):
        direct = CorrelatedEnergy(source, 'mp2', jk_backend='direct', max_memory=32.)
        direction_gpu = jax.device_put(direction, gpu)
        actual = jax.jit(jax.value_and_grad(direct))(direct.coordinates)
        actual_hvp = jax.jit(lambda r: jax.jvp(jax.grad(direct),
            (r,), (direction_gpu,))[1])(direct.coordinates)
        jax.block_until_ready((actual, actual_hvp))
    for a, b in zip(jax.tree.leaves((actual, actual_hvp)),
                    jax.tree.leaves((expected, expected_hvp))):
        assert a.devices() == {gpu}
        np.testing.assert_allclose(a, b, atol=3e-7, rtol=2e-6)


@pytest.mark.parametrize('operator', ['overlap', 'kinetic', 'nuclear', 'dipole'])
def test_one_electron_gpu_geometry_products_match_cpu(operator):
    from dataclasses import replace
    from gradscf import gto, integrals

    gpu, = jax.devices('gpu')
    cpu = jax.devices('cpu')[0]
    with jax.default_device(cpu):
        molecule = gto.M(atom='H 0 0 0; He .2 .1 1.5', basis='sto-3g',
                         spin=1, unit='Bohr', cart=True)
        topology, parameters = integrals.prepare_basis(molecule.to_spec(), molecule.basis, cart=True)
        plan = integrals.make_plan(topology, backend='native')
        coordinates = jnp.asarray(parameters.centers)
        direction = jnp.full_like(coordinates, .01).at[0, 2].set(.04)
        def value(r):
            moving = replace(parameters, centers=r, nuclear_coords=r)
            return plan.evaluate(operator, moving)
        def loss(r):
            return jnp.sum(value(r)**2)
        expected = jax.jit(jax.value_and_grad(loss))(coordinates)
        expected_hvp = jax.jit(lambda r: jax.jvp(jax.grad(loss),
                              (r,), (direction,))[1])(coordinates)
        jax.block_until_ready((expected, expected_hvp))
    with jax.default_device(gpu):
        r_gpu = jax.device_put(coordinates, gpu)
        d_gpu = jax.device_put(direction, gpu)
        actual = jax.jit(jax.value_and_grad(loss))(r_gpu)
        actual_hvp = jax.jit(lambda r: jax.jvp(jax.grad(loss),
                            (r,), (d_gpu,))[1])(r_gpu)
        jax.block_until_ready((actual, actual_hvp))
    for a, b in zip(jax.tree.leaves((actual, actual_hvp)),
                    jax.tree.leaves((expected, expected_hvp))):
        assert a.devices() == {gpu}
        np.testing.assert_allclose(a, b, atol=2e-9, rtol=2e-9)
