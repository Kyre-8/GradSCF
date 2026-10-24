"""Opt-in CPU/CUDA complete direct MP2 geometry response at degenerate N2.

STO-3G Cartesian, float64, Hartree/Bohr; native CPU integral bridges remain
explicit. This is a small correctness/placement gate, not a scaling benchmark.
"""
import os

import pytest

if os.environ.get('GRADSCF_TEST_GPU') != '1':
    pytest.skip('Select the allocated GPU explicitly.', allow_module_level=True)

import jax
import jax.numpy as jnp
import numpy as np

from gradscf import gto, scf
from gradscf_tools.correlated_response import CorrelatedEnergy
from test_correlated_direct import _forbid_full_integrals


def test_nitrogen_direct_mp2_geometry_cpu_cuda(monkeypatch):
    cpu, gpu = jax.devices('cpu')[0], jax.devices('gpu')[0]
    assert jax.default_backend() == 'gpu'
    direction = np.array([[0., 0., -.5], [0., 0., .5]])

    def calculate(device, forbid_full):
        print(f"MP2 geometry: {device} SCF", flush=True)
        with jax.default_device(device):
            mol = gto.M(atom='N 0 0 -1.04; N 0 0 1.04', basis='sto-3g',
                         unit='Bohr', cart=True)
            mf = scf.RHF(mol, integral_backend='native', jk_backend='direct',
                execution_device=('gpu' if device == gpu else 'cpu'), max_memory=64.,
                conv_tol=1e-13, conv_tol_density=1e-11, conv_tol_grad=1e-10,
                max_cycle=250, init_guess=1.4*np.eye(10)).run()
            assert mf.converged
            assert mf.mo_coeff.devices() == {device}
            assert np.min(np.abs(np.diff(np.asarray(mf.mo_energy)))) < 1e-9
            energy = CorrelatedEnergy(mf, 'mp2', jk_backend='direct',
                max_memory=64., gradient_tolerance=1e-10)
            if forbid_full:
                _forbid_full_integrals(monkeypatch)
            point = jax.device_put(energy.coordinates, device)
            tangent = jax.device_put(direction, device)
            derivative = jax.jit(jax.value_and_grad(energy))
            print(f"MP2 geometry: {device} gradient", flush=True)
            value, gradient = jax.block_until_ready(derivative(point))
            print(f"MP2 geometry: {device} HVP", flush=True)
            hvp = jax.block_until_ready(jax.jit(lambda r: jax.jvp(
                jax.grad(energy), (r,), (tangent,))[1])(point))
            for array in (value, gradient, hvp):
                assert array.devices() == {device}
                assert np.isfinite(np.asarray(array)).all()
            assert abs(float(value-mf.e_tot)) > 1e-6
            assert np.linalg.norm(np.asarray(gradient)) > 1e-6
            assert np.linalg.norm(np.asarray(hvp)) > 1e-6
            np.testing.assert_allclose(gradient.sum(axis=0), 0., atol=2e-8, rtol=0.)
            return tuple(np.asarray(x) for x in (value, gradient, hvp))

    expected = calculate(cpu, False)
    actual = calculate(gpu, True)
    for observed, reference, tolerance in zip(actual, expected, (3e-9, 3e-7, 3e-6)):
        np.testing.assert_allclose(observed, reference, atol=tolerance, rtol=0. if reference.ndim == 0 else 3e-6)
