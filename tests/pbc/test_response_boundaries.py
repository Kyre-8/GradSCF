"""Reject electrostatic models outside the implemented response equations."""
from types import SimpleNamespace

import numpy as np
import pytest

from gradscf.pbc import dft, gto, scf
from gradscf.pbc.momentum import momentum_transfer
from gradscf.pbc.response import q_kernel
from gradscf.pbc.tdscf import response_operators
from gradscf.dfpt.displacement import displacement_source
from gradscf.dfpt.finite_q import finite_q_problem
from gradscf.dfpt.force_constants import geometry_curvature


@pytest.fixture(scope='module', params=['hf', 'pbe0'])
def spherical_reference(request):
    cell = gto.M(atom='H .2 .3 .4; H 1.6 .3 .4', a=np.eye(3)*6,
        unit='Bohr', basis='gth-szv', pseudo='gth-pade', mesh=(7,)*3)
    options = dict(kpts=cell.make_kpts((2, 1, 1)), exxdiv='vcut_sph',
        conv_tol=1e-11, conv_tol_density=1e-10, conv_tol_grad=1e-9)
    mf = (scf.KRHF(cell, **options) if request.param == 'hf'
          else dft.KRKS(cell, xc=request.param, **options)).run()
    assert mf.converged
    return mf


def test_tda_rejects_spherical_exchange_reference(spherical_reference):
    with pytest.raises(NotImplementedError, match='vcut_sph'):
        response_operators(spherical_reference)


@pytest.mark.parametrize('consumer', [displacement_source, geometry_curvature])
def test_atomic_q_sources_reject_spherical_exchange(spherical_reference, consumer):
    direction = np.array([[.1, -.2, .3], [-.1, .2, -.3]])
    arguments = (direction,) if consumer is displacement_source else (direction, direction)
    with pytest.raises(NotImplementedError, match='vcut_sph'):
        consumer(spherical_reference, np.array([.5, 0., 0.]), *arguments)


@pytest.mark.parametrize('dimension', [1, 2])
@pytest.mark.parametrize('consumer', [q_kernel, finite_q_problem])
def test_array_q_response_rejects_reduced_electrostatics_before_assembly(dimension, consumer):
    # A boundary is sufficient to reject the unsupported model, before reading
    # any reference arrays or entering an expensive response construction.
    inputs = SimpleNamespace(boundary=SimpleNamespace(dimension=dimension))
    with pytest.raises(NotImplementedError, match='3D'):
        consumer(inputs, None, None, mesh=(7,)*3)


@pytest.mark.parametrize('exxdiv', ['ewald', None])
def test_existing_3d_q_kernels_remain_available(spherical_reference, exxdiv):
    mf = spherical_reference
    transfer = momentum_transfer(np.asarray(mf.kpts)@np.asarray(mf.cell.lattice).T/(2*np.pi),
                                 np.array([.5, 0., 0.]))
    action = q_kernel(mf.inputs, mf.result.density_spin, transfer,
                     mesh=mf.cell.mesh, xc=mf.xc, exxdiv=exxdiv)
    zero = np.zeros_like(mf.result.density_spin)
    np.testing.assert_allclose(action(zero), zero, atol=1e-14, rtol=0.)
