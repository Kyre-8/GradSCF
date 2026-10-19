"""Full signed HF query at source k points, with the actual SCF potential."""
import importlib

import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.integrals.periodic.fft import get_kpoint_jk
from gradscf.pbc import scf
from gradscf.solvers.nonlinear import ScalarRootConfig


query = importlib.import_module('gradscf.gw.pbc.query')


@pytest.fixture(scope='module')
def source_hf():
    # The original native two-k HF regression, whose folded response is not
    # certified on a finite FFT grid. Keep its reference unchanged.
    from test_kpoints_gw import _cell
    cell = _cell()
    mf = scf.KRHF(cell, kpts=cell.make_kpts((2, 1, 1)),
        conv_tol=1e-11, conv_tol_density=1e-10, max_cycle=150).run()
    assert mf.converged
    return mf


def test_native_hf_source_query_projects_the_actual_nonhartree_potential(source_hf):
    mf = source_hf
    data = query._prepare(mf, np.asarray(mf.inputs.kpoints), [0, 1], nw=16, eta=.001,
        fc=False, max_memory_bytes=512*1024**2, gap_tol=1e-7, pole_subtraction=True)
    _, exchange = get_kpoint_jk(mf.inputs, mf.result.density_spin, mesh=mf.cell.mesh,
                              exxdiv=mf.exxdiv, with_k=True)
    coefficient = data['coefficients']
    expected = -jnp.einsum('kam,kab,kbn->kmn', coefficient.conj(), exchange[0], coefficient)
    np.testing.assert_allclose(data['vxc'], jnp.diagonal(expected, axis1=-2, axis2=-1).real,
                               atol=3e-9, rtol=0)
    for k in range(2):
        np.testing.assert_allclose(data['vxc_blocks'][k, (0, 1)], expected[k], atol=3e-9, rtol=0)


def test_native_hf_source_query_has_a_full_signed_forward_path(source_hf):
    result = query.g0w0_bands(source_hf, np.asarray(source_hf.inputs.kpoints), [0, 1],
        nw=16, eta=.001, fc=False, pole_subtraction=True,
        root_config=ScalarRootConfig(method='hybrid', ftol=1e-9, xtol=1e-9, maxiter=120))
    assert bool(result.converged)
    assert np.asarray(result.qp_computed_mask).all()
    assert np.asarray(result.qp_branch_valid).all()
    assert np.isfinite(np.asarray(result.mo_energy)).all()
    assert np.isfinite(np.asarray(result.qp_weight)).all()
    assert np.max(np.abs(np.asarray(result.qp_residual))) < 1e-9


def test_hf_offmesh_query_is_rejected_before_band_construction(source_hf, monkeypatch):
    reciprocal = 2*np.pi*np.linalg.inv(np.asarray(source_hf.inputs.lattice)).T
    points = np.asarray(source_hf.inputs.kpoints[:1])+.13*reciprocal[:1]
    def forbidden(*args, **kwargs):
        pytest.fail('Unsupported off-mesh HF must be rejected before any band construction')
    monkeypatch.setattr(source_hf, 'get_bands', forbidden)
    with pytest.raises(NotImplementedError, match='HF.*source|source.*HF'):
        query.g0w0_bands(source_hf, points, [0], fc=False)


def test_hf_gamma_cell_query_does_not_silently_evaluate_offmesh_hf(source_hf):
    with pytest.raises(NotImplementedError, match='HF.*fc=False|fc=False.*HF'):
        query.g0w0_bands(source_hf, np.asarray(source_hf.inputs.kpoints), [0], fc=True)
