"""SCF-backed variational CI shares selected kernels for R/U/common RO frames."""
import importlib

import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import ci, gto, scf
from gradscf.ao2mo.scf import prepare_scf_source
from gradscf.scf.reference import unrestricted_reference_from_source
from gradscf.ci.integrals import reference_from_source


@pytest.fixture(scope='module', params=['rhf', 'uhf', 'rohf'])
def source(request):
    closed = request.param == 'rhf'
    molecule = gto.M(atom=('H 0 0 0; H 0 0 1.4' if closed else
                          'H 0 0 0; H 0 0 1.6; H 1.8 .2 .1'),
                     basis='sto-3g', spin=0 if closed else 1, unit='Bohr', cart=True)
    cls = {'rhf': scf.RHF, 'uhf': scf.UHF, 'rohf': scf.ROHF}[request.param]
    mf = cls(molecule, integral_backend='native', execution_device='cpu',
             conv_tol=1e-12, conv_tol_grad=1e-10, conv_tol_density=1e-11,
             max_cycle=250, max_memory=64.).run()
    assert mf.converged
    return request.param, mf


def test_scf_ci_uses_selected_integrals(source, monkeypatch):
    kind, mf = source
    unrestricted = kind != 'rhf'
    reference = (unrestricted_reference_from_source(mf) if unrestricted else reference_from_source(mf))
    expected = ci.CISD(reference, solver='dense', conv_tol=1e-10).run()
    module = importlib.import_module('gradscf.ci.api')
    def forbidden(*args, **kwargs):
        pytest.fail('SCF CI requested a complete MO Hamiltonian')
    monkeypatch.setattr(module, 'reference_from_source', forbidden)
    monkeypatch.setattr(module, 'unrestricted_reference_from_source', forbidden)
    actual = ci.CISD(mf, solver='dense', conv_tol=1e-10).run()
    assert actual.converged
    np.testing.assert_allclose(actual.e_tot, expected.e_tot, atol=3e-10)
    np.testing.assert_allclose(actual.e_corr, expected.e_corr, atol=3e-10)
    np.testing.assert_allclose(actual.make_rdm1(), expected.make_rdm1(), atol=3e-9)
    assert actual.max_memory == 64.


@pytest.mark.parametrize('source',['rohf'],indirect=True)
def test_rohf_adapter_is_opt_in_and_retains_saved_spin_fock(source):
    kind, mf = source
    with pytest.raises(NotImplementedError, match='ROHF'):
        prepare_scf_source(mf)
    data = prepare_scf_source(mf, allow_rohf=True)
    assert data.nocc == (2, 1)
    c = mf.mo_coeff
    for actual, f in zip(data.focks, (mf.scf_result.fock_matrix_alpha, mf.scf_result.fock_matrix_beta)):
        np.testing.assert_allclose(actual, c.T@f@c, atol=2e-12)
    np.testing.assert_allclose(data.reference_energy, mf.scf_result.total_energy, atol=0.)


@pytest.mark.parametrize('source', ['uhf'], indirect=True)
@pytest.mark.parametrize('frozen', [None, ([0], [2])])
def test_scf_ucis_uses_selected_integrals(source, frozen, monkeypatch):
    _, mf = source
    expected = ci.UCIS(unrestricted_reference_from_source(mf), frozen=frozen,
                       solver='dense').run()
    module = importlib.import_module('gradscf.ci.api')

    def forbidden(*args, **kwargs):
        pytest.fail('SCF UCIS materialized a complete MO/spin Hamiltonian')

    monkeypatch.setattr(module, 'unrestricted_reference_from_source', forbidden)
    monkeypatch.setattr(module, 'spin_orbital_integrals', forbidden, raising=False)
    actual = ci.UCIS(mf, frozen=frozen, solver='dense').run()
    assert np.all(actual.converged)
    np.testing.assert_allclose(actual.e, expected.e, atol=3e-10)
    a = np.concatenate([np.asarray(x).ravel() for x in actual.amplitudes])
    e = np.concatenate([np.asarray(x).ravel() for x in expected.amplitudes])
    np.testing.assert_allclose(np.outer(a, a), np.outer(e, e), atol=3e-10)


@pytest.mark.parametrize('source', ['rohf'], indirect=True)
def test_selected_ucis_keeps_the_nonstationary_rohf_boundary(source, monkeypatch):
    _, mf = source
    module = importlib.import_module('gradscf.ao2mo.scf')
    data = prepare_scf_source(mf, allow_rohf=True)
    bad = data._replace(focks=(data.focks[0].at[0, data.nocc[0]].add(.01), data.focks[1]))
    monkeypatch.setattr(module, 'prepare_scf_source', lambda *a, **kw: bad)
    api = importlib.import_module('gradscf.ci.api')

    def forbidden(*args, **kwargs):
        pytest.fail('ROHF stationarity validation built full spin ERIs')

    monkeypatch.setattr(api, 'unrestricted_reference_from_source', forbidden)
    with pytest.raises(ValueError, match='stationary'):
        ci.UCIS(mf, solver='dense').run()
